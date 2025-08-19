# main.py
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer
from sqlalchemy.orm import Session
from datetime import timedelta
import os
import shutil
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel

from .database import Base, engine, get_db
from . import models, security, schemas  # import Pydantic schemas

# --- NEW: Import the agent graph and ingestion utility ---
from .workflow import graph
from file_ingestion import ingest_pdf_to_chroma
from fastapi.responses import StreamingResponse

# --- NEW: Response Models for AI Chat ---
class FinalSummaryResponse(BaseModel):
    """Structured response containing summary and optional chart data."""
    summary: str
    chart_data: Optional[Dict[str, Any]] = None
    plotly_figures: Optional[Dict[str, Any]] = None
    chart_images: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    query: str

# Import the context management functions
from .tools.tools import set_request_user_id, clear_request_user_id

app = FastAPI(title="Pension AI API", version="1.0.0")

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Create tables
# ---------------------------
Base.metadata.create_all(bind=engine)

# ---------------------------
# Login endpoint
# ---------------------------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"user_id": user.id, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "env": "dev"}

# ---------------------------
# Resident endpoints
# ---------------------------
@app.get("/pension/me", response_model=list[schemas.PensionDataResponse])
def get_my_pension(current_user: models.User = Depends(security.get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "resident":
        raise HTTPException(status_code=403, detail="Only residents can view this endpoint")
    data = db.query(models.PensionData).filter(models.PensionData.user_id == current_user.id).all()
    return data

@app.post("/pension/me", response_model=schemas.PensionDataResponse)
def create_my_pension(pension_data: schemas.PensionDataCreate,
                      current_user: models.User = Depends(security.get_current_user),
                      db: Session = Depends(get_db)):
    if current_user.role != "resident":
        raise HTTPException(status_code=403, detail="Only residents can create data")
    pension = models.PensionData(**pension_data.dict(), user_id=current_user.id)
    db.add(pension)
    db.commit()
    db.refresh(pension)
    return pension

@app.put("/pension/me/{pension_id}", response_model=schemas.PensionDataResponse)
def update_my_pension(pension_id: int,
                      update_data: schemas.PensionDataCreate,
                      current_user: models.User = Depends(security.get_current_user),
                      db: Session = Depends(get_db)):
    if current_user.role != "resident":
        raise HTTPException(status_code=403, detail="Only residents can update data")
    pension = db.query(models.PensionData).filter(
        models.PensionData.id == pension_id,
        models.PensionData.user_id == current_user.id
    ).first()
    if not pension:
        raise HTTPException(status_code=404, detail="Pension data not found")
    for key, value in update_data.dict(exclude_unset=True).items():
        setattr(pension, key, value)
    db.commit()
    db.refresh(pension)
    return pension

# ---------------------------
# Advisor endpoints
# ---------------------------
@app.get("/pension/advisor", response_model=list[schemas.PensionDataResponse])
def get_advisor_clients(current_user: models.User = Depends(security.get_current_user),
                        db: Session = Depends(get_db)):
    if current_user.role != "advisor":
        raise HTTPException(status_code=403, detail="Only advisors can view this endpoint")
    client_ids = db.query(models.AdvisorClient.resident_id).filter(
        models.AdvisorClient.advisor_id == current_user.id
    ).all()
    client_ids = [c[0] for c in client_ids]
    data = db.query(models.PensionData).filter(models.PensionData.user_id.in_(client_ids)).all()
    return data

# ---------------------------
# Regulator endpoints
# ---------------------------
@app.get("/pension/regulator", response_model=list[schemas.PensionDataResponse])
def get_all_pension_data(current_user: models.User = Depends(security.get_current_user),
                         db: Session = Depends(get_db)):
    if current_user.role != "regulator":
        raise HTTPException(status_code=403, detail="Only regulators can view this endpoint")
    data = db.query(models.PensionData).all()
    return data

# ---------------------------
# User management endpoints (Admin-like)
# ---------------------------
@app.post("/users", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate,
                current_user: models.User = Depends(security.get_current_user),
                db: Session = Depends(get_db)):
    if current_user.role not in ["regulator", "advisor"]:
        raise HTTPException(status_code=403, detail="Only regulators/advisors can create users")
    db_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password=security.hash_password(user.password),
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users", response_model=list[schemas.UserResponse])
def get_users(current_user: models.User = Depends(security.get_current_user),
              db: Session = Depends(get_db)):
    if current_user.role not in ["regulator", "advisor"]:
        raise HTTPException(status_code=403, detail="Only regulators/advisors can view users")
    users = db.query(models.User).all()
    return users



# -----------------------------------------------------------------
# --- NEW SECTION 1: AI Knowledge Base Endpoint ---
# -----------------------------------------------------------------
@app.post("/pension/me/upload_document")
async def upload_pension_document(
    current_user: models.User = Depends(security.get_current_user),
    file: UploadFile = File(...)
):
    """
    Endpoint for an authenticated user to upload a PDF document.
    The document will be processed and ingested into the ChromaDB knowledge base.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Starting ingestion for user {current_user.id}...")
        result = ingest_pdf_to_chroma(file_path, user_id=current_user.id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {"status": "success", "filename": file.filename, "message": "Document ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# -----------------------------------------------------------------
# --- NEW SECTION 2: AI Agent Chat Endpoint ---
# -----------------------------------------------------------------
security = HTTPBearer()

# Models
class ChatRequest(BaseModel):
    message: str

class FinalSummaryResponse(BaseModel):
    summary: str
    chart_data: Optional[Dict[str, Any]] = None
    plotly_figures: Optional[Dict[str, Any]] = None
    chart_images: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Authentication dependency (this is what you'd implement for production)
async def get_current_user_id(token: str = Depends(security)) -> int:
    """
    Extract user_id from JWT token.
    This is where you'd implement your actual JWT validation logic.
    """
    try:
        # TODO: Implement actual JWT validation here
        # For now, we'll simulate by extracting from the token
        # In production, you'd decode the JWT and verify it
        
        # Simulate JWT decoding (replace with real implementation)
        if token.credentials == "valid_token_102":
            return 102
        elif token.credentials == "valid_token_103":
            return 103
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

@app.post("/chat", response_model=FinalSummaryResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    Main chat endpoint that processes pension queries.
    Uses request-scoped context for user authentication.
    """
    try:
        # Set user context for this request
        set_request_user_id(current_user_id)
        
        # Import and call the workflow
        from .workflow import graph
        result = graph.invoke({'messages': [('user', request.message)]})
        
        # Extract the final response
        final_response = result.get('final_response', {})
        
        # Return structured response
        return FinalSummaryResponse(
            summary=final_response.get('summary', 'No summary available'),
            chart_data=final_response.get('charts', {}),
            plotly_figures=final_response.get('plotly_figs', {}),
            chart_images=final_response.get('chart_images', {}),
            metadata={
                'user_id': current_user_id,
                'query': request.message,
                'workflow_completed': True
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")
    
    finally:
        # Clean up request context
        clear_request_user_id()

@app.get("/auth/status")
async def auth_status(current_user_id: int = Depends(get_current_user_id)):
    """Check authentication status"""
    return {"authenticated": True, "user_id": current_user_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Pension AI API"}

# -----------------------------------------------------------------
# --- NEW SECTION 3: Optional Streaming Chat Endpoint ---
# -----------------------------------------------------------------
@app.post("/chat/stream")
async def agent_chat_stream(
    request: ChatRequest,
    current_user: models.User = Depends(security.get_current_user)
):
    """
    Streaming endpoint for real-time updates from the AI workflow.
    Useful for showing progress and intermediate steps.
    """
    async def event_stream():
        query_with_context = f"User Info: id={current_user.id}, role={current_user.role}. Query: {request.query}"
        
        async for event in graph.astream({"messages": [("user", query_with_context)]}):
            print("--- STREAMING EVENT TO CLIENT ---", event)
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")