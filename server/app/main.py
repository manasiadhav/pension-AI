# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from .database import Base, engine, get_db
from . import models, security, schemas  # import Pydantic schemas

app = FastAPI(title="Pension AI")

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
