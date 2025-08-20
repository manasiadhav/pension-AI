import json
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field, validator, root_validator
import re

from ..database import SessionLocal
from .. import models
from ..chromadb_service import get_or_create_collection, query_collection
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set Google API key for LangChain with fallback
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
else:
    # Set a dummy key for testing (will fail gracefully)
    os.environ["GOOGLE_API_KEY"] = "dummy_key_for_testing"

json_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    response_mime_type="application/json"
)

# --- Tool 1: Risk Analysis ---
class RiskToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=RiskToolInput)
def analyze_risk_profile(user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes a user's risk profile based on their ID by fetching their data
    and evaluating it against fixed financial risk factors.
    Returns a structured JSON object with the complete risk assessment.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Running Risk Analysis for User ID: {user_id} ---")
    db: Session = SessionLocal()
    try:
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}

        user_data = {
            "Annual_Income": pension_data.annual_income,
            "Debt_Level": pension_data.debt_level,
            "Risk_Tolerance": pension_data.risk_tolerance,
            "Volatility": pension_data.volatility,
            "Portfolio_Diversity_Score": pension_data.portfolio_diversity_score,
            "Health_Status": pension_data.health_status
        }
        prompt = f"""
        **SYSTEM:** You are a Methodical Financial Risk Analyst System...
        **TASK:** Analyze the user's data below...
        **RISK ANALYSIS FACTORS:**
        1.  **Market Risk Mismatch**: `Risk_Tolerance` is 'Low' but `Volatility` > 3.5.
        2.  **Concentration Risk**: `Portfolio_Diversity_Score` < 0.5.
        3.  **High Debt-to-Income Ratio**: `Debt_Level` > 50% of `Annual_Income`.
        4.  **Longevity & Health Risk**: `Health_Status` is 'Poor'.
        **DATA TO ANALYZE:**
        ```json
        {json.dumps(user_data, indent=2)}
        ```
        **OUTPUT INSTRUCTIONS:**
        Return a single JSON object with this structure: {{"risk_level": "Low/Medium/High", "risk_score": float, "positive_factors": [], "risks_identified": [], "summary": "..."}}
        """
        response = json_llm.invoke(prompt)
        return json.loads(response.content)
    finally:
        db.close()

# --- Tool 2: Fraud Detection ---
class FraudToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=FraudToolInput)
def detect_fraud(user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes a user's recent transactions based on their ID to detect potential fraud.
    Evaluates data against fixed rules and returns a structured JSON assessment.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Running Fraud Detection for User ID: {user_id} ---")
    db: Session = SessionLocal()
    try:
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}
        
        user_data = {
            "Country": pension_data.country,
            "Transaction_Amount": pension_data.transaction_amount,
            "Suspicious_Flag": pension_data.suspicious_flag,
            "Anomaly_Score": pension_data.anomaly_score,
            "Geo_Location": pension_data.geo_location
        }
        
        prompt = f"""
        **SYSTEM:** You are a Financial Fraud Detection System...
        **TASK:** Analyze the user's transaction data below...
        **FRAUD DETECTION FACTORS:**
        1.  **Geographic Anomaly**: `Geo_Location` doesn't match `Country`.
        2.  **Suspicious Amount**: `Transaction_Amount` is unusually high/low.
        3.  **Flagged Transaction**: `Suspicious_Flag` is True.
        4.  **High Anomaly Score**: `Anomaly_Score` > 0.8.
        **DATA TO ANALYZE:**
        ```json
        {json.dumps(user_data, indent=2)}
        ```
        **OUTPUT INSTRUCTIONS:**
        Return a single JSON object with this structure: {{"fraud_risk": "Low/Medium/High", "fraud_score": float, "suspicious_factors": [], "recommendations": [], "summary": "..."}}
        """
        response = json_llm.invoke(prompt)
        return json.loads(response.content)
    finally:
        db.close()

# --- Tool 3: Pension Projection ---
class ProjectionToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=ProjectionToolInput)
def project_pension(user_id: int = None) -> Dict[str, Any]:
    """
    Provides a comprehensive pension overview including current savings, goal progress,
    years remaining, savings rate, and projected balance at retirement.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Running Comprehensive Pension Overview for User ID: {user_id} ---")
    db: Session = SessionLocal()
    try:
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}
        
        # Calculate comprehensive pension overview
        current_savings = pension_data.current_savings or 0
        annual_income = pension_data.annual_income or 0
        age = pension_data.age or 0
        retirement_goal = pension_data.retirement_age_goal or 65
        annual_contribution = pension_data.contribution_amount or 0
        employer_contribution = pension_data.employer_contribution or 0
        total_annual_contribution = annual_contribution + employer_contribution
        
        # Calculate years to retirement
        years_to_retirement = max(0, retirement_goal - age)
        
        # Calculate retirement goal (example: 10x annual income)
        retirement_goal_amount = annual_income * 10
        
        # Calculate progress
        progress_percentage = min(100, (current_savings / retirement_goal_amount) * 100) if retirement_goal_amount > 0 else 0
        
        # Determine status
        if age >= retirement_goal:
            status = "At Retirement Age"
        elif progress_percentage >= 80:
            status = "On Track"
        elif progress_percentage >= 50:
            status = "Good Progress"
        else:
            status = "Needs Attention"
        
        # Calculate savings rate
        savings_rate_percentage = (total_annual_contribution / annual_income) * 100 if annual_income > 0 else 0
        
        # Try to get projections from the projection service
        try:
            from ..agents.services.projection import run_projection_agent
            
            user_data = {
                "current_savings": current_savings,
                "annual_income": annual_income,
                "age": age,
                "retirement_age": retirement_goal,
                "annual_contribution": total_annual_contribution,
                "risk_tolerance": pension_data.risk_tolerance,
                "pension_type": pension_data.pension_type or "Defined Contribution"
            }
            
            scenario_params = {
                "inflation_rate": 0.025,  # 2.5% inflation
                "return_rate": 0.08,      # 8% annual return
                "years": years_to_retirement
            }
            
            projection_result = run_projection_agent(user_data, scenario_params)
            
            # Extract projection data
            projected_balance = projection_result.get("projected_balance", 0)
            nominal_projection = projection_result.get("nominal_projection", 0)
            inflation_adjusted = projection_result.get("inflation_adjusted", True)
            
        except Exception as e:
            print(f"Projection service unavailable: {e}")
            # Fallback calculations
            projected_balance = current_savings * (1.08 ** years_to_retirement) if years_to_retirement > 0 else current_savings
            nominal_projection = projected_balance
            inflation_adjusted = False
        
        return {
            "current_savings": f"${current_savings:,.0f}",
            "retirement_goal": f"${retirement_goal_amount:,.0f}",
            "progress_to_goal": f"{progress_percentage:.1f}%",
            "status": status,
            "years_remaining": years_to_retirement,
            "target_retirement_age": retirement_goal,
            "savings_rate": f"{savings_rate_percentage:.0f}%",
            "annual_income": f"${annual_income:,.0f}",
            "annual_contribution": f"${total_annual_contribution:,.0f}",
            "projected_balance_at_retirement": f"${projected_balance:,.0f}",
            "nominal_projection": f"${nominal_projection:,.0f}",
            "assumed_annual_return": "8.0%",
            "user_age": age,
            "pension_type": pension_data.pension_type or "Defined Contribution",
            "inflation_adjusted": inflation_adjusted
        }
        
    except Exception as e:
        return {"error": f"Error processing pension projection: {str(e)}"}
    finally:
        db.close()

# --- Tool 4: Knowledge Base Search ---
class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="The search query for the knowledge base.")
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

@tool(args_schema=KnowledgeSearchInput)
def knowledge_base_search(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Searches the knowledge base for relevant information about pensions, retirement planning,
    and financial advice. Returns structured information based on the query.
    This tool searches BOTH general pension knowledge AND the user's uploaded documents.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Searching Knowledge Base for User ID: {user_id} ---")
    
    all_results = []
    
    try:
        # SEARCH 1: General pension knowledge base
        print(f"🔍 Searching general pension knowledge base...")
        general_collection = get_or_create_collection("pension_knowledge")
        general_results = query_collection(general_collection, [query], n_results=2)
        
        # Handle ChromaDB results properly
        if general_results and isinstance(general_results, dict) and 'documents' in general_results:
            general_docs = general_results.get('documents', [])
            general_metadatas = general_results.get('metadatas', [])
            general_distances = general_results.get('distances', [])
            
            # ChromaDB returns nested lists, flatten them
            if general_docs and isinstance(general_docs[0], list):
                general_docs = general_docs[0]
            if general_metadatas and isinstance(general_metadatas[0], list):
                general_metadatas = general_metadatas[0]
            if general_distances and isinstance(general_distances[0], list):
                general_distances = general_distances[0]
            
            for i, (doc, metadata, distance) in enumerate(zip(general_docs, general_metadatas, general_distances)):
                if doc:  # Only add if document content exists
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": "General Pension Knowledge Base",
                        "type": "general_knowledge",
                        "relevance_score": 1 - distance if isinstance(distance, (int, float)) else 0.5
                    })
        
        # SEARCH 2: User's uploaded PDF documents
        print(f"🔍 Searching user's uploaded documents...")
        user_docs_collection = get_or_create_collection(f"user_{user_id}_docs")
        user_results = query_collection(user_docs_collection, [query], n_results=3)
        
        # Handle ChromaDB results properly
        if user_results:
            if isinstance(user_results, dict) and 'documents' in user_results:
                user_docs = user_results.get('documents', [])
                user_metadatas = user_results.get('metadatas', [])
                user_distances = user_results.get('distances', [])
                
                # ChromaDB returns nested lists, flatten them
                if user_docs and isinstance(user_docs[0], list):
                    user_docs = user_docs[0]
                if user_metadatas and isinstance(user_metadatas[0], list):
                    user_metadatas = user_metadatas[0]
                if user_distances and isinstance(user_distances[0], list):
                    user_distances = user_distances[0]
                    
            elif isinstance(user_results, list):
                # If it's a list, assume it's the documents directly
                user_docs = user_results
                user_metadatas = [{}] * len(user_results)
                user_distances = [0.0] * len(user_results)
            else:
                user_docs = []
                user_metadatas = []
                user_distances = []
            
            for i, (doc, metadata, distance) in enumerate(zip(user_docs, user_metadatas, user_distances)):
                if doc:  # Only add if document content exists
                    # Handle distance calculation safely
                    try:
                        if isinstance(distance, (int, float)):
                            relevance_score = 1 - distance
                        elif isinstance(distance, list) and len(distance) > 0:
                            first_distance = distance[0]
                            if isinstance(first_distance, (int, float)):
                                relevance_score = 1 - first_distance
                            else:
                                relevance_score = 0.5
                        else:
                            relevance_score = 0.5
                    except (TypeError, ValueError):
                        relevance_score = 0.5
                    
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": metadata.get('source', f'User {user_id} Document') if metadata else f'User {user_id} Document',
                        "type": "user_document",
                        "relevance_score": relevance_score
                    })
        
        # If no results found anywhere
        if not all_results:
            return {
                "found": False,
                "message": "No relevant information found in either the general knowledge base or your uploaded documents.",
                "suggestions": [
                    "Try rephrasing your question",
                    "Use more specific terms",
                    "Check if your question is related to pensions, retirement, or financial planning",
                    "Make sure you have uploaded relevant documents"
                ]
            }
        
        # Sort results by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update result numbers after sorting
        for i, result in enumerate(all_results):
            result["result"] = i + 1
        
        return {
            "found": True,
            "query": query,
            "results": all_results,
            "total_results": len(all_results),
            "search_summary": {
                "general_knowledge_results": len([r for r in all_results if r.get('type') == 'general_knowledge']),
                "user_document_results": len([r for r in all_results if r.get('type') == 'user_document']),
                "total_found": len(all_results)
            },
            "summary": f"Found {len(all_results)} relevant results for your query about '{query}'. "
                      f"This includes both general pension knowledge and information from your uploaded documents."
        }
        
    except Exception as e:
        return {"error": f"Error searching knowledge base: {str(e)}"}

# --- Tool 5: Document Analysis ---
class DocumentAnalysisInput(BaseModel):
    query: str = Field(description="The question or query about the uploaded document")
    user_id: Optional[int] = Field(description="User ID to identify which documents to search. If not provided, will be retrieved from context.")

@tool(args_schema=DocumentAnalysisInput)
def analyze_uploaded_document(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes the user's uploaded PDF documents to answer specific questions.
    This tool searches through all documents uploaded by the user and provides
    relevant information based on the query.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Analyzing Uploaded Documents for User ID: {user_id} ---")
    
    try:
        # Search user's uploaded documents
        user_docs_collection = get_or_create_collection(f"user_{user_id}_docs")
        user_results = query_collection(user_docs_collection, [query], n_results=5)
        
        # ChromaDB returns results in different formats, handle both
        if isinstance(user_results, dict):
            user_docs = user_results.get('documents', [])
            user_metadatas = user_results.get('metadatas', [])
            user_distances = user_results.get('distances', [])
            
            # ChromaDB returns nested lists, flatten them
            if user_docs and isinstance(user_docs[0], list):
                user_docs = user_docs[0]
            if user_metadatas and isinstance(user_metadatas[0], list):
                user_metadatas = user_metadatas[0]
            if user_distances and isinstance(user_distances[0], list):
                user_distances = user_distances[0]
                
        elif isinstance(user_results, list):
            # If it's a list, assume it's the documents directly
            user_docs = user_results
            user_metadatas = [{}] * len(user_results)
            user_distances = [0.0] * len(user_results)
        else:
            return {
                "found": False,
                "message": f"Unexpected result format from document search: {type(user_results)}",
                "suggestions": ["Try again or contact support"]
            }
        
        if not user_docs:
            return {
                "found": False,
                "message": "No uploaded documents found for analysis.",
                "suggestions": [
                    "Upload a PDF document first using the upload endpoint",
                    "Check if your document was successfully processed",
                    "Try a different query related to your uploaded content"
                ]
            }
        
        # Format the results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(user_docs, user_metadatas, user_distances)):
            if doc:  # Only add if document content exists
                # Handle distance calculation safely
                try:
                    if isinstance(distance, (int, float)):
                        relevance_score = 1 - distance
                    elif isinstance(distance, list) and len(distance) > 0:
                        first_distance = distance[0]
                        if isinstance(first_distance, (int, float)):
                            relevance_score = 1 - first_distance
                        else:
                            relevance_score = 0.5
                    else:
                        relevance_score = 0.5
                except (TypeError, ValueError):
                    relevance_score = 0.5
                
                formatted_results.append({
                    "result": i + 1,
                    "content": doc,
                    "source": metadata.get('source', f'User {user_id} Document') if metadata else f'User {user_id} Document',
                    "relevance_score": relevance_score,
                    "analysis": f"Document chunk {i+1} contains relevant information for your query: '{query}'"
                })
        
        # Sort by relevance
        formatted_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update result numbers after sorting
        for i, result in enumerate(formatted_results):
            result["result"] = i + 1
        
        return {
            "found": True,
            "query": query,
            "document_analysis": True,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "summary": f"Successfully analyzed your uploaded documents for '{query}'. "
                      f"Found {len(formatted_results)} relevant document sections.",
            "capabilities": [
                "Search through all your uploaded PDF documents",
                "Extract relevant information based on your questions",
                "Provide context-aware answers from your documents",
                "Support for multiple document types and formats"
            ]
        }
        
    except Exception as e:
        return {"error": f"Error analyzing uploaded documents: {str(e)}"}

# --- Tool 6: Knowledge Base Search (Enhanced) ---
class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="The search query for the knowledge base.")
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

@tool(args_schema=KnowledgeSearchInput)
def knowledge_base_search(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Searches the knowledge base for relevant information about pensions, retirement planning,
    and financial advice. Returns structured information based on the query.
    This tool searches BOTH general pension knowledge AND the user's uploaded documents.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Searching Knowledge Base for User ID: {user_id} ---")
    
    all_results = []
    
    try:
        # SEARCH 1: General pension knowledge base
        print(f"🔍 Searching general pension knowledge base...")
        general_collection = get_or_create_collection("pension_knowledge")
        general_results = query_collection(general_collection, [query], n_results=2)
        
        # Handle ChromaDB results properly
        if general_results and isinstance(general_results, dict) and 'documents' in general_results:
            general_docs = general_results.get('documents', [])
            general_metadatas = general_results.get('metadatas', [])
            general_distances = general_results.get('distances', [])
            
            for i, (doc, metadata, distance) in enumerate(zip(general_docs, general_metadatas, general_distances)):
                if doc:  # Only add if document content exists
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": "General Pension Knowledge Base",
                        "type": "general_knowledge",
                        "relevance_score": 1 - distance if isinstance(distance, (int, float)) else 0.5
                    })
        
        # SEARCH 2: User's uploaded PDF documents
        print(f"🔍 Searching user's uploaded documents...")
        user_docs_collection = get_or_create_collection(f"user_{user_id}_docs")
        user_results = query_collection(user_docs_collection, [query], n_results=3)
        
        # Handle ChromaDB results properly
        if user_results:
            if isinstance(user_results, dict) and 'documents' in user_results:
                user_docs = user_results.get('documents', [])
                user_metadatas = user_results.get('metadatas', [])
                user_distances = user_results.get('distances', [])
            elif isinstance(user_results, list):
                # If it's a list, assume it's the documents directly
                user_docs = user_results
                user_metadatas = [{}] * len(user_results)
                user_distances = [0.0] * len(user_results)
            else:
                user_docs = []
                user_metadatas = []
                user_distances = []
            
            for i, (doc, metadata, distance) in enumerate(zip(user_docs, user_metadatas, user_distances)):
                if doc:  # Only add if document content exists
                    # Handle distance calculation safely
                    try:
                        if isinstance(distance, (int, float)):
                            relevance_score = 1 - distance
                        elif isinstance(distance, list) and len(distance) > 0:
                            first_distance = distance[0]
                            if isinstance(first_distance, (int, float)):
                                relevance_score = 1 - first_distance
                            else:
                                relevance_score = 0.5
                        else:
                            relevance_score = 0.5
                    except (TypeError, ValueError):
                        relevance_score = 0.5
                    
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": metadata.get('source', f'User {user_id} Document') if metadata else f'User {user_id} Document',
                        "type": "user_document",
                        "relevance_score": relevance_score
                    })
        
        # If no results found anywhere
        if not all_results:
            return {
                "found": False,
                "message": "No relevant information found in either the general knowledge base or your uploaded documents.",
                "suggestions": [
                    "Try rephrasing your question",
                    "Use more specific terms",
                    "Check if your question is related to pensions, retirement, or financial planning",
                    "Make sure you have uploaded relevant documents"
                ]
            }
        
        # Sort results by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update result numbers after sorting
        for i, result in enumerate(all_results):
            result["result"] = i + 1
        
        return {
            "found": True,
            "query": query,
            "results": all_results,
            "total_results": len(all_results),
            "search_summary": {
                "general_knowledge_results": len([r for r in all_results if r.get('type') == 'general_knowledge']),
                "user_document_results": len([r for r in all_results if r.get('type') == 'user_document']),
                "total_found": len(all_results)
            },
            "summary": f"Found {len(all_results)} relevant results for your query about '{query}'. "
                      f"This includes both general pension knowledge and information from your uploaded documents."
        }
        
    except Exception as e:
        return {"error": f"Error searching knowledge base: {str(e)}"}

# Global context variable for user_id (shared across the module)
import contextvars
_user_id_context = contextvars.ContextVar('user_id', default=None)

# Helper function to get user_id from context
def get_current_user_id_from_context() -> Optional[int]:
    """
    Get user_id from current request context.
    This should be implemented based on your authentication system.
    """
    try:
        print(f"🔍 Context Debug: Attempting to get user_id from context...")
        
        # Option 1: Request-scoped context (production-ready)
        current_user_id = _user_id_context.get()
        print(f"🔍 Context Debug: Request context value: {current_user_id}")
        
        if current_user_id is not None:
            print(f"🔍 Context: Retrieved user_id={current_user_id} from request context")
            return current_user_id
        
        # Option 2: Thread-local storage (fallback for testing)
        import threading
        user_id = getattr(threading.current_thread(), 'user_id', None)
        print(f"🔍 Context Debug: Thread context value: {user_id}")
        
        if user_id is not None:
            print(f"🔍 Context: Retrieved user_id={user_id} from thread context (testing)")
            return user_id
        
        print(f"🔍 Context: No user_id found in context")
        return None
        
    except Exception as e:
        print(f"Error getting user_id from context: {e}")
        return None

def set_request_user_id(user_id: int):
    """
    Set the current user_id for the current request context.
    This is what your FastAPI endpoint should call.
    """
    try:
        _user_id_context.set(user_id)
        print(f"🔍 Context: Set user_id={user_id} in request context")
    except Exception as e:
        print(f"Error setting user_id in request context: {e}")
        # Fallback to thread-local for testing
        set_current_user_id(user_id)

def clear_request_user_id():
    """
    Clear the current user_id from the request context.
    This should be called at the end of each request.
    """
    try:
        _user_id_context.set(None)
        print(f"🔍 Context: Cleared user_id from request context")
    except Exception as e:
        print(f"Error clearing user_id from request context: {e}")
        # Fallback to thread-local for testing
        clear_current_user_id()

def extract_user_id_from_input(input_value) -> Optional[int]:
    """
    Extract user_id from various input formats and clean them up
    """
    if input_value is None:
        return None
    
    # If it's already an integer, return it
    if isinstance(input_value, int):
        return input_value
    
    # If it's a string, try to extract the number
    if isinstance(input_value, str):
        # Remove common prefixes and extract just the number
        import re
        match = re.search(r'(\d+)', input_value)
        if match:
            user_id = int(match.group(1))
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from '{input_value}'")
            return user_id
    
    # If it's a dict, try to get user_id
    if isinstance(input_value, dict):
        user_id = input_value.get('user_id')
        if user_id:
            return extract_user_id_from_input(user_id)
    
    print(f"🔍 Input Cleanup: Could not extract user_id from '{input_value}'")
    return None

# Context manager for setting user_id during testing
import threading

def set_current_user_id(user_id: int):
    """Set the current user_id for the current thread (for testing)"""
    threading.current_thread().user_id = user_id

def clear_current_user_id():
    """Clear the current user_id for the current thread"""
    if hasattr(threading.current_thread(), 'user_id'):
        delattr(threading.current_thread(), 'user_id')

# Export all tools for agents to use
all_pension_tools = [
    analyze_risk_profile,
    detect_fraud,
    project_pension,
    knowledge_base_search,
    analyze_uploaded_document
]


