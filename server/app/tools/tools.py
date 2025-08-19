import json
from typing import Dict, Any, List

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
    user_id: int = Field(description="The numeric database ID for the user.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        raise ValueError("user_id must be an integer or a string containing an integer")

@tool(args_schema=RiskToolInput)
def analyze_risk_profile(user_id: int) -> Dict[str, Any]:
    """
    Analyzes a user's risk profile based on their ID by fetching their data
    and evaluating it against fixed financial risk factors.
    Returns a structured JSON object with the complete risk assessment.
    """
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
    user_id: int = Field(description="The numeric database ID for the user.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        raise ValueError("user_id must be an integer or a string containing an integer")

@tool(args_schema=FraudToolInput)
def detect_fraud(user_id: int) -> Dict[str, Any]:
    """
    Analyzes a user's recent transactions based on their ID to detect potential fraud.
    Evaluates data against fixed rules and returns a structured JSON assessment.
    """
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
        **SYSTEM:** You are a Deterministic Fraud Detection System...
        **TASK:** Analyze the transaction data below against "Fraud Detection Rules".
        **FRAUD DETECTION RULES:**
        1.  **High Anomaly Score**: `Anomaly_Score` > 0.90.
        2.  **Explicit Suspicious Flag**: `Suspicious_Flag` is 'Yes'.
        3.  **Unusual Transaction Amount**: `Transaction_Amount` > $5,000.
        4.  **Mismatched Location**: `Geo_Location` country differs from user's home `Country`.
        **DATA TO ANALYZE:**
        ```json
        {json.dumps(user_data, indent=2)}
        ```
        **OUTPUT INSTRUCTIONS:**
        Return a single JSON object with this structure: {{"is_fraudulent": boolean, "confidence_score": float, "rules_triggered": [], "recommended_action": "Auto-Approve/Flag for Manual Review"}}
        """
        response = json_llm.invoke(prompt)
        return json.loads(response.content)
    finally:
        db.close()

# --- Tool 3: Pension Projection ---
class ProjectionToolInput(BaseModel):
    user_id: int = Field(description="The numeric database ID for the user.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        raise ValueError("user_id must be an integer or a string containing an integer")

@tool(args_schema=ProjectionToolInput)
def project_pension(user_id: int) -> Dict[str, Any]:
    """
    Calculates a comprehensive pension overview for a user including current savings, goal progress, status, years remaining, and savings rate.
    It fetches savings and contribution data and performs deterministic calculations.
    """
    print(f"\n--- TOOL: Running Comprehensive Pension Overview for User ID: {user_id} ---")
    db: Session = SessionLocal()
    try:
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}

        # Get user data
        current_savings = pension_data.current_savings or 0
        annual_contribution = pension_data.total_annual_contribution or 0
        return_rate = pension_data.annual_return_rate or 5.0
        target_retirement_age = 65
        user_age = pension_data.age or 35  # Default age if not provided
        annual_income = pension_data.annual_income or 50000  # Default income if not provided
        
        # Calculate years remaining until retirement
        years_to_retirement = max(0, target_retirement_age - user_age)
        
        # Calculate goal amount (using 4% rule: 25x annual expenses)
        # Assuming annual expenses are 80% of income
        annual_expenses = annual_income * 0.8
        retirement_goal = annual_expenses * 25
        
        # Calculate progress to goal
        progress_percentage = min(100, (current_savings / retirement_goal) * 100) if retirement_goal > 0 else 0
        
        # Determine status based on progress and age
        if years_to_retirement <= 0:
            status = "At Retirement Age"
        elif progress_percentage >= 80:
            status = "On Track"
        elif progress_percentage >= 50:
            status = "Good Progress"
        elif progress_percentage >= 25:
            status = "Needs Attention"
        else:
            status = "Needs Attention"
        
        # Calculate savings rate as percentage of income
        savings_rate_percentage = (annual_contribution / annual_income) * 100 if annual_income > 0 else 0
        
        # Calculate projected balance at retirement using the new projection service
        if years_to_retirement > 0:
            # Prepare data for projection service
            user_data = {
                "Pension_Type": "Defined Contribution",
                "Current_Savings": current_savings,
                "Total_Annual_Contribution": annual_contribution,
                "Annual_Return_Rate": return_rate,
                "Age": user_age,
                "Retirement_Age_Goal": target_retirement_age,
                "Fees_Percentage": 0.5  # Default 0.5% fees
            }
            
            scenario_params = {
                "new_retirement_age": target_retirement_age,
                "new_annual_contribution": annual_contribution,
                "new_return_rate": return_rate
            }
            
            # Import and use the projection service
            try:
                from ..agents.services.projection import run_projection_agent
                projection_result = run_projection_agent(user_data, scenario_params)
                
                if "error" not in projection_result:
                    future_value = projection_result.get("inflation_adjusted_projection", 0)
                    nominal_value = projection_result.get("nominal_projection", 0)
                else:
                    # Fallback to simple calculation if projection service fails
                    future_value = current_savings * ((1 + return_rate / 100) ** years_to_retirement) + \
                                  annual_contribution * ((((1 + return_rate / 100) ** years_to_retirement) - 1) / (return_rate / 100))
                    nominal_value = future_value
            except ImportError:
                # Fallback to simple calculation if projection service not available
                future_value = current_savings * ((1 + return_rate / 100) ** years_to_retirement) + \
                              annual_contribution * ((((1 + return_rate / 100) ** years_to_retirement) - 1) / (return_rate / 100))
                nominal_value = future_value
        else:
            future_value = current_savings
            nominal_value = current_savings

        return {
            "current_savings": f"${current_savings:,.0f}",
            "retirement_goal": f"${retirement_goal:,.0f}",
            "progress_to_goal": f"{progress_percentage:.1f}%",
            "status": status,
            "years_remaining": years_to_retirement,
            "target_retirement_age": target_retirement_age,
            "savings_rate": f"{savings_rate_percentage:.0f}%",
            "annual_income": f"${annual_income:,.0f}",
            "annual_contribution": f"${annual_contribution:,.0f}",
            "projected_balance_at_retirement": f"${future_value:,.0f}",
            "nominal_projection": f"${nominal_value:,.0f}",
            "assumed_annual_return": f"{return_rate}%",
            "user_age": user_age,
            "pension_type": "Defined Contribution",
            "inflation_adjusted": True
        }
    finally:
        db.close()
        
        
@tool("knowledge_base_search")
def knowledge_base_search(input: str) -> str:
    """
    Search the knowledge base (and optionally user docs) for a query.
    Accepted input:
    - "user_id=1, query=..."
    - "query=..."
    - any free text → treated as the query
    """
    # Extract user_id and query from a free-form input string
    uid_match = re.search(r"(?:^|[,\s])user_id\s*[:=]\s*(\d+)(?:\b|,)", input, flags=re.IGNORECASE)
    user_id = int(uid_match.group(1)) if uid_match else None
    q_match = re.search(r"(?:^|[,\s])query\s*[:=]\s*(\".*?\"|\'.*?\'|[^,]+)", input, flags=re.IGNORECASE)
    if q_match:
        q_raw = q_match.group(1).strip()
        if (q_raw.startswith('"') and q_raw.endswith('"')) or (q_raw.startswith("'") and q_raw.endswith("'")):
            q_raw = q_raw[1:-1]
        query = q_raw.strip()
    else:
        query = input.strip()

    print(f"\n--- TOOL: Searching knowledge base for query: '{query}' ---")
    context = ""
    try:
        # 1. Search the general FAQ collection
        faq_collection = get_or_create_collection("faq_collection")
        faq_results = query_collection(faq_collection, query_texts=[query], n_results=5)
        if faq_results.get('documents') and faq_results['documents'][0]:
            # Prefer metadata answers if present
            answers = []
            for docs, metas in zip(faq_results.get('documents', []), faq_results.get('metadatas', [])):
                for meta in metas:
                    if isinstance(meta, dict) and meta.get('answer'):
                        answers.append(meta['answer'])
            if answers:
                context += "--- Relevant General Info ---\n" + "\n".join(answers[:3])
            else:
                context += "--- Relevant General Info ---\n" + "\n".join(faq_results['documents'][0])

        # 2. Search the user's private document collection only when user_id is present
        if user_id is not None:
            user_collection_name = f"user_{user_id}_docs"
            user_collection = get_or_create_collection(user_collection_name)
            user_doc_results = query_collection(user_collection, query_texts=[query], n_results=3)
            if user_doc_results.get('documents') and user_doc_results['documents'][0]:
                context += "\n\n--- Relevant Info From Your Documents ---\n" + "\n".join(user_doc_results['documents'][0])

        return context if context else "No relevant information found in the knowledge base."
    except Exception as e:
        return f"An error occurred while searching the knowledge base: {e}"

# --- The complete list of tools available to the agents ---
all_pension_tools = [analyze_risk_profile, detect_fraud, project_pension, knowledge_base_search]


