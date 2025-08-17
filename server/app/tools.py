import json
from typing import Dict, Any, List

from sqlalchemy.orm import Session
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field

from .database import SessionLocal
from . import models
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Set Google API key for LangChain

json_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    response_mime_type="application/json"
)

# --- Tool 1: Risk Analysis ---
class RiskToolInput(BaseModel):
    user_id: int = Field(description="The numeric database ID for the user.")

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

@tool(args_schema=ProjectionToolInput)
def project_pension(user_id: int) -> Dict[str, Any]:
    """
    Calculates a simple 10-year pension projection for a user based on their ID.
    It fetches savings and contribution data and performs a deterministic calculation.
    """
    print(f"\n--- TOOL: Running Pension Projection for User ID: {user_id} ---")
    db: Session = SessionLocal()
    try:
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}

        # Perform calculation in reliable Python code, not with the LLM
        years = 10
        current_savings = pension_data.current_savings or 0
        annual_contribution = pension_data.total_annual_contribution or 0
        return_rate = pension_data.annual_return_rate or 5.0 # Default to 5%
        
        future_value = current_savings * ((1 + return_rate / 100) ** years) + \
                       annual_contribution * ((((1 + return_rate / 100) ** years) - 1) / (return_rate / 100))

        return {
            "projection_period_years": years,
            "starting_balance": f"${current_savings:,.2f}",
            "projected_balance": f"${future_value:,.2f}",
            "assumed_annual_return": f"{return_rate}%"
        }
    finally:
        db.close()

# List of all tools for the supervisor to use
all_pension_tools = [analyze_risk_profile, detect_fraud, project_pension]