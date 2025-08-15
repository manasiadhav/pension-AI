from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

# ---------------------------
# User Schemas
# ---------------------------
class UserBase(BaseModel):
    full_name: str
    email: EmailStr
    role: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        orm_mode = True

# ---------------------------
# Pension Data Schemas
# ---------------------------
class PensionDataBase(BaseModel):
    age: Optional[int]
    gender: Optional[str]
    country: Optional[str]
    employment_status: Optional[str]
    annual_income: Optional[float]
    current_savings: Optional[float]
    retirement_age_goal: Optional[int]
    risk_tolerance: Optional[str]
    contribution_amount: Optional[float]
    contribution_frequency: Optional[str]
    employer_contribution: Optional[float]
    total_annual_contribution: Optional[float]
    years_contributed: Optional[int]
    investment_type: Optional[str]
    fund_name: Optional[str]
    annual_return_rate: Optional[float]
    volatility: Optional[float]
    fees_percentage: Optional[float]
    projected_pension_amount: Optional[float]
    expected_annual_payout: Optional[float]
    inflation_adjusted_payout: Optional[float]
    years_of_payout: Optional[int]
    survivor_benefits: Optional[str]
    transaction_id: Optional[str]
    transaction_amount: Optional[float]
    transaction_date: Optional[datetime]
    suspicious_flag: Optional[str]
    anomaly_score: Optional[float]
    marital_status: Optional[str]
    number_of_dependents: Optional[int]
    education_level: Optional[str]
    health_status: Optional[str]
    life_expectancy_estimate: Optional[int]
    home_ownership_status: Optional[str]
    debt_level: Optional[float]
    monthly_expenses: Optional[float]
    savings_rate: Optional[float]
    investment_experience_level: Optional[str]
    financial_goals: Optional[str]
    insurance_coverage: Optional[str]
    portfolio_diversity_score: Optional[float]
    tax_benefits_eligibility: Optional[str]
    government_pension_eligibility: Optional[str]
    private_pension_eligibility: Optional[str]
    pension_type: Optional[str]
    withdrawal_strategy: Optional[str]
    transaction_channel: Optional[str]
    ip_address: Optional[str]
    device_id: Optional[str]
    geo_location: Optional[str]
    time_of_transaction: Optional[datetime]
    transaction_pattern_score: Optional[float]
    previous_fraud_flag: Optional[str]
    account_age: Optional[int]

class PensionDataCreate(PensionDataBase):
    pass

class PensionDataResponse(PensionDataBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True

# ---------------------------
# Advisor Client Schemas
# ---------------------------
class AdvisorClientBase(BaseModel):
    advisor_id: int
    resident_id: int

class AdvisorClientCreate(AdvisorClientBase):
    pass

class AdvisorClientResponse(AdvisorClientBase):
    id: int

    class Config:
        orm_mode = True
