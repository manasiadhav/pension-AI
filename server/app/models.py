from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import validates
from .database import Base
from .security import hash_password

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False)

    @validates('password')
    def _hash_password(self, key, password):
        return hash_password(password)

class PensionData(Base):
    __tablename__ = "pension_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # only residents

    # Demographics & financial info
    age = Column(Integer)
    gender = Column(String(10))
    country = Column(String(50))
    employment_status = Column(String(50))
    annual_income = Column(Float)
    current_savings = Column(Float)
    retirement_age_goal = Column(Integer)
    risk_tolerance = Column(String(20))
    contribution_amount = Column(Float)
    contribution_frequency = Column(String(20))
    employer_contribution = Column(Float)
    total_annual_contribution = Column(Float)
    years_contributed = Column(Integer)
    investment_type = Column(String(50))
    fund_name = Column(String(100))
    annual_return_rate = Column(Float)
    volatility = Column(Float)
    fees_percentage = Column(Float)
    projected_pension_amount = Column(Float)
    expected_annual_payout = Column(Float)
    inflation_adjusted_payout = Column(Float)
    years_of_payout = Column(Integer)
    survivor_benefits = Column(String(10))
    transaction_id = Column(String(50))
    transaction_amount = Column(Float)
    transaction_date = Column(DateTime)
    suspicious_flag = Column(String(10))
    anomaly_score = Column(Float)
    marital_status = Column(String(20))
    number_of_dependents = Column(Integer)
    education_level = Column(String(50))
    health_status = Column(String(50))
    life_expectancy_estimate = Column(Integer)
    home_ownership_status = Column(String(50))
    debt_level = Column(Float)
    monthly_expenses = Column(Float)
    savings_rate = Column(Float)
    investment_experience_level = Column(String(50))
    financial_goals = Column(String(100))
    insurance_coverage = Column(String(10))
    portfolio_diversity_score = Column(Float)
    tax_benefits_eligibility = Column(String(10))
    government_pension_eligibility = Column(String(10))
    private_pension_eligibility = Column(String(10))
    pension_type = Column(String(50))
    withdrawal_strategy = Column(String(50))
    transaction_channel = Column(String(50))
    ip_address = Column(String(50))
    device_id = Column(String(50))
    geo_location = Column(String(100))
    time_of_transaction = Column(DateTime)
    transaction_pattern_score = Column(Float)
    previous_fraud_flag = Column(String(10))
    account_age = Column(Integer)

class AdvisorClient(Base):
    __tablename__ = "advisor_clients"

    id = Column(Integer, primary_key=True, index=True)
    advisor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    resident_id = Column(Integer, ForeignKey("users.id"), nullable=False)