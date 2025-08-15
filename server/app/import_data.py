import os
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base
from . import models, security
import pandas as pd  # new import

# ---------------------------
# Create tables first
# ---------------------------
Base.metadata.create_all(bind=engine)

# ---------------------------
# Excel file path (absolute or relative)
# ---------------------------
EXCEL_FILE = r"C:\Users\marya\OneDrive\Desktop\pension_data.xlsx"

def import_data():
    db: Session = SessionLocal()
    try:
        # Read Excel using pandas
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")  # ensure openpyxl is installed
        df.fillna("", inplace=True)  # replace NaNs with empty strings

        for idx, row in df.iterrows():
            user_id_value = str(row.get("User_ID")).strip()
            if not user_id_value:
                print(f"⚠️ Row {idx+1} missing User_ID. Skipping.")
                continue

            user_email = f"{user_id_value}@example.com"

            # 1️⃣ Create or get user
            user = db.query(models.User).filter(models.User.email == user_email).first()
            if not user:
                user = models.User(
                    full_name=user_id_value,
                    email=user_email,
                    password=security.hash_password("password123"),
                    role="resident"
                )
                db.add(user)
                db.commit()
                db.refresh(user)

            # 2️⃣ Create pension data
            pension = models.PensionData(
                user_id=user.id,
                age=int(row.get("Age") or 0),
                gender=row.get("Gender"),
                country=row.get("Country"),
                employment_status=row.get("Employment_Status"),
                annual_income=float(row.get("Annual_Income") or 0.0),
                current_savings=float(row.get("Current_Savings") or 0.0),
                retirement_age_goal=int(row.get("Retirement_Age_Goal") or 0),
                risk_tolerance=row.get("Risk_Tolerance"),
                contribution_amount=float(row.get("Contribution_Amount") or 0.0),
                contribution_frequency=row.get("Contribution_Frequency"),
                employer_contribution=float(row.get("Employer_Contribution") or 0.0),
                total_annual_contribution=float(row.get("Total_Annual_Contribution") or 0.0),
                years_contributed=int(row.get("Years_Contributed") or 0),
                investment_type=row.get("Investment_Type"),
                fund_name=row.get("Fund_Name"),
                annual_return_rate=float(row.get("Annual_Return_Rate") or 0.0),
                volatility=float(row.get("Volatility") or 0.0),
                fees_percentage=float(row.get("Fees_Percentage") or 0.0),
                projected_pension_amount=float(row.get("Projected_Pension_Amount") or 0.0),
                expected_annual_payout=float(row.get("Expected_Annual_Payout") or 0.0),
                inflation_adjusted_payout=float(row.get("Inflation_Adjusted_Payout") or 0.0),
                years_of_payout=int(row.get("Years_of_Payout") or 0),
                survivor_benefits=row.get("Survivor_Benefits"),
                transaction_id=row.get("Transaction_ID"),
                transaction_amount=float(row.get("Transaction_Amount") or 0.0),
                transaction_date=row.get("Transaction_Date"),
                suspicious_flag=row.get("Suspicious_Flag"),
                anomaly_score=float(row.get("Anomaly_Score") or 0.0),
                marital_status=row.get("Marital_Status"),
                number_of_dependents=int(row.get("Number_of_Dependents") or 0),
                education_level=row.get("Education_Level"),
                health_status=row.get("Health_Status"),
                life_expectancy_estimate=int(row.get("Life_Expectancy_Estimate") or 0),
                home_ownership_status=row.get("Home_Ownership_Status"),
                debt_level=float(row.get("Debt_Level") or 0.0),
                monthly_expenses=float(row.get("Monthly_Expenses") or 0.0),
                savings_rate=float(row.get("Savings_Rate") or 0.0),
                investment_experience_level=row.get("Investment_Experience_Level"),
                financial_goals=row.get("Financial_Goals"),
                insurance_coverage=row.get("Insurance_Coverage"),
                portfolio_diversity_score=float(row.get("Portfolio_Diversity_Score") or 0.0),
                tax_benefits_eligibility=row.get("Tax_Benefits_Eligibility"),
                government_pension_eligibility=row.get("Government_Pension_Eligibility"),
                private_pension_eligibility=row.get("Private_Pension_Eligibility"),
                pension_type=row.get("Pension_Type"),
                withdrawal_strategy=row.get("Withdrawal_Strategy"),
                transaction_channel=row.get("Transaction_Channel"),
                ip_address=row.get("IP_Address"),
                device_id=row.get("Device_ID"),
                geo_location=row.get("Geo_Location"),
                time_of_transaction=row.get("Time_of_Transaction"),
                transaction_pattern_score=float(row.get("Transaction_Pattern_Score") or 0.0),
                previous_fraud_flag=row.get("Previous_Fraud_Flag"),
                account_age=int(row.get("Account_Age") or 0)
            )
            db.add(pension)
        db.commit()
        print("✅ Data import completed successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    import_data()
