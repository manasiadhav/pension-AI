def run_projection_agent(user_data: dict, scenario_params: dict) -> dict:
    """
    A pure calculation tool that projects pension value.
    It checks the Pension_Type and runs the appropriate logic.
    """
    try:
        pension_type = user_data.get("Pension_Type")
        
        if not pension_type:
            return {"error": "Pension type not specified"}

        if pension_type == "Defined Contribution":
            # --- Logic for Defined Contribution Plans (Market-based Projection) ---
            
            # 1. Get parameters with validation
            retirement_age = int(scenario_params.get("new_retirement_age") or user_data.get("Retirement_Age_Goal", 65))
            annual_contribution = float(scenario_params.get("new_annual_contribution") or user_data.get("Total_Annual_Contribution", 0.0))
            annual_return_rate = float(scenario_params.get("new_return_rate") or user_data.get("Annual_Return_Rate", 5.0))
            
            # 2. Get fixed data with validation
            current_savings = float(user_data.get("Current_Savings", 0) or 0)
            fees = float(user_data.get("Fees_Percentage", 0) or 0) / 100
            current_age = int(user_data.get("Age", 0) or 0)

            # 3. Validate inputs
            if current_age >= retirement_age:
                return {
                    "pension_type": "Defined Contribution",
                    "error": "Current age must be less than retirement age",
                    "current_age": current_age,
                    "retirement_age": retirement_age
                }

            # 4. Perform the Nominal Projection Calculation
            net_return_rate = (annual_return_rate / 100) - fees
            years_to_grow = retirement_age - current_age

            # Future Value (FV) calculation with safety checks
            if net_return_rate != 0:
                fv_of_savings = current_savings * ((1 + net_return_rate) ** years_to_grow)
                fv_of_contributions = annual_contribution * ((((1 + net_return_rate) ** years_to_grow) - 1) / net_return_rate)
            else:
                fv_of_savings = current_savings
                fv_of_contributions = annual_contribution * years_to_grow
                
            nominal_projection = fv_of_savings + fv_of_contributions

            # 5. Perform the Inflation-Adjusted Calculation
            assumed_inflation_rate = 0.025  # Assume a standard 2.5% inflation rate
            if years_to_grow > 0:
                inflation_adjusted_projection = nominal_projection / ((1 + assumed_inflation_rate) ** years_to_grow)
            else:
                inflation_adjusted_projection = nominal_projection

            return {
                "pension_type": "Defined Contribution",
                "nominal_projection": round(nominal_projection, 2),
                "inflation_adjusted_projection": round(inflation_adjusted_projection, 2),
                "parameters_used": {
                    "retirement_age": retirement_age,
                    "annual_contribution": annual_contribution,
                    "annual_return_rate": annual_return_rate,
                    "current_age": current_age,
                    "years_to_grow": years_to_grow,
                    "fees_percentage": fees * 100
                }
            }

        elif pension_type == "Defined Benefit":
            # --- Logic for Defined Benefit Plans (Guaranteed Payout) ---
            return {
                "pension_type": "Defined Benefit",
                "projected_amount": user_data.get("Projected_Pension_Amount"),
                "expected_annual_payout": user_data.get("Expected_Annual_Payout"),
                "survivor_benefits": user_data.get("Survivor_Benefits"),
                "parameters_used": {
                    "pension_type": "Defined Benefit"
                }
            }
        
        else:
            return {"error": f"Unknown pension type: {pension_type}"}

    except Exception as e:
        print(f"An error occurred in the projection agent: {e}")
        return {"error": f"Could not calculate the projection: {str(e)}"}
