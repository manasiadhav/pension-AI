#!/usr/bin/env python3
"""
Check what data exists in the database
"""

def check_database():
    """Check what users and pension data exist"""
    
    try:
        from app.database import SessionLocal
        from app import models
        
        db = SessionLocal()
        
        print("🔍 Checking Database Contents...")
        
        # Check users
        users = db.query(models.User).all()
        print(f"\n👥 Users found: {len(users)}")
        for user in users:
            print(f"  - ID: {user.id}, Name: {user.full_name}, Email: {user.email}, Role: {user.role}")
        
        # Check pension data
        pension_data = db.query(models.PensionData).all()
        print(f"\n💰 Pension Data found: {len(pension_data)}")
        for data in pension_data:
            print(f"  - User ID: {data.user_id}, Age: {data.age}, Savings: ${data.current_savings or 0:,.0f}")
        
        # Check specific user
        if users:
            first_user_id = users[0].id
            print(f"\n🔍 Checking pension data for User ID: {first_user_id}")
            user_pension = db.query(models.PensionData).filter(models.PensionData.user_id == first_user_id).first()
            if user_pension:
                print(f"  ✅ Found pension data for user {first_user_id}")
                print(f"     Current Savings: ${user_pension.current_savings or 0:,.0f}")
                print(f"     Annual Income: ${user_pension.annual_income or 0:,.0f}")
                print(f"     Age: {user_pension.age or 'Not set'}")
            else:
                print(f"  ❌ No pension data found for user {first_user_id}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database()
