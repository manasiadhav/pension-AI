# server/app/setup_db.py

import os
import sys
from typing import List, Dict, Any
from uuid import uuid4

# Add the 'app' directory to the path so we can import our services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chromadb_service import get_or_create_collection, add_documents_to_collection

FAQS = [
    {"question": "What is a defined contribution plan?", "answer": "A defined contribution plan is a type of retirement plan where an employee and/or employer contribute to an individual's account. The final payout depends on the contributions made and the investment returns."},
    {"question": "How do I check my pension balance?", "answer": "You can check your pension balance by logging into your account on the member portal or by contacting your financial advisor."},
    {"question": "What is my investment risk tolerance?", "answer": "Your investment risk tolerance is a measure of your ability and willingness to take risks to achieve a financial goal. It is often determined by your age, financial situation, and comfort level with market volatility."},
    {"question": "Can I withdraw from my pension early?", "answer": "Early withdrawals from a pension plan are generally not recommended and may result in significant tax penalties. Consult with a financial advisor before making any decisions."},
    {"question": "What happens to my pension if I change jobs?", "answer": "When you change jobs, you have several options for your pension, including rolling it over to your new employer's plan, an individual retirement account (IRA), or leaving it with your old employer. You can also cash it out, but this is not advised."},
    {"question": "What is a beneficiary?", "answer": "A beneficiary is a person or entity you name to receive the assets from your pension plan after you pass away. It is important to keep your beneficiary information up to date."},
    {"question": "How do I update my personal information?", "answer": "You can update your personal information, such as your address or name, by logging into the member portal and navigating to your profile settings, or by contacting our support team."},
    {"question": "What is the difference between a traditional IRA and a Roth IRA?", "answer": "A traditional IRA may allow for tax-deductible contributions with taxes paid upon withdrawal in retirement. A Roth IRA uses after-tax contributions, allowing for tax-free withdrawals in retirement."},
    {"question": "What is an annuity?", "answer": "An annuity is a financial product sold by financial institutions that is designed to accept and grow funds from an individual and then, upon annuitization, pay out a stream of payments to the individual at a later point in time."},
    {"question": "How are pension contributions calculated?", "answer": "Pension contributions are often calculated as a percentage of your salary, with both you and your employer contributing to the fund. The exact percentage can vary based on your plan rules."},
    {"question": "What is a vesting period?", "answer": "A vesting period is the time you must work for an employer before you are fully entitled to their pension contributions. If you leave before this period ends, you may forfeit some or all of the employer contributions."},
    {"question": "How is my money invested?", "answer": "Your pension money is invested in a portfolio of assets, which can include stocks, bonds, and mutual funds, based on your chosen investment strategy and risk tolerance."},
    {"question": "What are my investment options?", "answer": "Your investment options typically include a range of funds with different risk profiles, such as low-risk bond funds, medium-risk balanced funds, and high-risk equity funds."},
    {"question": "How do I change my investment strategy?", "answer": "You can change your investment strategy by logging into your account and selecting a different portfolio. It's often recommended to speak with a financial advisor before making this change."},
    {"question": "What are administrative fees?", "answer": "Administrative fees are charges for managing your pension account, including record-keeping and customer service. These fees can impact your overall investment returns."},
    {"question": "Is my pension protected from market crashes?", "answer": "The value of your pension fund is subject to market fluctuations. It is not fully protected from market crashes, but diversification and a long-term investment horizon can help mitigate risk."},
    {"question": "How do I add or change my beneficiary?", "answer": "You can add or change a beneficiary by filling out and submitting a beneficiary designation form, available on the member portal or from your HR department."},
    {"question": "When can I retire?", "answer": "The standard retirement age is typically 65, but you may be able to retire earlier or later depending on your plan's rules and your financial goals."},
    {"question": "What is inflation and how does it affect my pension?", "answer": "Inflation is the rate at which the price of goods and services rises. It reduces the purchasing power of your pension, so it's important for your investments to grow at a rate higher than inflation."},
    {"question": "What is a pension portability?", "answer": "Pension portability allows you to move your accumulated pension benefits from one employer's plan to another or to a personal retirement account, making it easier to manage your retirement savings if you change jobs."},
    {"question": "What is a defined benefit plan?", "answer": "A defined benefit plan, or traditional pension plan, promises a specific monthly payment in retirement based on your salary, years of service, and age. The employer bears the investment risk."},
    {"question": "How do I get a tax statement for my contributions?", "answer": "You can download your tax statements from your account portal, typically available at the end of the fiscal year. You can also contact support for assistance."},
    {"question": "What if I can't make my contributions?", "answer": "If you are unable to make contributions, you may have the option to pause them temporarily. Contact your plan administrator to understand the rules and implications."},
    {"question": "What is a spousal consent form?", "answer": "A spousal consent form is a document that may be required for certain pension transactions, such as withdrawing funds or naming a non-spouse beneficiary. It ensures your spouse is aware and agrees to the action."},
    {"question": "How do I report a suspicious transaction?", "answer": "If you notice a suspicious transaction, you should immediately report it by contacting the fund's security or fraud department using the contact information on your statements."},
    {"question": "What is a pension fund manager?", "answer": "A pension fund manager is a professional who is responsible for the investment decisions of the pension fund, aiming to grow the assets to meet future pension obligations."},
    {"question": "What is compound interest?", "answer": "Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods. It is a key factor in pension growth."},
    {"question": "How often can I view my account statement?", "answer": "Account statements are typically available quarterly or annually, but you can usually check your balance and recent activity in real-time by logging into the member portal."},
    {"question": "What are the tax implications of my pension?", "answer": "The tax implications of your pension depend on the type of plan and your country's tax laws. Contributions may be tax-deductible, while withdrawals may be taxable. It is best to consult with a tax professional."},
    {"question": "What is the role of the Pension Fund Regulatory and Development Authority (PFRDA)?", "answer": "The PFRDA is the regulatory body in India that oversees pension funds. It is responsible for promoting and ensuring the orderly growth of the pension sector and protecting the interests of scheme members."},
]

def ingest_faqs_to_chroma():
    """
    Ingests a predefined list of FAQs into a dedicated ChromaDB collection.
    """
    print("🚀 Starting FAQ ingestion into ChromaDB...")
    try:
        # 1. Get or create a dedicated collection for FAQs
        faq_collection = get_or_create_collection(name="faq_collection")

        # 2. Prepare the data for ingestion
        documents = [faq["question"] for faq in FAQS]
        metadatas = [
            {
                "question": faq["question"],
                "answer": faq["answer"]
            }
            for faq in FAQS
        ]
        ids = [str(uuid4()) for _ in FAQS]

        # 3. Add the documents to the collection
        add_documents_to_collection(faq_collection, documents, ids, metadatas)
        
        print("✅ FAQ ingestion complete.")
        return {"status": "success", "message": "FAQ collection populated."}

    except Exception as e:
        print(f"❌ An error occurred during FAQ ingestion: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    from app.chromadb_service import get_or_create_collection, add_documents_to_collection
    ingest_faqs_to_chroma()