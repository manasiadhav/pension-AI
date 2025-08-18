# File: verify_chroma.py
import sys
import os
import shutil
from app.database import SessionLocal
from app.models import User, PensionData
from app.import_data import import_data
from app.file_ingestion import ingest_pdf_to_chroma
from app.chromadb_service import get_or_create_collection, query_collection
from app.setup_chroma_db import ingest_faqs_to_chroma
from app.tools.tools import analyze_risk_profile, detect_fraud, project_pension, knowledge_base_search
from app.workflow import graph
from langchain_core.messages import HumanMessage
import asyncio
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))



# Clean up any existing ChromaDB data to start fresh (if using persistent client)
# If using in-memory, this is not needed
# try:
#     shutil.rmtree("./chroma-db")
# except FileNotFoundError:
#     pass

# Create a dummy PDF file for testing
dummy_pdf_content = """
My Pension Statement
This document outlines my pension details.
My name is U1000. My account number is P-12345.
The policy on early withdrawals states that a 10% penalty applies to any withdrawal before age 59.5.
This penalty is in addition to any income tax owed.
"""
dummy_file_path = "test_doc.pdf"
from reportlab.pdfgen import canvas
c = canvas.Canvas(dummy_file_path)
c.drawString(100, 750, dummy_pdf_content)
c.save()
print("✅ Created a dummy PDF for testing.")

# A. First, ingest the general FAQs
print("\n--- Step 1: Ingesting general FAQs into ChromaDB... ---")
ingest_faqs_to_chroma()
print("FAQ ingestion complete.")

# B. Now, ingest a user's document
user_id = 1000
print(f"\n--- Step 2: Ingesting a dummy PDF for user {user_id}... ---")
ingest_pdf_to_chroma(dummy_file_path, user_id)
print("Dummy PDF ingestion complete.")

# C. Finally, test the search
print("\n--- Step 3: Testing Retrieval-Augmented Generation (RAG) search... ---")
faq_collection = get_or_create_collection("faq_collection")
user_doc_collection = get_or_create_collection(f"user_{user_id}_docs")

# Test a general query
general_query = "What is a benefit plan?"
faq_results = query_collection(faq_collection, query_texts=[general_query], n_results=1)
print(f"🔍 Query: '{general_query}'")
print(f"Found in FAQ collection: {faq_results['documents']}")

# Test a user-specific query
user_query = "What is the penalty for early withdrawal?"
user_results = query_collection(user_doc_collection, query_texts=[user_query], n_results=1)
print(f"🔍 Query: '{user_query}'")
print(f"Found in user's documents: {user_results['documents']}")

# Clean up the dummy file
os.remove(dummy_file_path)
print("✅ Cleaned up dummy PDF.")