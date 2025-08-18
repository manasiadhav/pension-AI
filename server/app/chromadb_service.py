# File: app/chromadb_service.py
import chromadb
from typing import List, Dict, Any, Optional
import os
import json
from uuid import uuid4
from datetime import datetime, timezone

# Use a persistent client so data survives across process runs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(CHROMA_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_create_collection(name: str):
    """
    Retrieves an existing collection or creates a new one if it doesn't exist.
    """
    return client.get_or_create_collection(name=name)

def add_documents_to_collection(
    collection,
    documents: List[str],
    ids: List[str],
    metadatas: List[Dict[str, Any]]
):
    """
    Adds a list of documents with their IDs and metadata to a collection.
    """
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Added {len(documents)} documents to ChromaDB collection '{collection.name}'.")

def query_collection(
    collection,
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Queries a collection for relevant documents based on a query string.
    """
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where
    )

# --- NEW FUNCTION FOR CONVERSATIONAL MEMORY ---
def log_conversation_to_chroma(user_id: int, user_query: str, agent_answer: Dict[str, Any]):
    """
    Logs a user's query and the agent's answer to a dedicated ChromaDB collection
    for that user, creating a conversational memory.
    """
    try:
        collection_name = f"user_{user_id}_conversations"
        collection = get_or_create_collection(name=collection_name)
        document = user_query
        metadata = {
            "user_id": user_id,
            "query": user_query,
            "answer": json.dumps(agent_answer), # Store the full JSON answer
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        entry_id = str(uuid4())
        add_documents_to_collection(
            collection=collection,
            documents=[document],
            ids=[entry_id],
            metadatas=[metadata]
        )
        print(f"✅ Logged conversation for user {user_id} to ChromaDB.")
    except Exception as e:
        print(f"❌ Failed to log conversation to ChromaDB: {e}")