import os
import sys
import chromadb
import shutil

# Ensure the root directory and Marcelo implementation are in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_root, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

from src.guesser.context_db.collections import (
    COLLECTION_MATHS,
    COLLECTION_SCIENCE_NATURE,
    COLLECTION_HISTORY_POLITICS,
    COLLECTION_ENTERTAINMENT,
    COLLECTION_NEWS,
    COLLECTION_PHILOSOPHY_PSYCHOLOGY,
    COLLECTION_DEFAULT
)

def recreate_db():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    
    # 1. Force delete the existing folder to ensure clean L2 -> Cosine transition
    if os.path.exists(db_path):
        print(f"Purging old database at: {db_path}")
        shutil.rmtree(db_path)
    
    print(f"Creating persistent ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    collections = [
        COLLECTION_MATHS,
        COLLECTION_SCIENCE_NATURE,
        COLLECTION_HISTORY_POLITICS,
        COLLECTION_ENTERTAINMENT,
        COLLECTION_NEWS,
        COLLECTION_PHILOSOPHY_PSYCHOLOGY,
        COLLECTION_DEFAULT
    ]
    
    for col_name in collections:
        print(f" -> Initializing collection: '{col_name}' with Cosine Similarity...")
        client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"}
        )

    print("\n✅ Database folder recreated successfully with all 5 collections.")
    print("Unified Names: science_nature, entertainment, maths, history_politics.")
    print("Metric: Cosine Similarity.")

if __name__ == "__main__":
    recreate_db()
