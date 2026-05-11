import os
import sys
import chromadb

# Ensure the root directory is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

marcelo_root = os.path.abspath(os.path.dirname(__file__))
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

def inspect_db():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    
    if not os.path.exists(db_path):
        print(f"Error: Database path not found at {db_path}")
        return

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()

    print(f"\n--- ChromaDB Status Report ---")
    print(f"Location: {db_path}")
    print(f"Total Collections: {len(collections)}")
    print("-" * 45)

    total_items = 0
    for col in collections:
        count = col.count()
        total_items += count
        print(f"Collection: {col.name:<25} | Items: {count:,}")
        
        # Peek at the last 2 items to verify content
        if count > 0:
            peek = col.get(limit=2, include=['documents', 'metadatas'])
            for i in range(len(peek['documents'])):
                text_snippet = peek['documents'][i][:150].replace('\n', ' ')
                print(f"   [Sample {i+1}] Metadata: {peek['metadatas'][i]}")
                print(f"   [Sample {i+1}] Text: {text_snippet}...")
            print("-" * 15)

    print("-" * 45)
    print(f"TOTAL ITEMS IN DB: {total_items:,}")
    print(f"---------------------------------------------\n")

if __name__ == "__main__":
    inspect_db()
