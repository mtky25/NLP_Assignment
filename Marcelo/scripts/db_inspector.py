import os
import sys
import chromadb

# Ensure the root directory and Marcelo implementation are in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_root, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

def inspect_db():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    
    if not os.path.exists(db_path):
        print(f"Error: Database path not found at {db_path}")
        return

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()

    # Summary List at the beginning
    print(f"\n--- Database Summary ---")
    summary_total = 0
    for col in collections:
        c = col.count()
        summary_total += c
        print(f" - {col.name:<25}: {c:,} documents")
    print(f" TOTAL DOCUMENTS IN DB: {summary_total:,}")
    print("-" * 45)

    print(f"\n--- ChromaDB Detailed Report ---")
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
