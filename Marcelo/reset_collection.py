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

def list_collections(client):
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return
    print("\nAvailable collections:")
    for col in collections:
        print(f" - {col.name}")
    print("")

def reset_collection(collection_name):
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    
    if not os.path.exists(db_path):
        print(f"Error: Database path not found at {db_path}")
        return

    client = chromadb.PersistentClient(path=db_path)
    
    try:
        # Check if it exists first
        collections = [c.name for c in client.list_collections()]
        if collection_name not in collections:
            print(f"❌ Error: Collection '{collection_name}' does not exist.")
            list_collections(client)
            return

        client.delete_collection(name=collection_name)
        print(f"✅ Collection '{collection_name}' has been successfully deleted and reset.")
        print("It will be recreated as an empty collection next time you run an ingestion.")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python Marcelo/reset_collection.py <collection_name>")
        
        # Connect just to show available collections
        db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
        if os.path.exists(db_path):
            client = chromadb.PersistentClient(path=db_path)
            list_collections(client)
    else:
        target = sys.argv[1]
        reset_collection(target)
