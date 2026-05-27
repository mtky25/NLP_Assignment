import os
import sys
import chromadb

scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))

db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
client = chromadb.PersistentClient(path=db_path)

existing = {c.name for c in client.list_collections()}

TO_ADD = ["news", "philosophy_psychology"]

for name in TO_ADD:
    if name in existing:
        print(f"  already exists: '{name}'")
    else:
        client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        print(f"  created: '{name}'")

print("\nCollections now in DB:")
for c in client.list_collections():
    print(f"  - {c.name}")
