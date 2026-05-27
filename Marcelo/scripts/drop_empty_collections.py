import os
import sys
import chromadb

scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))

db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
client = chromadb.PersistentClient(path=db_path)

# Legacy / duplicated empty collections to remove. Add others here if needed.
TO_DROP = ["science_nature"]  # superseded by "science_and_nature"

existing = {c.name: c for c in client.list_collections()}

for name in TO_DROP:
    if name not in existing:
        print(f"  not found: '{name}'")
        continue
    count = existing[name].count()
    if count > 0:
        print(f"  SKIPPED '{name}': has {count} docs (not empty). Manually verify before dropping.")
        continue
    client.delete_collection(name=name)
    print(f"  dropped empty collection: '{name}'")

print("\nCollections now in DB:")
for c in client.list_collections():
    print(f"  - {c.name}: {c.count():,} docs")
