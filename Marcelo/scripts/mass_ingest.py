import asyncio
import os
import sys

# Ensure the root directory and Marcelo implementation are in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_root, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

from src.guesser.ingestion.pipelines.zim_ingestion import ZimIngestionPipeline
from src.guesser.engine.configs import EMBEDDING_MODEL
from src.guesser.context_db.collections import (
    COLLECTION_MATHS,
    COLLECTION_HISTORY_POLITICS,
    COLLECTION_SCIENCE_NATURE,
    COLLECTION_ENTERTAINMENT
)

async def mass_ingest():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    raw_data_dir = os.path.join(marcelo_root, "src", "guesser", "data", "raw")
    
    # Configuration: (zim_file_path, collection_name, limit)
    tasks = [
        # History
        (os.path.join(raw_data_dir, "ancient_history_and_politics", "wikipedia_en_history_nopic_2026-04.zim"), 
         COLLECTION_HISTORY_POLITICS, 500),
        
        # Science
        (os.path.join(raw_data_dir, "science_nature", "wikipedia_en_physics_nopic_2026-04.zim"), 
         COLLECTION_SCIENCE_NATURE, 500),
        (os.path.join(raw_data_dir, "science_nature", "wikipedia_en_chemistry_nopic_2026-04.zim"), 
         COLLECTION_SCIENCE_NATURE, 500),
        
        # Maths
        (os.path.join(raw_data_dir, "maths", "wikipedia_en_mathematics_nopic_2026-03.zim"), 
         COLLECTION_MATHS, 500),
         
        # Entertainment / General
        (os.path.join(raw_data_dir, "wikipedia_en_simple_all_maxi_2026-02.zim"), 
         COLLECTION_ENTERTAINMENT, 500),
    ]

    print(f"Starting Mass Ingestion into: {db_path}")

    for file_path, collection, limit in tasks:
        if not os.path.exists(file_path):
            print(f"SKIPPING: File not found: {file_path}")
            continue
            
        print(f"\n>>> Ingesting {limit} random articles into '{collection}' from {os.path.basename(file_path)}")
        
        pipeline = ZimIngestionPipeline(
            raw_data_file_path=file_path,
            embedding_model=EMBEDDING_MODEL,
            db_path=db_path,
            colletion_name=collection
        )
        
        # Using a fixed seed for reproducibility, or dynamic for true randomness
        await pipeline.process(limit=limit, random_seed=42, batch_size=20, summary_only=True)

    print("\n✅ MASS INGESTION COMPLETED!")

if __name__ == "__main__":
    asyncio.run(mass_ingest())
