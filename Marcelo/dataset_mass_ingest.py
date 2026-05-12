import asyncio
import os
import sys

# Ensure the root directory is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the Marcelo path for implementation details
marcelo_root = os.path.abspath(os.path.dirname(__file__))
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

from src.guesser.ingestion.pipelines.dataset_ingestion import DatasetIngestionPipeline
from src.guesser.ingestion.extractors.configs import (
    HF_SCIENCE_DATASET, 
    HF_MATH_GSM8K_DATASET,
    HF_MATH500_DATASET,
    HF_ANCIENT_HISTORY_DATASET,
    HF_ENTERTAINMENT_DATASET
)

from src.guesser.configs import EMBEDDING_MODEL
from src.guesser.context_db.collections import (
    COLLECTION_MATH,
    COLLECTION_SCIENCE,
    COLLECTION_HISTORY,
    COLLECTION_ENTERTAINMENT
)

async def dataset_mass_ingest():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    
    # Configuration: (dataset_config, collection_name, limit)
    tasks = [
        # Science (SciQ)
        #(HF_SCIENCE_DATASET, COLLECTION_SCIENCE, 20000),
        
        # Maths (GSM8K)
        #(HF_MATH_GSM8K_DATASET, COLLECTION_MATH, 10000),
        #(HF_MATH500_DATASET, COLLECTION_MATH, 10000),

        # Ancient History
        #(HF_ANCIENT_HISTORY_DATASET, COLLECTION_HISTORY, 12000),
        
        # Entertainment
        (HF_ENTERTAINMENT_DATASET,COLLECTION_ENTERTAINMENT, 50000)
    ]

    print(f"Starting Dataset Mass Ingestion into: {db_path}")

    for config, collection, limit in tasks:
        print(f"\n>>> Ingesting {limit} items into '{collection}' from dataset: {config['hf_dataset']}")
        
        pipeline = DatasetIngestionPipeline(
            dataset_config=config,
            embedding_model=EMBEDDING_MODEL,
            db_path=db_path,
            colletion_name=collection
        )
        
        await pipeline.process(limit=limit, batch_size=50, random_seed=42)

    print("\n✅ DATASET MASS INGESTION COMPLETED!")

if __name__ == "__main__":
    asyncio.run(dataset_mass_ingest())
