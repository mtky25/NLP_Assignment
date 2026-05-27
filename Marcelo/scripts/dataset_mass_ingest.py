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

from src.guesser.ingestion.pipelines.dataset_ingestion import DatasetIngestionPipeline
from src.guesser.ingestion.extractors.configs import (
    HF_SCIENCE_DATASET, 
    HF_PHYSICS_DATASET_2,
    HF_BIOLOGY_DATASET,
    HF_CHEMISTRY_DATASET,
    HF_MATH_GSM8K_DATASET,
    HF_MATH500_DATASET,
    HF_ANCIENT_HISTORY_DATASET,
    HF_ANCIENT_HISTORY_DATASET_2,
    HF_ENTERTAINMENT_DATASET,
    HF_PHILOSOPHY_DATASET,
    HF_PSYCHOLOGY_DATASET

)

from src.guesser.engine.configs import EMBEDDING_MODEL
from src.guesser.context_db.collections import (
    COLLECTION_MATHS,
    COLLECTION_SCIENCE_NATURE,
    COLLECTION_HISTORY_POLITICS,
    COLLECTION_ENTERTAINMENT,
    COLLECTION_NEWS,
    COLLECTION_PHILOSOPHY_PSYCHOLOGY
)
from dotenv import load_dotenv

async def dataset_mass_ingest():
    db_path = os.path.join(marcelo_root, "src", "guesser", "context_db")
    env_path = os.path.join(marcelo_root, ".env")
    load_dotenv(env_path)
    hf_token = os.getenv("HF_TOKEN")
    
    # Configuration: (dataset_config, collection_name, limit)
    tasks = [
        # Science (SciQ)
        #(HF_SCIENCE_DATASET, COLLECTION_SCIENCE_NATURE, 20000),
        # (HF_CHEMISTRY_DATASET, COLLECTION_SCIENCE_NATURE, 20000),
        # (HF_PHYSICS_DATASET_2, COLLECTION_SCIENCE_NATURE, 20000),
        # (HF_BIOLOGY_DATASET, COLLECTION_SCIENCE_NATURE, 20000),
        
        # Maths (GSM8K)
        #(HF_MATH_GSM8K_DATASET, COLLECTION_MATHS, 10000),
        #(HF_MATH500_DATASET, COLLECTION_MATHS, 10000),

        # Ancient History
        #(HF_ANCIENT_HISTORY_DATASET, COLLECTION_HISTORY_POLITICS, 12000),
        #(HF_ANCIENT_HISTORY_DATASET_2, COLLECTION_HISTORY_POLITICS, 12000),
        
        
        # Entertainment
        #(HF_ENTERTAINMENT_DATASET,COLLECTION_ENTERTAINMENT, 50000)

        # Philosophy
        #(HF_PHILOSOPHY_DATASET,COLLECTION_PHILOSOPHY_PSYCHOLOGY, 15000),

        # Psychology
        (HF_PSYCHOLOGY_DATASET,COLLECTION_PHILOSOPHY_PSYCHOLOGY, 100000)
    ]

    print(f"Starting Dataset Mass Ingestion into: {db_path}")

    for config, collection, limit in tasks:
        print(f"\n>>> Ingesting {limit} items into '{collection}' from dataset: {config['hf_dataset']}")
        
        pipeline = DatasetIngestionPipeline(
            dataset_config=config,
            embedding_model=EMBEDDING_MODEL,
            db_path=db_path,
            colletion_name=collection,
            hf_token=hf_token
        )
        
        await pipeline.process(limit=limit, batch_size=50, random_seed=42)

    print("\n✅ DATASET MASS INGESTION COMPLETED!")

if __name__ == "__main__":
    asyncio.run(dataset_mass_ingest())
