import random
from datasets import load_dataset
import hashlib
from llama_index.core.schema import Document
from src.guesser.ingestion.extractors.extractor import BaseExtractor

class DatasetExtractor(BaseExtractor):
    def __init__(self, dataset_configs: dict, hf_token: str = None):
        self.configs = dataset_configs
        self.hf_token = hf_token

    
    def extract(self, limit=None, starting_id=0, random_seed=None):
        dataset = load_dataset(
            self.configs["hf_dataset"],
            self.configs.get("hf_subset"),
            split=self.configs["split"],
            token=self.hf_token
        )
        
        total_items = len(dataset)
        
        # Determine the range of indices to consider
        indices = list(range(starting_id, total_items))
        
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(indices)
            
        if limit is not None:
            indices = indices[:limit]

        filters = self.configs.get("filters", {})

        for idx in indices:
            item = dataset[idx]
            
            # Apply filters if they exist
            skip_item = False
            for col, allowed_values in filters.items():
                val = str(item.get(col, ""))
                
                if isinstance(allowed_values, list):
                    if val not in allowed_values:
                        skip_item = True
                        break
                else:
                    if val != str(allowed_values):
                        skip_item = True
                        break
            
            if skip_item:
                continue

            # Handle col_text as a single string or a list of columns
            col_text_config = self.configs.get("col_text", "")
            
            if isinstance(col_text_config, list):
                # Combine multiple columns into one text block
                text_parts = []
                for col in col_text_config:
                    col_val = str(item.get(col, "")).strip()
                    if col_val:
                        # Capitalize first letter of col name for better formatting
                        label = col.replace("_", " ").title()
                        text_parts.append(f"{label}: {col_val}")
                text = "\n".join(text_parts)
            else:
                # Original behavior for single column
                text = str(item.get(col_text_config, ""))

            if not text.strip():
                continue

            doc_id = hashlib.md5(text.encode()).hexdigest()
            meta = {key: str(item.get(key, "")) for key in self.configs["cols_meta"]}
            doc = Document(text=text, metadata=meta, doc_id=doc_id)
            yield doc
