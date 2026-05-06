from datasets import load_dataset
import hashlib
from llama_index.core.schema import Document
from src.guesser.ingestion.extractors.extractor import BaseExtractor

class DatasetExtractor(BaseExtractor):
    def __init__(self,dataset_configs:dict):
        self.configs = dataset_configs
        self.documents = []         

    
    def extract(self,limit=None):
        dataset = load_dataset(
            self.configs["hf_dataset"],
            self.configs.get("hf_subset"),
            split=self.configs["split"])
        
        if limit == None:
            limit = len(dataset)

        for item in dataset.select(range(limit)):
            text = str(item.get(self.configs["col_text"], ""))

            if not text.strip():
                continue

            meta = {key: str(item.get(key, "")) for key in self.configs["cols_meta"]}
            doc = Document(text=text, metadata=meta)
            self.documents.append(doc)
        return self.documents
