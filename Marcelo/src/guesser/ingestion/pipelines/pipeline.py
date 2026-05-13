from src.guesser.ingestion.extractors.zim_extractor import ZimExtractor
from src.guesser.ingestion.chunking import Chunker
from src.guesser.ingestion.loader import Loader
from abc import ABC,abstractmethod

class Pipeline(ABC):
    def __init__(self,raw_data_file_path,embedding_model,db_path,colletion_name):
        self.path = raw_data_file_path
        self.extractor = ZimExtractor(raw_data_file_path)
        self.loader = Loader(
                            db_path=db_path,
                            model_name=embedding_model,
                            collection_name=colletion_name)
    
    @abstractmethod
    def process(self, limit):
        pass