from abc import ABC, abstractmethod
from typing import Iterable
from llama_index.core import Document

class BaseExtractor(ABC):
    """
    Generic interface for all RAG data extractors.
    """
    
    @abstractmethod
    def extract(self, limit: int = None) -> Iterable[Document]:
        """
        Must extract data from the source and return an iterable of Documents.
        """
        pass