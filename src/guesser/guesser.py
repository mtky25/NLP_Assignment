from abc import ABC, abstractmethod
from typing import Dict, Any
import pprint

class Guesser(ABC):     
    """
    Generic interface for all Guessers.
    """
    def __init__(self,config:Dict[str, Any]):
        if not isinstance(config, dict):
            raise ValueError("config must be a dict")
        self.config = config


    def print(self):
        pprint.pprint(self.config)
 
    @abstractmethod
    def infer_answer(self,question) -> int:
        pass
    