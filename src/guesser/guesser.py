from abc import ABC, abstractmethod
from typing import Dict, Any
import pprint
from src.millionaire_client.models import Question
from src.models import ExperimentConfig
class Guesser(ABC):     
    """
    Generic interface for all Guessers.
    """
    def __init__(self, config:ExperimentConfig):
        if not isinstance(config, ExperimentConfig):
            raise ValueError(f"config must be a {type(ExperimentConfig)}")
        self.config = config


    def print(self):
        pprint.pprint(self.config)
 
    @abstractmethod
    def infer_answer(self, question: Question) -> int:
        pass
    
    def format_question_for_llm(self, question: Question) -> str:
        """
        Format question
         """
        prompt_lines = [f"Question: {question.text}\n", "Options:"]
        
        for index, option in enumerate(question.options):
            prompt_lines.append(f"[{index}] {option.text}")
            
        return "\n".join(prompt_lines)