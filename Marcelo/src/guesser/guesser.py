from src.guesser.guesser import Guesser
from Marcelo.src.guesser.ingestion.loader import Loader
from Marcelo.src.guesser.engine.guesser_engine import GuesserEngine
from Marcelo.src.guesser.configs import INFERENCE_MODEL, EMBEDDING_MODEL
from src.models import ExperimentConfig
from src.millionaire_client.models import Question
import re


class MarceloGuesser(Guesser):
    def __init__(self, config: ExperimentConfig, 
                 db_path="Marcelo/src/guesser/context_db/",
                 embedding_model_name=EMBEDDING_MODEL,
                 inference_model_name=INFERENCE_MODEL,
                 theme: str = "Science and Nature"):        
        super().__init__(config)

        self.engine = GuesserEngine(inference_model_name,db_path,embedding_model_name,temperature=0.0,theme=theme)
        
    def infer_answer(self, question: Question,theme: str=None) -> int:
        result = self.engine.answer(question)
        raw = result["answer"].strip()
        match = re.search(r"[0-3]", raw)
        if match:
            return int(match.group())
        raise ValueError(f"Could not extract answer digit from: {raw[:80]}")
        
