from src.guesser.guesser import Guesser
from Marcelo.src.guesser.ingestion.loader import Loader
from Marcelo.src.guesser.engine.guesser_engine import GuesserEngine
from Marcelo.src.guesser.configs import INFERENCE_MODEL, EMBEDDING_MODEL
from Marcelo.src.guesser.engine.prompts import MCQ_PROMPT
from src.models import ExperimentConfig
from src.millionaire_client.models import Question

class MarceloGuesser(Guesser):
    def __init__(self, config: ExperimentConfig, 
                 db_path="Marcelo/src/guesser/context_db/",
                 collection_name="Science_Nature",
                 embedding_model_name=EMBEDDING_MODEL,
                 inference_model_name=INFERENCE_MODEL):        
        super().__init__(config)

        self.index = Loader(db_path,embedding_model_name).get_index(collection_name)
        self.engine = GuesserEngine(self.index,inference_model_name,prompt=MCQ_PROMPT)
        
    def infer_answer(self, question: Question) -> int:
        import re
        formatted = super().format_question_for_llm(question)
        result = self.engine.answer_question(formatted)
        raw = result["answer"].strip()
        match = re.search(r"[0-3]", raw)
        if match:
            return int(match.group())
        raise ValueError(f"Could not extract answer digit from: {raw[:80]}")
        
