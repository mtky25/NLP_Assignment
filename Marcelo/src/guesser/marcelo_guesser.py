from src.guesser.guesser import Guesser
from src.guesser.ingestion.loader import Loader
from src.guesser.engine.guesser_engine import GuesserEngine
from src.guesser.configs import INFERENCE_MODEL, EMBEDDING_MODEL
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
import re
import os


class MarceloGuesser(Guesser):
    def __init__(self, config: ExperimentConfig, 
                 db_path=None,
                 embedding_model_name=EMBEDDING_MODEL,
                 inference_model_name=INFERENCE_MODEL,
                 theme: str = "Science and Nature"):        
        super().__init__(config)

        if db_path is None:
            # Default to the context_db folder relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "context_db")

        self.engine = GuesserEngine(inference_model_name,db_path,embedding_model_name,temperature=0.0,theme=theme)
        
        # If the approach is RAG or DIRECT_LLM, force that for all themes.
        # If it is HYBRID, let the Router decide the best approach per theme.
        if self.config.approach != ApproachType.HYBRID:
            self.engine.approach_type = self.config.approach

    def debug_chat(self, text: str) -> str:
        """
        Directly chat with the LLM for debugging.
        """
        return self.engine.chat(text)

    def infer_answer(self, question: Question, theme: str = None) -> int:
        self.search_time = 0.0
        self.reasoning_time = 0.0
        
        if theme:
            self.engine.set_theme(theme)
            
            # Re-apply global approach override if not HYBRID
            if self.config.approach != ApproachType.HYBRID:
                self.engine.approach_type = self.config.approach

        try:
            result = self.engine.answer(question)
            
            # Propagate separated timing metrics
            self.search_time = result.get('search_time', 0.0)
            self.reasoning_time = result.get('reasoning_time', 0.0)

            raw = result["answer"].strip()
            
            # 1. Try to find the structured FINAL_INDEX (new math format)
            match = re.search(r"FINAL_INDEX:\s*([0-3])", raw)
            if match:
                return int(match.group(1))
            
            # 2. Fallback to general digit search (original behavior for other themes)
            match = re.search(r"[0-3]", raw)
            if match:
                return int(match.group())
            
            raise ValueError(f"Could not extract answer digit from: {raw[:80]}")
        except Exception:
            # Capture partial times from engine even on failure/timeout
            self.search_time = getattr(self.engine, 'last_search_time', 0.0)
            self.reasoning_time = getattr(self.engine, 'last_reasoning_time', 0.0)
            raise
        
