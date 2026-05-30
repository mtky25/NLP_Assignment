from src.guesser.guesser import Guesser
from typing import Any, Dict
from src.guesser.ingestion.loader import Loader
from src.guesser.engine.guesser_engine import GuesserEngine
from src.guesser.engine.configs import INFERENCE_MODEL, EMBEDDING_MODEL, PRE_LOAD_MODELS, FALLBACK_INFERENCE_MODEL, MATH_INFERENCE_MODEL, TRANSLATOR_MODEL
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
import re
import os
import random


class MarceloGuesser(Guesser):
    def __init__(self, config: ExperimentConfig,
                 db_path=None,
                 embedding_model_name=EMBEDDING_MODEL,
                 inference_model_name=None,
                 fallback_model_name: str = None,
                 math_model_name: str = None,
                 translator_model_name: str = None,
                 theme: str = "Science and Nature"):
        super().__init__(
            config,
            mode=getattr(config, 'mode', 'text'),
            transcription_model=getattr(config, 'transcription_model', 'tiny'),
        )

        # Priority: explicit arg > config > global default
        if inference_model_name is None:
            inference_model_name = getattr(self.config, 'inference_model', INFERENCE_MODEL)

        _fallback  = fallback_model_name  or FALLBACK_INFERENCE_MODEL
        _math      = math_model_name      or MATH_INFERENCE_MODEL
        _translator = translator_model_name or TRANSLATOR_MODEL

        if db_path is None:
            # Default to the context_db folder relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "context_db")

        # Build the pre-load list from the actual models this experiment will use
        # (deduplicated so no model is warmed up twice)
        _pre_load = list(dict.fromkeys([
            _translator,
            inference_model_name,
            _fallback,
            _math,
        ]))

        self.engine = GuesserEngine(
            inference_model_name,
            db_path,
            embedding_model_name,
            temperature=0.0,
            theme=theme,
            debug=self.config.debug,
            pre_load_models=_pre_load,
            fallback_model_override=_fallback,
            math_model_override=_math,
            translator_model_override=_translator,
        )
        

        if self.config.approach != ApproachType.HYBRID:
            self.engine.approach_type = self.config.approach

    def debug_chat(self, text: str) -> str:
        """
        Directly chat with the LLM for debugging.
        """
        return self.engine.chat(text)

    def infer_answer(self, question: Question, theme: str = None, game_session: Any = None) -> int:
        self.search_time = 0.0
        self.reasoning_time = 0.0
        
        if theme:
            self.engine.set_theme(theme)
            
            if self.config.approach != ApproachType.HYBRID:
                self.engine.approach_type = self.config.approach

        try:
            result = self.engine.answer(question)
            
            self.search_time = result.get('search_time', 0.0)
            self.reasoning_time = result.get('reasoning_time', 0.0)
            self.last_chunks = result.get('chunks_metadata', [])

            raw = result["answer"].strip()
            
            if self.engine.theme in ["maths", "math"]:
                print(f"\n--- [MATH DEBUG] RAW RESPONSE ---\n{raw}\n--------------------------------")

            # Prioritize PoT result if it's already a single digit string
            if len(raw) == 1 and raw.isdigit() and 0 <= int(raw) <= 3:
                return int(raw)

            match = re.search(r"FINAL_INDEX:.*?([0-3])", raw, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Fallback for complex math responses
            if self.engine.theme in ["maths", "math"]:
                if "</scratchpad>" in raw:
                    post_scratchpad = raw.split("</scratchpad>")[-1]
                    match = re.search(r"([0-3])", post_scratchpad)
                else:
                    match = re.search(r"([0-3])", raw)
                    
                if not match:
                    match = re.search(r"(?:index|option|answer):?\s*([0-3])", raw, re.IGNORECASE)
            else:
                # General case: find the first digit that looks like an answer
                match = re.search(r"[0-3]", raw)
            
            if match:
                return int(match.group(1) if match.groups() else match.group())

            # FALLBACK: model refused or output unparseable. Random guess beats crashing the game.
            fallback = random.randint(0, 3)
            print(f" [Guesser] Could not extract digit from response ('{raw[:80]}...'). Random fallback: {fallback}")
            return fallback
        except Exception as e:
            # Capture partial times from engine even on failure/timeout
            self.search_time = getattr(self.engine, 'last_search_time', 0.0)
            self.reasoning_time = getattr(self.engine, 'last_reasoning_time', 0.0)
            # Last-resort fallback: random guess instead of propagating the error
            fallback = random.randint(0, 3)
            print(f" [Guesser] Exception during inference ({e}). Random fallback: {fallback}")
            return fallback
        
