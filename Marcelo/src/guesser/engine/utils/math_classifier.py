import time
from src.guesser.engine.llmprovider import get_llm
from src.guesser.engine.prompts import MATH_CLASSIFIER_PROMPT_STR


class MathClassifier:
    """
    Lightweight classifier that decides whether a math question is
    'calculation' (route to PoT) or 'theory' (route to direct inference).
    """

    def __init__(self, model_name: str = "qwen2.5:0.5b", debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self.llm = get_llm(
            model_name=model_name,
            temperature=0.0,
            num_predict=3,
            stop=["\n"]
        )

    def classify(self, question_text: str) -> str:
        """
        Returns 'calculation' or 'theory'. Defaults to 'calculation' on ambiguity.
        """
        prompt = MATH_CLASSIFIER_PROMPT_STR.format(question=question_text)

        if self.debug:
            print(f"[DEBUG] MathClassifier model: {self.model_name}")
            print(f"[DEBUG] MathClassifier question: {question_text[:120]}...")

        start = time.time()
        try:
            response = self.llm.complete(prompt)
            raw = str(response).strip().lower()
        except Exception as e:
            elapsed = time.time() - start
            print(f" [MathClassifier] Error after {elapsed:.2f}s: {e}. Defaulting to 'calculation'.")
            return "calculation"
        elapsed = time.time() - start

        if "theory" in raw:
            label = "theory"
        elif "calc" in raw:
            label = "calculation"
        else:
            label = "calculation"
            if self.debug:
                print(f"[DEBUG] MathClassifier ambiguous output '{raw}' -> defaulted to 'calculation'")

        if self.debug:
            print(f"[DEBUG] MathClassifier raw='{raw}' -> '{label}' (latency {elapsed:.2f}s)")
        return label
