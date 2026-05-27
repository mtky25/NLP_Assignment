from src.guesser.engine.llmprovider import get_llm
from src.guesser.engine.prompts import QUERY_TRANSLATOR_PROMPT_STR

class QueryTranslator:
    def __init__(self, model_name="llama3.2", debug:bool=False):
        self.model_name = model_name
        self.debug = debug
        self.llm = get_llm(
            model_name=model_name,
            temperature=0.0,
            num_predict=32,
            stop=["\n"]
        )

    def translate(self, question_text: str) -> str:
        """
        Translates a noisy question into search-optimized keywords.
        """
        if self.debug:
            print(f"[DEBUG] Translating Query (Model: {self.model_name})")
            
        prompt = QUERY_TRANSLATOR_PROMPT_STR.format(question=question_text)
        response = self.llm.complete(prompt)

        translated_query = str(response).strip()

        # Dedupe keywords (case-insensitive). Split by comma if present, else by whitespace.
        if "," in translated_query:
            parts = [p.strip() for p in translated_query.split(",") if p.strip()]
            sep = ", "
        else:
            parts = translated_query.split()
            sep = " "

        seen = set()
        deduped = []
        for kw in parts:
            k_lower = kw.lower()
            if k_lower not in seen:
                seen.add(k_lower)
                deduped.append(kw)

        # Cap at 3 keywords max
        deduped = deduped[:3]
        translated_query = sep.join(deduped)

        # Log for visibility
        print(f" [Translator] Raw: \"{question_text[:50]}...\" -> Keywords: \"{translated_query}\"")

        return translated_query
