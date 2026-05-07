from Marcelo.src.guesser.context_db.collections import (
    COLLECTION_ENTERTAINMENT,
    COLLECTION_HISTORY,
    COLLECTION_MATH,
    COLLECTION_SCIENCE,
    COLLECTION_DEFAULT 
)
from Marcelo.src.guesser.engine.prompts import (
    MCQ_PROMPT_ENTERTAINMENT,
    MCQ_PROMPT_HISTORY_POLITICS,
    MCQ_PROMPT_MATHS,
    MCQ_PROMPT_SCIENCE_NATURE
)



class Router:
    def __init__(self, theme: str):
        self.theme = theme.lower().strip()

    def route(self):
        """
        Return a tuple (collection_name, prompt) based on the theme.
        """
        # Normalize theme: replace underscores with spaces and strip
        normalized_theme = self.theme.replace("_", " ")
        
        mapping = {
            "entertainment": (COLLECTION_ENTERTAINMENT, MCQ_PROMPT_ENTERTAINMENT),
            "ancient history and politics": (COLLECTION_HISTORY, MCQ_PROMPT_HISTORY_POLITICS),
            "maths": (COLLECTION_MATH, MCQ_PROMPT_MATHS),
            "science and nature": (COLLECTION_SCIENCE, MCQ_PROMPT_SCIENCE_NATURE),
        }

        return mapping.get(normalized_theme, (COLLECTION_DEFAULT, MCQ_PROMPT_SCIENCE_NATURE))