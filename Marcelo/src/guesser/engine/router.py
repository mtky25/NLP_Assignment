from src.models import ThemeConfig
from src.guesser.engine.configs import (
    MATHS_CONFIG,
    HISTORY_POLITICS_CONFIG,
    SCIENCE_NATURE_CONFIG,
    ENTERTAINMENT_CONFIG,
    NEWS_CONFIG,
    PHILOSOPHY_PSYCHOLOGY_CONFIG,
    DEFAULT_CONFIG
)

class Router:
    def __init__(self, theme: str):
        self.theme = theme.lower().strip()

    def route(self) -> ThemeConfig:
        """
        Return a ThemeConfig based on keyword matching in the theme string.
        """
        if "math" in self.theme:
            return MATHS_CONFIG

        if "history" in self.theme or "politics" in self.theme:
            return HISTORY_POLITICS_CONFIG

        if "science" in self.theme or "nature" in self.theme:
            return SCIENCE_NATURE_CONFIG

        if "entertainment" in self.theme:
            return ENTERTAINMENT_CONFIG

        if "news" in self.theme:
            return NEWS_CONFIG

        if "philosophy" in self.theme or "psychology" in self.theme:
            return PHILOSOPHY_PSYCHOLOGY_CONFIG

        return DEFAULT_CONFIG
