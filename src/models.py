from __future__ import annotations

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from uuid import uuid4


class ApproachType(str, Enum):
    DIRECT_LLM = "direct_llm"
    RAG = "rag"
    HYBRID = "hybrid"
    POT = "pot"
    SEARCH = "search"


class CompetitionTheme(str, Enum):
    ENTERTAINMENT = "Entertainment"
    ANCIENT_HISTORY_POLITICS = "Ancient History & Politics"
    SCIENCE_AND_NATURE = "Science & Nature"
    MATHS = "Maths"
    NEWS = "News"
    PHILOSOPHY_PSYCHOLOGY = "Philosophy and Psychology"


class QuestionOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"
    ERROR = "error"  

@dataclass
class ExperimentConfig:
    experiment_id: str = field(init=False, default_factory=lambda: uuid4().hex)
    username: str = ""
    notes: str = ""
    approach: Optional[ApproachType] = None
    inference_model: str = ""
    inference_model_size: str = ""
    debug: bool = False
    mode: str = "text"
    transcription_model: str = "tiny"

    is_rag: bool = False
    embedding_model: Optional[str] = None
    embedding_model_size: Optional[str] = None

    mean_question_accuracy: float = 0.0
    mean_time: float = 0.0
    mean_search_time: float = 0.0
    mean_reasoning_time: float = 0.0
    mean_transcription_time: float = 0.0

    entertainment_mean_time: float = 0.0
    entertainment_mean_question_accuracy: float = 0.0
    entertainment_mean_search_time: float = 0.0
    entertainment_mean_reasoning_time: float = 0.0
    entertainment_mean_transcription_time: float = 0.0

    science_nature_mean_time: float = 0.0
    science_nature_mean_question_accuracy: float = 0.0
    science_nature_mean_search_time: float = 0.0
    science_nature_mean_reasoning_time: float = 0.0
    science_nature_mean_transcription_time: float = 0.0

    ancient_history_mean_time: float = 0.0
    ancient_history_mean_question_accuracy: float = 0.0
    ancient_history_mean_search_time: float = 0.0
    ancient_history_mean_reasoning_time: float = 0.0
    ancient_history_mean_transcription_time: float = 0.0

    maths_mean_time: float = 0.0
    maths_mean_question_accuracy: float = 0.0
    maths_mean_search_time: float = 0.0
    maths_mean_reasoning_time: float = 0.0
    maths_mean_transcription_time: float = 0.0

    news_mean_time: float = 0.0
    news_mean_question_accuracy: float = 0.0
    news_mean_search_time: float = 0.0
    news_mean_reasoning_time: float = 0.0

    philosophy_psychology_mean_time: float = 0.0
    philosophy_psychology_mean_question_accuracy: float = 0.0
    philosophy_psychology_mean_search_time: float = 0.0
    philosophy_psychology_mean_reasoning_time: float = 0.0

@dataclass
class QuestionResult:
    theme: str = ""
    question_outcome: QuestionOutcome = QuestionOutcome.ERROR
    answer_time: float = 0.0
    search_time: float = 0.0
    reasoning_time: float = 0.0
    transcription_time: float = 0.0
    level: int = 0

@dataclass
class ThemeConfig:
    collection_name: str
    prompt_template: any # PromptTemplate
    approach_type: ApproachType
    model_name: str
    fallback_model: Optional[str] = None
    num_predict: int = 128
    temperature: float = 0.1
    top_k: int = 2
    similarity_threshold: float = 0.6
    translator_model: Optional[str] = None
    two_vector_search: bool = False
    is_rag: bool = True
    timeout: float = 120.0
    max_search_results: int = 5
