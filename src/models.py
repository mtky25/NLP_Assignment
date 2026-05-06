from __future__ import annotations

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from uuid import uuid4


class ApproachType(str, Enum):
    DIRECT_LLM = "direct_llm"
    RAG = "rag"
    HYBRID = "hybrid"


class CompetitionTheme(str, Enum):
    ENTERTAINMENT = "Entertainment"
    ANCIENT_HISTORY_POLITICS = "Ancient History & Politics"
    SCIENCE_AND_NATURE = "Science & Nature"
    MATHS = "Maths"


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
    inference_model_size: int = 0

    is_rag: bool = False
    embedding_model: Optional[str] = None
    embedding_model_size: Optional[str] = None

    mean_question_accuracy: float = 0.0
    mean_time: float = 0.0
    entertainment_mean_time: float = 0.0
    entertainment_mean_question_accuracy: float = 0.0
    science_nature_mean_time: float = 0.0
    science_nature_mean_question_accuracy: float = 0.0
    ancient_history_mean_time: float = 0.0
    ancient_history_mean_question_accuracy: float = 0.0
    maths_mean_time: float = 0.0
    maths_mean_question_accuracy: float = 0.0

@dataclass
class QuestionResult:
    theme: str = ""
    question_outcome: QuestionOutcome = QuestionOutcome.ERROR
    answer_time: float = 0.0
    level: int = 0

    


    

    
