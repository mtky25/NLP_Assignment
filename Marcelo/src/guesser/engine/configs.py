from src.models import ApproachType, ThemeConfig
from src.guesser.context_db.collections import (
    COLLECTION_ENTERTAINMENT,
    COLLECTION_HISTORY_POLITICS,
    COLLECTION_MATHS,
    COLLECTION_SCIENCE_NATURE,
    COLLECTION_NEWS,
    COLLECTION_PHILOSOPHY_PSYCHOLOGY,
    COLLECTION_DEFAULT
)
from src.guesser.engine.prompts import (
    MCQ_PROMPT_ENTERTAINMENT,
    MCQ_PROMPT_HISTORY_POLITICS,
    MCQ_PROMPT_MATHS,
    MCQ_PROMPT_MATHS_POT,
    MCQ_PROMPT_SCIENCE_NATURE,
    MCQ_PROMPT_NEWS,
    MCQ_PROMPT_PHILOSOPHY_PSYCHOLOGY,
    MCQ_PROMPT
)

# Global Configs
CHROMA_DB_PATH="./context_db"
INFERENCE_MODEL="llama3.2:latest"
FALLBACK_INFERENCE_MODEL="llama3.2:latest"
TRANSLATOR_MODEL="qwen2.5:0.5b"
EMBEDDING_MODEL="nomic-embed-text"
MATH_INFERENCE_MODEL="qwen2-math:1.5b"
MATH_TRANSLATOR_MODEL="qwen2.5:0.5b"

PRE_LOAD_MODELS = [
    TRANSLATOR_MODEL,
    INFERENCE_MODEL,
    FALLBACK_INFERENCE_MODEL,
    MATH_INFERENCE_MODEL,
]

ENTERTAINMENT_CONFIG = ThemeConfig(
    collection_name=COLLECTION_ENTERTAINMENT,
    prompt_template=MCQ_PROMPT_ENTERTAINMENT,
    approach_type=ApproachType.RAG,
    model_name=INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=5,
    top_k=1,
    similarity_threshold=0.82,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=True,
    is_rag=True
)

HISTORY_POLITICS_CONFIG = ThemeConfig(
    collection_name=COLLECTION_HISTORY_POLITICS,
    prompt_template=MCQ_PROMPT_HISTORY_POLITICS,
    approach_type=ApproachType.SEARCH,
    model_name=FALLBACK_INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=5,
    top_k=1,
    similarity_threshold=0.88,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=False,
    is_rag=False,
    max_search_results=4
)

SCIENCE_NATURE_CONFIG = ThemeConfig(
    collection_name=COLLECTION_SCIENCE_NATURE,
    prompt_template=MCQ_PROMPT_SCIENCE_NATURE,
    approach_type=ApproachType.RAG,
    model_name=INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=5,
    top_k=1,
    similarity_threshold=0.82,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=True,
    is_rag=True
)

MATHS_CONFIG = ThemeConfig(
    collection_name=COLLECTION_MATHS,
    prompt_template=MCQ_PROMPT_MATHS_POT,
    approach_type=ApproachType.POT,
    model_name=MATH_INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=350,
    temperature=0.1,
    top_k=2,
    similarity_threshold=0.8,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=True,
    is_rag=False,
    timeout=30.0
)


NEWS_CONFIG = ThemeConfig(
    collection_name=COLLECTION_NEWS,
    prompt_template=MCQ_PROMPT_NEWS,
    approach_type=ApproachType.SEARCH,
    model_name=FALLBACK_INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=5,
    top_k=1,
    similarity_threshold=0.75,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=False,
    is_rag=False,
    max_search_results=4
)

PHILOSOPHY_PSYCHOLOGY_CONFIG = ThemeConfig(
    collection_name=COLLECTION_PHILOSOPHY_PSYCHOLOGY,
    prompt_template=MCQ_PROMPT_PHILOSOPHY_PSYCHOLOGY,
    approach_type=ApproachType.RAG,
    model_name=INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=5,
    top_k=1,
    similarity_threshold=0.82,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=True,
    is_rag=True
)

DEFAULT_CONFIG = ThemeConfig(
    collection_name=COLLECTION_DEFAULT, 
    prompt_template=MCQ_PROMPT, 
    approach_type=ApproachType.RAG,
    model_name=INFERENCE_MODEL,
    fallback_model=FALLBACK_INFERENCE_MODEL,
    num_predict=32,
    top_k=2,
    similarity_threshold=0.6,
    translator_model=TRANSLATOR_MODEL,
    two_vector_search=False,
    is_rag=True
)
