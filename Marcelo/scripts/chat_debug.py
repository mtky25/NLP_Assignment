import sys
import os
import time

# Ensure the root directory and Marcelo implementation are in sys.path
# This handles the namespace package merging for 'src'
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_dir = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_dir not in sys.path:
    sys.path.append(marcelo_dir)

from src.guesser.marcelo_guesser import MarceloGuesser
from src.models import ExperimentConfig, ApproachType
from src.guesser.engine.utils.ollama_utils import populate_experiment_config_sizes
from src.guesser.engine.configs import INFERENCE_MODEL, EMBEDDING_MODEL, MATH_INFERENCE_MODEL
from src.millionaire_client.models import Question, Option
from src.guesser.context_db.collections import (
    COLLECTION_HISTORY_POLITICS, 
    COLLECTION_SCIENCE_NATURE, 
    COLLECTION_MATHS, 
    COLLECTION_ENTERTAINMENT
)

def run_debug_chat(question_text: str, options: list, theme: str = COLLECTION_HISTORY_POLITICS):
    """
    Runs a simulation of a game question through the RAG pipeline 
    and prints detailed metrics and context.
    """
    # Select appropriate model based on theme
    model = MATH_INFERENCE_MODEL if theme in [COLLECTION_MATHS, "maths", "math"] else INFERENCE_MODEL
    model = "llama3.2:latest"
    config = ExperimentConfig(
        username="cli_debug",
        approach=ApproachType.HYBRID, # GuesserEngine will override this if not HYBRID, but we use HYBRID to let theme config decide
        inference_model=model,
        embedding_model=EMBEDDING_MODEL,
        debug=True
    )
    populate_experiment_config_sizes(config)

    # Initialize guesser...
    # We pass the db_path explicitly to ensure it finds it from the new script location
    db_path = os.path.join(marcelo_dir, "src", "guesser", "context_db")
    guesser = MarceloGuesser(config=config, theme=theme, db_path=db_path)
    
    # Create mock question object
    mock_q = Question(
        id=999, 
        text=question_text, 
        options=[Option(id=i, text=t) for i, t in enumerate(options)]
    )

    print("\n" + "="*60)
    print(f"🤔 TESTING QUESTION: {question_text}")
    print(f"📊 THEME: {theme}")
    print(f"🤖 MODEL: {config.inference_model} ({config.inference_model_size})")
    print(f"🧠 EMBEDDING: {config.embedding_model} ({config.embedding_model_size})")
    print("="*60)

    # Execute inference using the actual guesser to test parsing logic
    try:
        ans_idx = guesser.infer_answer(mock_q, theme=theme)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return

    print("\n" + "-"*30)
    print("⏱️  EXECUTION TIMING")
    print("-"*30)
    print(f"Total Time:      {guesser.search_time + guesser.reasoning_time:.2f}s")
    print(f"Search Time:     {guesser.search_time:.2f}s")
    print(f"Reasoning Time:  {guesser.reasoning_time:.2f}s")

    print("\n" + "-"*30)
    print("🤖 GUESSER RESULT")
    print("-"*30)
    print(f"Chosen Index:    {ans_idx}")
    if ans_idx is not None and ans_idx < len(options):
        print(f"Final Answer:    {options[ans_idx]}")
    else:
        print(f"Final Answer:    Error in parsing or out of bounds")

    print("\n" + "-"*30)
    print("📚 RETRIEVED CONTEXT CHUNKS")
    print("-"*30)
    
    last_chunks = getattr(guesser, 'last_chunks', [])
    if not last_chunks:
        print("No chunks were retrieved or met the similarity threshold.")
    else:
        for i, chunk in enumerate(last_chunks):
            print(f"\n[CHUNK {i+1}] (Similarity: {chunk.get('similarity', 0):.4f})")
            print(f"Content: {chunk.get('text', '')}")
            print("-" * 20)

if __name__ == "__main__":
    # EXAMPLE USAGE
#     test_q = "Q: A point $(x,y)$ is randomly picked from inside the rectangle with vertices $(0,0)$, $(3,0)$, $(3,2)$, and $(0,2)$. What is the probability that $x < y$?"


#     test_opts = [
#   "[0] \frac{1}{12}",
#   "[1] \frac{1}{3}",
#   "[2] \frac{1}{6}",
#   "[3] \frac{2}{3}",
#     ]
    # test_q = "Q: How does the structure of DNA relate to its function in protein synthesis?"
    # test_opts = [  
    # "[0] The sequence of bases in DNA dictates the sequence of amino acids in proteins.",
    # "[1] The double helix structure prevents any gene expression.",
    # "[2] The structure is irrelevant to protein synthesis.",
    # "[3] The structure allows DNA to directly interact with proteins without transcription.",]
    

    test_q = "What is the capital of a fictional country named 'Geminiland'?"
    test_opts = [  
    "[0] London",
    "[1] Paris",
    "[2] Tokyo",
    "[3] New York",]
    run_debug_chat(test_q, test_opts, theme=COLLECTION_HISTORY_POLITICS)

