import sys
import os
import time

# Ensure the root directory and Marcelo implementation are in sys.path
# This handles the namespace package merging for 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.guesser.marcelo_guesser import MarceloGuesser
from src.models import ExperimentConfig, ApproachType
from src.guesser.configs import INFERENCE_MODEL, EMBEDDING_MODEL
from src.millionaire_client.models import Question, Option
from src.guesser.context_db.collections import (
    COLLECTION_HISTORY, 
    COLLECTION_SCIENCE, 
    COLLECTION_MATH, 
    COLLECTION_ENTERTAINMENT
)

def run_debug_chat(question_text: str, options: list, theme: str = COLLECTION_HISTORY):
    """
    Runs a simulation of a game question through the RAG pipeline 
    and prints detailed metrics and context.
    """
    config = ExperimentConfig(
        username="cli_debug",
        approach=ApproachType.HYBRID,
        inference_model=INFERENCE_MODEL,
        embedding_model=EMBEDDING_MODEL
    )

    # Initialize guesser (it handles engine theme-switching internally)
    guesser = MarceloGuesser(config=config, theme=theme)
    
    # Create mock question object
    mock_q = Question(
        id=999, 
        text=question_text, 
        options=[Option(id=i, text=t) for i, t in enumerate(options)]
    )

    print("\n" + "="*60)
    print(f"🤔 TESTING QUESTION: {question_text}")
    print(f"📊 THEME: {theme}")
    print("="*60)

    # Execute inference
    result = guesser.engine.answer(mock_q)

    print("\n" + "-"*30)
    print("⏱️  EXECUTION TIMING")
    print("-"*30)
    print(f"Total Time:      {result.get('total_time'):.2f}s")
    print(f"Search Time:     {result.get('search_time'):.2f}s")
    print(f"Reasoning Time:  {result.get('reasoning_time'):.2f}s")

    print("\n" + "-"*30)
    print("🤖 LLM RESPONSE")
    print("-"*30)
    ans_idx = result['answer']
    print(f"Chosen Index:    {ans_idx}")
    if ans_idx.isdigit() and int(ans_idx) < len(options):
        print(f"Final Answer:    {options[int(ans_idx)]}")
    else:
        print(f"Final Answer:    Error in response format")

    print("\n" + "-"*30)
    print("📚 RETRIEVED CONTEXT CHUNKS")
    print("-"*30)
    
    for i, meta in enumerate(result.get('chunks_metadata', [])):
        print(f"\n[CHUNK {i+1}] | Similarity: {meta['similarity']:.4f} | RRF: {meta['rrf']:.4f}")
        # Print a snippet of the text
        clean_text = meta['text'].replace('\n', ' ')
        print(f"Text: {clean_text[:400]}...")
        print("-" * 15)

if __name__ == "__main__":
    # EXAMPLE USAGE
    # You can change these to test specific RAG scenarios
    test_q = "Q: Statement 1 | Every group of order 42 has a normal subgroup of order 7. Statement 2 | Every group of order 42 has a normal subgroup of order 8."
    test_opts = [
        "[0] False, True",
        "[1] False, False",
        "[2] True, False",
        "[3] True, True",
    ]
    

    
    run_debug_chat(test_q, test_opts, theme=COLLECTION_MATH)
