import os
import sys

# Ensure the root directory and Marcelo implementation are in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_root, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_root not in sys.path:
    sys.path.append(marcelo_root)

from src.guesser.engine.query_translator import QueryTranslator

def test_translator():
    translator = QueryTranslator()
    
    test_questions = [
        "Q: Statement 1 | Every group of order 42 has a normal subgroup of order 7. Statement 2 | Every group of order 42 has a normal subgroup of order 8.",
        "Which of the following elements is a transition metal found in the fourth period of the periodic table?",
        "In the movie 'Inception', what is the name of the character played by Leonardo DiCaprio?",
        "Who was the prime minister of the United Kingdom during the majority of World War II?"
    ]
    
    print("\n" + "="*60)
    print("🧪 QUERY TRANSLATOR DEBUG")
    print("="*60)
    
    for i, q in enumerate(test_questions):
        print(f"\n[TEST {i+1}]")
        # The translate method already prints the log line, but we'll capture it here too
        translated = translator.translate(q)
        print(f"Resulting Search Keywords: {translated}")
        print("-" * 30)

if __name__ == "__main__":
    test_translator()
