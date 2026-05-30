"""
Model Comparison Runner
=======================
Runs multiple model experiments back-to-back, saving every result to the same
Excel file so you can compare them side-by-side afterwards.

HOW TO USE
----------
1. Edit the EXPERIMENTS list and DEFAULT_GAMES_PER_THEME below.
2. Run from the project root:
       python -m Marcelo.scripts.model_runner
   Or call run_all(client) from the notebook after logging in.

CONFIGURABLE SECTION — only edit between the dashed lines.
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# Each dict becomes one full benchmark run. Keys:
#   name              – label stored in the Excel "notes" column
#   inference_model   – main RAG / SEARCH reasoning model
#   fallback_model    – used when RAG finds no context (knowledge fallback)
#   math_model        – model used exclusively for Maths theme
#   translator_model  – QueryTranslator + MathClassifier (keep small)
#   embedding_model   – ChromaDB retrieval model (must match the indexed DB)
#   games_per_theme   – (optional) overrides DEFAULT_GAMES_PER_THEME for this run
# ─────────────────────────────────────────────────────────────────────────────

# Ordered from lightest to heaviest. Experiments 6–5 explore variations around
# the best base so far (gemma3:4b as inference). Translator/classifier is always
# qwen2.5:0.5b, embedding is always nomic-embed-text:latest, and the Maths model is kept
# under 2b in every run.
EXPERIMENTS = [
    {   # 1 — lightest baseline (floor)
        "name": "llama3.2:1b/qwen2.5:1.5b",
        "inference_model":  "llama3.2:1b",
        "fallback_model":   "qwen2.5:1.5b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 2 — light, swap the 1b/1.5b roles
        "name": "qwen2.5:1.5b/llama3.2:1b",
        "inference_model":  "qwen2.5:1.5b",
        "fallback_model":   "llama3.2:1b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 3 — light-medium, inference up to 3b with cheap fallback
        "name": "llama3.2:3b/llama3.2:1b",
        "inference_model":  "llama3.2:3b",
        "fallback_model":   "llama3.2:1b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 4 — medium, qwen family at 3b
        "name": "qwen2.5:3b/llama3.2:3b",
        "inference_model":  "qwen2.5:3b",
        "fallback_model":   "llama3.2:3b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 5 — medium, phi3.5 (~3.8b) as inference
        "name": "phi3.5:latest/llama3.2:3b",
        "inference_model":  "phi3.5:latest",
        "fallback_model":   "llama3.2:3b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 6 — best inference + ultralight fallback (cost/benefit)
        "name": "gemma3:4b/llama3.2:1b",
        "inference_model":  "gemma3:4b",
        "fallback_model":   "llama3.2:1b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 7 — best base so far, balanced fallback
        "name": "gemma3:4b/llama3.2:3b",
        "inference_model":  "gemma3:4b",
        "fallback_model":   "llama3.2:3b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 8 — gemma3:4b with qwen 3b fallback
        "name": "gemma3:4b/qwen2.5:3b",
        "inference_model":  "gemma3:4b",
        "fallback_model":   "qwen2.5:3b",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 9 — gemma3:4b with phi3.5 fallback
        "name": "gemma3:4b/phi3.5:latest",
        "inference_model":  "gemma3:4b",
        "fallback_model":   "phi3.5:latest",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
    {   # 5 — heaviest ceiling: best inference + gemma2 (~9b) fallback
        "name": "gemma3:4b/gemma2:latest",
        "inference_model":  "gemma3:4b",
        "fallback_model":   "gemma2:latest",
        "math_model":       "qwen2-math:1.5b",
        "translator_model": "qwen2.5:0.5b",
        "embedding_model":  "nomic-embed-text:latest",
        "games_per_theme": 10,
    },
]

# Default number of games played per theme when a spec doesn't set its own value.
DEFAULT_GAMES_PER_THEME = 2

# All experiment results are appended to this single file.
OUTPUT_FILE = "Marcelo/model_comparison.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# END OF CONFIGURABLE SECTION — no need to edit below
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import subprocess
import time

# Allow running as a script from the project root
_MARCELO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure the root directory and Marcelo implementation are in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
marcelo_dir = os.path.abspath(os.path.join(scripts_dir, ".."))
project_root = os.path.abspath(os.path.join(marcelo_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if marcelo_dir not in sys.path:
    sys.path.append(marcelo_dir)

from src.millionaire_client import MillionaireClient, AuthenticationError
from src.benchmark import Benchmark
from src.models import ExperimentConfig, ApproachType
from src.guesser.marcelo_guesser import MarceloGuesser


# ── Ollama helpers ─────────────────────────────────────────────────────────

def _get_loaded_models() -> list[str]:
    """Return the list of model names currently loaded in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:          # skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        print(f" [OllamaManager] Warning: could not list models — {e}")
        return []


def stop_all_ollama_models():
    """
    Stop every model currently loaded in Ollama to free RAM before the next
    experiment starts with a clean slate.
    """
    models = _get_loaded_models()
    if not models:
        print(" [OllamaManager] No models currently loaded — nothing to stop.")
        return

    print(f" [OllamaManager] Stopping {len(models)} loaded model(s): {', '.join(models)}")
    for model in models:
        try:
            subprocess.run(
                ["ollama", "stop", model],
                capture_output=True, timeout=15
            )
            print(f"   ✓ Stopped: {model}")
        except Exception as e:
            print(f"   ✗ Could not stop {model}: {e}")

    # Brief pause so Ollama fully releases memory before the next load
    time.sleep(2)


# ── Runner ─────────────────────────────────────────────────────────────────

def run_all(client: MillionaireClient):
    """
    Iterate through EXPERIMENTS, run a full benchmark for each, and save
    all results to OUTPUT_FILE (rows are appended, not overwritten).
    """
    total = len(EXPERIMENTS)
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)

    for idx, spec in enumerate(EXPERIMENTS, start=1):
        name             = spec.get("name", f"experiment_{idx}")
        inference_model  = spec["inference_model"]
        fallback_model   = spec["fallback_model"]
        math_model       = spec["math_model"]
        translator_model = spec.get("translator_model", "qwen2.5:0.5b")
        embedding_model  = spec.get("embedding_model",  "nomic-embed-text:latest")
        games_per_theme  = spec.get("games_per_theme",  DEFAULT_GAMES_PER_THEME)

        print(f"\n{'='*65}")
        print(f" EXPERIMENT {idx}/{total}: {name}")
        print(f"   inference  : {inference_model}")
        print(f"   fallback   : {fallback_model}")
        print(f"   math       : {math_model}")
        print(f"   translator : {translator_model}")
        print(f"   embedding  : {embedding_model}")
        print(f"   games/theme: {games_per_theme}")
        print(f"{'='*65}\n")

        # ── Stop any leftover models from the previous experiment ──────────
        if idx > 1:
            print(" [OllamaManager] Clearing Ollama memory before new experiment...")
            stop_all_ollama_models()

        # ── Build ExperimentConfig ─────────────────────────────────────────
        config = ExperimentConfig(
            username="Marcelo",
            notes=name,
            approach=ApproachType.HYBRID,
            inference_model=inference_model,
            inference_model_size="",
            embedding_model=embedding_model,
            embedding_model_size="",
            is_rag=True,
        )

        # ── Build Guesser ──────────────────────────────────────────────────
        try:
            guesser = MarceloGuesser(
                config,
                inference_model_name=inference_model,
                fallback_model_name=fallback_model,
                math_model_name=math_model,
                translator_model_name=translator_model,
                embedding_model_name=embedding_model,
            )
        except Exception as e:
            print(f" [Runner] ERROR: could not initialise guesser for '{name}': {e}")
            print(f" [Runner] Skipping this experiment.\n")
            continue

        # ── Run benchmark ──────────────────────────────────────────────────
        try:
            benchmark = Benchmark(config, guesser, client)
            benchmark.run(games_per_theme, filename=OUTPUT_FILE)
        except Exception as e:
            print(f" [Runner] ERROR during benchmark for '{name}': {e}")
        finally:
            # Release the guesser object (helps Python GC)
            del guesser

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" All {total} experiment(s) complete.")
    print(f" Results saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"{'='*65}\n")

    stop_all_ollama_models()


# ── Standalone entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(_MARCELO, ".env"))

    API_URL  = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("MILLIONAIRE_USERNAME")
    PASSWORD = os.getenv("MILLIONAIRE_PASSWORD")

    client = MillionaireClient(API_URL)
    try:
        user = client.login(USERNAME, PASSWORD)
        print(f" Logged in as: {user.username}")
    except AuthenticationError as e:
        print(f" Login failed: {e}")
        sys.exit(1)

    run_all(client)
