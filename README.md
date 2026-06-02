# NLP Assignment — Poli-Millionaire Benchmark

This repository contains the code developed for the NLP course project at Politecnico di Milano.
The goal is to build an automated agent that plays a "Who Wants to Be a Millionaire"-style quiz game
and benchmark different NLP approaches against each other.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Framework](#3-framework)
4. [Approaches implemented](#4-approaches-implemented)
5. [Setup & running](#6-setup--running)

---

## 1. Project overview

The benchmark connects to the Poli-Millionaire API, plays multiple quiz games across competition
themes, and records the performance of each NLP approach. Results are written to
`benchmark_results.xlsx` with per-theme accuracy and response times.

**Competition themes:**

| ID | Theme |
|----|-------|
| 0 | Entertainment |
| 1 | Ancient History and Politics |
| 2 | Science and Nature |
| 3 | Maths |
| 4 | News |
| 5 | Philosophy and Psychology


Each game has up to 15 questions. Every question carries a **30-second hard timeout** — any
pipeline that exceeds it scores zero on that question and the game ends immediately.

---

## 2. Repository structure

```
NLP_Assignment/
│
├── src/
│   ├── millionaire_client/     # API client
│   ├── guesser/
│   │   └── guesser.py          # Abstract Guesser base class
│   ├── game/game.py            # Game loop
│   ├── analysis/analysis.py    # Metric computation
│   ├── models.py               # ExperimentConfig, QuestionResult, enums
│   └── benchmark.py            # Benchmark runner
│
├── Luca/                       # Sentence Transformer baselines
├── Gianpaolo/                  # Theme-routed RAG on Colab T4
├── Marcelo/                    # Offline hybrid RAG with ChromaDB
├── Albana/                     # Theme-routed RAG on Colab T4 with single LLM
│
├── plots/                      # Performance plots
└── benchmark_results.xlsx      # Aggregated results (auto-generated)
```

---

## 3. Framework

### The `Guesser` interface

All approaches extend the abstract `Guesser` class in [`src/guesser/guesser.py`](src/guesser/guesser.py).
The contract is a single method:

```python
def infer_answer(self, question: Question, theme: str, game_session=None) -> int:
    # must return 0, 1, 2, or 3 within 30 seconds
```

The base class also provides two shared capabilities used across implementations:

- **Speech mode** — when `mode="speech"` is passed, the base class fetches audio clips from the
  server and transcribes them with [OpenAI Whisper](https://github.com/openai/whisper) before
  calling `infer_answer`. Transcription is parallelised (question audio starts transcribing while
  option clips are still being fetched). A regex post-processing pass strips Whisper artifacts
  (laughs, filler words, false option prefixes).
- **Timing instrumentation** — subclasses populate `self.search_time` and `self.reasoning_time`;
  the benchmark framework reads these after every question and aggregates them per theme.

### The `Benchmark` runner

[`src/benchmark.py`](src/benchmark.py) orchestrates the full evaluation loop: it logs in,
starts games across all competitions, calls `infer_answer` per question, and writes a results
row to `benchmark_results.xlsx`. Each row captures per-theme accuracy, mean answer time,
search time, and reasoning time alongside the experiment configuration.

### `ExperimentConfig`

[`src/models.py`](src/models.py) defines `ExperimentConfig`, the metadata object that travels
with every run. It records the approach type, model names, sizes, and all computed metrics, and
is serialised directly into the results file.

---

## 4. Approaches implemented

### Luca — Sentence Transformer baselines (`Luca/`)

Three LLM-free approaches using [sentence-transformers](https://www.sbert.net/) and Wikipedia.
See [`Luca/sentence_transformer_only.ipynb`](Luca/sentence_transformer_only.ipynb) and [`Luca/cross_encoder.ipynb`](Luca/cross_encoder.ipynb).

| File | Approach |
|------|----------|
| [`bert_baseline.py`](Luca/bert_baseline.py) | **Cosine similarity** — embeds question and options with `nomic-embed-text-v1.5`, picks the highest cosine match. No retrieval. |
| [`bert_rag_baseline.py`](Luca/bert_rag_baseline.py) | **Bi-encoder + Wikipedia** — fetches Wikipedia per option, picks the option whose retrieved content best matches the question embedding. |
| [`cross_encoder_rag_baseline.py`](Luca/cross_encoder_rag_baseline.py) | **Bi-encoder + Cross-Encoder re-rank** — bi-encoder retrieves top Wikipedia sections, then `ettin-reranker-1b-v1` re-scores each *(option, section)* pair. Also evaluated in **speech mode**. |

---

### Gianpaolo — Theme-routed RAG on Colab T4 (`Gianpaolo/`)

Three-branch pipeline running 14B models (4-bit NF4) on a Colab T4.
See [`Gianpaolo/RAG.ipynb`](Gianpaolo/RAG.ipynb) and [`Gianpaolo/rag_benchmark_guesser.py`](Gianpaolo/rag_benchmark_guesser.py).

- **Wikipedia RAG** (most themes) — `Qwen2.5-14B` generates search queries → parallel Wikipedia fetch → BM25 prefilter → `bge-reranker-base` rerank → LLM reads top-5 chunks and answers.
- **News via Guardian API** — same pipeline but hits the Guardian Open Platform with a ±3-day date window parsed from the question text.
- **Math with streaming watchdog** — `DeepSeek-R1-14B` runs chain-of-thought with a wall-clock `StoppingCriteria` at 25 s; if no answer is found in the partial trace, a short deterministic refine call commits to one.

---

### Marcelo — Offline hybrid RAG with ChromaDB (`Marcelo/`)

Fully local system running via [Ollama](https://ollama.com), no cloud GPU required.
See [`Marcelo/marcelo_notebook.ipynb`](Marcelo/marcelo_notebook.ipynb) and [`Marcelo/src/guesser/marcelo_guesser.py`](Marcelo/src/guesser/marcelo_guesser.py).

A `Router` maps each competition theme to a strategy: Entertainment and Science/Nature use RAG against a pre-built [ChromaDB](https://www.trychroma.com/) store (`gemma3:4b`); Ancient History falls back to live web search (`phi3.5`); Maths uses **Program-of-Thought** — `qwen2-math:1.5b` generates Python code that is executed to compute the answer. Questions are first translated into retrieval-friendly keywords by a `qwen2.5:0.5b` model before hitting the vector store.

---

### Albana - One single LLM multi-pipeline

Three-branch pipeline running Mistral-7B-Instruct-v0.3 on a Google Colab notebook environment.
See [Albana/albana_notebook.ipynb]

- **Wikipedia Sniper (General Trivia)** — Mistral extracts dense keyword context via a custom automated information retrieval agent and an option-matching semantic paragraph re-ranker to resolve general-knowledge trivia.

- **News via Guardian API** — same pipeline as the previous case, but using Guardian API and ISO date-parsed temporal filtering.

- **Agentic Math Engine (ReAct Pipeline)** — ReAct tool-calling framework routing to a sandboxed Python execution kernel (`numpy`, `scipy`, `sympy`, `networkx`) or a theoretical web crawler, guarded by a JSON auto-healer and tolerance-based (`math.isclose`) float evaluator.

## 5. Setup & running

<details>
<summary>Prerequisites, installation, and credentials</summary>

**Python 3.10+** and **Ollama** (for local approaches) are required.

```bash
pip install ollama requests pandas openpyxl python-dotenv \
    llama-index-embeddings-ollama llama-index-vector-stores-chroma llama-index-core \
    sentence-transformers rank-bm25 wikipedia-api
```

Credentials go in a `.env` file at the project root:

```
MILLIONAIRE_USERNAME=YourUsername
MILLIONAIRE_PASSWORD=YourPassword
```

</details>

<details>
<summary>Running a benchmark</summary>

All local approaches share the same runner pattern:

```python
from dotenv import load_dotenv
import os
from src.millionaire_client import MillionaireClient
from src.benchmark import Benchmark
from src.models import ExperimentConfig, ApproachType

load_dotenv()
client = MillionaireClient("http://131.175.15.22:51111/")
client.login(os.getenv("MILLIONAIRE_USERNAME"), os.getenv("MILLIONAIRE_PASSWORD"))

config = ExperimentConfig(
    username="YourUsername",
    notes="experiment description",
    approach=ApproachType.RAG,
    inference_model="model-name",
    inference_model_size=1,
    is_rag=True,
)

guesser = YourGuesser(config)
benchmark = Benchmark(config, guesser, client)
benchmark.run(times_per_competition=5, filename="your_results.xlsx")
```

Gianpaolo's notebook manages its own game loop and writes to `experiment_results.csv`.
Luca's Colab notebooks are self-contained and portable across Colab sessions.

</details>
