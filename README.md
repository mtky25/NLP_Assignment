# NLP Assignment — Poli-Millionaire Benchmark

This repository contains the code developed for the NLP course project at Politecnico di Milano.
The goal is to build an automated agent that plays a "Who Wants to Be a Millionaire"-style quiz game
and benchmark different NLP approaches (direct LLM, RAG, etc.) against each other.

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Credentials setup (.env)](#4-credentials-setup-env)
5. [Project structure](#5-project-structure)
6. [Step 1 — Implement your Guesser](#6-step-1--implement-your-guesser)
7. [Step 2 — Configure your experiment](#7-step-2--configure-your-experiment)
8. [Step 3 — Run the benchmark](#8-step-3--run-the-benchmark)
9. [Understanding the results](#9-understanding-the-results)
10. [Reference implementation](#10-reference-implementation)

---

## 1. What this project does

The benchmark connects to the Poli-Millionaire API, plays multiple quiz games across four
competition themes, and records the performance of your NLP model. At the end it writes a
results row to `benchmark_results.xlsx` with per-theme accuracy and response times.

**The four competitions are:**
| ID | Theme |
|----|-------|
| 0 | Entertainment |
| 1 | Ancient History and Politics |
| 2 | Science and Nature |
| 3 | Maths |

Each game has up to 15 questions. Every question has a **30-second timeout** — if your model
takes longer than 30 seconds to answer, the question is marked as timed out and the game ends.

---

## 2. Prerequisites

Before you start, make sure you have the following installed on your machine:

- **Python 3.10 or newer** — download from https://www.python.org/downloads/
- **Ollama** — a tool that runs LLM models locally. Download from https://ollama.com/download

After installing Ollama, open a terminal and pull the model you want to use. For example:

```bash
ollama pull llama3.2
```

You can also pull the embedding model if you plan to use RAG:

```bash
ollama pull nomic-embed-text
```

To verify Ollama is running, open a terminal and type:

```bash
ollama list
```

You should see the models you just downloaded listed there.

---

## 3. Installation

**Step 1.** Clone or download this repository to your computer.

**Step 2.** Open a terminal in the root folder of the project
(the folder that contains this README file).

**Step 3.** Install all Python dependencies by running:

```bash
pip install ollama requests pandas openpyxl python-dotenv llama-index-embeddings-ollama llama-index-vector-stores-chroma llama-index-core
```

> **Tip:** If you are using a virtual environment (recommended), activate it first before running
> the command above.

---

## 4. Credentials setup (.env)

The game API requires a username and password. These are stored in a file called `.env`
in the **root folder** of the project (the same folder as this README).

**Step 1.** Create a new file named exactly `.env` (note the dot at the start).

**Step 2.** Add the following two lines, replacing the values with your own credentials:

```
MILLIONAIRE_USERNAME=YourUsername
MILLIONAIRE_PASSWORD=YourPassword
```

> **Important:** Never share this file or commit it to git. It contains your personal credentials.

**Step 3.** In your notebook or script, load the credentials like this:

```python
from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("MILLIONAIRE_USERNAME")
password = os.getenv("MILLIONAIRE_PASSWORD")
```

---

## 5. Project structure

```
NLP_Assignment/
│
├── src/
│   ├── millionaire_client/     # API client — DO NOT MODIFY
│   │   └── ...
│   │
│   ├── guesser/
│   │   └── guesser.py          # Abstract Guesser base class — your class must extend this
│   │
│   ├── game/
│   │   └── game.py             # Game loop — handles question/answer flow
│   │
│   ├── analysis/
│   │   └── analysis.py         # Computes metrics after all games finish
│   │
│   ├── models.py               # Data models: ExperimentConfig, QuestionResult, enums
│   └── benchmark.py            # Benchmark runner — orchestrates everything
│
├── Marcelo/                    # Reference implementation (RAG-based guesser)
│
├── .env                        # Your credentials (you create this)
└── benchmark_results.xlsx      # Results file (created automatically after first run)
```

The only folders you need to work with are:

- `src/models.py` — to understand `ExperimentConfig`
- `src/guesser/guesser.py` — to understand the interface your class must implement
- Your own file/folder — where you write your custom `Guesser` class

---

## 6. Step 1 — Implement your Guesser

A **Guesser** is a Python class that receives a quiz question and returns the index of the
answer it thinks is correct (a number from `0` to `3`).

You must create a class that **inherits** from the base `Guesser` class and implements the
`infer_answer` method.

### The interface

Open `src/guesser/guesser.py`. The base class looks like this:

```python
class Guesser(ABC):
    def __init__(self, config: ExperimentConfig):
        ...

    def infer_answer(self, question: Question) -> int:
        # You MUST implement this method
        # It must return an integer: 0, 1, 2, or 3
        pass

    def format_question_for_llm(self, question: Question) -> str:
        # Helper already provided — formats the question as a text prompt:
        # "Question: ...\nOptions:\n[0] ...\n[1] ...\n[2] ...\n[3] ..."
        ...
```

### The `question` object

Inside `infer_answer`, the `question` parameter has these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `question.text` | `str` | The question text |
| `question.options` | `list` | List of 4 options |
| `question.options[i].text` | `str` | The text of option `i` |
| `question.options[i].id` | `int` | The ID of option `i` (used internally by the API) |

> **Important:** `infer_answer` must return the **index** (0, 1, 2, or 3), not the option ID.
> The game loop handles converting the index to the correct option ID automatically.

### Minimal example — random guesser

This is the simplest possible guesser (answers randomly). Use it to verify everything works
before building a real model:

```python
import random
from src.guesser.guesser import Guesser
from src.millionaire_client.models import Question
from src.models import ExperimentConfig

class RandomGuesser(Guesser):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    def infer_answer(self, question: Question) -> int:
        return random.randint(0, 3)
```

### Example — direct LLM guesser with Ollama

```python
import re
import ollama
from src.guesser.guesser import Guesser
from src.millionaire_client.models import Question
from src.models import ExperimentConfig

SYSTEM_PROMPT = (
    "You answer multiple-choice questions. "
    "Reply with ONLY the digit of the correct option (0, 1, 2, or 3). "
    "No explanation. No punctuation. Just the digit."
)

class OllamaGuesser(Guesser):
    def __init__(self, config: ExperimentConfig, model_name: str = "llama3.2"):
        super().__init__(config)
        self.model_name = model_name

    def infer_answer(self, question: Question) -> int:
        prompt = self.format_question_for_llm(question)  # uses the built-in helper

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
        )

        raw = response["message"]["content"].strip()

        # Extract the first digit 0-3 from the response (defensive against extra text)
        match = re.search(r"[0-3]", raw)
        if match:
            return int(match.group())

        raise ValueError(f"Could not extract answer from: {raw[:80]}")
```

### Rules for your `infer_answer` method

- It **must** return an `int` between `0` and `3`.
- It **must** finish within **30 seconds** — the game server will time out the question otherwise.
- If it raises an exception, the benchmark catches it, records the question as an error,
  and moves to the next game. The game does not crash.

---

## 7. Step 2 — Configure your experiment

`ExperimentConfig` is a data class in `src/models.py` that describes what you are testing.
Its values get saved to the results Excel file so you can compare runs later.

```python
from src.models import ExperimentConfig, ApproachType

config = ExperimentConfig(
    username="YourUsername",          # your game username (for reference)
    notes="my first test",            # any free-text note you want to remember this run by
    approach=ApproachType.DIRECT_LLM, # ApproachType.DIRECT_LLM or ApproachType.RAG
    inference_model="llama3.2",       # the name of the model you are using
    inference_model_size=3,           # approximate size in billions of parameters (e.g. 3 for 3B)
    is_rag=False,                     # True if your guesser uses RAG / vector retrieval
    embedding_model=None,             # name of the embedding model if is_rag=True, else None
    embedding_model_size=None,        # size of the embedding model if is_rag=True, else None
)
```

> **Note:** `experiment_id` is generated automatically — do not set it manually.

### ApproachType options

| Value | When to use |
|-------|-------------|
| `ApproachType.DIRECT_LLM` | Your guesser calls an LLM directly with no external knowledge |
| `ApproachType.RAG` | Your guesser retrieves documents from a vector database first |
| `ApproachType.HYBRID` | A combination of both |

---

## 8. Step 3 — Run the benchmark

Once you have your Guesser class and your ExperimentConfig, running the benchmark takes
four lines of code:

```python
from dotenv import load_dotenv
import os
from src.millionaire_client import MillionaireClient
from src.benchmark import Benchmark

# 1. Load credentials
load_dotenv()
username = os.getenv("MILLIONAIRE_USERNAME")
password = os.getenv("MILLIONAIRE_PASSWORD")

# 2. Connect to the game server
API_URL = "http://131.175.15.22:51111/"
client = MillionaireClient(API_URL)
client.login(username, password)

# 3. Create your config and guesser
config = ExperimentConfig(
    username=username,
    notes="direct llm baseline",
    approach=ApproachType.DIRECT_LLM,
    inference_model="llama3.2",
    inference_model_size=3,
    is_rag=False,
)
guesser = OllamaGuesser(config, model_name="llama3.2")  # your class here

# 4. Run the benchmark
benchmark = Benchmark(config, guesser, client)
benchmark.run(times_per_competition=3)  # plays 3 games per competition = 12 games total
```

### `benchmark.run()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `times_per_competition` | `5` | How many games to play per competition theme |
| `save` | `True` | Whether to save results to `benchmark_results.xlsx` |

A full run with `times_per_competition=5` plays **20 games** (5 × 4 competitions).
Each game has up to 15 questions. Plan for roughly **5–30 seconds per question** depending
on your model's speed.

---

## 9. Understanding the results

After the benchmark finishes, `benchmark_results.xlsx` is created (or updated) in the
root folder of the project. Each row corresponds to one complete benchmark run.

### Columns explained

| Column | Description |
|--------|-------------|
| `experiment_id` | Auto-generated unique ID for this run |
| `username` | Your username |
| `notes` | The note you wrote in `ExperimentConfig` |
| `approach` | `direct_llm`, `rag`, or `hybrid` |
| `inference_model` | Model name (e.g. `llama3.2`) |
| `inference_model_size` | Model size in billions of parameters |
| `is_rag` | Whether RAG was used |
| `embedding_model` | Embedding model name (if RAG) |
| `mean_question_accuracy` | Overall accuracy across all games and competitions (0.0–1.0) |
| `mean_time` | Average seconds per question across all games |
| `entertainment_mean_time` | Average seconds per question in Entertainment games |
| `entertainment_mean_question_accuracy` | Accuracy in Entertainment games (0.0–1.0) |
| `science_nature_mean_time` | Average seconds per question in Science and Nature games |
| `science_nature_mean_question_accuracy` | Accuracy in Science and Nature games |
| `ancient_history_mean_time` | Average seconds per question in Ancient History games |
| `ancient_history_mean_question_accuracy` | Accuracy in Ancient History games |
| `maths_mean_time` | Average seconds per question in Maths games |
| `maths_mean_question_accuracy` | Accuracy in Maths games |
| `timestamp` | When this run was saved |

> **Accuracy** is calculated as: correct answers ÷ total questions answered.
> A value of `0.75` means 75% of questions were answered correctly.

### Reading per-theme accuracy

If a theme column shows `0.0` for both time and accuracy, it means no questions from that
theme were recorded — usually because all games for that theme ended on an inference error
before any answer was submitted. Check the console output for `Inference error:` messages.

---

## 10. Reference implementation

A complete RAG-based implementation is available in the `Marcelo/` folder. It uses:

- **LlamaIndex** + **ChromaDB** for vector storage and retrieval
- **nomic-embed-text** (via Ollama) for embeddings
- **llama3.2** or **phi3.5** for inference

You can import and use it as follows (after building the ChromaDB collection):

```python
from Marcelo.src.guesser.guesser import MarceloGuesser
from src.models import ExperimentConfig, ApproachType

config = ExperimentConfig(
    username="YourUsername",
    notes="rag test",
    approach=ApproachType.RAG,
    inference_model="llama3.2",
    inference_model_size=3,
    is_rag=True,
    embedding_model="nomic-embed-text",
    embedding_model_size="0.1",
)

guesser = MarceloGuesser(
    config=config,
    db_path="Marcelo/src/guesser/context_db/",
    collection_name="Science_Nature",   # must match a collection in your ChromaDB
    embedding_model_name="nomic-embed-text",
    inference_model_name="llama3.2",
)
```

> The ChromaDB collection must be built before running. See the notebooks in `Marcelo/`
> for ingestion pipelines that load Wikipedia (ZIM) and HuggingFace datasets into ChromaDB.
