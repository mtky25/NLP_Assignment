# Project Overview
Poli-Millionaire is an automated game-playing agent for a "Who Wants to Be a Millionaire" style competition. It combines a custom API client (`millionaire_client`) with an LLM-based reasoning engine (`src.guesser`) powered by Ollama to participate in competitions, answer questions, and track performance metrics.

# Critical Constraints
- **Client Immortality**: You MUST NOT modify any files within the `millionaire_client/` directory. This is a fixed dependency.

## Key Technologies
- **Python**: Core programming language.
- **Ollama**: Local LLM orchestration (default model: `llama3.2`).
- **Pandas & Openpyxl**: Data management and Excel result exporting.
- **Python-Dotenv**: Environment variable configuration.
- **Requests**: Backend for the `MillionaireClient` API interactions.

## Architecture
- `main.py`: The entry point that orchestrates authentication, competition selection, and the game loop.
- `millionaire_client/`: A comprehensive, modular API client for the game server.
- `src/game/`: Manages individual game sessions, level progression, and metric calculation.
- `src/guesser/`: Integrates with Ollama to provide multiple-choice answers based on system prompts.
- `src/prompts/`: Contains the system instructions and formatting for the LLM guesser.

---

# Building and Running

### Prerequisites
- Python 3.x installed.
- Ollama installed and running with the `llama3.2` model (or the model ID specified in `main.py`).
- A `.env` file in the root directory (or as configured in `main.py`).

### Environment Configuration
Create a `.env` file with the following keys:
```env
MILLIONAIRE_USERNAME=your_username
MILLIONAIRE_PASSWORD=your_password
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Project
```bash
python main.py
```

### Outputs
- **Console**: Real-time game progress, levels, and guesser reasoning.
- **game_results.xlsx**: An Excel file containing historical performance data (model name, correct answers, average response time, theme).

---

# Development Conventions

### Coding Style
- **Modularity**: Logic is strictly separated into the API client, game orchestration, and inference engine.
- **Object-Oriented**: Core components like `Game`, `Guesser`, and `MillionaireClient` are implemented as classes.
- **Type Hinting**: Used in critical paths (e.g., `millionaire_client`) to ensure clarity.

### Configuration
- Constants like `API_URL` and `MODEL_ID` are currently defined in `main.py`.
- Secrets (username/password) MUST be kept in `.env` and never hardcoded.

### Testing and Validation
- Currently, validation is performed through live game runs.
- Metrics are automatically appended to `game_results.xlsx` for post-run analysis.

---

# Project-Specific Instructions
- **Ollama Availability**: Ensure the Ollama service is reachable at the configured `MODEL_ID` before starting.
- **Excel Locking**: Close `game_results.xlsx` before running the project to avoid permission errors when saving new results.
- **API Connectivity**: The project targets a specific research API (default: `http://131.175.15.22:51111/`). Ensure network access is available.
