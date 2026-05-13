# Who Wants to Be a Millionaire - RAG Guesser

This project implements an automated guesser for the "Who Wants to Be a Millionaire" game, utilizing a Retrieval-Augmented Generation (RAG) approach to answer questions based on Wikipedia data.

## 🚀 Approach

The guesser follows a RAG architecture:
1.  **Data Ingestion**: Wikipedia articles (from ZIM files) or Hugging Face datasets are processed, chunked, and stored in a **ChromaDB** vector database.
2.  **Retrieval**: When a question is presented, the system searches the vector database for the most relevant context.
3.  **Generation**: An LLM (via **Ollama**) receives the retrieved context along with the question and options to infer the correct answer.

## 🛠️ Dependencies

To run this project, you need the following:

- **Python 3.10+**
- **Ollama**: Must be installed and running locally.
- **System Libraries**: `libzim` is required for processing `.zim` files.

### Python Libraries
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `ollama`: For LLM inference.
- `libzim`: To read Wikipedia archives.
- `llama-index`: For the RAG framework and orchestration.
- `chromadb`: As the vector database.
- `millionaire_client`: A custom client to interact with the game server.

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the `Marcelo/` directory with your credentials:

```env
MILLIONAIRE_USERNAME=your_username
MILLIONAIRE_PASSWORD=your_password
```

### Guesser & DB Config
General configuration for the guesser and the vector database can be found in `src/guesser/configs.py`:
- `CHROMA_DB_PATH`: Where the vector store is saved.
- `INFERENCE_MODEL`: The Ollama model to use for answering (e.g., `phi3.5`, `llama3.2`).
- `EMBEDDING_MODEL`: The model used for generating embeddings (e.g., `nomic-embed-text`).

### Data Extraction Config
Configuration for Hugging Face datasets and other extraction sources is located in `src/guesser/ingestion/extractors/configs.py`.

## 📂 Data Extraction & Ingestion

To populate your knowledge base, you can use the provided ingestion pipelines:

### 1. Wikipedia (ZIM) Ingestion
Use the `ZimIngestionPipeline` located in `src/guesser/ingestion/pipelines/zim_ingestion.py`. You will need a `.zim` file (e.g., Simple English Wikipedia).

### 2. Dataset Ingestion
Use the `DatasetIngestionPipeline` in `src/guesser/ingestion/pipelines/dataset_ingestion.py` to ingest data from Hugging Face datasets like `sciq` or `gsm8k`.

You can also use `dataset_mass_ingest.py` to ingest multiple datasets at once:
```bash
python Marcelo/dataset_mass_ingest.py
```

## 📜 Development Contracts

### Guesser Contract
Any guesser implementation used by the `Game` class must follow this interface:
- `add_question(question: Question)`: Receives the current question object.
- `infer_answer() -> int`: Must return the `id` of the chosen option.
- `model_name`: A property/attribute identifying the model.

### Extractor Contract
Custom data extractors must inherit from `BaseExtractor` (`src/guesser/ingestion/extractors/extractor.py`) and implement:
- `extract(limit: int) -> Iterable[Document]`: Returns an iterable of `llama-index` Documents.

## 🎮 How to Run

1.  Ensure Ollama is running and the required models are pulled (`ollama pull llama3.2`, etc.).
2.  Configure your `.env` file.
3.  (Optional) Run an ingestion pipeline if the database is not yet populated.
4.  Execute the main script:

```bash
python main.py
```

The results will be exported to `game_results.xlsx` and `detailed_game_results.xlsx` after each game session.
