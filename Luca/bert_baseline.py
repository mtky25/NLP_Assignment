import torch
import os
import sys
import time
import requests
import re
import numpy as np
from typing import Any, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

# Ensure we can import from the root src directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.guesser.guesser import Guesser
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient

load_dotenv()

class BertBaseline(Guesser):
    """
    A baseline guesser that uses BERT embeddings (SentenceTransformers) to select the most
    semantically similar option to the question.
    Now supports RAG (Retrieval-Augmented Generation) mode.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Use nomic-ai/modernbert-embed-base as a modern default if not specified
        self.model_id = config.embedding_model or "nomic-ai/modernbert-embed-base"
        
        print(f"Loading SentenceTransformer model: {self.model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_id, device=device)
        
        # Nomic ModernBERT embed requires prefixes for optimal performance
        self.use_prefixes = "modernbert-embed" in self.model_id.lower()

        # RAG setup
        self.is_rag = config.is_rag
        if self.is_rag:
            # LLM is not used in this RAG approach, only the embedding model.
            self.email = os.getenv("POLI_EMAIL", "luca_rag_bot@example.com")

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()

        if self.is_rag:
            result_idx = self._infer_with_rag(question, theme, game_session)
            self.reasoning_time = time.time() - start_time - self.search_time
            if result_idx is not None:
                return question.options[result_idx].id
            print("[RAG] Fallback to BERT similarity due to RAG failure.")

        # Default BERT similarity logic
        question_text = question.text
        options = question.options
        option_texts = [opt.text for opt in options]
        
        # 1. Prepare texts with optional prefixes for ModernBERT
        if self.use_prefixes:
            query_text = f"search_query: {question_text}"
            doc_texts = [f"search_document: {opt_text}" for opt_text in option_texts]
        else:
            query_text = question_text
            doc_texts = option_texts
            
        # 2. Embed the question and the options
        with torch.no_grad():
            question_embedding = self.model.encode(query_text, convert_to_tensor=True)
            option_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
            
            # 3. Compute Cosine Similarity
            cosine_scores = util.cos_sim(question_embedding, option_embeddings)[0]
            
            # 4. Select the option with the highest similarity
            best_idx = torch.argmax(cosine_scores).item()
        
        self.reasoning_time = time.time() - start_time - self.search_time
        
        print(f"[BERT Baseline] Question: {question_text[:50]}...")
        for i, score in enumerate(cosine_scores):
            print(f"  Option {i} ({option_texts[i][:30]}): Similarity = {score:.4f}")
            
        return options[best_idx].id

    def _infer_with_rag(self, question: Question, theme: str, game_session: Any = None) -> Optional[int]:
        print(f"\n[RAG] Processing question with context-enhanced similarity: {question.text[:100]}...")
        search_start = time.time()

        # 1. Create queries for the question and each option
        queries = [question.text] + [opt.text for opt in question.options]

        # 2. Fetch the best document for each query in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            docs = list(executor.map(self._find_best_document, queries))
        
        self.search_time = time.time() - search_start
        print(f"[RAG] Document fetching took {self.search_time:.2f}s")

        question_doc = docs[0]
        option_docs = docs[1:]

        # 3. Create context-enhanced texts
        question_context = f"{question_doc}\n\n{question.text}" if question_doc else question.text
        option_contexts = [
            f"{doc}\n\n{opt.text}" if doc else opt.text 
            for doc, opt in zip(option_docs, question.options)
        ]

        # 4. Embed and compare
        if self.use_prefixes:
            query_text = f"search_query: {question_context}"
            doc_texts = [f"search_document: {opt_ctx}" for opt_ctx in option_contexts]
        else:
            query_text = question_context
            doc_texts = option_contexts

        with torch.no_grad():
            question_embedding = self.model.encode(query_text, convert_to_tensor=True)
            option_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)

            cosine_scores = util.cos_sim(question_embedding, option_embeddings)[0]
            best_idx = torch.argmax(cosine_scores).item()

        print(f"[RAG] Question (context-enhanced): {question_context[:80]}...")
        for i, score in enumerate(cosine_scores):
            print(f"  Option {i} ({question.options[i].text[:30]}...): Similarity = {score:.4f}")

        return best_idx

    def _find_best_document(self, query: str) -> str:
        """Searches Wikipedia for a query and returns the most relevant document chunk."""
        print(f"[RAG] Searching for: {query[:100]}...")
        chunks = self._wikisearch_single(query)
        if not chunks:
            print(f"[RAG] No documents found for query: {query[:100]}...")
            return ""

        # Use the embedding model to find the most relevant chunk
        if self.use_prefixes:
            query_text = f"search_query: {query}"
            chunk_texts = [f"search_document: {chunk}" for chunk in chunks]
        else:
            query_text = query
            chunk_texts = chunks

        with torch.no_grad():
            query_emb = self.model.encode(query_text, convert_to_tensor=True)
            chunk_embs = self.model.encode(chunk_texts, convert_to_tensor=True)
            
            scores = util.cos_sim(query_emb, chunk_embs)[0]
            best_idx = torch.argmax(scores).item()
            
        print(f"[RAG] Best chunk similarity for '{query[:30]}...': {scores[best_idx]:.4f}")
        return chunks[best_idx]

    def _wikisearch_single(self, query: str) -> List[str]:
        search_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": f"PoliMillionaire-Bot/1.0 ({self.email})"}
        
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3
            }
            res = requests.get(search_url, params=params, headers=headers, timeout=5)
            if res.status_code != 200:
                return []
            
            search_data = res.json()
            if "query" not in search_data:
                return []
            
            titles = [r["title"] for r in search_data["query"]["search"]]
            
            chunks = []
            for title in titles:
                extract_params = {
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json"
                }
                eres = requests.get(search_url, params=extract_params, headers=headers, timeout=5)
                if eres.status_code != 200:
                    continue
                
                pages = eres.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    if "extract" in page:
                        chunks.extend(self._chunk_text(page["extract"]))
            return chunks
        except Exception as e:
            print(f"Wiki search error for query '{query}': {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def play_baseline():
    API_URL = "http://131.175.15.22:51111/"
    # Load credentials from .env
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    
    if not USERNAME or not PASSWORD:
        print("Error: POLI_USERNAME and POLI_PASSWORD must be set in .env")
        return

    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 1. Run standard BERT similarity benchmark
    print("\n>>> Starting Standard BERT Baseline Benchmark (5 runs per competition)")
    config_sim = ExperimentConfig(
        username="Luca",
        notes="BERT baseline using ModernBERT cosine similarity",
        approach=ApproachType.DIRECT_LLM,
        embedding_model="nomic-ai/modernbert-embed-base",
        is_rag=False
    )
    baseline_sim = BertBaseline(config_sim)
    benchmark_sim = Benchmark(config_sim, baseline_sim, client)
    benchmark_sim.run(times_per_competition=5, filename="luca_bert_benchmark_results.xlsx")

    # 2. Run BERT + RAG benchmark
    print("\n>>> Starting BERT + RAG Benchmark (5 runs per competition)")
    config_rag = ExperimentConfig(
        username="Luca",
        notes="Context-enhanced BERT similarity with Wikipedia lookup.",
        approach=ApproachType.RAG,
        embedding_model="nomic-ai/modernbert-embed-base",
        is_rag=True
    )
    baseline_rag = BertBaseline(config_rag)
    benchmark_rag = Benchmark(config_rag, baseline_rag, client)
    benchmark_rag.run(times_per_competition=5, filename="luca_bert_benchmark_results.xlsx")

if __name__ == "__main__":
    play_baseline()
