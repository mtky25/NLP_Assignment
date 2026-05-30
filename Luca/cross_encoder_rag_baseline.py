import torch
import os
import time
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import wikipediaapi
from dotenv import load_dotenv

from src.guesser.guesser import Guesser
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient


class CrossEncoderRAGBaseline(Guesser):
    """
    A Guesser using a Retrieve and Re-rank approach.
    It retrieves candidate Wikipedia sections using a fast Bi-Encoder,
    and then uses a Cross-Encoder to score how well the context + option answers the question.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        
        # 1. Bi-Encoder for
        self.bi_encoder_id = config.embedding_model
        # 2. Cross-Encoder for accurate re-ranking/classification
        self.cross_encoder_id = config.inference_model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(self.bi_encoder_id, device=device)
        self.cross_encoder = CrossEncoder(self.cross_encoder_id, device=device)

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()
        
        wiki = wikipediaapi.Wikipedia(user_agent='Polimillionaire', language='en')
        search_start = time.time()

        # 1. Fetch sections for the Question and all Options
        # We retrieve sections specifically for the question and also for each option to ensure coverage
        all_candidate_sections = set()
        
        # Sections for the Question
        q_title, _ = self._fetch_question_data(wiki, question.text)
        all_candidate_sections.update(self._get_sections_for_query(wiki, question.text, exclude_title=None))
        
        # Sections for each Option
        for option in question.options:
            all_candidate_sections.update(self._get_sections_for_query(wiki, f"{option.text} {question.text}", exclude_title=q_title))

        candidate_sections = list(all_candidate_sections)
        self.search_time = time.time() - search_start

        if not candidate_sections:
            # Fallback: Score options directly against the question
            model_inputs = [(question.text, option.text) for option in question.options]
            scores = self.cross_encoder.predict(model_inputs, activation_fn=torch.nn.Sigmoid())
            best_option_idx = np.argmax(scores)
            best_overall_score = scores[best_option_idx]
        else:
            # 2. STEP 1: RETRIEVAL (Bi-Encoder) - Filter down to top sections globally
            TOP_K_RETRIEVAL = 10
            with torch.no_grad():
                question_emb = self.bi_encoder.encode(question.text, convert_to_tensor=True)
                section_embeddings = self.bi_encoder.encode(candidate_sections, convert_to_tensor=True)
                cos_scores = util.cos_sim(question_emb, section_embeddings)[0]
                
                top_k = min(TOP_K_RETRIEVAL, len(candidate_sections))
                top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
                top_sections = [candidate_sections[i] for i in top_indices]

            # 3. STEP 2: RE-RANKING (Cross-Encoder) - Batch predict all (Option, Section) pairs
            # We want to find the max score for each option across all top sections
            all_pairs = []
            for option in question.options:
                for section in top_sections:
                    all_pairs.append((question.text, f"{option.text}. {section}"))
            
            print(f"[Cross-RAG] Batch predicting {len(all_pairs)} pairs...")
            # activation_fct=torch.nn.Sigmoid() ensures scores are in [0, 1]
            all_scores = self.cross_encoder.predict(all_pairs, activation_fn=torch.nn.Sigmoid(), batch_size=32)
            
            # Reshape scores to (num_options, num_sections) and take the max per option
            all_scores = all_scores.reshape(len(question.options), len(top_sections))
            option_max_scores = np.max(all_scores, axis=1)
            
            for opt_idx, score in enumerate(option_max_scores):
                print(f"  Option {opt_idx} ({question.options[opt_idx].text[:20]}...): Confidence = {score:.4f}")
            
            best_option_idx = np.argmax(option_max_scores)
            best_overall_score = option_max_scores[best_option_idx]

        self.reasoning_time = time.time() - start_time - self.search_time

        print(f"[Cross-RAG] Search & Reasoning took {self.search_time + self.reasoning_time:.2f}s")
        print(f"[Cross-RAG] Selected Option {best_option_idx} with Confidence: {best_overall_score:.4f}")

        return question.options[best_option_idx].id

    def _fetch_question_data(self, wiki: wikipediaapi.Wikipedia, text: str) -> tuple[str, str]:
        """Fetches the top article title and summary for the question."""
        try:
            results = wiki.search(text)
            if results.pages:
                title = list(results.pages.keys())[0]
                return title, results.pages[title].summary
        except Exception as e:
            print(f"[Cross-RAG] Error searching for question: {e}")
        return "", ""

    def _get_sections_for_query(self, wiki: wikipediaapi.Wikipedia, query: str, exclude_title: str = None) -> list[str]:
        """Retrieves sections from Wikipedia for a given query."""
        try:
            TOP_N_PAGES = 1
            results = wiki.search(query)
            if not results.pages:
                return []

            top_titles = []
            for title in results.pages.keys():
                if title != exclude_title:
                    top_titles.append(title)
                if len(top_titles) == TOP_N_PAGES:
                    break
            
            sections = []
            for title in top_titles:
                page = results.pages[title]
                sections.extend(self._get_page_sections(page))
            return sections
        except Exception as e:
            print(f"[Cross-RAG] Error fetching sections for '{query}': {e}")
            return []

    def _get_page_sections(self, page: wikipediaapi.WikipediaPage) -> list[str]:
        """Extracts clean text sections from a Wikipedia page."""
        sections_text = [page.summary]
        exclude = {"See also", "References", "Further reading", "External links", "Notes", "Sources", "Bibliography", "Gallery"}
        
        for s in page.sections:
            if s.title not in exclude and s.text.strip():
                sections_text.append(s.text)
        return sections_text


def play_cross_rag_baseline():
    load_dotenv()
    API_URL = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    ATTEMPTS_PER_COMPETITION = 2
    
    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Login failed for User {USERNAME}: {e}")
        return

    config_rag = ExperimentConfig(
        username="Luca",
        notes="Retrieve & Re-rank RAG (Bi-Encoder + Cross-Encoder)",
        approach=ApproachType.RAG,
        is_rag=True,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_model_size=0.1, # 0.1B
        inference_model="cross-encoder/ms-marco-MiniLM-L12-v2",
        inference_model_size=0.033, # 33.4M
    )
    baseline_rag = CrossEncoderRAGBaseline(config_rag)
    benchmark_rag = Benchmark(config_rag, baseline_rag, client)
    benchmark_rag.run(times_per_competition=ATTEMPTS_PER_COMPETITION, filename="luca_cross_rag_results.xlsx")

if __name__ == "__main__":
    play_cross_rag_baseline()
