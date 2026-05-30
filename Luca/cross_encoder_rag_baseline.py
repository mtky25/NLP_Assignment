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
        
        # 1. Bi-Encoder for retrieval
        self.bi_encoder_id = config.embedding_model
        # 2. Cross-Encoder for accurate re-ranking/classification
        self.cross_encoder_id = config.inference_model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(self.bi_encoder_id, device=device)
        
        # Ettin-1B is a significantly larger model (1B params).
        # We use bfloat16 to reduce VRAM usage and trust_remote_code for ModernBERT architecture.
        self.cross_encoder = CrossEncoder(
            self.cross_encoder_id, 
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {}
        )

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()
        
        wiki = wikipediaapi.Wikipedia(user_agent='Polimillionaire', language='en')
        search_start = time.time()

        # 1. FETCH & CONTEXTUALIZE CANDIDATE SECTIONS
        all_candidate_sections = set()
        
        # Phase 1: Search for the Question Topic
        q_title, _ = self._fetch_question_data(wiki, question.text)
        if q_title:
            all_candidate_sections.update(self._get_sections_for_query(wiki, q_title))
        
        # Phase 2: Expanded Search for each Option
        # We context-anchor the option search using the question's main article title (e.g., "2 - Adele")
        for option in question.options:
            search_query = f"{option.text} {q_title}" if q_title else f"{option.text} {question.text}"
            all_candidate_sections.update(self._get_sections_for_query(wiki, search_query))

        candidate_sections = list(all_candidate_sections)
        self.search_time = time.time() - search_start

        if not candidate_sections:
            # Fallback: Score options directly against the question
            model_inputs = [(question.text, option.text) for option in question.options]
            scores = self.cross_encoder.predict(model_inputs, activation_fn=torch.nn.Sigmoid())
            best_option_idx = np.argmax(scores)
        else:
            # 2. STEP 1: RETRIEVAL (Bi-Encoder)
            # Filter a large pool of sections down to the top K globally for re-ranking
            TOP_K_RETRIEVAL = 15
            with torch.no_grad():
                question_emb = self.bi_encoder.encode(question.text, convert_to_tensor=True)
                section_embeddings = self.bi_encoder.encode(candidate_sections, convert_to_tensor=True)
                cos_scores = util.cos_sim(question_emb, section_embeddings)[0]
                
                top_k = min(TOP_K_RETRIEVAL, len(candidate_sections))
                top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
                top_sections = [candidate_sections[i] for i in top_indices]

            # 3. STEP 2: RE-RANKING (Cross-Encoder)
            # We use an NLI-style declarative statement ("Answer is Option") to verify against context
            all_pairs = []
            for option in question.options:
                statement = f"The answer to '{question.text}' is {option.text}."
                for section in top_sections:
                    all_pairs.append((statement, section))
            
            print(f"[Cross-RAG] Batch predicting {len(all_pairs)} pairs using Ettin-1B...")
            all_scores = self.cross_encoder.predict(all_pairs, activation_fn=torch.nn.Sigmoid(), batch_size=4)
            
            # Reshape scores to (num_options, num_sections)
            all_scores = all_scores.reshape(len(question.options), len(top_sections))
            
            # 4. SCORE AGGREGATION
            # Sum the top 3 scores per option to find the most consistently supported answer
            option_final_scores = []
            for opt_idx in range(len(question.options)):
                opt_scores = all_scores[opt_idx]
                top_3_sum = np.sum(np.sort(opt_scores)[-3:])
                option_final_scores.append(top_3_sum)
                print(f"  Option {opt_idx} ({question.options[opt_idx].text[:20]}...): Aggregate Confidence = {top_3_sum:.4f}")
            
            best_option_idx = np.argmax(option_final_scores)

        self.reasoning_time = time.time() - start_time - self.search_time

        print(f"[Cross-RAG] Search & Reasoning took {self.search_time + self.reasoning_time:.2f}s")
        print(f"[Cross-RAG] Selected Option {best_option_idx}")

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

    def _get_sections_for_query(self, wiki: wikipediaapi.Wikipedia, query: str) -> list[str]:
        """Retrieves contextualized sections from Wikipedia for a given query."""
        try:
            TOP_N_PAGES = 1
            results = wiki.search(query)
            if not results.pages:
                return []

            top_titles = list(results.pages.keys())[:TOP_N_PAGES]
            
            sections = []
            for title in top_titles:
                page = results.pages[title]
                sections.extend(self._get_page_sections(page))
            return sections
        except Exception as e:
            print(f"[Cross-RAG] Error fetching sections for '{query}': {e}")
            return []

    def _get_page_sections(self, page: wikipediaapi.WikipediaPage) -> list[str]:
        """Extracts clean text sections from a Wikipedia page with context headers."""
        sections_text = [f"[{page.title}] {page.summary}"]
        exclude = {"See also", "References", "Further reading", "External links", "Notes", "Sources", "Bibliography", "Gallery"}
        
        for s in page.sections:
            if s.title not in exclude and s.text.strip():
                sections_text.append(f"[{page.title} - {s.title}] {s.text}")
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
        notes="Retrieve & Re-rank RAG (Bi-Encoder + Ettin-1B Reranker)",
        approach=ApproachType.RAG,
        is_rag=True,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_model_size=0.1, # 0.1B
        inference_model="cross-encoder/ettin-reranker-1b-v1",
        inference_model_size=1.0, # 1B
    )
    baseline_rag = CrossEncoderRAGBaseline(config_rag)
    benchmark_rag = Benchmark(config_rag, baseline_rag, client)
    benchmark_rag.run(times_per_competition=ATTEMPTS_PER_COMPETITION, filename="luca_cross_rag_results.xlsx")

if __name__ == "__main__":
    play_cross_rag_baseline()
