import torch
import os
import time
from typing import Any
from sentence_transformers import SentenceTransformer, util
import wikipediaapi
from dotenv import load_dotenv
from src.guesser.guesser import Guesser
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient


class BertRAGBaseline(Guesser):
    """
    A Baseline Guesser using a BERT sentence transformer
    For the question and each option we search for the appropriate Wikipedia document
    We embed the returned document and choose the option with the highest cosine similarity 
    to the embedding of questions document
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model_id = config.embedding_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_id, device=device)

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()
        
        wiki = wikipediaapi.Wikipedia(user_agent='Polimillionaire (lucaadrian.ruf@mail.polimi.it)', language='en')
        search_start = time.time()

        # 1. Prepare Question Context
        question_title, question_summary = self._fetch_question_data(wiki, question.text)
        question_context = f"{question.text}\n\n{question_summary}" if question_summary else question.text
        
        with torch.no_grad():
            question_embedding = self.model.encode(question_context, convert_to_tensor=True)

        # 2. Score Options
        best_option_idx = 0
        best_overall_score = -1.0

        for opt_idx, option in enumerate(question.options):
            score = self._score_option(wiki, option.text, question_embedding, question_title or question.text[:50], question_title)
            
            print(f"  Option {opt_idx} ({option.text[:20]}...): Best Section Similarity = {score:.4f}")
            if score > best_overall_score:
                best_overall_score = score
                best_option_idx = opt_idx

        self.search_time = time.time() - search_start
        self.reasoning_time = time.time() - start_time - self.search_time

        print(f"[RAG] Search & Reasoning took {self.search_time + self.reasoning_time:.2f}s")
        print(f"[RAG] Selected Option {best_option_idx} with score: {best_overall_score:.4f}")

        return question.options[best_option_idx].id

    def _fetch_question_data(self, wiki: wikipediaapi.Wikipedia, text: str) -> tuple[str, str]:
        """Fetches the top article title and summary for the question."""
        try:
            results = wiki.search(text)
            if results.pages:
                title = list(results.pages.keys())[0]
                return title, results.pages[title].summary
        except Exception as e:
            print(f"[RAG] Error searching for question: {e}")
        return "", ""

    def _score_option(self, wiki: wikipediaapi.Wikipedia, opt_text: str, question_emb: torch.Tensor, context_query: str, exclude_title: str) -> float:
        """Calculates the best similarity score for an option using Wikipedia sections from top articles."""
        try:
            TOP_N_PAGES = 1
            TOP_N_SECTIONS = 4
            query = f"{opt_text} - {context_query}"
            results = wiki.search(query)
            
            if not results.pages:
                return self._get_direct_similarity(opt_text, question_emb)

            # Evaluate up to top n articles, excluding the article found for the question
            top_titles = []
            for title in results.pages.keys():
                if title != exclude_title:
                    top_titles.append(title)
                if len(top_titles) == TOP_N_PAGES:
                    break
            
            all_sections_text = []
            for title in top_titles:
                page = results.pages[title]
                sections = self._get_page_sections(page)
                # Append the option text to the end of each section to provide more context for the similarity score
                sections_with_opt = [f"{s}\n\n{opt_text}" for s in sections]
                for section in sections_with_opt:
                    all_sections_text.append(section)
                    if len(all_sections_text) == TOP_N_SECTIONS:
                        break
                
            if not all_sections_text:
                return self._get_direct_similarity(opt_text, question_emb)
            
            with torch.no_grad():
                section_embeddings = self.model.encode(all_sections_text, convert_to_tensor=True)
                scores = util.cos_sim(question_emb, section_embeddings)[0]
                return torch.max(scores).item()
                
        except Exception as e:
            print(f"[RAG] Error scoring option '{opt_text}': {e}")
            return self._get_direct_similarity(opt_text, question_emb)

    def _get_page_sections(self, page: wikipediaapi.WikipediaPage) -> list[str]:
        """Extracts clean text sections from a Wikipedia page."""
        sections_text = [page.summary]
        exclude = {"See also", "References", "Further reading", "External links", "Notes", "Sources", "Bibliography", "Gallery"}
        
        for s in page.sections:
            if s.title not in exclude and s.text.strip():
                sections_text.append(s.text)
        return sections_text

    def _get_direct_similarity(self, text: str, question_emb: torch.Tensor) -> float:
        """Fallback: computes direct similarity between text and question embedding."""
        with torch.no_grad():
            opt_emb = self.model.encode(text, convert_to_tensor=True)
            return util.cos_sim(question_emb, opt_emb)[0].item()


def play_rag_baseline():
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
        notes="BERT + RAG",
        approach=ApproachType.RAG,
        #embedding_model="nomic-ai/modernbert-embed-base", # big + slow
        #embedding_model="all-MiniLM-L6-v2", # small + fast
        # embedding_model="BAAI/bge-small-en-v1.5", # medium small
        embedding_model="nomic-ai/nomic-embed-text-v1.5", # medium large
        #embedding_model_size=0,
        is_rag=True
    )
    baseline_rag = BertRAGBaseline(config_rag)
    benchmark_rag = Benchmark(config_rag, baseline_rag, client)
    benchmark_rag.run(times_per_competition=ATTEMPTS_PER_COMPETITION, filename="luca_bert_rag_results.xlsx")

if __name__ == "__main__":
    play_rag_baseline()
