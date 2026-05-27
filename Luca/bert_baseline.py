import torch
import os
import time
from typing import Any
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from src.guesser.guesser import Guesser
from src.models import ExperimentConfig
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient


class BertBaseline(Guesser):
    """
    A Baseline Guesser using a BERT sentence transformer
    The option with the highest cosine similarity to the question is chosen
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = config.embedding_model
        
        print(f"Loading SentenceTransformer model: {self.model_id}")
        self.model = SentenceTransformer(self.model_id, device=device)

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()

        # extract questions and options
        question_text = question.text
        options = question.options
        option_texts = [opt.text for opt in options]
        
        with torch.no_grad():
            # calculate embedding
            question_embedding = self.model.encode(question_text, convert_to_tensor=True)
            option_embeddings = self.model.encode(option_texts, convert_to_tensor=True)
            
            # compute cosine similarity
            cosine_sim = util.cos_sim(question_embedding, option_embeddings)[0]
            best_idx = torch.argmax(cosine_sim).item()
        
        self.reasoning_time = time.time() - start_time - self.search_time
        
        # print out scores
        print(f"Question: {question_text[:50]}...")
        for i, score in enumerate(cosine_sim):
            print(f"  Option {i} ({option_texts[i][:30]}): Similarity = {score:.4f}")
            
        return options[best_idx].id

def play_baseline():
    load_dotenv()
    API_URL = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    ATTEMPTS_PER_COMPETITION = 5
    
    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Login failed for User {USERNAME}: {e}")
        return

    config_sim = ExperimentConfig(
        username="Luca",
        notes="BERT baseline",
        #approach="SentenceTransformer",
        #embedding_model="nomic-ai/modernbert-embed-base",
        embedding_model="nomic-ai/nomic-embed-text-v1.5", # medium large
        #embedding_model_size=0,
        is_rag=False,
    )

    baseline_sim = BertBaseline(config_sim)
    benchmark_sim = Benchmark(config_sim, baseline_sim, client)
    benchmark_sim.run(times_per_competition=ATTEMPTS_PER_COMPETITION, filename="luca_bert_benchmark_results.xlsx")

if __name__ == "__main__":
    play_baseline()
