import os
import time
from typing import Any
from dotenv import load_dotenv
from random import Random
from src.guesser.guesser import Guesser
from src.models import ExperimentConfig
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient


class RandomBaseline(Guesser):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.random_answer_generator = Random()

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        self.reasoning_time = 0.0

        answer = self.random_answer_generator.randint(0, 3)
        
        time.sleep(0.2)
        return answer

def play_at_random():
    load_dotenv()
    API_URL = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    ATTEMPTS_PER_COMPETITION = 2000
    
    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"Login failed for User {USERNAME}: {e}")
        return

    config_sim = ExperimentConfig(
        username="Luca",
        notes="random baseline",
        is_rag=False,
    )

    baseline_sim = RandomBaseline(config_sim)
    benchmark_sim = Benchmark(config_sim, baseline_sim, client)
    benchmark_sim.run(times_per_competition=ATTEMPTS_PER_COMPETITION, filename="random_benchmark_results.xlsx")

if __name__ == "__main__":
    play_at_random()
