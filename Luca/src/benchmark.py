import os
import dataclasses
import pandas as pd
from datetime import datetime
from typing import List
from src.guesser.guesser import Guesser
from src.analysis.analysis import Analysis
from src.game.game import Game 
from src.models import ExperimentConfig
from src.millionaire_client import MillionaireClient

class Benchmark:
    def __init__(self, experiment: ExperimentConfig, guesser: Guesser, client: MillionaireClient):
        self.experiment = experiment
        self.guesser = guesser
        self.client = client
        self.competitions = [0,1,2,3] 

    def run(self, times_per_competition: int = 5, save: bool = True, filename: str = "benchmark_results.xlsx"):
        all_results = []

        for comp_id in self.competitions:
            for game_num in range(times_per_competition):
                try:
                    game = Game(self.client, guesser=self.guesser, competition_id=comp_id)
                    game.play_game()
                    
                    results = game.get_game_results()
                    all_results.extend(results)
                    
                    # Save correct questions to a dataset file
                    self.save_questions_to_excel(results)
                    
                except Exception as e: 
                    print(f"Error on game {game_num + 1} of competition {comp_id}: {e}")

        print("Calculating Metrics")
        analysis = Analysis(all_results, self.experiment)
        analysis.calculate_experiments_metrics()
        
        if save:
            self.save_to_excel(filename)

    def save_questions_to_excel(self, results: List[QuestionResult], filename: str = "collected_questions.xlsx"):
        # Filter for correct answers only
        correct_results = [res for res in results if res.question_outcome.name == "CORRECT"]
        if not correct_results:
            return

        df_new = pd.DataFrame([dataclasses.asdict(res) for res in correct_results])
        
        try:
            if os.path.exists(filename):
                df_existing = pd.read_excel(filename)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                # Drop duplicates based on question text to avoid redundancy
                df_final = df_final.drop_duplicates(subset=['question_text'])
            else:
                df_final = df_new

            df_final.to_excel(filename, index=False)
            print(f"Saved {len(correct_results)} correct questions to {filename}")
        except Exception as e:
            print(f"Error saving questions to excel: {e}")

    def save_to_excel(self, filename: str = "benchmark_results.xlsx"):
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        data = dataclasses.asdict(self.experiment)

        if data.get('approach'):
            data['approach'] = data['approach'].value

        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_new = pd.DataFrame([data])

        try:
            if os.path.exists(filename):
                df_existing = pd.read_excel(filename)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = df_new

            df_final.to_excel(filename, index=False)
            print(f"Results successfully saved at: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")