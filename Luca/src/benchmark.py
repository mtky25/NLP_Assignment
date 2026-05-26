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
                    
                except Exception as e: 
                    print(f"Error on game {game_num + 1} of competition {comp_id}: {e}")

        print("Calculating Metrics")
        analysis = Analysis(all_results, self.experiment)
        analysis.calculate_experiments_metrics()
        
        if save:
            self.save_to_excel(filename)

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