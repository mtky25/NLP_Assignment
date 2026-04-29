import pandas as pd
import os
from datetime import datetime
import time
import json
from src.guesser.guesser import Guesser
from millionaire_client import MillionaireClient

class Game:
    def __init__(self, client : MillionaireClient, guesser : Guesser):
        self.guesser = guesser
        self.client = client
        self.game = None
        self.times = []
        self.question_details = []

    def init_game(self, competition_id=1):
        # Start the game using the client's game module
        self.game = self.client.game.start(competition_id=competition_id)
        print(f"Started game in competition: {self.game.state.competition.name}")

    def play_game(self):
        # Reset trackers for this specific game
        self.times = []
        self.question_details = []
        
        while self.game.in_progress:
            question = self.game.current_question
            if not question:
                break
            
            self.guesser.add_question(question)

            print(f"\n--- Level {self.game.current_level} ---")
            print(f"Q: {question.text}")
            
            # Print options for the user to see
            for i, opt in enumerate(question.options):
                print(f"  [{i}] {opt.text}")
            
            # Automated Guessing
            try:
                print("\nGuesser is thinking...")
                start_t = time.time()
                answer_id = self.guesser.infer_answer()
                duration = time.time() - start_t
                self.times.append(duration)
                print(f"Guesser chose option: {answer_id} (Time: {duration:.2f}s)")
            except Exception as e:
                print(f"Inference error: {e}")
                break

            # Submit answer
            result = self.game.answer(answer_id)

            # Store per-question details
            self.question_details.append({
                "model": self.guesser.model_name,
                "question": question.text,
                "answer_options": json.dumps([opt.text for opt in question.options]),
                "theme": self.game.state.competition.name,
                "time": round(duration, 2),
                "guesser_answer": answer_id,
                "correct_answer": result.correct,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            if result.correct:
                print(" CORRECT!")
                if result.game_over:
                    print(f"CONGRATULATIONS! Final earnings: ${result.earned_amount:,.2f}")
            else:
                status_text = result.status if result.status else self.game.state.status.value
                print(f" GAME OVER! Result: {status_text.upper()}")
                break

        # Export results
        self.export_to_excel()
        self.export_detailed_results()

    def export_to_excel(self, filename="game_results.xlsx"):
        full_path = os.path.join("Marcelo", filename)
        
        # Calculate metrics
        avg_time = sum(self.times) / len(self.times) if self.times else 0
        
        # correct_answers: current_level - 1 if game ended, 
        # unless it was the final correct answer completing the game.
        correct_count = self.game.current_level - 1
        if self.game.state.status.value == "completed":
             correct_count = self.game.current_level

        new_entry = {
            "model_name": [self.guesser.model_name],
            "correct_answers": [correct_count],
            "average_time": [round(avg_time, 2)],
            "questions_theme": [self.game.state.competition.name],
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        df_new = pd.DataFrame(new_entry)
        
        if os.path.exists(full_path):
            try:
                df_old = pd.read_excel(full_path)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                print(f"Could not read existing Excel file: {e}. Creating new one.")
                df_final = df_new
        else:
            df_final = df_new
            
        df_final.to_excel(full_path, index=False)
        print(f"\nGame metrics for '{self.game.state.competition.name}' saved to {full_path}")

    def export_detailed_results(self, filename="detailed_game_results.xlsx"):
        full_path = os.path.join("Marcelo", filename)
        
        if not self.question_details:
            return

        df_new = pd.DataFrame(self.question_details)
        
        if os.path.exists(full_path):
            try:
                df_old = pd.read_excel(full_path)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                print(f"Could not read existing detailed Excel file: {e}. Creating new one.")
                df_final = df_new
        else:
            df_final = df_new
            
        df_final.to_excel(full_path, index=False)
        print(f"Detailed question results saved to {full_path}")
