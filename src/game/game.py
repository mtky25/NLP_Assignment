import time
from src.guesser.guesser import Guesser
from src.millionaire_client import MillionaireClient
from src.models import QuestionResult, QuestionOutcome

class Game:
    def __init__(self, client : MillionaireClient, guesser : Guesser, competition_id=1):
        self.guesser = guesser
        self.client = client
        self.game = None
        self.competition_id = competition_id
        self.results = []

    def play_game(self):
        self.game = self.client.game.start(competition_id=self.competition_id)
        print(f"Started game in competition: {self.game.state.competition.name}")
        while self.game.in_progress:
            question = self.game.current_question
            if not question:
                break
            
            print(f"\n--- Level {self.game.current_level} ---")
            print(f"Q: {question.text}")
            # Print options for the user to see
            for i, opt in enumerate(question.options):
                print(f"  [{i}] {opt.text}")
            
            # Automated Guessing
            duration = 0.0
            start_t = time.time()
            try:
                print("\nGuesser is thinking...")
                answer_id = self.guesser.infer_answer(question, self.game.state.competition.name, game_session=self.game)
                duration = time.time() - start_t
                
                # Retrieve separated metrics if available
                search_time = getattr(self.guesser, 'search_time', 0.0)
                reasoning_time = getattr(self.guesser, 'reasoning_time', 0.0)
                
                print(f"Guesser chose option: {answer_id} (Time: {duration:.2f}s, Search: {search_time:.2f}s, Reasoning: {reasoning_time:.2f}s)")
            except Exception as e:
                duration = time.time() - start_t
                search_time = getattr(self.guesser, 'search_time', 0.0)
                reasoning_time = getattr(self.guesser, 'reasoning_time', 0.0)
                print(f"Inference error: {e} (Time: {duration:.2f}s, Search: {search_time:.2f}s, Reasoning: {reasoning_time:.2f}s)")
                self.results.append(QuestionResult(
                    theme=self.game.state.competition.name,
                    question_outcome=QuestionOutcome.ERROR,
                    answer_time=duration,
                    search_time=search_time,
                    reasoning_time=reasoning_time,
                    level=self.game.state.current_level,
                ))
                break

            # Submit answer
            result = self.game.answer(answer_id)

            if result.correct:
                print(" CORRECT!")
                question_result = QuestionResult(
                    theme=self.game.state.competition.name,
                    question_outcome=QuestionOutcome.CORRECT,
                    answer_time=duration,
                    search_time=search_time,
                    reasoning_time=reasoning_time,
                    level=self.game.state.current_level,
                )
            elif result.timed_out:
                question_result = QuestionResult(
                    theme=self.game.state.competition.name,
                    question_outcome=QuestionOutcome.TIMEOUT,
                    answer_time=duration,
                    search_time=search_time,
                    reasoning_time=reasoning_time,
                    level=self.game.state.current_level,
                )
            else:
                question_result = QuestionResult(
                    theme=self.game.state.competition.name,
                    question_outcome=QuestionOutcome.INCORRECT,
                    answer_time=duration,
                    search_time=search_time,
                    reasoning_time=reasoning_time,
                    level=self.game.state.current_level,
                )
            self.results.append(question_result)
            if result.game_over:
                status = result.status
                if status is None:
                    # Fallback for when result.status is None (e.g. incorrect answer causing game over)
                    status_text = "INCORRECT" if result.correct is False else "FINISHED"
                else:
                    status_text = status.value if hasattr(status, "value") else str(status)
                
                if result.correct:
                    print(f"CONGRATULATIONS! Final earnings: ${result.earned_amount:,.2f}")
                else:
                    print(f"GAME OVER! Result: {status_text.upper()}")
                break
            

    def get_game_results(self):
        return self.results
