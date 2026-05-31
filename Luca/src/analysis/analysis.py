from collections import defaultdict
from src.models import ExperimentConfig, QuestionResult

class Analysis:
    def __init__(self, results: list[QuestionResult], experiment: ExperimentConfig):
        self.results = results
        self.experiment = experiment

    def calculate_experiments_metrics(self) -> None:
        if not self.results:
            return

        total_time = 0.0
        total_search_time = 0.0
        total_reasoning_time = 0.0
        total_transcription_time = 0.0
        total_correct = 0
        total_count = len(self.results)
        
        theme_stats = defaultdict(lambda: {
            "total_time": 0.0, 
            "total_search_time": 0.0,
            "total_reasoning_time": 0.0,
            "total_transcription_time": 0.0,
            "correct_count": 0, 
            "count": 0
        })

        for res in self.results:
            is_correct = 1 if res.question_outcome.name == "CORRECT" else 0
            total_time += res.answer_time
            total_search_time += res.search_time
            total_reasoning_time += res.reasoning_time
            total_transcription_time += res.transcription_time
            total_correct += is_correct

            theme_stats[res.theme]["total_time"] += res.answer_time
            theme_stats[res.theme]["total_search_time"] += res.search_time
            theme_stats[res.theme]["total_reasoning_time"] += res.reasoning_time
            theme_stats[res.theme]["total_transcription_time"] += res.transcription_time
            theme_stats[res.theme]["correct_count"] += is_correct
            theme_stats[res.theme]["count"] += 1

        self.experiment.mean_time = total_time / total_count
        self.experiment.mean_search_time = total_search_time / total_count
        self.experiment.mean_reasoning_time = total_reasoning_time / total_count
        self.experiment.mean_transcription_time = total_transcription_time / total_count
        self.experiment.mean_question_accuracy = total_correct / total_count

        theme_map = {
            "Entertainment": "entertainment",
            "Ancient History and Politics": "ancient_history",
            "Science and Nature": "science_nature",
            "Maths": "maths",
            "News": "news",
            "Philosophy and Psychology": "philosophy_psychology",
        }

        for theme_name, stats in theme_stats.items():
            prefix = theme_map.get(theme_name)
            
            if prefix and stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                avg_search = stats["total_search_time"] / stats["count"]
                avg_reasoning = stats["total_reasoning_time"] / stats["count"]
                avg_transcription = stats["total_transcription_time"] / stats["count"]
                avg_acc = stats["correct_count"] / stats["count"]
                
                # Attribute names
                attr_time = f"{prefix}_mean_time"
                attr_search = f"{prefix}_mean_search_time"
                attr_reasoning = f"{prefix}_mean_reasoning_time"
                attr_transcription = f"{prefix}_mean_transcription_time"
                attr_acc = f"{prefix}_mean_question_accuracy"
                
                if hasattr(self.experiment, attr_time):
                    setattr(self.experiment, attr_time, avg_time)
                if hasattr(self.experiment, attr_search):
                    setattr(self.experiment, attr_search, avg_search)
                if hasattr(self.experiment, attr_reasoning):
                    setattr(self.experiment, attr_reasoning, avg_reasoning)
                if hasattr(self.experiment, attr_transcription):
                    setattr(self.experiment, attr_transcription, avg_transcription)
                if hasattr(self.experiment, attr_acc):
                    setattr(self.experiment, attr_acc, avg_acc)
