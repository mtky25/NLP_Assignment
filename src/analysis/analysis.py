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
        total_correct = 0
        total_count = len(self.results)
        theme_stats = defaultdict(lambda: {"total_time": 0.0, "correct_count": 0, "count": 0})

        for res in self.results:
            is_correct = 1 if res.question_outcome.name == "CORRECT" else 0
            total_time += res.answer_time
            total_correct += is_correct

            theme_stats[res.theme]["total_time"] += res.answer_time
            theme_stats[res.theme]["correct_count"] += is_correct
            theme_stats[res.theme]["count"] += 1

        self.experiment.mean_time = total_time / total_count
        self.experiment.mean_question_accuracy = total_correct / total_count

        theme_map = {
            "Entertainment": "entertainment",
            "Ancient History and Politics": "ancient_history",
            "Science and Nature": "science_nature",
            "Maths": "maths",
        }

        for theme_name, stats in theme_stats.items():
            prefix = theme_map.get(theme_name)
            
            if prefix and stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                avg_acc = stats["correct_count"] / stats["count"]
                
                attr_time = f"{prefix}_mean_time"
                attr_acc = f"{prefix}_mean_question_accuracy"
                
                if hasattr(self.experiment, attr_time):
                    setattr(self.experiment, attr_time, avg_time)
                if hasattr(self.experiment, attr_acc):
                    setattr(self.experiment, attr_acc, avg_acc)

