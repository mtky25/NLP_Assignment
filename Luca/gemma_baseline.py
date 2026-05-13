import torch
import os
import sys
import time
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from dotenv import load_dotenv

# Ensure we can import from the root src directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.guesser.guesser import Guesser
from src.models import ExperimentConfig, ApproachType
from src.millionaire_client.models import Question
from src.benchmark import Benchmark
from src.millionaire_client import MillionaireClient

load_dotenv()

class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, game_session, threshold=5.0):
        self.game_session = game_session
        self.threshold = threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.game_session is None:
            return False
        try:
            remaining = self.game_session.time_remaining
            if remaining is not None and remaining < self.threshold:
                print(f"\n[Warning] Time low ({remaining:.2f}s), stopping reasoning.")
                return True
        except Exception:
            pass
        return False

class GemmaBaseline(Guesser):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Use the model ID from config, or fallback to the baseline default
        model_id = config.inference_model or "google/gemma-4-e2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        self.option_tokens = ["A", "B", "C", "D"]
        self.option_ids = [self.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in self.option_tokens]

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_reasoning = time.time()
        
        question_text = question.text
        options = question.options
        
        opts_str = "\n".join([f"{chr(65+i)}. {opt.text}" for i, opt in enumerate(options)])
        
        prompt = (
            "<|turn>system\n"
            "<|think|>You are a helpful assistant for the 'Who wants to be a PoliMillionaire?' quiz. "
            "Reason step-by-step inside the thought channel.<turn|>\n"
            "<|turn>user\n"
            f"Question: {question_text}\n"
            f"Options:\n{opts_str}\n"
            "Analyze and then state the correct option letter.<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stop_token = "<channel|>"
        stop_token_id = self.tokenizer.convert_tokens_to_ids(stop_token)
        
        # We need a safe buffer for the final prediction on CPU. 8s is a good compromise.
        stopping_criteria = StoppingCriteriaList([TimeStoppingCriteria(game_session, threshold=8.0)])
        
        with torch.no_grad():
            # 1. Generate reasoning with KV caching enabled
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                eos_token_id=stop_token_id,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                use_cache=True
            )
            
            # Get the generated tokens
            gen_tokens = outputs.sequences[0]
            
            # Prepare the final trigger for the answer
            trigger_text = "The correct answer is "
            trigger_ids = self.tokenizer.encode(trigger_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            
            # To make the final pass FAST, we only process the trigger tokens
            final_input_ids = torch.cat([gen_tokens.unsqueeze(0), trigger_ids], dim=-1)
            
            # Final forward pass to get logits for the options
            final_outputs = self.model(final_input_ids)
            logits = final_outputs.logits[0, -1, :]
            
            relevant_logits = logits[self.option_ids]
            best_idx = torch.argmax(relevant_logits).item()
            
            self.reasoning_time = time.time() - start_reasoning
            
            return options[best_idx].id

def play_baseline():
    API_URL = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    
    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"login failed: {e}")
        return

    config = ExperimentConfig(
        username="Luca",
        notes="Gemma baseline using standardized benchmark logic",
        approach=ApproachType.DIRECT_LLM,
        inference_model="google/gemma-4-e2b-it",
        inference_model_size=4,
        is_rag=False
    )
    
    baseline = GemmaBaseline(config)
    benchmark = Benchmark(config, baseline, client)
    
    # Run 1 game per competition for a quick check
    benchmark.run(times_per_competition=1, filename="luca_benchmark_results.xlsx")

if __name__ == "__main__":
    play_baseline()
