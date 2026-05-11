import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from millionaire_client import MillionaireClient
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://131.175.15.22:51111/"
USERNAME = os.getenv("POLI_USERNAME", "")
PASSWORD = os.getenv("POLI_PASSWORD", "")

class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, game, threshold=5.0):
        self.game = game
        self.threshold = threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        remaining = self.game.time_remaining
        if remaining is not None and remaining < self.threshold:
            print(f"\n[Warning] Time low ({remaining:.2f}s), stopping reasoning.")
            return True
        return False

class GemmaBaseline:
    def __init__(self, model_id="google/gemma-4-e2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        self.option_tokens = ["A", "B", "C", "D"]
        self.option_ids = [self.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in self.option_tokens]

    def select_answer(self, game, question_text, options):
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
        stopping_criteria = StoppingCriteriaList([TimeStoppingCriteria(game, threshold=8.0)])
        
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
            # We combine the previous sequences with the trigger
            final_input_ids = torch.cat([gen_tokens.unsqueeze(0), trigger_ids], dim=-1)
            
            # Final forward pass to get logits for the options
            final_outputs = self.model(final_input_ids)
            logits = final_outputs.logits[0, -1, :]
            
            relevant_logits = logits[self.option_ids]
            best_idx = torch.argmax(relevant_logits).item()
            
            return options[best_idx].id

def play_baseline():
    client = MillionaireClient(API_URL)
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"login failed: {e}")
        return

    baseline = GemmaBaseline()
    comp_ids = [0, 1, 2, 3]
    runs_per_comp = 5
    results = defaultdict(list)
    
    for comp_id in comp_ids:
        print(f"starting games for competition {comp_id}")
        for run in range(runs_per_comp):
            try:
                game = client.game.start(competition_id=comp_id)
                print(f"run {run + 1}/{runs_per_comp} for competition {comp_id} started")
                
                while game.in_progress:
                    question = game.current_question
                    if not question:
                        break
                        
                    print(f"\n--- New Question ---")
                    print(f"Time remaining: {game.time_remaining:.1f}s")
                    
                    answer_id = baseline.select_answer(game, question.text, question.options)
                    
                    result = game.answer(answer_id)
                    status = "Correct" if result.correct else "Wrong"
                    if result.timed_out: status = "Timed Out"
                    
                    print(f"Result: {status} | Level: {result.current_level or 0}")
                    
                earnings = game.earned_amount
                results[comp_id].append(earnings)
                print(f"run {run + 1} finished. earnings: {earnings}")
            except Exception as e:
                print(f"error during run {run + 1}: {e}")
                results[comp_id].append(0)

    all_scores = [score for comp_scores in results.values() for score in comp_scores]
    print("\n--- final statistics ---")
    if all_scores:
        avg_overall = sum(all_scores) / len(all_scores)
        print(f"overall average score: {avg_overall:.2f}")
        for comp_id in comp_ids:
            comp_scores = results[comp_id]
            if comp_scores:
                print(f"competition {comp_id} average: {sum(comp_scores) / len(comp_scores):.2f}")
    else:
        print("no games were completed.")

if __name__ == "__main__":
    play_baseline()
