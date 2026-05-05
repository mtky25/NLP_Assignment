import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from millionaire_client import MillionaireClient
import os
from collections import defaultdict
import time

API_URL = "http://131.175.15.22:51111/"
USERNAME = os.getenv("POLI_USERNAME", "")
PASSWORD = os.getenv("POLI_PASSWORD", "")

class GemmaBaseline:
    def __init__(self, model_id="google/gemma-4-e2b-it"):
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # the model stays on cpu, it is small enough
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu", 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # find token ids for a, b, c, d
        self.option_tokens = ["A", "B", "C", "D"]
        self.option_ids = [self.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in self.option_tokens]

    def select_answer(self, question_text, options):
        # construct the prompt - simple format for fast logit extraction
        opts_str = ", ".join([f"{chr(65+i)}. {opt.text}" for i, opt in enumerate(options)])
        prompt = f"please answer the following quiz question: {question_text}. options: {opts_str}. the correct answer is option"
        
        # the prompt goes into tokens
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # run a single forward pass (fast)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # take the last token's logits
            logits = outputs.logits[0, -1, :]
            
            # look for a, b, c, d only
            relevant_logits = logits[self.option_ids]
            # choose the highest logit
            best_idx = torch.argmax(relevant_logits).item()
            
            # return the option id from the game
            return options[best_idx].id

def play_baseline():
    # initialize the client
    client = MillionaireClient(API_URL)
    
    # login
    try:
        client.login(USERNAME, PASSWORD)
    except Exception as e:
        print(f"login failed: {e}")
        return

    # instantiate the baseline model
    baseline = GemmaBaseline()
    
    # competitions to play
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
                        
                    # the model decides the answer (fast logit method)
                    answer_id = baseline.select_answer(question.text, question.options)
                    
                    # submit answer
                    result = game.answer(answer_id)
                    
                earnings = game.earned_amount
                results[comp_id].append(earnings)
                print(f"run {run + 1} finished. earnings: {earnings}")
            except Exception as e:
                print(f"error during run {run + 1}: {e}")
                results[comp_id].append(0)

    # calculate statistics
    all_scores = [score for comp_scores in results.values() for score in comp_scores]
    
    print("\n--- final statistics ---")
    if all_scores:
        avg_overall = sum(all_scores) / len(all_scores)
        max_overall = max(all_scores)
        min_overall = min(all_scores)
        print(f"overall average score: {avg_overall:.2f}")
        print(f"overall highest score: {max_overall}")
        print(f"overall lowest score: {min_overall}")
        
        for comp_id in comp_ids:
            comp_scores = results[comp_id]
            if comp_scores:
                avg_comp = sum(comp_scores) / len(comp_scores)
                max_comp = max(comp_scores)
                min_comp = min(comp_scores)
                print(f"competition {comp_id} average: {avg_comp:.2f}")
                print(f"competition {comp_id} highest: {max_comp}")
                print(f"competition {comp_id} lowest: {min_comp}")
    else:
        print("no games were completed.")

if __name__ == "__main__":
    play_baseline()
