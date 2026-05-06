HF_SCIENCE_DATASET = {
    "hf_dataset": "sciq",
    "hf_subset": None,
    "split": "train",
    "col_text": "support",
    "cols_meta": ["question", "correct_answer"],
}

HF_MATH_DATASET = {
    "hf_dataset": "gsm8k",
    "hf_subset": "main",
    "split": "train",
    "col_text": "question", 
    "cols_meta": ["answer"],
}