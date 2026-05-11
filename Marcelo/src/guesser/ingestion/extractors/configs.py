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

HF_ANCIENT_HISTORY_DATASET = {
    "hf_dataset": "kth8/OpenTriviaQA",
    "hf_subset": "default",
    "split": "train",
    "col_text": "answer", 
    "cols_meta": ["question"],
     "filters": {
        "category": ["history", "world"]
    }
}

HF_ENTERTAINMENT_DATASET = {
    "hf_dataset": "kth8/OpenTriviaQA",
    "hf_subset": "default",
    "split": "train",
    "col_text": "answer", 
    "cols_meta": ["question"],
     "filters": {
        "category": ["music",
                      "television",
                      "movies",
                      "celebrities",
                      "sports",
                      "newest",
                      "entertainment",
                      "literature",
                      ]
    }
}