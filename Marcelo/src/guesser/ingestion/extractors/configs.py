HF_SCIENCE_DATASET = {
    "hf_dataset": "sciq",
    "hf_subset": None,
    "split": "train",
    "col_text": ["support ","question"],     
    "cols_meta": ["correct_answer"],
}

HF_MATH_GSM8K_DATASET = {
    "hf_dataset": "gsm8k",
    "hf_subset": "main",
    "split": "train",
    "col_text": ["question","answer"],     
    "cols_meta": [],
}

HF_MATH500_DATASET = {
    "hf_dataset": "HuggingFaceH4/MATH-500",
    "hf_subset": "default",
    "split": "test",
    "col_text": ["problem","solution"], 
    "cols_meta": ["subject"],
}

HF_ANCIENT_HISTORY_DATASET = {
    "hf_dataset": "kth8/OpenTriviaQA",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["question","answer"],     
    "cols_meta": ["question"],
     "filters": {
        "category": ["history", "world"]
    }
}

HF_ENTERTAINMENT_DATASET = {
    "hf_dataset": "kth8/OpenTriviaQA",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["question","answer"],     
    "cols_meta": ["category"],
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

