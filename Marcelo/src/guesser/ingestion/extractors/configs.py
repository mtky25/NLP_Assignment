#-----------------------------SCIENCE AND NATURE---------------------------

HF_SCIENCE_DATASET = {
    "hf_dataset": "sciq",
    "hf_subset": None,
    "split": "train",
    "col_text": ["support ","question"],     
    "cols_meta": ["correct_answer"],
}
#NOT INGESTED
HF_PHYSICS_DATASET = {
    "hf_dataset": "gallen881/arxiv-physics",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["support ","question"],     
    "cols_meta": ["question"],
}
#NOT INGESTED
HF_PHYSICS_DATASET_2 = {
    "hf_dataset": "camel-ai/physics",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["message_1 ","message_2"],     
    "cols_meta": ["topic;","sub_topic"],
}
#NOT INGESTED
HF_CHEMISTRY_DATASET = {
    "hf_dataset": "camel-ai/chemistry",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["message_1 ","message_2"],     
    "cols_meta": ["topic;","sub_topic"],
}
#NOT INGESTED
HF_BIOLOGY_DATASET = {
    "hf_dataset": "camel-ai/biology",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["message_1 ","message_2"],     
    "cols_meta": ["topic;","sub_topic"],
}
#-----------------------------MATHS---------------------------

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
#-----------------------------ANCIENT HISTORY AND POLITICS---------------------------

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

HF_ANCIENT_HISTORY_DATASET_2 = {
    "hf_dataset": "mattwesney/CoT_Reasoning_The_Ancient_Past",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["question","answer"],     
    "cols_meta": ["id"]
}
#-----------------------------NEWS---------------------------

HF_NEWS_DATASET = {
    "hf_dataset": "cc_news",
    "hf_subset": "plain_text",
    "split": "train",
    "col_text": ["title", "text"],
    "cols_meta": ["date", "domain"],
}

#-----------------------------PHILOSOPHY AND PSYCHOLOGY---------------------------

HF_PHILOSOPHY_DATASET = {
    "hf_dataset": "bingbangboom/philosophia-QA",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["question", "answer"],
    "cols_meta": ["category"],
}

HF_PSYCHOLOGY_DATASET = {
    "hf_dataset": "BoltMonkey/psychology-question-answer",
    "hf_subset": "default",
    "split": "train",
    "col_text": ["answer", "question"],
    "cols_meta": ["question"],
}

#-----------------------------ENTERTAINMENT---------------------------

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

