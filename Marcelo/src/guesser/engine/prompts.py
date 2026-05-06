from llama_index.core import PromptTemplate

QA_PROMPT_TMPL_STR = (
    "You are an expert data extraction and Q&A assistant. Your task is to read the provided context and answer the multiple-choice question.\n"
    "CRITICAL RULE: You must reply ONLY with the exact number of the correct option (0, 1, 2, or 3). "
    "Do not provide any explanations, introductory phrases, or punctuation other than the single digit.\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and no prior knowledge, answer the question.\n\n"
    "Example:\n"
    "Question: Which term refers to the younger and passive partner in a male homosexual relationship in ancient Greece?\n"
    "[0] Erana\n"
    "[1] Pais\n"
    "[2] Eromenos\n"
    "[3] Erastes\n"
    "Answer: 2\n\n"
    "Now it is your turn:\n"
    "{query_str}\n"
    "Answer: "
)

MCQ_PROMPT = PromptTemplate(QA_PROMPT_TMPL_STR)

QA_PROMPT_TMPL_STR_MATHS = (
    "You are an expert math problem solver. Use the context to solve the question.\n"
    "--- CONTEXT ---\n"
    "{context_str}\n"
    "----------------\n"
    "QUESTION:\n{query_str}\n\n"
    "CRITICAL: Read the resolution in the context. Find the final numerical result.\n"
    "Match that result to the correct option index (0, 1, 2, or 3).\n"
    "Output ONLY the index number. No text, no symbols, just the digit.\n\n"
    "Final Option Index: " # Isso aqui é o 'Anchor' - o modelo tende a completar apenas com o número
)

MCQ_PROMPT_MATHS = PromptTemplate(QA_PROMPT_TMPL_STR_MATHS)