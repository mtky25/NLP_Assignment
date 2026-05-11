from llama_index.core import PromptTemplate

# A general prompt template that can be used as a base or default
GENERAL_MCQ_PROMPT_STR = (
    "You are an expert assistant. Use the following context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "CRITICAL: Find the correct option index (0, 1, 2, or 3).\n"
    "If there's NO INFORMATION about the question in the CONTEXT, you should REASON with your own knowledge to answer.\n"
    "Output ONLY the index number. No text, no symbols, just the digit.\n\n"
    "Final Option Index: "
)

MCQ_PROMPT = PromptTemplate(GENERAL_MCQ_PROMPT_STR)

QA_PROMPT_TMPL_STR_MATHS = (
    "You are an expert math problem solver. Use the context to solve the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "CRITICAL: Read the resolution in the context. Find the final numerical result.\n"
    "If there's NO INFORMATION n about the question in the CONTEXT, you should REASON with your own knowledge to answer.\n"
    "Match that result to the correct option index (0, 1, 2, or 3).\n"
    "Output ONLY the index number. No text, no symbols, just the digit.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_MATHS = PromptTemplate(QA_PROMPT_TMPL_STR_MATHS)

#---------

QA_PROMPT_TMPL_STR_ENTERTAINMENT = (
    "You are an expert in entertainment and pop culture. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n\n"
    "If there's NO INFORMATION n about the question in the CONTEXT, you should REASON with your own knowledge to answer.\n"
    "Final Option Index: "
)
MCQ_PROMPT_ENTERTAINMENT = PromptTemplate(QA_PROMPT_TMPL_STR_ENTERTAINMENT)

#---------

QA_PROMPT_TMPL_STR_SCIENCE_NATURE = ( 
    "You are an expert in science and nature. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n\n"
    "If there's NO INFORMATION n about the question in the CONTEXT, you should REASON with your own knowledge to answer.\n"
    "Final Option Index: "
)
MCQ_PROMPT_SCIENCE_NATURE = PromptTemplate(QA_PROMPT_TMPL_STR_SCIENCE_NATURE)

#---------

QA_PROMPT_TMPL_STR_HISTORY_POLITICS = (
    "You are an expert in history and politics. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n\n"
    "If there's NO INFORMATION n about the question in the CONTEXT, you should REASON with your own knowledge to answer.\n"
    "Final Option Index: "
)
MCQ_PROMPT_HISTORY_POLITICS = PromptTemplate(QA_PROMPT_TMPL_STR_HISTORY_POLITICS)
