from llama_index.core import PromptTemplate

GENERAL_MCQ_PROMPT_STR = (
    "You are an expert assistant. Use the following context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "INSTRUCTIONS:\n"
    "1. Find the correct option index (0, 1, 2, or 3).\n"
    "2. If the CONTEXT is unrelated to the QUESTION, IGNORE it and reason with your own knowledge.\n"
    "3. You MUST output exactly ONE digit (0, 1, 2, or 3).\n"
    "4. Do NOT include any other text, symbols, or explanations.\n\n"
    "Final Option Index: "
)

MCQ_PROMPT = PromptTemplate(GENERAL_MCQ_PROMPT_STR)

QA_PROMPT_TMPL_STR_MATHS_POT = (
    "Solve this math problem using Python. Output ONLY one ```python code block.\n\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION: {question}\n\n"
    "OPTIONS: {options}\n\n"
    "Environment (already available):\n"
    "- `from sympy import *` (symbols, solve, integrate, diff, Matrix, etc.)\n"
    "- `import math, statistics`\n"
    "- `PYTHON_OPTIONS = {options_list}` (list of option strings)\n\n"
    "Rules:\n"
    "- If the context is unrelated to the math problem, ignore it.\n"
    "- Compute the answer numerically or symbolically.\n"
    "- Last line MUST be `print(index)` where index is 0, 1, 2, or 3.\n\n"
    "Your solution:\n"
)
MCQ_PROMPT_MATHS_POT = PromptTemplate(QA_PROMPT_TMPL_STR_MATHS_POT)

QA_PROMPT_TMPL_STR_MATHS_THEORY = (
    "You are an expert in mathematical theory. Answer the multiple choice question.\n"
    "If the CONTEXT is irrelevant, ignore it.\n\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: Do NOT include any other text, words, or explanations. Just the DIGIT.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_MATHS_THEORY = PromptTemplate(QA_PROMPT_TMPL_STR_MATHS_THEORY)

MATH_CLASSIFIER_PROMPT_STR = (
    "Classify the math question as ONE word: \"calculation\" or \"theory\".\n"
    "- \"calculation\" = requires computing/finding a numeric answer, including word problems with numbers, measurements, fractions, counting, geometry with values, etc.\n"
    "- \"theory\" = asks about names, history, definitions, descriptions of theorems/concepts WITHOUT any computation required.\n"
    "- When in doubt, answer \"calculation\".\n\n"
    "Examples:\n"
    "Q: What is the integral of x^2 from 0 to 2?\n"
    "A: calculation\n\n"
    "Q: An 8.5-by-11-inch paper is folded in half. What is the length of the longest side after the fold?\n"
    "A: calculation\n\n"
    "Q: In a group of 11 people, find the sum of all possible values of T such that the constraints hold.\n"
    "A: calculation\n\n"
    "Q: A triangle has sides 3, 4, 5. What is its area?\n"
    "A: calculation\n\n"
    "Q: How many distinct ways can 5 books be arranged on a shelf?\n"
    "A: calculation\n\n"
    "Q: Who proved Fermat's Last Theorem?\n"
    "A: theory\n\n"
    "Q: What is the definition of a vector space?\n"
    "A: theory\n\n"
    "Q: Which of the following best describes the Pythagorean theorem?\n"
    "A: theory\n\n"
    "Q: {question}\n"
    "A:"
)


QA_PROMPT_TMPL_STR_ENTERTAINMENT = (
    "You are an expert in entertainment and pop culture. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: If the CONTEXT is unrelated to the QUESTION, IGNORE it and use your own knowledge.\n"
    "MANDATORY: You MUST output exactly ONE digit (0, 1, 2, or 3). Do NOT include other text.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_ENTERTAINMENT = PromptTemplate(QA_PROMPT_TMPL_STR_ENTERTAINMENT)


QA_PROMPT_TMPL_STR_SCIENCE_NATURE = ( 
    "You are an expert in science and nature. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: If the CONTEXT is unrelated to the QUESTION, IGNORE it and use your own knowledge.\n"
    "MANDATORY: You MUST output exactly ONE digit (0, 1, 2, or 3). Do NOT include other text.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_SCIENCE_NATURE = PromptTemplate(QA_PROMPT_TMPL_STR_SCIENCE_NATURE)


QA_PROMPT_TMPL_STR_HISTORY_POLITICS = (
    "You are an expert in history and politics. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: If the CONTEXT is unrelated to the QUESTION, IGNORE it and use your own knowledge.\n"
    "MANDATORY: You MUST output exactly ONE digit (0, 1, 2, or 3). Do NOT include other text.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_HISTORY_POLITICS = PromptTemplate(QA_PROMPT_TMPL_STR_HISTORY_POLITICS)


QA_PROMPT_TMPL_STR_NEWS = (
    "You are an expert in current events and news. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: If the CONTEXT is unrelated to the QUESTION, IGNORE it and use your own knowledge.\n"
    "MANDATORY: You MUST output exactly ONE digit (0, 1, 2, or 3). Do NOT include other text.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_NEWS = PromptTemplate(QA_PROMPT_TMPL_STR_NEWS)


QA_PROMPT_TMPL_STR_PHILOSOPHY_PSYCHOLOGY = (
    "You are an expert in philosophy and psychology. Use the context to answer the question.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "----------------\n"
    "QUESTION:\n{question}\n\n"
    "OPTIONS:\n{options}\n\n"
    "Output ONLY the index number (0, 1, 2, or 3) of the correct answer.\n"
    "CRITICAL: If the CONTEXT is unrelated to the QUESTION, IGNORE it and use your own knowledge.\n"
    "MANDATORY: You MUST output exactly ONE digit (0, 1, 2, or 3). Do NOT include other text.\n\n"
    "Final Option Index: "
)
MCQ_PROMPT_PHILOSOPHY_PSYCHOLOGY = PromptTemplate(QA_PROMPT_TMPL_STR_PHILOSOPHY_PSYCHOLOGY)


QUERY_TRANSLATOR_PROMPT_STR = (
    "Your task is to extract 1-3 UNIQUE search keywords from the question below.\n"
    "INSTRUCTIONS:\n"
    "1. Prioritize Proper Nouns (Names of people, places, specific events, or titles).\n"
    "2. If no proper nouns exist, extract the core subject nouns.\n"
    "3. NEVER repeat a keyword. Each keyword must appear at most ONCE.\n"
    "4. Maximum of 3 keywords total.\n"
    "5. Output ONLY the keywords separated by spaces. No explanations, no filler.\n\n"
    "EXAMPLES:\n"
    "Question: Who is the author of 'The Great Gatsby'?\n"
    "Keywords: Great Gatsby\n"
    "Question: What is the chemical symbol for Gold?\n"
    "Keywords: Gold\n"
    "Question: How many planets are in our solar system?\n"
    "Keywords: solar system\n"
    "Question: When did the French Revolution start?\n"
    "Keywords: French Revolution\n"
    "Question: What material is used in the balls of a Newton's cradle?\n"
    "Keywords: Newton cradle material\n\n"
    "QUESTION: {question}\n"
    "Keywords:"
)
