from llama_index.core import PromptTemplate

QA_PROMPT_TMPL_STR = (
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "{query_str}\n\n"
    "Examples of correct output format:\n"
    "Question: Who painted the Mona Lisa?\n"
    "[0] Michelangelo [1] Leonardo da Vinci [2] Raphael [3] Caravaggio\n"
    "Answer: 1\n\n"
    "Question: What is the chemical symbol for water?\n"
    "[0] CO2 [1] NaCl [2] H2O [3] O2\n"
    "Answer: 2\n\n"
    "Question: Which planet is closest to the Sun?\n"
    "[0] Venus [1] Earth [2] Mars [3] Mercury\n"
    "Answer: 3\n\n"
    "WRONG: 'Leonardo da Vinci painted it because...'\n"
    "CORRECT: 1\n\n"
    "RULE: Reply with ONLY the single digit (0, 1, 2, or 3). No words. No punctuation. No explanation.\n"
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