# ==============================================================================
# SETTINGS — fill these in
# ==============================================================================

user_name = "Pastasciutta"
password  = "pastasciutta"

HEADERS = {
    "User-Agent": (
        "WhoWantsToBeAMillionaire-Bot/1.0 (research project;"
        " bianchigianpaolo2@gmail.com)"
    )
}

# ==============================================================================
# IMPORTS
# ==============================================================================

import re
import random
import time
import requests
import numpy as np
import pandas as pd
import ollama
import sys
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==============================================================================
# CLIENT SETUP
# ==============================================================================

API_URL   = "http://131.175.15.22:51111/"

# Adjust this path if your millionaire_client package lives elsewhere
sys.path.append(".")
from millionaire_client import MillionaireClient, AuthenticationError

client = MillionaireClient(API_URL)
try:
    user = client.login(user_name, password)
    print(f"\nWelcome, {user.username}! (Role: {user.role})")
except AuthenticationError as e:
    print(f"Login failed: {e}")

# ==============================================================================
# GLOBAL MODEL INIT
# ==============================================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker  = CrossEncoder("BAAI/bge-reranker-base")

# ==============================================================================
# PROMPT BUILDERS
# ==============================================================================

# From some previous experiments, actually adding the theme of the questions may
# result in lower performance (extra tokens confusing the model or biasing the
# type of query)
def build_query_prompt(question, options):
    system_prompt = {
        "role": "system",
        "content": """You are a Wikipedia search assistant. Given a multiple choice question, output EXACTLY three Wikipedia search queries, one per line, nothing else.

Rules:
- Write short keyword phrases, not full sentences.
- Target Wikipedia article titles, named entities, historical events, or scientific concepts.
- Each query must be independently searchable and cover a different angle of the question.
- No filler words: avoid "what is", "explain", "who was", "describe".
- No numbering, no bullets, no extra text.

Example:
QUESTION: What element has atomic number 79?
A) Silver  B) Gold  C) Platinum  D) Copper
OUTPUT:
Gold chemical element
Atomic number periodic table
Noble metals chemistry""",
    }
    
    user_prompt = {
        "role": "user",
        "content": f"""#QUESTION: {question}

#OPTIONS:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}""",
    }

    return [system_prompt, user_prompt]


def build_doc_prompt(question, options, documents):
    system_prompt = {
        "role": "system",
        "content": """You are an expert quiz player.

Use the documents as evidence.
Reason silently.

Output exactly one line:

FINAL_ANSWER: X

where X is A, B, C, or D.""",
    }

    formatted_docs = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)
    )
    print(formatted_docs)
    print( )
    
    user_prompt = {
        "role": "user",
        "content": f"""#QUESTION: {question}
#OPTIONS:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

#REFERENCE DOCUMENTS:
{formatted_docs}""",
    }

    return [system_prompt, user_prompt]


# Numerous configurations have been tried: both letting the model think and then
# output or simply output. By letting the model think, we are not able to enforce
# the 30s time budget, as we are not sure to receive an answer before that time.
# Moreover, in thinking mode, we cannot enforce the model to make checkpoints
# (like write CURRENT_BEST) as there are no constraints for the thinking part.
# Also, putting a token constraint would also incur in the same problem.
# This solution simply tells the model to OUTPUT within a given sentence limit
# (often not met though) while making sure the model "think:False"
def build_math_prompt(question, options):
    system = {
        "role": "system",
        "content": """
You are a mathematics and statistics expert solving timed multiple-choice questions.

Rules:
- Reason carefully but concisely.
- Use short, precise steps.
- Verify definitions and theorem statements exactly.
- Do not guess based on intuition alone.
- Avoid unnecessary explanations.
- Stop once the answer is determined.

Output format:

Key steps:
- ...

FINAL_ANSWER: X

Where X is A, B, C, or D.
"""
    }
    user = {
        "role": "user",
        "content": f"""Question:
{question}

Options:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
    }

    return [system, user]

# ==============================================================================
# TEXT PROCESSING & CHUNKING HELPERS
# ==============================================================================

# Splits Wikipedia by paragraphs (following \n\n).
# Appends sentences until max_words is reached (stops at a sentence boundary to
# avoid cutting off mid-thought).
def chunk_text(text, max_words=250):
    paragraphs = text.split("\n\n")
    chunks = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        words = p.split()
        if len(words) <= max_words:
            chunks.append(p)
            continue

        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', p)

        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            if current_word_count + sentence_word_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = sentence_words
                current_word_count = sentence_word_count
            else:
                current_chunk.extend(sentence_words)
                current_word_count += sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks


# Cleans a query.
# This was done because it was usual for the model to output strange values,
# such as chinese characters, words stitched together, long sentences...
# BUG: sometimes those still happen 0_0
def clean_query(q):
    q = q.strip().replace('"', "")
    q = q.split("\n")[0]
    q = re.sub(r"[^a-zA-Z0-9\s\-']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:120]

# ==============================================================================
# RERANKING & RETRIEVAL (RAG Core)
# ==============================================================================

def get_top_chunks(question, options, chunks, top_k=5):
    start_time = time.time()
    question_embedding = embedder.encode(question)
    option_embeddings  = embedder.encode(options)
    chunk_embeddings   = embedder.encode(chunks)

    chunks_norm   = chunk_embeddings   / np.linalg.norm(chunk_embeddings,   axis=1, keepdims=True)
    question_norm = question_embedding / np.linalg.norm(question_embedding)
    options_norm  = option_embeddings  / np.linalg.norm(option_embeddings,  axis=1, keepdims=True)

    question_sim    = chunks_norm @ question_norm
    option_sims     = chunks_norm @ options_norm.T
    best_option_sim = option_sims.max(axis=1)
    combined        = 0.6 * question_sim + 0.4 * best_option_sim

    top_indices = np.argsort(combined)[::-1][:top_k]
    print(f"Time to chunk documents {time.time() - start_time:.1f}")
    return [chunks[i] for i in top_indices]


# Takes the best chunks. In particular it takes all 4 possible [question,option]
# pairs and scores all chunks against that pair. Then the best <top_k> chunks are taken.
def rerank_chunks(question, options, chunks, top_k=3):
    start_time = time.time()
    query = question + " " + " ".join(options)

    pairs  = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    seen  = set()
    final = []
    for score, chunk in ranked:
        norm = re.sub(r"\s+", " ", chunk.strip().lower())
        if norm not in seen:
            seen.add(norm)
            final.append(chunk)
        if len(final) == top_k:
            break

    print(f"Time to rerank documents {time.time() - start_time:.1f}")
    return final

#Fast bm25 prefiltering of the top_k chunks
def bm25_prefilter(question, options, chunks, top_k=10):
    start_time = time.time()
    query            = f"{question} {' '.join(options)}"
    tokenized_query  = query.lower().split()
    tokenized_chunks = [c.lower().split() for c in chunks]

    bm25   = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    print(f"Time to bm25 prefilter documents {time.time() - start_time:.3f}")
    return [chunk for _, chunk in ranked[:top_k]]

# ==============================================================================
# WIKIPEDIA SEARCH
# ==============================================================================

# Performs a single Wikipedia search API call.
# It takes a query and returns the <srlimit> best possible titles/pages, then
# extracts the full content of those pages.
# So if we have 3 queries generated, it will take up to 6 total pages.
def wikisearch_single(query):
    search_url    = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "format":   "json",
        "srlimit":  2,
    }
    search_response = requests.get(search_url, params=search_params, headers=HEADERS)

    if not search_response.text or search_response.status_code != 200:
        return []

    titles = [r["title"] for r in search_response.json()["query"]["search"]]
    print(f"Titles found for query {query}: {titles}")

    pages = []
    for title in titles:
        extract_params = {
            "action":      "query",
            "titles":      title,
            "prop":        "extracts",
            "explaintext": True,
            "format":      "json",
        }
        extract_response = requests.get(search_url, params=extract_params, headers=HEADERS)

        if not extract_response.text or extract_response.status_code != 200:
            continue

        for page in extract_response.json()["query"]["pages"].values():
            if "extract" in page:
                pages.append((title, page["extract"]))

    return pages


# Performs multiple Wikipedia search API calls in parallel.
def wikisearch_multi(queries, question, options):

    def search_one(query):
        if not query:
            return []
        return wikisearch_single(query)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(search_one, queries))

    all_pages = [page for query_pages in results for page in query_pages]

    seen_titles  = set()
    unique_pages = []
    for title, text in all_pages:
        if title not in seen_titles:
            seen_titles.add(title)
            unique_pages.append((title, text))

    # Title prefix gives the reranker article-level context
    all_chunks = []
    for title, text in unique_pages:
        for chunk in chunk_text(text):
            all_chunks.append(f"[{title}] {chunk}")

    if not all_chunks:
        return [], 0, 0, 0

    search_doc_time = time.time() - start_time
    print(f"Time to search the documents {search_doc_time:.1f}")

    t_prefilter   = time.time()
    prefiltered   = bm25_prefilter(question, options, all_chunks, top_k=10)
    prefilter_time = time.time() - t_prefilter

    t_rerank       = time.time()
    final_chunks   = rerank_chunks(question, options, prefiltered, top_k=5)
    rerank_doc_time = time.time() - t_rerank

    return final_chunks, search_doc_time, prefilter_time, rerank_doc_time

# ==============================================================================
# MODEL INFERENCE CALLS
# ==============================================================================

#The errors made are pure semantic errors, model does not seem to understand or gets confused even when the answer is present in the documents provided.
def call_model(prompt, max_tokens):
    response = ollama.chat(
        model="qwen2.5:14b-instruct",
        messages=prompt,
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]

#Low temperature yields best results. Token limitation does not guarantee results.
#Multiple calls as a consensus did not seem to yield better results.
#Calculation tooling may improve accuracy but introduces latency.
#Thinking introduces pipeline uncertainty in token limitation and latency.
def call_math_model(prompt):
    response = ollama.chat(
        model="deepseek-r1:14b-qwen-distill-q8_0",
        messages=prompt,
        think=False,
        options={"temperature": 0.1},
    )
    content = response["message"].get("content", "")
    print(f"Math model reasoning: {content}")
    # 1. Your original format: FINAL_ANSWER: B
    match = re.search(r"\*{0,2}FINAL_ANSWER\*{0,2}:\s*([ABCD])", content)
    if match:
        return match.group(1)

    # 2. "Final Answer: C) ..." or "Final Answer:** C"
    match = re.search(r"[Ff]inal\s+[Aa]nswer\*{0,2}:?\*{0,2}\s*([ABCD])\b", content)
    if match:
        return match.group(1)

    # 3. "the answer is C" / "answer is (C)"
    match = re.search(r"[Aa]nswer\s+is\s+\(?([ABCD])\)?", content)
    if match:
        return match.group(1)

    # 4. Last standalone A/B/C/D in the entire response
    matches = re.findall(r'\b([ABCD])\b', content)
    if matches:
        return matches[-1]

    return None

# ==============================================================================
# CORE INFERENCE FUNCTION
# ==============================================================================

def infer_answer(question, comp_id):
    search_doc_time = 0.0
    prefilter_time  = 0.0
    rerank_doc_time = 0.0
    reasoning_time  = 0.0

    try:
        options = [opt.text for opt in question.options]

        if comp_id == 3:
            model_prompt = build_math_prompt(question.text, options) 
            t0 = time.time() 
            response = call_math_model(model_prompt) 
            reasoning_time = time.time() - t0 
            print(f"Model answered: {response}") 
            answer_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(response, 0)

        else:
            model_prompt = build_query_prompt(question.text, options)
            raw_queries  = call_model(model_prompt, 300)
            #print(f"Raw queries: {raw_queries}")
            queries = [clean_query(q) for q in raw_queries.strip().split("\n")]
            queries = [q for q in queries if 3 < len(q) < 120]
            queries = list(dict.fromkeys(queries))[:3]
            print(f"Queries: {queries}")

            found_docs, search_doc_time, prefilter_time, rerank_doc_time = wikisearch_multi(
                queries, question.text, options
            )
            #print(f"Retrieved chunks: {found_docs}")

            model_prompt = build_doc_prompt(question.text, options, found_docs)
            t1 = time.time()
            response = call_model(model_prompt, 80)
            reasoning_time = time.time() - t1
            print(f"Model answered: {response}")

            match = re.search(r"FINAL_ANSWER:\s*([ABCD])", response)
            answer_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(
                match.group(1) if match else None, random.randint(0, 3)
            )
        print(f"Reasoning time: {reasoning_time:.1f}")
        metrics = {
            "search_doc_time": search_doc_time,
            "prefilter_time":  prefilter_time,
            "rerank_doc_time": rerank_doc_time,
            "reasoning_time":  reasoning_time,
        }

        return answer_index, metrics

    except Exception as e:
        print(f"Error in infer_answer: {e}")
        metrics = {
            "search_doc_time": 0.0,
            "prefilter_time":  0.0,
            "rerank_doc_time": 0.0,
            "reasoning_time":  0.0,
        }
        return random.randint(0, 3), metrics

# ==============================================================================
# GAME LOOP
# ==============================================================================

wrong = []

def play_game(game, comp_id):
    start_time = time.time()

    question_search_doc_times = []
    question_prefilter_times  = []
    question_rerank_doc_times = []
    question_reasoning_times  = []

    num_questions_answered = 0
    num_correct_answers    = 0
    time_to_answer         = 0.0

    while game.in_progress:
        question = game.current_question
        if not question:
            print("No question available. Game may have ended.")
            break

        print(f"\n--- Level {game.current_level} ---")
        print(f"Q: {question.text}")
        for opt in question.options:
            print(f"  {opt.id}: {opt.text}")
        print()

        response_id, metrics = infer_answer(question, comp_id)
        print(f"Selected answer: {response_id}")

        question_search_doc_times.append(metrics["search_doc_time"])
        question_prefilter_times.append(metrics["prefilter_time"])
        question_rerank_doc_times.append(metrics["rerank_doc_time"])
        question_reasoning_times.append(metrics["reasoning_time"])

        result         = game.answer(response_id)
        time_to_answer = time.time() - start_time
        num_questions_answered += 1

        if result.correct:
            print(" CORRECT!")
            num_correct_answers += 1
            if result.game_over:
                print(f"\n CONGRATULATIONS! You completed the game!")
                print(f" Final earnings: ${result.earned_amount:,.2f}")
        elif result.timed_out:
            print("TIMED OUT!")
            print(f"\n Game Over! Final earnings: ${result.earned_amount:,.2f}")
            #break
        else:
            print(" WRONG ANSWER!")
            print(f"\n Game Over! Final earnings: ${result.earned_amount:,.2f}")

            wrong.append( {"question": question, "model_answer": response_id})
            #break

    print("\n=== Game Summary ===")
    print(f"Reached Level: {game.current_level}")
    print(f"Total Earnings: ${game.earned_amount:,.2f}")
    time.sleep(20)
    n = num_questions_answered
    avg_response_time   = time_to_answer                        / n if n > 0 else 0
    avg_search_doc_time = sum(question_search_doc_times)        / n if n > 0 else 0
    avg_prefilter_time  = sum(question_prefilter_times)         / n if n > 0 else 0
    avg_rerank_doc_time = sum(question_rerank_doc_times)        / n if n > 0 else 0
    avg_reasoning_time  = sum(question_reasoning_times)         / n if n > 0 else 0
    accuracy_per_run    = num_correct_answers                   / n if n > 0 else 0

    return {
        "level_reached":       game.current_level,
        "avg_response_time":   avg_response_time,
        "avg_search_doc_time": avg_search_doc_time,
        "avg_prefilter_time":  avg_prefilter_time,
        "avg_rerank_doc_time": avg_rerank_doc_time,
        "avg_reasoning_time":  avg_reasoning_time,
        "accuracy_per_run":    accuracy_per_run,
    }


def game_start(comp_id):
    print("\n=== Starting Game ===")
    game = client.game.start(competition_id=comp_id)
    print(f"Session ID: {game.session_id}")
    print(f"Total number of questions: {game.state.competition.max_levels}")
    print()
    return play_game(game, comp_id)

# ==============================================================================
# RUN EXPERIMENTS LOOP
# ==============================================================================

def main(comp_ids=range(0, 4), runs_per_comp=3, csv_path="experiment_results.csv"):
    all_stats = []

    for comp_id in comp_ids:
        for run in range(1, runs_per_comp + 1):
            print(f"\n>>> Competition {comp_id} | Run {run}")
            game_metrics = game_start(comp_id)

            stats_entry = {
                "comp_id":             comp_id,
                "run":                 run,
                "level_reached":       game_metrics["level_reached"],
                "avg_response_time":   game_metrics["avg_response_time"],
                "avg_search_doc_time": game_metrics["avg_search_doc_time"],
                "avg_prefilter_time":  game_metrics["avg_prefilter_time"],
                "avg_rerank_doc_time": game_metrics["avg_rerank_doc_time"],
                "avg_reasoning_time":  game_metrics["avg_reasoning_time"],
                "accuracy_per_run":    game_metrics["accuracy_per_run"],
            }
            all_stats.append(stats_entry)


    df = pd.DataFrame(all_stats)
    print("\n--- Experiment Results Summary ---")
    print(df)

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    overall = df.groupby("comp_id")["accuracy_per_run"].mean().reset_index()
    print("\n--- Overall Average Accuracy by Competition ---")
    print(overall)

    print(f"Overall per-question accuracy mean: {df['accuracy_per_run'].mean()}")

    print(f"Wrong answers (not timeout): {wrong}")

    return df


if __name__ == "__main__":
    main(comp_ids=range(3, 4), runs_per_comp=3)