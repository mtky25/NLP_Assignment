import re
import random
import time
import numpy as np
import ollama
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from datetime import datetime, timedelta
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.guesser.guesser import Guesser
from src.millionaire_client.models import Question
from src.models import ExperimentConfig

# ==============================================================================
# GLOBAL CONFIGURATIONS & MODELS INITIALIZATION
# ==============================================================================

GUARDIAN_KEY = ""
email = ""

HEADERS = {
    "User-Agent": (
        "WhoWantsToBeAMillionaire-Bot/1.0 (research project;"
        f" {email})"
    )
}

reranker = CrossEncoder("BAAI/bge-reranker-base")


# ==============================================================================
# GENERAL PROMPT BUILDERS
# ==============================================================================

#From some previous experiments, actually adding the theme of the questions may result in lower performance (extra tokens confusing the model or biasing the type of query)
def build_query_prompt(question, options, theme):
    system_prompt = {
        "role": "system",
        "content": f"""You are a Wikipedia search assistant. Given a multiple choice question, output EXACTLY three Wikipedia search queries, one per line, nothing else.

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
        "content": """You are an expert quiz player. Use the provided documents as evidence.

Procedure:
1. Identify which document(s) contain the answer. If none do, rely on your own knowledge.
2. Quote (silently) the single sentence that decides the question.
3. Map that sentence to A, B, C, or D.

If two options are partially supported, pick the one that matches the question's exact wording (dates, names, units).

Output exactly one line:

FINAL_ANSWER: X

where X is A, B, C, or D.""",
    }

    formatted_docs = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)
    )

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

# ==============================================================================
# TEXT PROCESSING & CHUNKING HELPERS
# ==============================================================================

#Splits Wikipedia by paragraphs (following \n\n).
#Appends sentences until max_words is reached (stops actually at a dot to avoid cutting off paragraphs).
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
        
        #Split paragraph into sentences
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

#Cleans a query
#This was done because it was usual for the model to output strange values, such as chinese characters, words stitched together, long sentences...
#The model may still output strange queries, but this reduces most of the noise
def clean_query(q):
    q = q.strip().replace('"', "")
    q = q.split("\n")[0]  # first line only
    q = re.sub(r"[^a-zA-Z0-9\s\-']", " ", q) #leave only letter, digit, whitespace, hyphen or apostrophe with a space
    q = re.sub(r"\s+", " ", q).strip()
    return q[:120]


# ==============================================================================
# RERANKING & RETRIEVAL (RAG Core)
# ==============================================================================

#Takes the best chunks
#In particular it takes all 4 possible [question,option] pairs and scores all chunks against that pair. Then the best <top_k> chunks are taken
def rerank_chunks(question, options, chunks, top_k=3):
    # Single query combining question + all options
    start_time = time.time()
    query = question + " " + " ".join(options)
    
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Rank all chunks, take top_k directly
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    
    seen = set()
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

def bm25_prefilter(question, options, chunks, top_k=10):
    start_time = time.time()
    # Build query from question + all options combined
    query = f"{question} {' '.join(options)}"
    tokenized_query = query.lower().split()
    tokenized_chunks = [c.lower().split() for c in chunks]
    
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)
    
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    print(f"Time to bm25 prefilter documents {time.time() - start_time:.3f}")
    return [chunk for _, chunk in ranked[:top_k]]

#Performs a single wikipedia search API call
#It takes a query and returns the <srlimit> best possible titles/pages. It then extracts the full content of those pages.
#So if we have 3 queries generated, it will take 6 total pages.
def wikisearch_single(query):
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 2,
    }
    search_response = requests.get(
        search_url, params=search_params, headers=HEADERS
    )

    if not search_response.text or search_response.status_code != 200:
        return []

    titles = [r["title"] for r in search_response.json()["query"]["search"]]
    print(f"Titles found for query {query}: {titles}")

    pages = []
    # downloading full wikipedia page
    for title in titles:
        extract_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        extract_response = requests.get(
            search_url, params=extract_params, headers=HEADERS
        )

        if not extract_response.text or extract_response.status_code != 200:
            continue

        for page in extract_response.json()["query"]["pages"].values():
            if "extract" in page:
                pages.append((title, page["extract"]))

    return pages

#Performs multiple wikipedia search API calls in parallel
def wikisearch_multi(queries, question, options):

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(wikisearch_single, queries))

    # Pool all chunks from all queries
    all_pages = [page for query_pages in results for page in query_pages]

    # Now deduplicate by title
    seen_titles = set()
    unique_pages = []
    for title, text in all_pages:
        if title not in seen_titles:
            seen_titles.add(title)
            unique_pages.append((title, text))

    # Title prefix gives the reranker article-level context at zero cost
    all_chunks = []
    for title, text in unique_pages:
        for chunk in chunk_text(text):
            all_chunks.append(f"[{title}] {chunk}")

    if not all_chunks:
        return []

    print(f"Time to search the documents {time.time()-start_time:.1f}")
    prefiltered = bm25_prefilter(question, options, all_chunks, top_k=10)
    return rerank_chunks(question, options, prefiltered, top_k=5)


# ==============================================================================
# GENERAL MODEL INFERENCE CALLS
# ==============================================================================

def final_answer_regex(text: str):
    match = re.search(r"\*{0,2}FINAL_ANSWER\*{0,2}:\s*([ABCD])", text)
    if match:
        return match.group(1)

    match = re.search(r"[Ff]inal\s+[Aa]nswer\*{0,2}:?\*{0,2}\s*([ABCD])\b", text)
    if match:
        return match.group(1)
    
    match = re.search(r"[Aa]nswer\s+is\s+\(?([ABCD])\)?", text)
    if match:
        return match.group(1)

    return False

#The errors made are pure semantic errors, model does not seem to understand or gets confused even when the answer is present in the documents provided.
#Model size may be the main bottleneck.
def call_model(prompt, max_tokens):
    response = ollama.chat(
        model="qwen2.5:14b-instruct",
        messages=prompt,
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]

# ==============================================================================
# MATH
# ==============================================================================

#Low temperature yields best results. Token limitation does not guarantee results.
#Multiple calls as a consensus did not seem to yield better results.
#Calculation tooling may improve accuracy but introduces latency.
def call_math_with_budget(prompt, hard_budget=25):
    buf = {"content": ""}
    stop_flag = threading.Event()

    def stream():
        for chunk in ollama.chat(
            model="deepseek-r1:14b-qwen-distill-q8_0",
            messages=prompt, stream=True, think=True,
            options={"temperature": 0.1},
        ):
            if stop_flag.is_set(): break
            buf["content"] += (chunk["message"].content or "")
            buf["content"] += (getattr(chunk["message"], "thinking", "") or "")

    t = threading.Thread(target=stream); t.start()
    t.join(timeout=hard_budget)
    stop_flag.set()

    print(f"Buf content {buf['content']}")

    #Check if it produced an answer
    letter = final_answer_regex(buf["content"])
    if letter: return letter

    refine_prompt = prompt + [
        {"role": "assistant", "content": buf["content"]},
        {"role": "user", "content": "Based on the reasoning above, output ONLY: FINAL_ANSWER: X"},
    ]
    try:
        r = ollama.chat(model="deepseek-r1:14b-qwen-distill-q8_0",
                        messages=refine_prompt, think=False,
                        options={"num_predict": 10, "temperature": 0.1})
        letter = final_answer_regex(r["message"]["content"])
        if letter: return letter
    except Exception:
        pass
    
    #Random fallback
    return random.choice(['A', 'B', 'C', 'D'])

#Numerous configurations have been tried: both letting the model think and then output or simply output
#By letting the model think, we are not able to enforce the 30s time budget, as we are not sure to receive an answer before that time.
#Moreover, in thinking mode, we cannot enforce the model to make checkpoints (like write CURRENT_BEST) as there are no constraints for the thinking part.
#Also, putting a token constraint would also incur in the same problem.
def build_math_prompt(question, options):
    system = {
        "role": "system",
        "content": """You are solving timed multiple-choice math and statistics problems.

Strategy — pick whichever resolves the question fastest:
1. SUBSTITUTION: plug each option into the equation/condition and check which satisfies it.
2. ELIMINATION: rule out options that fail a quick check (sign, magnitude, units, parity, bounds).
3. DIRECT: solve and match.

Rules:
- Pick ONE strategy and execute. Do not re-derive what you've already computed.
- One sanity check is fine; repeated verification wastes time.
- If torn between two options after checking, commit to the one with the stronger constraint match.
- Definitions and theorem statements must be matched exactly — do not paraphrase.

Output exactly one line at the end:

FINAL_ANSWER: X

where X is A, B, C, or D.""",
    }
    user = {
        "role": "user",
        "content": f"""Question:
{question}

Options:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}""",
    }
    return [system, user]

# ==============================================================================
# NEWS
# ==============================================================================

def build_news_query_prompt(question, options):
    system = {
        "role": "system",
        "content": """You are a news search assistant. Given a multiple choice question about a published news article, output EXACTLY three news search queries, one per line.

- Each query MUST include the most distinctive named entity from the QUESTION
  (place, person, organization, event name).
- 2–4 words. Pick distinctive nouns over generic ones.
- All three queries should orbit the SAME entity, varying only the secondary term.
- Use ONLY terms from the QUESTION. NEVER use words that appear in the OPTIONS.

Example:
QUESTION: Where are white-tailed eagles set to be reintroduced?
OUTPUT:
White-tailed eagle reintroduction
White-tailed eagle UK
White-tailed eagle rewilding""",
    }
    user = {
        "role": "user",
        "content": f"""#QUESTION: {question}

#OPTIONS:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
    }
    return [system, user]

#Parses a date (2000 onwards) from the question
def parse_question_date(text):
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def news_search_single(query, date_obj=None, window_days=3, page_size=5):
    if not GUARDIAN_KEY:
        print("NO KEY")
        return []
    params = {
        "q": query,
        "api-key": GUARDIAN_KEY,
        "show-fields": "bodyText", #full article
        "page-size": page_size, #how many articles to return
        "order-by": "relevance",
    }

    #Window for catching edge cases by filtering by date
    if date_obj:
        params["from-date"] = (date_obj - timedelta(days=window_days)).date().isoformat()
        params["to-date"]   = (date_obj + timedelta(days=window_days)).date().isoformat()
    try:
        url = "https://content.guardianapis.com/search"
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        results = data.get("response", {}).get("results", [])

        #Keeps all articles that actually contain information
        kept = [(it["webTitle"], it["fields"].get("bodyText", ""))
                for it in results
                if it.get("fields", {}).get("bodyText")]
        return kept
    except Exception as e:
        print(f"Guardian fetch failed: {e}")
        return []

#Call in parallel multiple news search queries
def news_search_multi(queries, question, options, date_obj=None):
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda q: news_search_single(q, date_obj), queries))

    #Deduplication
    seen_titles, unique_pages = set(), []
    for pages in results:
        for title, text in pages:
            if title not in seen_titles:
                seen_titles.add(title)
                unique_pages.append((title, text))

    #Chunking
    all_chunks = []
    for title, text in unique_pages:
        for chunk in chunk_text(text):
            all_chunks.append(f"[{title}] {chunk}")

    if not all_chunks:
        return []

    #Filtering and ranking
    print(f"Time to search news: {time.time()-start:.1f}")
    prefiltered = bm25_prefilter(question, options, all_chunks, top_k=10)
    return rerank_chunks(question, options, prefiltered, top_k=5)

# ==============================================================================
# CORE GUESSER PIPELINE CLASS
# ==============================================================================

class WikiRAGGuesser(Guesser):

    def __init__(self, config: ExperimentConfig, mode: str = "text",
                 transcription_model: str = "tiny", **kwargs):
        super().__init__(config, mode=mode, transcription_model=transcription_model)

    def infer_answer(self, question: Question, theme: str = None, game_session=None) -> int:
        self.search_time = 0.0
        self.reasoning_time = 0.0
        try:
            options = [opt.text for opt in question.options]
            if theme == "Maths":
                model_prompt = build_math_prompt(question.text, options)
                t0 = time.time()
                response = call_math_with_budget(model_prompt,26)
                print(f"Model answered: {response}")
                self.reasoning_time = time.time() - t0
                return {"A": 0, "B": 1, "C": 2, "D": 3}.get(response, random.randint(0,3))
            elif theme == "News":
                date_obj = parse_question_date(question.text)
                model_prompt = build_news_query_prompt(question.text, options)
                raw_queries  = call_model(model_prompt, 300)
                queries = [clean_query(q) for q in raw_queries.strip().split("\n")]
                queries = [q for q in queries if 3 < len(q) < 120]
                queries = list(dict.fromkeys(queries))[:3]
                print(f"News queries: {queries}")
                t0 = time.time()
                found_docs = news_search_multi(queries, question.text, options, date_obj)
                self.search_time = time.time() - t0

                if not found_docs:
                    return random.randint(0, 3)

                t1 = time.time()
                model_prompt = build_doc_prompt(question.text, options, found_docs)
                response = call_model(model_prompt, 80)
                self.reasoning_time = time.time() - t1

                letter = final_answer_regex(response)
                return {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter, random.randint(0,3))
            else:
                model_prompt = build_query_prompt(question.text, options, theme)
                raw_queries = call_model(model_prompt, 300)
                #print(f"Raw queries: {raw_queries}")
                queries = [clean_query(q) for q in raw_queries.strip().split("\n")]
                queries = [q for q in queries if 3 < len(q) < 120]
                queries = list(dict.fromkeys(queries))[
                    :3
                ]
                print(f"Queries: {queries}")
                t0 = time.time()
                found_docs = wikisearch_multi(queries, question.text, options)
                #print(f"Retrieved chunks: {found_docs}")
                self.search_time = time.time() - t0

                t1 = time.time()
                model_prompt = build_doc_prompt(question.text, options, found_docs)
                response = call_model(model_prompt, 80)
                print(f"Model answered: {response}")
                self.reasoning_time = time.time() - t1

                letter = final_answer_regex(response)
                return {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter, random.randint(0,3))

        except Exception:
            raise
