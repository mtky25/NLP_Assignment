#Add tools to Math model
#Check if chunking and reranking can be improved

#Math model: call model, if it does not answer before 30s, we in parallel are asking some smaller models what do they think

import re
import random
import time
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import numpy as np
import ollama
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.guesser.guesser import Guesser
from src.millionaire_client.models import Question
from src.models import ExperimentConfig

# ==============================================================================
# GLOBAL CONFIGURATIONS & MODELS INITIALIZATION
# ==============================================================================

HEADERS = {
    "User-Agent": (
        "WhoWantsToBeAMillionaire-Bot/1.0 (research project;"
        " bianchigianpaolo2@gmail.com)"
    )
}

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ==============================================================================
# PROMPT BUILDERS
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

#Numerous configurations have been tried: both letting the model think and then output or simply output
#By letting the model think, we are not able to enforce the 30s time budget, as we are not sure to receive an answer before that time.
#Moreover, in thinking mode, we cannot enforce the model to make checkpoints (like write CURRENT_BEST) as there are no constraints for the thinking part.
#Also, putting a token constraint would also incur in the same problem.
#This solution simply tells the model to OUTPUT within a given sentence limit (often not met though) while making sure the model "think:False"
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

#Splits Wikipedia by paragraphs (following \n\n).
#Appends sentences until max_words is reached (stops actually at a dot to avoid cutting off paragraphs).
'''
def chunk_text(text, max_words=250):
    paragraphs = text.split("\n\n")
    chunks = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        words = p.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                chunks.append(chunk)
        else:
            chunks.append(p)

    return chunks
'''
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
                # Current chunk is full — flush it
                chunks.append(" ".join(current_chunk))
                current_chunk = sentence_words
                current_word_count = sentence_word_count
            else:
                current_chunk.extend(sentence_words)
                current_word_count += sentence_word_count

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks

#Cleans a query
#This was done because it was usual for the model to output strange values, such as chinese characters, words stitched together, long sentences...
#BUG sometimes those still happen 0_0
def clean_query(q):
    q = q.strip().replace('"', "")
    q = q.split("\n")[0]  # first line only
    q = re.sub(r"[^a-zA-Z0-9\s\-']", " ", q) #leave only letter, digit, whitespace, hyphen or apostrophe with a space
    q = re.sub(r"\s+", " ", q).strip()
    return q[:120]


# ==============================================================================
# RERANKING & RETRIEVAL (RAG Core)
# ==============================================================================


def get_top_chunks(question, options, chunks, top_k=5):
    # Embed question and each option separately
    start_time = time.time()
    question_embedding = embedder.encode(question)
    option_embeddings = embedder.encode(options)
    chunk_embeddings = embedder.encode(chunks)

    # Normalize everything
    chunks_norm = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True
    )
    question_norm = question_embedding / np.linalg.norm(question_embedding)
    options_norm = option_embeddings / np.linalg.norm(
        option_embeddings, axis=1, keepdims=True
    )

    # Similarity with question
    question_sim = chunks_norm @ question_norm

    # Max similarity across all options (not average — max preserves signal)
    option_sims = chunks_norm @ options_norm.T  # shape: (n_chunks, 4)
    best_option_sim = option_sims.max(axis=1)  # best matching option per chunk

    # Combine: weight question more heavily than options
    combined = 0.6 * question_sim + 0.4 * best_option_sim

    top_indices = np.argsort(combined)[::-1][:top_k]
    print(f"Time to chunk documents {time.time() - start_time:.1f}")
    return [chunks[i] for i in top_indices]


#Takes the best chunks
#In particular it takes all 4 possible [question,option] pairs and scores all chunks against that pair. Then the best <top_k> chunks are taken
def rerank_chunks_v2(question, options, chunks, top_k=3):
    best_chunks = []

    for option in options:
        query = f"{question} {option}"
        pairs = [[query, chunk] for chunk in chunks]
        scores = reranker.predict(pairs)

        best_idx = np.argmax(scores)
        best_chunks.append((scores[best_idx], chunks[best_idx]))

    best_chunks.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    final = []
    for score, chunk in best_chunks:
        norm = re.sub(r"\s+", " ", chunk.strip().lower())
        if norm not in seen:
            seen.add(norm)
            final.append(chunk)
        if len(final) == top_k:
            break

    return final

def rerank_chunks_v3(question, options, chunks, top_k=3):
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

    # No titles or rate limited, a log could be made here
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

        # No documents or rate limited, a log could be made here
        if not extract_response.text or extract_response.status_code != 200:
            continue

        for page in extract_response.json()["query"]["pages"].values():
            if "extract" in page:
                pages.append((title, page["extract"]))

    return pages

#Performs multiple wikipedia search API calls in parallel
def wikisearch_multi(queries, question, options):

    def search_one(query):
        #query = query.strip().replace('"', "")
        if not query:
            return []
        return wikisearch_single(query)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(search_one, queries))

    # Pool all chunks from all queries
    all_pages = [page for query_pages in results for page in query_pages]

    # Now deduplicate by title
    seen_titles = set()
    unique_pages = []
    for title, text in all_pages:
        if title not in seen_titles:
            seen_titles.add(title)
            unique_pages.append((title, text))

    # Now chunk the unique full texts
    # Title prefix gives the reranker article-level context at zero cost
    all_chunks = []
    for title, text in unique_pages:
        for chunk in chunk_text(text):
            all_chunks.append(f"[{title}] {chunk}")

    if not all_chunks:
        return []

    print(f"Time to search the documents {time.time()-start_time:.1f}")
    #prefiltered = get_top_chunks(question, options, all_chunks, top_k=10)
    prefiltered = bm25_prefilter(question, options, all_chunks, top_k=10)
    return rerank_chunks_v3(question, options, prefiltered, top_k=5)


# ==============================================================================
# MODEL INFERENCE CALLS (LLM Engine)
# ==============================================================================

def call_model(prompt, max_tokens):
    response = ollama.chat(
        model="qwen2.5:14b-instruct",
        messages=prompt,
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]


def call_math_model(prompt):
    response = ollama.chat(
        model="deepseek-r1:14b-qwen-distill-q8_0", messages=prompt, think=False, options={"temperature": 0.1}
    )
    content = response["message"].get("content", "")
    print(f"Math model reasoning: {content}")
    text_to_search = content if content.strip() else ""
    match = re.search(r"FINAL_ANSWER:\s*([ABCD])", text_to_search)
    return match.group(1) if match else None



# ==============================================================================
# CORE GUESSER PIPELINE CLASS
# ==============================================================================


class WikiRAGGuesser(Guesser):

    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(config)

    def infer_answer(self, question: Question, theme: str = None) -> int:
        self.search_time = 0.0
        self.reasoning_time = 0.0
        try:
            options = [opt.text for opt in question.options]
            t0 = time.time()
            if theme == "Maths":
                model_prompt = build_math_prompt(question.text, options)
                response = call_math_model(model_prompt)
                print(f"Model answered: {response}")
                self.reasoning_time = time.time() - t0
                return {"A": 0, "B": 1, "C": 2, "D": 3}.get(response, 0)
            else:
                model_prompt = build_query_prompt(question.text, options, theme)
                raw_queries = call_model(model_prompt, 300)
                print(f"Raw queries: {raw_queries}")
                queries = [clean_query(q) for q in raw_queries.strip().split("\n")]
                queries = [q for q in queries if 3 < len(q) < 120]
                queries = list(dict.fromkeys(queries))[
                    :3
                ]  # hardcoding of just 3 queries
                print(f"Queries: {queries}")
                found_docs = wikisearch_multi(queries, question.text, options)
                print(f"Retrieved chunks: {found_docs}")
                self.search_time = time.time() - t0

                t1 = time.time()
                model_prompt = build_doc_prompt(
                    question.text, options, found_docs
                )
                response = call_model(model_prompt, 80)
                print(f"Model answered: {response}")
                self.reasoning_time = time.time() - t1

            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            match = re.search(r"FINAL_ANSWER:\s*([ABCD])", response)
            if match:
                return letter_to_index[match.group(1)]
            return random.randint(0, 3)

        except Exception:
            raise


# ==============================================================================
# LEGACY / EXPERIMENTAL CODE BLOCKS (DEPRECATED COMMENTED CODE)
# ==============================================================================
"""
def filter_top_pages(question, options, search_results, top_p=3):
    \"\"\"
    Ranks Wikipedia search results based on snippets before downloading full pages.
    \"\"\"
    if not search_results:
        return []

    # Combine question and options for a rich search context
    query_context = f"{question} {' '.join(options)}"
    
    # Extract snippets and titles
    texts_to_rank = [f"{res['title']} {res.get('snippet', '')}" for res in search_results]
    
    # Use cross-encoder for high-precision page selection (fast on small N)
    pairs = [[query_context, text] for text in texts_to_rank]
    scores = reranker.predict(pairs)
    
    # Get indices of the best pages
    top_indices = np.argsort(scores)[::-1][:top_p]
    return [search_results[i]["title"] for i in top_indices]

# General call to LLM model via tokenizer local deployment
def call_model(tokenizer, model, prompt, max_tokens):
    inputs = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    ).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

!pip install groq
from groq import Groq
groq_client = Groq(api_key=userdata.get('API_KEY'))

def call_api_model(prompt, max_tokens):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


import threading

def call_with_timeout(prompt, timeout=27):
    final_ans_re = re.compile(r"FINAL_ANSWER:\s*([ABCD])")
    checkpoint_re = re.compile(r"CURRENT_BEST:\s*([ABCD])")
    
    result = {}

    def stream_call():
        thinking_buf = ""
        content_buf = ""
        
        for chunk in ollama.chat(
            model="deepseek-r1:14b-qwen-distill-q8_0",
            messages=prompt,
            stream=True,
            think=False,
            options={"temperature": 0.3}
        ):
            msg = chunk["message"]
            #thinking_buf += msg.thinking or ""
            content_buf += msg.content or ""
            #result["thinking"] = thinking_buf
            result["content"] = content_buf

    thread = threading.Thread(target=stream_call)
    thread.start()
    thread.join(timeout=timeout)
    # at this point result has whatever was streamed before timeout

    #thinking = result.get("thinking", "")
    content = result.get("content", "")

    #print(thinking)
    print(content)

    match = final_ans_re.search(content)
    if match:
        return match.group(1)
    checkpoints = checkpoint_re.findall(content)
    if checkpoints:
        return checkpoints[-1]
    return None

def deduplicate_chunks(chunks):
   unique = []

    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk.lower()).strip()

        is_duplicate = False
        for existing in unique:
            ex = re.sub(r"\s+", " ", existing.lower()).strip()

            if normalized[:200] == ex[:200]:
                is_duplicate = True
                break

            # containment check
            if normalized in ex or ex in normalized:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(chunk)

    return unique

FIRST MATH PROMPT (SECOND BEST WORKING ACCURACY GIVEN EMPIRICAL EXAMINATION)
You are a mathematics expert. You will receive a question and 4 possible options, only one is correct. CRITICAL OPERATIONAL CONSTRAINTS: 1. TOKEN BUDGET / BREVITY: You have a strict generation limit. Do NOT drift into long-winded explanations. Keep your internal reasoning extremely concise, direct, and under 10 sentences total. Focus only on the core formula and calculation. At the very end, output exactly one line in this precise format: FINAL_ANSWER: X where X is A, B, C, or D.
"""