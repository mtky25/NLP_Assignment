import time


def web_search(query: str, max_results: int = 5, debug: bool = False) -> str:
    """
    Search DuckDuckGo and return concatenated snippets as a context string.
    Returns empty string on any failure (caller falls back to LLM knowledge).
    """
    if debug:
        print(f"[DEBUG] WebSearch query: '{query}'")
        print(f"[DEBUG] WebSearch max_results: {max_results}")

    start = time.time()
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
    except Exception as e:
        elapsed = time.time() - start
        print(f" [WebSearch] Failed after {elapsed:.2f}s: {e}")
        return ""
    elapsed = time.time() - start

    if not results:
        if debug:
            print(f"[DEBUG] WebSearch returned 0 results in {elapsed:.2f}s")
        return ""

    snippets = [f"{r['title']}: {r['body']}" for r in results if r.get('body')]

    if debug:
        print(f"[DEBUG] WebSearch got {len(snippets)} snippets in {elapsed:.2f}s")
        for i, r in enumerate(results):
            print(f"[DEBUG]   [{i}] {r.get('title', '')[:80]}")
            print(f"[DEBUG]       {r.get('body', '')[:200]}")
            print(f"[DEBUG]       {r.get('href', '')}")

    return "\n\n".join(snippets)
