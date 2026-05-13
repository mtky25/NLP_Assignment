import time
from src.guesser.engine.llmprovider import get_llm
from src.guesser.engine.router import Router
from src.millionaire_client.models import Question
from src.guesser.ingestion.loader import Loader
from src.models import ApproachType

class GuesserEngine:
    
    def __init__(self, llm_model="llama3.2", db_path:str="", embedding_model:str="", top_k=2, temperature=0.0, theme:str=""):
        self.llm = get_llm(
            model_name=llm_model,
            temperature=temperature
        )
        
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.theme = None
        self.approach_type = None
        self.theme_cache = {} # {theme_name: (retriever, prompt_template, approach_type)}
        
        if theme:
            self.set_theme(theme)
        else:
            self.set_theme("science and nature")

    def set_theme(self, theme: str):
        """
        Dynamically switch the theme (index, prompt, and approach) used by the engine.
        """
        if theme == self.theme:
            return

        if theme in self.theme_cache:
            self.retriever, self.prompt_template, self.approach_type = self.theme_cache[theme]
            self.theme = theme
            return

        router_data = Router(theme).route() 
        collection_name = router_data[0]
        prompt_template = router_data[1]
        approach_type = router_data[2]
        
        index = Loader(self.db_path, self.embedding_model).get_index(collection_name)
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        
        self.theme_cache[theme] = (retriever, prompt_template, approach_type)
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.approach_type = approach_type
        self.theme = theme

    def _reciprocal_rank_fusion(self, results_list, k=60):
        node_scores = {}
        for nodes in results_list:
            for rank, node_with_score in enumerate(nodes):
                node_id = node_with_score.node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = {
                        "rrf_score": 0.0, 
                        "node": node_with_score.node,
                        "original_score": node_with_score.score
                    }
                else:
                    # Keep the highest original similarity score
                    node_scores[node_id]["original_score"] = max(node_scores[node_id]["original_score"], node_with_score.score)
                node_scores[node_id]["rrf_score"] += 1.0 / (k + rank + 1)
        
        sorted_nodes = sorted(node_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_nodes

    def generate_queries(self, question: Question):
        only_question = question.text
        
        options_list = []
        for index, option in enumerate(question.options):
            options_list.append(f"[{index}] {option.text}")
        
        options_text = " ".join(options_list)
        question_and_options = f"Question: {only_question} Options: {options_text}"
        
        return only_question, question_and_options, options_text

    def chat(self, prompt: str) -> dict:
        """
        Directly chat with the LLM using RAG context for debugging.
        """
        nodes = self.retriever.retrieve(prompt)
        
        # De-duplicate by content to avoid redundant information
        unique_contents = []
        seen_contents = set()
        for n in nodes:
            content = n.get_content().strip()
            if content not in seen_contents:
                unique_contents.append(content)
                seen_contents.add(content)
        
        context_str = "\n\n".join(unique_contents)
        
        final_prompt = f"Context:\n{context_str}\n\nUser Question: {prompt}\n\nAnswer:"
        response = self.llm.complete(final_prompt)
        
        return {
            "response": str(response).strip(),
            "context_chunks": unique_contents
        }

    def answer(self, question: Question):
        start_total = time.time()
        self.last_search_time = 0.0
        self.last_reasoning_time = 0.0

        q_only, q_full, options_str = self.generate_queries(question)

        context_str = ""
        final_nodes = []
        
        # Decide if retrieval is needed
        if self.approach_type in [ApproachType.RAG, ApproachType.HYBRID]:
            search_start = time.time()
            try:
                nodes_only = self.retriever.retrieve(q_only)
                nodes_full = self.retriever.retrieve(q_full)

                fused_items = self._reciprocal_rank_fusion([nodes_only, nodes_full])
                
                # De-duplicate by content
                seen_contents = set()
                for item in fused_items:
                    content = item["node"].get_content().strip()
                    if content not in seen_contents:
                        final_nodes.append(item)
                        seen_contents.add(content)
                
                final_nodes = final_nodes[:self.top_k]
                context_str = "\n\n".join([item["node"].get_content() for item in final_nodes])
                
                if final_nodes:
                    print(f" [Retriever] Top {len(final_nodes)} chunks similarity scores:")
                    for idx, item in enumerate(final_nodes):
                        print(f"   - Chunk {idx+1}: Similarity={item['original_score']:.4f} | RRF={item['rrf_score']:.4f}")
            finally:
                self.last_search_time = time.time() - search_start

        if self.approach_type == ApproachType.DIRECT_LLM:
            final_prompt = f"You are an expert assistant. Answer the following multiple choice question.\n\nQUESTION:\n{q_only}\n\nOPTIONS:\n{options_str}\n\nOutput ONLY the correct option index (0, 1, 2, or 3).\nFinal Option Index: "
        else:
            final_prompt = self.prompt_template.format(
                context=context_str,
                question=q_only,
                options=options_str
            )

        reasoning_start = time.time()
        try:
            response = self.llm.complete(final_prompt)
        finally:
            self.last_reasoning_time = time.time() - reasoning_start

        fonts = [item["node"].get_content() for item in final_nodes]
        resolution = [item["node"].metadata.get('answer', 'No Resolution Found') for item in final_nodes]

        total_elapsed = time.time() - start_total
        print(f"[{self.approach_type.upper()}] Total: {total_elapsed:.2f}s | Search: {self.last_search_time:.2f}s | Reasoning: {self.last_reasoning_time:.2f}s")

        return {
            "answer": str(response).strip(),
            "sources": fonts,
            "resolution_method": resolution,
            "search_time": self.last_search_time,
            "reasoning_time": self.last_reasoning_time,
            "total_time": total_elapsed,
            "chunks_metadata": [
                {"text": item["node"].get_content(), "similarity": item["original_score"], "rrf": item["rrf_score"]}
                for item in final_nodes
            ]
        }