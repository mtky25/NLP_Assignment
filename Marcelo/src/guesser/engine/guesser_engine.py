import time
from Marcelo.src.guesser.engine.llmprovider import get_llm
from Marcelo.src.guesser.engine.router import Router
from src.millionaire_client.models import Question
from Marcelo.src.guesser.ingestion.loader import Loader

class GuesserEngine:
    
    def __init__(self, llm_model="llama3.2", db_path:str="", embedding_model:str="", top_k=5, temperature=0.0, theme:str=""):
        self.llm = get_llm(
            model_name=llm_model,
            temperature=temperature
        )
        
        self.theme = theme
        self.router_data = Router(theme).route() 
        self.collection_name = self.router_data[0]
        self.prompt_template = self.router_data[1]
        
        self.index = Loader(db_path, embedding_model).get_index(self.collection_name)
        
        self.retriever = self.index.as_retriever(similarity_top_k=top_k)
        self.top_k = top_k

    def _reciprocal_rank_fusion(self, results_list, k=60):
        node_scores = {}
        for nodes in results_list:
            for rank, node_with_score in enumerate(nodes):
                node_id = node_with_score.node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = {"score": 0.0, "node": node_with_score.node}
                node_scores[node_id]["score"] += 1.0 / (k + rank + 1)
        
        sorted_nodes = sorted(node_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["node"] for item in sorted_nodes]

    def generate_queries(self, question: Question):
        only_question = question.text
        
        options_list = []
        for index, option in enumerate(question.options):
            options_list.append(f"[{index}] {option.text}")
        
        options_text = " ".join(options_list)
        question_and_options = f"Question: {only_question} Options: {options_text}"
        
        return only_question, question_and_options, options_text

    def answer(self, question: Question):
        start_time = time.time()

        q_only, q_full, options_str = self.generate_queries(question)

        nodes_only = self.retriever.retrieve(q_only)
        nodes_full = self.retriever.retrieve(q_full)

        fused_nodes = self._reciprocal_rank_fusion([nodes_only, nodes_full])
        
        final_nodes = fused_nodes[:self.top_k]
        context_str = "\n\n".join([n.get_content() for n in final_nodes])
        final_prompt = self.prompt_template.format(
            context=context_str,
            question=q_only,
            options=options_str
        )

        response = self.llm.complete(final_prompt)
        fonts = [node.get_content() for node in final_nodes]
        resolution = [node.metadata.get('answer', 'No Resolution Found') for node in final_nodes]

        elapsed_time = time.time() - start_time
        print(f"Total time: {elapsed_time:.2f}s")

        return {
            "answer": str(response).strip(),
            "sources": fonts,
            "resolution_method": resolution,
            "time_taken": f"{elapsed_time:.2f}s"
        }