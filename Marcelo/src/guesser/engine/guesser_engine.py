from Marcelo.src.guesser.engine.llmprovider import get_llm
from Marcelo.src.guesser.engine.prompts import MCQ_PROMPT
from src.millionaire_client.models import Question
class GuesserEngine:
    
    def __init__(self, index, llm_model="llama3.2:2b",top_k=5,temperature=0.0,prompt=MCQ_PROMPT):
        self.llm = get_llm(
            model_name=llm_model,
            temperature=temperature
        )
        self.engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=top_k,
            prompt=MCQ_PROMPT
        )

    def answer_question(self, question):

        response = self.engine.query(question)
        fonts = []
        resolution = []
        
        for node in response.source_nodes:
            fonts.append(node.get_content())
            resolution.append(node.metadata.get('answer', 'No Resolution Found'))

        return {
            "answer": str(response).strip(),
            "sources": fonts,
            "resolution_method": resolution
        }