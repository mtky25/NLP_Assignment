from llama_index.core.node_parser import SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from src.guesser.configs import EMBEDDING_MODEL,CHUNK_SIZE,CHUNK_OVERLAP
import pprint

class Chunker:
    def __init__(self,document:Document):
        self.document = document
        self.all_nodes = []

        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model
        )
        
        self.token_fallback_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP 
        )

    def chunk_article(self):
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model
        )
        
        semantic_nodes = semantic_splitter.get_nodes_from_documents([self.document])

        for node in semantic_nodes:
            if len(node.get_content()) > 4000: 
                safe_nodes = self.token_fallback_splitter.get_nodes_from_documents(
                    [Document(text=node.get_content(), metadata=node.metadata)]
                )
                self.all_nodes.extend(safe_nodes)
            else:
                self.all_nodes.append(node)

            pprint.pprint(node)


