from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

class Loader: 
    def __init__(self, db_path="../context_db", embedding_model_name="nomic-embed-text"):
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.embed_model = OllamaEmbedding(model_name=embedding_model_name)
    
    def load_nodes(self, nodes, collection_name="wikipedia_mcq"):
        chroma_collection = self.db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes, 
            storage_context=storage_context, 
            embed_model=self.embed_model
        )

    def get_index(self, collection_name: str):        
        chroma_collection = self.db_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        return index