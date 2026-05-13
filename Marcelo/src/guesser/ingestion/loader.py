from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

class Loader: 
    def __init__(self, db_path="../context_db", embedding_model_name="nomic-embed-text"):
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.embed_model = OllamaEmbedding(model_name=embedding_model_name)
        self._indices = {}

    def _get_or_create_index(self, collection_name):
        if collection_name not in self._indices:
            chroma_collection = self.db_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self._indices[collection_name] = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
        return self._indices[collection_name]
    
    async def aload_nodes(self, nodes, collection_name="wikipedia_mcq"):
        index = self._get_or_create_index(collection_name)
        await index.ainsert_nodes(nodes)
        return index

    def get_index(self, collection_name: str):        
        return self._get_or_create_index(collection_name)

    def get_all_document_ids(self, collection_name):
        """Retrieves all existing document IDs from the collection."""
        try:
            collection = self.db_client.get_collection(collection_name)
            # We need metadatas because 'ids' are node-level UUIDs, 
            # while 'doc_id' in metadata is the content-based hash we use.
            results = collection.get(include=['metadatas'])

            if not results or not results['metadatas']:
                return set()

            # Extract doc_id from each metadata dictionary
            existing_hashes = set()
            for meta in results['metadatas']:
                if meta and 'doc_id' in meta:
                    existing_hashes.add(meta['doc_id'])

            return existing_hashes
        except Exception:
            # Collection might not exist yet
            return set()