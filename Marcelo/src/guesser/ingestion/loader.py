from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

class Loader: 
    # Class-level cache for ChromaDB clients to prevent multiple heavy connections to the same path
    _clients = {}

    def __init__(self, db_path="../context_db", embedding_model_name="nomic-embed-text"):
        # Resolve path to ensure consistent keys in _clients
        import os
        abs_db_path = os.path.abspath(db_path)
        
        if abs_db_path not in Loader._clients:
            Loader._clients[abs_db_path] = chromadb.PersistentClient(path=abs_db_path)
        
        self.db_client = Loader._clients[abs_db_path]
        self.embed_model = OllamaEmbedding(model_name=embedding_model_name)
        self._indices = {}

    def _get_or_create_index(self, collection_name):
        if collection_name not in self._indices:
            # Use Cosine Similarity instead of the default L2 distance
            chroma_collection = self.db_client.get_or_create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
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

    def get_all_document_ids(self, collection_name, batch_size=10000):
        """
        Retrieves all existing document IDs from the collection using batching
        to prevent memory exhaustion on large datasets.
        """
        try:
            collection = self.db_client.get_collection(collection_name)
            existing_hashes = set()
            offset = 0
            
            while True:
                # Fetch in batches to keep memory usage stable
                results = collection.get(
                    include=['metadatas'],
                    limit=batch_size,
                    offset=offset
                )
                
                if not results or not results['metadatas']:
                    break
                
                for meta in results['metadatas']:
                    if meta and 'doc_id' in meta:
                        existing_hashes.add(meta['doc_id'])
                
                if len(results['metadatas']) < batch_size:
                    break
                    
                offset += batch_size
                print(f" [Loader] Loaded {len(existing_hashes)} unique hashes so far...")

            return existing_hashes
        except Exception:
            # Collection might not exist yet
            return set()