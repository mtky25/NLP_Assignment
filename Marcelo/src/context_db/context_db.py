import chromadb
from chromadb.utils import embedding_functions
import os
import uuid


class contextDB: 
    def __init__(self, path=None, model_embedding_name=None):
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
        self.client = chromadb.PersistentClient(path=path)
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name="nomic-embed-text" if model_embedding_name is None else model_embedding_name
            )  
        
    def create_collection(self, collection_name):
        return self.client.get_or_create_collection(name=collection_name,embedding_function=self.ollama_ef)
    
    def get_all_collections(self):
        collections = self.client.list_collections()
        print([c.name for c in collections])
        return collections

    def add_document_to_collection(self, collection_name, documents: list, metadata=None):
        ids = [str(uuid.uuid4()) for _ in documents]

        # Ensure we are working with the collection object with the correct embedding function
        collection = self.client.get_collection(name=collection_name, embedding_function=self.ollama_ef)
        
        if metadata is not None:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadata
            )
        else:
            collection.upsert(
                ids=ids,
                documents=documents
            )
        
        print(f"{len(documents)} documents read and processed.")
    
    def query(self, collection_name, texts: list, n_results):
        collection = self.client.get_collection(name=collection_name, embedding_function=self.ollama_ef)
        results = collection.query(
            query_texts=texts,
            n_results=n_results
        )
        return results





# collection = client.get_collection(name="context")

# # A. Contar quantos itens existem
# print(f"Total de registros: {collection.count()}")

# # B. Espiar os primeiros 5 registros (Documentos, IDs e Metadados)
# # O método .get() retorna os dados brutos sem a necessidade de uma busca semântica
# dados = collection.get(limit=5)
# print(dados)

# # C. Ver os metadados de tudo o que está lá
# print(dados['metadatas'])

# import chromadb
# from chromadb.config import Settings

# client = chromadb.PersistentClient(path="./src/guesser")

# collection = client.get_or_create_collection(name="context")

# # 3. Adiciona dados ao banco
# # O ChromaDB, por padrão, usa um modelo de embedding leve (all-MiniLM-L6-v2) 
# # que ele baixa automaticamente na primeira execução.
# collection.add(
#     documents=[
#         "Apple is a good fruit for the health."
#     ],
#     ids=["id1"],
#     metadatas=[{"fonte": "healthy"}]
# )

# # 4. Realiza uma consulta (Query)
# results = collection.query(
#     query_texts=["fruits"],
#     n_results=1
# )

# # 5. Exibe o resultado
# print(f"Texto mais relevante encontrado: {results['documents'][0][0]}")
# print(f"Metadados: {results['metadatas'][0][0]}")