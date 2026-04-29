import sys
import os

# Add the project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.context_db.context_db import contextDB

def populate():
    # Initialize the database
    # It will default to the src/context_db/ folder
    db = contextDB()
    
    # Create or get the collection
    collection_name = "context"
    db.create_collection(collection_name)
    
    # Define documents to add
    documents = [
        "Marcelo tem 20 anos",
        "A casa é verde",
        "A internet é tóxica",
        "O Sol é uma estrela de tipo G2V que está no centro do Sistema Solar.",
        "A Lua é o único satélite natural da Terra.",
        "A capital da França é Paris.",
        "A inteligência artificial está transformando a tecnologia moderna.",
        "O aprendizado de máquina é um subcampo da IA.",
        "Python é uma linguagem de programação popular para ciência de dados.",
        "ChromaDB é um banco de dados vetorial para construir aplicações de IA com embeddings."
    ]
    
    # Add metadata for each document
    metadata = [
        {"topic": "personal", "source": "user_input"},
        {"topic": "house", "source": "user_input"},
        {"topic": "social", "source": "user_input"},
        {"topic": "astronomy", "source": "general_knowledge"},
        {"topic": "astronomy", "source": "general_knowledge"},
        {"topic": "geography", "source": "general_knowledge"},
        {"topic": "technology", "source": "general_knowledge"},
        {"topic": "technology", "source": "general_knowledge"},
        {"topic": "technology", "source": "general_knowledge"},
        {"topic": "technology", "source": "documentation"}
    ]
    
    # Add documents to the collection
    print(f"Adding {len(documents)} documents to collection '{collection_name}'...")
    db.add_document_to_collection(collection_name, documents, metadata)
    
    print("\nDatabase populated successfully!")

if __name__ == "__main__":
    populate()
