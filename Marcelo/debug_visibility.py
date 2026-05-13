import os
import sys
from llama_index.core.schema import Document

# Ensure the root directory is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_metadata_visibility():
    # Simulate a document created by our current DatasetExtractor
    text = "What is 2 + 2?"
    meta = {
        "answer": "4",
        "solution": "Step 1: take 2. Step 2: add 2. Result is 4."
    }
    
    doc = Document(text=text, metadata=meta)
    
    print("\n--- Current LlamaIndex Default Representation ---")
    print("This is exactly what is sent to the LLM context:")
    print("-" * 50)
    # get_content() with MetadataMode.LLM is what the prompt template uses
    from llama_index.core.schema import MetadataMode
    print(doc.get_content(metadata_mode=MetadataMode.LLM))
    print("-" * 50)

if __name__ == "__main__":
    test_metadata_visibility()
