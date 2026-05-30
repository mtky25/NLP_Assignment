from llama_index.core.node_parser import SentenceSplitter
from src.guesser.ingestion.configs import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:
    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    async def achunk_documents(self, documents):
        return await self.splitter.aget_nodes_from_documents(documents)
