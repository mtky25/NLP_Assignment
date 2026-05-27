from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from src.guesser.ingestion.configs import CHUNK_SIZE, CHUNK_OVERLAP
import asyncio

class Chunker:
    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    async def achunk_documents(self, documents):
        """
        Asynchronously chunk a list of documents into nodes.
        """
        # SentenceSplitter's aget_nodes_from_documents is async
        return await self.splitter.aget_nodes_from_documents(documents)

    def chunk_article(self, document: Document):
        """
        Synchronous version for backward compatibility if needed, 
        though pipelines should migrate to achunk_documents.
        """
        return self.splitter.get_nodes_from_documents([document])



