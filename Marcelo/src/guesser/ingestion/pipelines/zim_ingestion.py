import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.guesser.ingestion.extractors.zim_extractor import ZimExtractor
from src.guesser.ingestion.chunking import Chunker
from src.guesser.ingestion.loader import Loader

class ZimIngestionPipeline:
    def __init__(self,raw_data_file_path,embedding_model,db_path,colletion_name):
        self.path = raw_data_file_path
        self.extractor = ZimExtractor(raw_data_file_path)
        self.loader = Loader(
                            db_path=db_path,
                            embedding_model_name=embedding_model,
                            )
        self.collection_name = colletion_name

    async def _producer(self, queue, limit, starting_id, summary_only, num_paragraphs, random_seed, batch_size):
        """
        Runs document extraction and pushes batches to an async queue.
        """
        loop = asyncio.get_event_loop()
        print(f" [Producer] Starting extraction from ZIM...")
        
        # Function to run the synchronous generator in a separate thread
        def run_extraction():
            count = 0
            batch = []
            for doc in self.extractor.extract(
                limit=limit, 
                starting_id=starting_id, 
                summary_only=summary_only, 
                num_paragraphs=num_paragraphs, 
                random_seed=random_seed
            ):
                batch.append(doc)
                count += 1
                
                if len(batch) >= batch_size:
                    # Put a whole batch in the queue at once to reduce overhead
                    asyncio.run_coroutine_threadsafe(queue.put(batch), loop).result()
                    if count % 100 == 0 or count == limit:
                        print(f" [Producer] Extracted and queued {count} documents...")
                    batch = []
            
            # Put the final partial batch in the queue
            if batch:
                asyncio.run_coroutine_threadsafe(queue.put(batch), loop).result()
                print(f" [Producer] Extracted and queued {count} documents...")

            print(f" [Producer] Extraction finished. Total docs extracted: {count}")
            # Signal completion
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Extraction is I/O and CPU bound (HTML cleaning), so we run it in a thread
        await loop.run_in_executor(None, run_extraction)

    async def _consumer(self, queue, limit):
        """
        Pulls document batches from queue, chunks them, and loads into ChromaDB.
        """
        chunker = Chunker()
        processed_count = 0
        total_nodes = 0

        while True:
            batch = await queue.get()
            
            if batch is None:
                queue.task_done()
                break
                
            processed_count += len(batch)
            print(f" [Consumer] Chunking and Embedding batch of {len(batch)} docs (Total seen: {processed_count}/{limit or 'All'})...")
            
            # Chunks documents and loads them
            nodes = await chunker.achunk_documents(batch)
            total_nodes += len(nodes)
            
            await self.loader.aload_nodes(nodes, collection_name=self.collection_name)
            print(f" [Consumer] Successfully loaded {len(nodes)} nodes into ChromaDB. (Total Nodes: {total_nodes})")
            
            queue.task_done()
        
        return processed_count, total_nodes

    async def process(self, limit=10, starting_id=0, batch_size=50, summary_only=False, num_paragraphs=10, random_seed=None):
        print(f"Starting Fast Parallel Ingestion (limit={limit}, batch_size={batch_size}, random={random_seed is not None})...")
        
        # The queue now holds lists of documents (batches)
        queue = asyncio.Queue(maxsize=10) # Enough for 10 batches in the pipeline
        
        # Start producer and consumer tasks
        producer_task = asyncio.create_task(self._producer(queue, limit, starting_id, summary_only, num_paragraphs, random_seed, batch_size))
        consumer_task = asyncio.create_task(self._consumer(queue, limit))
        
        # Wait for everything to finish
        await asyncio.gather(producer_task, consumer_task)
        
        processed_count, total_nodes = await consumer_task
        print(f"\nIngestion Concluded!")
        print(f"Total articles processed: {processed_count}")
        print(f"Total nodes generated: {total_nodes}")