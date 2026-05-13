import asyncio
import hashlib
from src.guesser.ingestion.extractors.dataset_extractor import DatasetExtractor
from src.guesser.ingestion.chunking import Chunker
from src.guesser.ingestion.loader import Loader

class DatasetIngestionPipeline:
    def __init__(self,dataset_config,embedding_model,db_path,colletion_name):
        self.extractor = DatasetExtractor(dataset_config)
        self.loader = Loader(
                            db_path=db_path,
                            embedding_model_name=embedding_model,
                            )
        self.collection_name = colletion_name

    async def _producer(self, queue, limit, starting_id, batch_size, random_seed, initial_seen_hashes):
        """
        Runs document extraction and pushes batches to an async queue.
        """
        loop = asyncio.get_event_loop()
        print(f" [Producer] Starting extraction from Dataset...")

        def run_extraction():
            count = 0
            duplicates = 0
            seen_hashes = initial_seen_hashes.copy()
            batch = []
            
            for doc in self.extractor.extract(limit=limit, starting_id=starting_id, random_seed=random_seed):
                # doc.doc_id is already the MD5 hash set in the extractor
                doc_hash = doc.doc_id
                
                if doc_hash in seen_hashes:
                    duplicates += 1
                    continue
                
                seen_hashes.add(doc_hash)
                batch.append(doc)
                count += 1

                if len(batch) >= batch_size:
                    asyncio.run_coroutine_threadsafe(queue.put(batch), loop).result()
                    if count % 100 == 0 or count == (limit if limit else count):
                        print(f" [Producer] Extracted and queued {count} unique documents...")
                    batch = []

            if batch:
                asyncio.run_coroutine_threadsafe(queue.put(batch), loop).result()
                print(f" [Producer] Extracted and queued {count} unique documents...")

            total_attempted = count + duplicates
            dup_percent = (duplicates / total_attempted * 100) if total_attempted > 0 else 0
            
            print(f" [Producer] Extraction finished. Total unique: {count} | Duplicates skipped: {duplicates} ({dup_percent:.2f}%)")
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
            return count, duplicates

        # We need the return values from the thread
        return await loop.run_in_executor(None, run_extraction)

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

            nodes = await chunker.achunk_documents(batch)
            total_nodes += len(nodes)

            await self.loader.aload_nodes(nodes, collection_name=self.collection_name)
            print(f" [Consumer] Successfully loaded {len(nodes)} nodes into ChromaDB. (Total Nodes: {total_nodes})")

            queue.task_done()

        return processed_count, total_nodes

    async def process(self, limit=10, starting_id=0, batch_size=50, random_seed=None):
        print(f"Starting Fast Parallel Ingestion (limit={limit}, batch_size={batch_size}, random={random_seed is not None})...")

        # Fetch existing IDs from ChromaDB for global duplicate detection
        existing_ids = self.loader.get_all_document_ids(self.collection_name)
        if existing_ids:
            print(f" [Pipeline] Found {len(existing_ids)} existing documents in collection '{self.collection_name}'.")

        queue = asyncio.Queue(maxsize=10)

        producer_task = asyncio.create_task(self._producer(queue, limit, starting_id, batch_size, random_seed, existing_ids))
        consumer_task = asyncio.create_task(self._consumer(queue, limit))

        await asyncio.gather(producer_task, consumer_task)

        unique_count, duplicate_count = await producer_task
        processed_count, total_nodes = await consumer_task

        total_attempted = unique_count + duplicate_count
        dup_percent = (duplicate_count / total_attempted * 100) if total_attempted > 0 else 0

        print(f"\nIngestion Concluded!")
        print(f"Total articles attempted: {total_attempted}")
        print(f"Unique articles processed: {processed_count}")
        print(f"Duplicates skipped: {duplicate_count} ({dup_percent:.2f}%)")
        print(f"Total nodes generated: {total_nodes}")