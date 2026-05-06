from Marcelo.src.guesser.ingestion.extractors.zim_extractor import ZimExtractor
from src.guesser.ingestion.chunking import Chunker
from src.guesser.ingestion.loader import Loader

class ZimIngestionPipeline:
    def __init__(self,raw_data_file_path,embedding_model,db_path,colletion_name):
        self.path = raw_data_file_path
        self.extractor = ZimExtractor(raw_data_file_path)
        self.loader = Loader(
                            db_path=db_path,
                            model_name=embedding_model,
                            collection_name=colletion_name)

    def process(self, limit=10, starting_id=0):
        batch_nodes = []
        print("Starting extraction and chunking...")
        for doc in self.extractor.extract_articles(limit=limit, starting_id=starting_id):
            chunker = Chunker(doc)
            chunker.chunk_article()
            batch_nodes.extend(chunker.all_nodes)   
        print(f"Total of {len(batch_nodes)} nodes generated. Sending to ChromaDB...") 

        if batch_nodes:
            self.loader.load_nodes(batch_nodes)
            print("Ingestion Concluded!")
        else:
            print("No articles found to be processed.")