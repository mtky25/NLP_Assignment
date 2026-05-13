import bs4
import pprint
import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor
from src.guesser.ingestion.extractors.extractor import BaseExtractor
from libzim.reader import Archive
from llama_index.core import Document


class ZimExtractor(BaseExtractor):
    def __init__(self, file_name, max_workers=None):
        self.file_name = file_name
        self.archive = Archive(file_name)
        # Snapdragon X has 10-12 cores. We use 8 workers to leave breathing room for Ollama.
        self.max_workers = max_workers or min(8, (os.cpu_count() or 4))

    def count_archives(self):
        pprint.pprint(f"Total of Articles: {self.archive.entry_count}")

    def _process_single_entry(self, entry_id, summary_only, num_paragraphs):
        """
        Processes a single ZIM entry: reads, cleans, and generates a Document with ID.
        Runs in a ThreadPool for parallel HTML cleaning.
        """
        try:
            entry = self.archive._get_entry_by_id(entry_id)
            item = entry.get_item()
            
            if item.mimetype == "text/html":
                html_content = item.content.tobytes().decode('utf-8')
                clean_text = self._clean_html(html_content, summary_only=summary_only, num_paragraphs=num_paragraphs)
                
                if not clean_text.strip():
                    return None
                
                base_name = os.path.basename(self.file_name)
                doc_id = hashlib.md5(f"{base_name}_{entry.path}".encode()).hexdigest()
                
                return Document(
                    text=clean_text,
                    doc_id=doc_id,
                    metadata={
                        "title": entry.title,
                        "path": entry.path,
                        "font": "Kiwix_ZIM",
                        "content": "article_wikipedia"
                    }
                )
        except Exception:
            pass
        return None

    def extract(self, limit=None, starting_id=0, summary_only=False, num_paragraphs=10, random_seed=None):
        total_articles = self.archive.entry_count
        
        # Determine the range of IDs to scan
        if random_seed is not None:
            random.seed(random_seed)
            indices = list(range(starting_id, total_articles))
            random.shuffle(indices)
        else:
            indices = range(starting_id, total_articles)

        count = 0
        
        # Process in batches to efficiently feed the ThreadPool
        internal_batch_size = self.max_workers * 4
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(indices), internal_batch_size):
                chunk = indices[i:i + internal_batch_size]
                
                # Clean multiple HTML files simultaneously
                futures = [executor.submit(self._process_single_entry, idx, summary_only, num_paragraphs) for idx in chunk]
                
                for future in futures:
                    doc = future.result()
                    if doc:
                        yield doc
                        count += 1
                        if limit and count >= limit:
                            return
                
    def _clean_html(self, html, summary_only=False, num_paragraphs=10):
        # Use lxml parser if available for significantly better performance
        parser = 'lxml' if 'lxml' in bs4.builder.builder_registry.builders else 'html.parser'
        soup = bs4.BeautifulSoup(html, parser) 
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'table', 'aside', 'figure']):
            tag.decompose()
            
        if summary_only:
            # Wikipedia articles usually keep the summary in the lead section
            # which consists of the first few paragraphs before the first heading.
            paragraphs = []
            p_count = 0
            # Iterate through siblings of the body or main content
            content_div = soup.find('div', {'id': 'mw-content-text'}) or soup.body
            if content_div:
                for element in content_div.find_all('p', recursive=True):
                    text = element.get_text().strip()
                    if text:
                        paragraphs.append(text)
                        p_count += 1
                    if p_count >= num_paragraphs:
                        break
            return " ".join(paragraphs)
            
        return soup.get_text(separator=' ', strip=True)
