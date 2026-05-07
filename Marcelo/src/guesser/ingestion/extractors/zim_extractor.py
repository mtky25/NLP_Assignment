from libzim.reader import Archive
from llama_index.core import Document 
import bs4
import pprint
from src.guesser.ingestion.extractors.extractor import BaseExtractor


class ZimExtractor(BaseExtractor):
    def __init__(self, file_name):
        self.archive = Archive(file_name)

    def count_archives(self):
        pprint.pprint(f"Total of Articles: {self.archive.entry_count}")
        
    def extract(self, limit=None, starting_id=0):
        count = 0
        total_articles = self.archive.entry_count

        for i in range(starting_id, total_articles):
            entry = self.archive._get_entry_by_id(i)
            
            if entry.get_item().mimetype == "text/html":
                try:
                    html_content = entry.get_item().content.tobytes().decode('utf-8')
                    title = entry.title
                    
                    clean_text = self._clean_html(html_content)
                    if not clean_text.strip():
                        continue
                    
                    yield Document(
                        text=clean_text,
                        metadata={
                            "title": title,
                            "path": entry.path,
                            "font": "Kiwix_ZIM",
                            "content": "article_wikipedia"
                        }
                    )
                    
                    count += 1
                    if limit and count >= limit:
                        break
                        
                except Exception as e:
                    print(f"Error Processing {entry.title}: {e}")
                
    def _clean_html(self, html):

        soup = bs4.BeautifulSoup(html, 'html.parser') 
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'table']):
            tag.decompose()
            
        return soup.get_text(separator=' ', strip=True)