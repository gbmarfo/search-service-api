from http.client import HTTPException
from text_search import TextSearch
from vector_search import VectorSearch
import logging
import pickle

class HybridSearch:
    def __init__(self, index_file=None):
        self.index_file = index_file
        self.inverted_index = {}
        self.documents = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.load_text_index()

    def search(self, query, top_k=5):
        """
        Perform a hybrid search using both vector embeddings and keyword search.

        - **query**: The query string to search for.
        - **top_k**: The number of top results to return.
        """
        try:
            vector_search = VectorSearch(self.index_file)
            vector_results = vector_search.search(query, top_k)

            text_search = TextSearch(self.index_file)
            text_results = text_search.boolean_search(query)

            # Combine the results from both searches
            combined_results = {
                "vector_results": vector_results,
                "text_results": text_results
            }

            return combined_results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    
    def load_text_index(self):
        """
        Load the inverted index from the file.
        """
        try:
            with open(self.index_file, 'r') as f:
                data = pickle.load(f)
                self.inverted_index = data.get('inverted_index', {})
                self.documents = data.get('documents', {})
                self.doc_lengths = data.get('doc_lengths', {})
                logging.info(f"Loaded index from {self.index_file}.")
                
        except EOFError:
            self.inverted_index = {}
            logging.error(f"Index file {self.index_file} is empty or corrupted.")
        except FileNotFoundError:
            self.inverted_index = {}
            logging.error(f"Index file {self.index_file} not found.")
        

    