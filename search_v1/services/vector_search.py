import logging
import os
import numpy as np
import faiss
import torch
import pickle
import csv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from services.text_search import TextSearch

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class VectorSearch:
    def __init__(self, file_id:str =None):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(os.getenv('BASE_EMBEDDING_MODEL'), 
                                                   device=self.torch_device)
        self.index = None
        self.documents = {}
        self.doc_embeddings = None
        self.embedding_file = f"data/{file_id}_emb.npy" 
        self.doc_file = f"data/{file_id}_text.txt"
        self.vector_index_file = f"data/{file_id}_faiss.index"
        self.file_id = file_id
        
        self.load_index(self.vector_index_file, self.doc_file, self.embedding_file)
    
    def get_embeddings(self, texts):
        return self.embedding_model.encode(texts, convert_to_tensor=True)

    def create_index(self, data, text_column, id_column):
        self.documents = {row[id_column]: row[text_column] for row in data}
        self.doc_embeddings = self.get_embeddings([row[text_column] for row in data])
        doc_ids = [row[id_column] for row in data]

        # create vector embeddings and their IDs
        text_vectors = np.array(self.doc_embeddings).astype('float32')
        text_ids = np.array([hash(str(doc_id)) for doc_id in doc_ids]).astype('int')

        faiss.normalize_L2(text_vectors)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(text_vectors.shape[1]))
        self.index.add_with_ids(text_vectors, text_ids)

        # save the index to a file
        self.save_index(self.vector_index_file, self.doc_file, self.embedding_file)


    def save_index(self, vector_index_path, data_path, embedding_path=None):
        # Check if the index is created before saving
        if self.index is None:
            raise ValueError("Index has not been created. Call create_index first.")
        
        # save the vector index
        faiss.write_index(self.index, vector_index_path)

        with open(data_path, 'w') as f:
            for doc_id, text in self.documents.items():
                doc_id = "NO_ID" if not doc_id else doc_id.replace('\r', '').replace('\n', '')
                text = "NO DESCRIPTION" if not text else text.replace('\r', '').replace('\n', '')
                f.write(f"{doc_id}, {text}\n")

        if embedding_path:
            np.save(embedding_path, self.doc_embeddings)


    def load_index(self, vector_index_path, data_path, embedding_path=None):
        if os.path.exists(vector_index_path):
            self.index = faiss.read_index(vector_index_path)
            
            with open(data_path, 'r') as f:
                for line in f:
                    doc_id, text = line.strip().split(',', 1)
                    self.documents[doc_id] = text
        
        if os.path.exists(embedding_path):
            self.doc_embeddings = np.load(embedding_path, allow_pickle=True)
        

    def add_documents(self, new_data, text_column, id_column):
        if self.index is None:
            raise ValueError("Index has not been created. Call create_index first.")

        new_doc_embeddings = self.get_embeddings([row[text_column] for row in new_data])
        new_doc_ids = [row[id_column] for row in new_data]

        new_text_vectors = np.array(new_doc_embeddings).astype('float32')
        new_text_ids = np.array(new_doc_ids).astype('int')

        faiss.normalize_L2(new_text_vectors)
        self.index.add_with_ids(new_text_vectors, new_text_ids)

        # Append new data to the existing data
        for row in new_data:
            self.documents[row[id_column]] = row[text_column]
    

    def similarity_search_lite(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been created. Call create_index first.")
        
        query_embedding = self.get_embeddings([query])
        doc_embeddings = self.doc_embeddings

        doc_scores = {doc_id: cos_sim(query_embedding, doc_embedding).flatten().tolist()
                      for doc_id, doc_embedding in zip(self.documents.keys(), doc_embeddings)}
        
        sorted_doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_doc_scores[:top_k]

        return [{'text': self.documents[doc_id], 'score': score, 'id': doc_id} for doc_id, score in top_results]
        

    def boolean_semantic_search(self, query):
        """
        Perform a boolean semantic search on the provided query.
        This method first performs a boolean search using the index and then
        refines the results by calculating the cosine similarity between the
        query embedding and the embeddings of the boolean search results.
        Args:
            query (str): The search query string.
        Returns:
            list: A list of dictionaries containing the top search results. Each
                  dictionary includes the following keys:
                  - 'text': The text of the document.
                  - 'score': The cosine similarity score of the document.
                  - 'id': The ID of the document.
        Raises:
            ValueError: If the index has not been created or if the query is empty.
        """

        if self.index is None:
            raise ValueError("Index has not been created.")
        
        query_words = query.split()
        if not query_words:
            return []
        
        try:
            # Perform a boolean search using the index
            text_search = TextSearch(
                index_file=self.file_id
            )
            boolean_results = text_search.boolean_search(query)

            # Get the embeddings of the boolean search results
            boolean_doc_ids = [doc['id'] for doc in boolean_results]

            boolean_doc_texts = [self.documents[doc_id] for doc_id in boolean_doc_ids]

            boolean_doc_embeddings = self.get_embeddings(boolean_doc_texts)
            boolean_doc_embeddings = np.array(boolean_doc_embeddings).astype('float32')

            query_embedding = self.get_embeddings([query])

            doc_scores = {doc_id: cos_sim(query_embedding, doc_embedding).flatten().tolist()
                          for doc_id, doc_embedding in zip(boolean_doc_ids, boolean_doc_embeddings)}
            
           
            sorted_doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_results = sorted_doc_scores[:5]  # Adjust the number of results as needed

            return [{'text': self.documents[doc_id], 'score': score, 'id': doc_id} for doc_id, score in top_results]
        
        except KeyError as e:
            logging.error(f"KeyError encountered: {e}")
            raise ValueError(f"Document ID not found in the documents dictionary: {e}")
        except Exception as e:
            logging.error(f"An error occurred during boolean semantic search: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}")