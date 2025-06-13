import logging
import os
import numpy as np
import faiss
import torch
import pickle
import csv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class VectorSearch:
    def __init__(self, file_id:str =None):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(os.getenv('BASE_EMBEDDING_MODEL'), 
                                                   device=self.torch_device)
        self.index = None
        self.data = {}
        self.doc_embeddings = None
        self.embedding_file = f"data/{file_id}_emb.npy" 
        self.doc_file = f"data/{file_id}_text.txt"
        self.vector_index_file = f"data/{file_id}_faiss.index"
        
        self.load_index(self.vector_index_file, self.doc_file, self.embedding_file)
    
    def get_embeddings(self, texts):
        return self.embedding_model.encode(texts, convert_to_tensor=True)

    def create_index(self, data, text_column, id_column):
        self.data = {row[id_column]: row[text_column] for row in data}
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
            for doc_id, text in self.data.items():
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
                    self.data[doc_id] = text
        
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
            self.data[row[id_column]] = row[text_column]


    def semantic_score(self, query_embedding, doc_embedding):
        return cos_sim(query_embedding, doc_embedding).flatten().tolist()
    

    def similarity_search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been created. Call create_index first.")
        
        query_embedding = self.get_embeddings([query])
        doc_embeddings = self.doc_embeddings

        doc_scores = {doc_id: self.semantic_similarity(query_embedding, doc_embedding) 
                      for doc_id, doc_embedding in zip(self.documents.keys(), doc_embeddings)}
        # query_embedding = self.get_embeddings([query])
        # similarities, similarities_ids = self.index.search(query_embedding, top_k)
        # similarities = np.clip(similarities, 0, 1)

    
        similarities = self.semantic_score(query_embedding, doc_embeddings)
        print(similarities)

        # results = []
        # print(similarities, similarities_ids)
        # for i in range(len(similarities[0])):
        #     doc_id = similarities_ids[0][i]
        #     text = self.data[doc_id]

            # results.append({
            #     'doc_id': doc_id,
            #     'text': text,
            #     'score': similarities[0][i]
            # })
            # print(f"Doc ID: {doc_id}, Score: {similarities[0][i]}, Text: {text}")

        # return results
