from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from database import schemas, models, search_crud
from database.database import SessionLocal, engine, get_db
from sqlalchemy.orm import Session
from services.text_search import TextSearch
from services.vector_search import VectorSearch

models.Base.metadata.create_all(bind=engine)

router = APIRouter()

IDX_SRC = "data/7a9a67d9-9062-47a1-a8d1-d72ba2913523_text.pkl"

@router.get("/{index_id}/ranked_naive", summary="Ranked Search using TF-IDF",
            description="Search for documents based on the query and search type.")
async def ranked_search(query: str, index_id: str):
    try:
        selected_documents = TextSearch(
            index_file=index_id
        ).ranked_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{index_id}/full_text", summary="Ranked Search using BM25",
            description="Search for documents based on the query and search type.")
async def ranked_search_bm25(query: str, index_id: str):
    try:
        selected_documents = TextSearch(
            index_file=index_id
        ).bm25_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{index_id}/boolean_ranked", summary="Ranked Search with Boolean Search First",
            description="Perform a ranked search (TF-IDF) on documents after performing a boolean search.")
async def boolean_ranked_search(query: str, index_id: str):
    try:
        selected_documents = TextSearch(
            index_file=index_id
        ).boolean_ranked_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{index_id}/exact", summary="Exact Search with strict boolean search",
            description="Search for documents based on keywords.")
async def keyword_search(query: str, index_id: str):
    """
    Search for documents based on keywords.

    - **keywords**: A list of keywords to search for.
    """
    try:
        selected_documents = TextSearch(
            index_file=index_id
        ).boolean_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/{index_id}/fuzzy", summary="Fuzzy Search",
            description="Perform a fuzzy search on documents.")
async def fuzzy_search(query: str, index_id: str):
    """
    Perform a fuzzy search on documents.

    - **query**: The fuzzy search query string.
    """
    try:
        selected_documents = TextSearch(
            index_file=index_id
        ).fuzzy_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{index_id}/similarity", summary="Similarity Search",
            description="Perform a similarity search on documents.")
async def similarity_search(query: str, index_id: str):
    """
    Perform a similarity search on documents.

    - **query**: The similarity search query string.
    """
    try:
        # Perform similarity search using the VectorSearch class
        vSearch = VectorSearch(
            file_id=index_id
        )
        
        selected_documents = vSearch.similarity_search_lite(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{index_id}/exact_similarity", summary="Exact Similarity Search",
            description="Perform an exact similarity search on documents.")
async def exact_similarity_search(query: str, index_id: str):
    """
    Perform an exact similarity search on documents.

    - **query**: The exact similarity search query string.
    """
    try:
        # Perform exact similarity search using the VectorSearch class
        selected_documents = VectorSearch(file_id=index_id).boolean_semantic_search(query)

        return {
            "results": selected_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))