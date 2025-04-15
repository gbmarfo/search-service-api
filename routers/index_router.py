from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from database import schemas, models, search_crud
from database.database import SessionLocal, engine, get_db
from sqlalchemy.orm import Session
from services.text_search import TextSearch
from services.vector_search import VectorSearch

router = APIRouter()

@router.post("/create", summary="Create a new search index",
            description="Create a new search index in the database.")
def create_index_from_db(search_index: schemas.SearchIndexCreate, db: Session = Depends(get_db)):
    """
    Create a new search index in the database.
    """
    search_index = search_crud.create_search_index(db=db, search_index=search_index)
    search_index_id = search_index.global_id

    # create the index using the TextSearch class
    text_search = TextSearch(
        index_file=search_index_id
    )

    # get searchable columns from the search index
    searchable_columns = search_index.text_columns.split(",")

    # get the table name from the search index
    table_name = search_index.table_name

    # get the id column from the search index
    id_column = search_index.id_col

    # get the schema name from the search index
    schema_name = search_index.schema_name

    # extract data from the database and add it to the index
    text_search.add_data(
        db=db,
        table_name=table_name,
        text_columns=searchable_columns,
        id_column=id_column,
        schema=schema_name
    )

    # create the vector index
    vector_search = VectorSearch(
        file_id=search_index_id,
    )
    
    vector_search.create_index(
        data=[{id_column: row[0], "concatenated_text": row[1]} for row in text_search.data],
        text_column="concatenated_text",
        id_column=id_column
    )

    return {
        "message": "Search index created successfully",
        "id": search_index.global_id
    }
