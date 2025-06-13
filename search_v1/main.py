from fastapi import FastAPI
from routers import search_router, index_router, account_router

# Define the title, description, and version of the API
title = "Search Service API"
description = """
📝 This is an API for Search Service and provides endpoints for searching and indexing text data.
"""
version = "0.1"

# Create the FastAPI app instance
app = FastAPI(
    title=title,
    description=description,
    version=version
)


# add routers to the app
app.include_router(search_router.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(index_router.router, prefix="/api/v1/index", tags=["Index"])
app.include_router(account_router.router, prefix="/api/v1/account", tags=["Account"])

# # Define the root endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the PxMO Search API"}
