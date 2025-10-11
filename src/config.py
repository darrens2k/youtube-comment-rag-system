import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import chromadb
from sentence_transformers import SentenceTransformer

"""
Configuration Script

This script sets up the environment and core components required for collecting 
and embedding YouTube comments in a local Chroma vector database. It performs the 
following tasks:

1. Loads environment variables from a .env file to access API credentials.
2. Initializes a YouTube Data API client using the `googleapiclient` library.
3. Loads a lightweight sentence embedding model (`all-MiniLM-L6-v2`) from Hugging Face 
   via the `sentence-transformers` library to convert text into embeddings.
4. Creates or retrieves a persistent Chroma database collection to store comment 
   embeddings for later retrieval and semantic search.

Dependencies:
- python-dotenv
- google-api-python-client
- chromadb
- sentence-transformers

Artifacts:
- A persistent Chroma database located at ./youtube_comment_database
- A collection named "youtube_comments" containing embedded YouTube comment data
"""


# load env file
load_dotenv()

# initialize youtube client
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# import sentence embedder from huggingface
# using all-MiniLM-L6-v2 since llama2:7B doesn't have an encoder and this one is light enough for me to run
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# initialize an instance of the database
# persist ensures that the database is saved to the computer so I can reference it in other scripts
database = chromadb.PersistentClient(path="./youtube_comment_database")
# create a collection (group of documents and their embeddings)
collection = database.get_or_create_collection(name="youtube_comments")
