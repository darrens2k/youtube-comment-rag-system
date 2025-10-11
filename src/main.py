from youtube_scraper import fetchYouTubeComments
from data_cleaning import cleanData
from upload_vector_db import uploadToVectorDB
from semantic_search import getSemanticSearchResults
from llm_interface import callLLM

"""
YouTube RAG Pipeline Script

This script demonstrates a complete Retrieval-Augmented Generation (RAG) workflow 
for generating YouTube comment replies using a local LLM. It connects all stages 
of the pipeline: data collection, cleaning, embedding, semantic search, and 
LLM-based response generation.

Pipeline Steps:
---------------
1. Fetch YouTube comments and their most liked replies for a list of video IDs.
2. Clean the collected comment–reply data (remove duplicates, links, HTML tags, emojis, etc.).
3. Upload the cleaned comments and replies into a persistent Chroma vector database 
   with embeddings for semantic search.
4. Perform semantic search to retrieve the top-N most relevant comment–reply pairs 
   based on a user-provided prompt.
5. Generate a context-aware reply using a local LLM (LLaMA 2:7B) through the Ollama API.

Usage:
------
- Define a user prompt and a list of YouTube video IDs.
- Run the pipeline to fetch, clean, store, retrieve, and respond to comments.
- The final LLM-generated response is printed to the console.

Dependencies:
-------------
- src.youtube_scraper.fetchYouTubeComments
- src.data_cleaning.cleanData
- src.upload_vector_db.uploadToVectorDB
- src.semantic_search.getSemanticSearchResults
- src.llm_interface.callLLM
- ChromaDB, SentenceTransformers, Ollama

Example:
--------
prompt = "I love the Civic Si"
video_ids = ["RrZSuz-e9NY", "gGmdz9tA1Y8", "rGMWjQX5LG8", ...]
output = fetchYouTubeComments(video_ids)
output = cleanData(output)
uploadToVectorDB(output)
comments, replies = getSemanticSearchResults(prompt)
response = callLLM(comments, replies, prompt)
print(response)
"""

# Define user input and video IDs
prompt = "I love the Civic Si"
video_ids = [
    "RrZSuz-e9NY", "gGmdz9tA1Y8", "rGMWjQX5LG8", "nafje4-tv-w", "JcvQC0eYwJA",
    "DkW0Fr5KGf0", "ezgZCGM-_bg", "f6WAqT6073w", "F8IEZHeycS4", "VY91tZ3m-qU",
    "wczsTzaIgcE", "1h4MB5K_w1I", "JOp1xZrbuQM", "_e5mIqafwMA", "evTLpZZp6R0", "pUTj3C-Owx8"
]

# 1. Fetch comments
output = fetchYouTubeComments(video_ids)

# 2. Clean data
output = cleanData(output)

# 3. Upload to vector DB
uploadToVectorDB(output)

# 4. Perform semantic search
comments, replies = getSemanticSearchResults(prompt)

# 5. Generate LLM response
response = callLLM(comments, replies, prompt)

print(response)
