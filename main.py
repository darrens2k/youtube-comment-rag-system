from youtube_scraper import fetchYouTubeComments
from data_cleaning import cleanData
from upload_vector_db import uploadToVectorDB
from semantic_search import getSemanticSearchResults
from llm_interface import callLLM

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
