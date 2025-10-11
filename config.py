import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# YouTube API client
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# Embedding model (used globally)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent Chroma database
database = chromadb.PersistentClient(path="./youtube_comment_database")
collection = database.get_or_create_collection(name="youtube_comments")
