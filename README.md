# YouTube RAG Project: Comment Reply Generation with LLM

This project showcases a Retrieval-Augmented Generation (RAG) workflow for generating replies to YouTube comments using a local LLM. The pipeline collects and cleans comments, stores embeddings in a vector database, retrieves the most relevant comments, and produces context-aware responses. For this project, the focus is limited to comments on Honda Civic Si videos.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Steps](#pipeline-steps)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Notes](#notes)  

---

## Project Overview

The goal of this project is to build an intelligent system that can **reply to YouTube comments** in a contextually relevant way, leveraging semantic search and a local LLM. The workflow is designed to be **end-to-end**, from data collection to response generation, and demonstrates key concepts in:

- Natural Language Processing (NLP)  
- Sentence embeddings and semantic search  
- Retrieval-Augmented Generation (RAG)  
- Large Language Model (LLM) prompt engineering  

---

## Pipeline Steps

1. **Data Collection**  
   - Fetch comments and their most liked replies from a list of YouTube videos using the YouTube Data API.

2. **Data Cleaning**  
   - Remove duplicates, links, HTML tags, special characters, punctuation, and emojis.
   - Converts data into a structured format suitable for embedding.

3. **Vector Database Upload**  
   - Embed comments using `all-MiniLM-L6-v2` from Hugging Face.
   - Store embeddings in a **ChromaDB** vector database with replies as metadata.

4. **Semantic Search**  
   - Encode a user prompt.
   - Retrieve the top-N most relevant comment–reply pairs from the vector database.

5. **LLM Response Generation**  
   - Construct a prompt with retrieved examples for the LLaMA 2:7B model.
   - Generate a context-aware reply using **Ollama**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/youtube-rag.git
cd youtube-rag
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a .env file with your YouTube API key:

```bash
YOUTUBE_API_KEY=your_api_key_here
```

## Usage

1. Define a user prompt and a list of YouTube video IDs in main.py.

2. Run the pipeline:
```bash
python main.py
```

3. The final output is a generated LLM reply printed to the console.

## Dependencies

- google-api-python-client — YouTube Data API

- pandas — Data cleaning and manipulation

- chromadb — Vector database for embeddings

- sentence-transformers — Pretrained sentence embedding model

- ollama — Local LLM interface (LLaMA 2:7B)

- python-dotenv — Load environment variables

## File Structure

youtube-rag/  
│  
├── src/  
│   ├── youtube_scraper.py       # Fetch comments from YouTube  
│   ├── data_cleaning.py         # Clean and preprocess comments  
│   ├── upload_vector_db.py      # Embed and store comments in ChromaDB  
│   ├── semantic_search.py       # Retrieve top-N comments via semantic search  
│   ├── llm_interface.py         # Generate replies using LLaMA 2:7B  
│   ├── main.py                  # Full RAG pipeline execution  
│   └── config.py                # Initialize YouTube client, embedding model, and DB  
│  
├── Archive/                     # Contains archived files that were used during development and testing phases  
├── youtube_comment_database/    # Vector database storing the youtube comment-reply pairs  
├── requirements.txt             # Python dependencies  
└── README.md                     # Project documentation  

## Notes

- Only comments with replies are included for RAG retrieval.

- ChromaDB stores embeddings of comments and metadata of replies separately.

- Ollama must be running locally to interface with LLaMA 2:7B.

- The system is designed for English comments related to Honda Civic Si videos, but can be extended to other topics.
