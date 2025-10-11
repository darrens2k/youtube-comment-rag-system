from config import collection, embedding_model

"""
Semantic Search

This module provides functionality to retrieve semantically similar YouTube comment–reply
pairs from a Chroma vector database based on a user’s input prompt. It encodes the prompt
into an embedding vector, performs a similarity search, and returns the most relevant
stored comments and their corresponding replies.

Function:
----------
getSemanticSearchResults(user_prompt: str, comments_to_return: int = 5) -> tuple[list[str], list[list[str]]]
    - Encodes the user prompt using the preloaded sentence embedding model.
    - Queries the Chroma database for the top-N most similar comment–reply pairs using
      cosine similarity.
    - Returns lists of retrieved comments and their associated replies.

Parameters:
------------
user_prompt : str
    The input comment or query from the user to use for semantic search.
comments_to_return : int, optional (default = 5)
    The number of most relevant comment–reply pairs to return.

Returns:
---------
tuple[list[str], list[list[str]]]
    A tuple containing:
        1. A list of retrieved comment strings.
        2. A list of lists, where each inner list contains replies corresponding
           to the associated comment.

Dependencies:
--------------
- src.config (for the initialized `collection` and `embedding_model`)
- ChromaDB
- sentence-transformers
"""

# function to embed the prompt and perform semantic search on vector database (retrieval)
# user_prompt == prompt from user
# comments_to_return == number of comments to return from semantic search
def getSemanticSearchResults(user_prompt, comments_to_return=5):

    # encode the prompt
    promptEncoded = embedding_model.encode(user_prompt)

    # search the database using the encoded query and get 5 most related comment-reply pairs
    # distance metric is cosine similarity by default, need to set it when I set up the collection
    semantic_search_results = collection.query(query_embeddings=promptEncoded, n_results=comments_to_return)

    # get the comments 
    comments = semantic_search_results['documents'][0]

    # get replies
    replies = semantic_search_results['metadatas'][0]
    # convert the list of dictionaries to a list of the replies (replies are dictionary values)
    replies = [list(reply.values()) for reply in replies]

    return comments, replies
