from config import collection, embedding_model

def getSemanticSearchResults(user_prompt):
    promptEncoded = embedding_model.encode(user_prompt)
    semantic_search_results = collection.query(query_embeddings=promptEncoded, n_results=5)

    comments = semantic_search_results['documents'][0]
    replies = semantic_search_results['metadatas'][0]
    replies = [list(reply.values()) for reply in replies]

    return comments, replies
