from sentence_transformers import SentenceTransformer
import chromadb

"""
Vector Database Upload Script

This module defines a function for embedding and uploading cleaned YouTube comment–reply
data into a persistent Chroma vector database. It uses a pretrained sentence embedding
model to convert comments into vector representations and stores their corresponding
replies as metadata for future retrieval and semantic search.

Function:
----------
uploadToVectorDB(df: pandas.DataFrame) -> None
    - Initializes a persistent Chroma database and creates or retrieves a collection.
    - Loads the 'all-MiniLM-L6-v2' model from Hugging Face for sentence embeddings.
    - Extracts comments and replies from the input DataFrame.
    - Encodes comments into embeddings for storage.
    - Uploads embedded comments and reply metadata into the Chroma collection.

Parameters:
------------
df : pandas.DataFrame
    The cleaned DataFrame containing 'comment' and 'reply' columns, typically output 
    from the `cleanData` function.

Returns:
---------
None
    The function does not return a value. It adds data directly to the Chroma database.

Dependencies:
--------------
- pandas
- chromadb
- sentence-transformers

Artifacts:
-----------
- A persistent Chroma database stored at ./youtube_comment_database
- A collection named "youtube_comments" containing comment embeddings and reply metadata
"""

# function to add data into vector database
# accepts output from the cleanData function

def uploadToVectorDB(df):

    # initialize an instance of the database
    # persist ensures that the database is saved to the computer so I can reference it in other scripts
    database = chromadb.PersistentClient(path="./youtube_comment_database")
    # create a collection (group of documents and their embeddings)
    collection = database.get_or_create_collection(name="youtube_comments")

    # import sentence embedder from huggingface
    # using all-MiniLM-L6-v2 since llama2:7B doesn't have an encoder and this one is light enough for me to run
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # prepping data for embedding model

    # only going to embed the comments
    # users will prompt the llm with a comment and the llm will draft a reply to the comment based on the comment-reply pairs the semantic search returns
    # when the semantic search is happening, we should only be searching the comments, I want to see the comment-reply pairs for the most similar comments
    # therefore the replies will be stored as metadata in the database while the only the comments will be embedded

    # convert dataframe items to lists
    comments = df["comment"].to_list()
    replies = df["reply"].to_list()

    # convert replies to list of dictionaries so I can pass it as metadata
    replies_dict = [{"reply":reply} for reply in replies]

    # embed comments
    encoded_comments = embedding_model.encode(comments)

    # add data into database
    collection.add(
        ids=[str(i) for i in range(len(comments))],
        embeddings=encoded_comments,
        documents=comments,
        metadatas=replies_dict
    )

    return