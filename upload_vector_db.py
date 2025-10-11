from sentence_transformers import SentenceTransformer
import chromadb

def uploadToVectorDB(df):
    database = chromadb.PersistentClient(path="./youtube_comment_database")
    collection = database.get_or_create_collection(name="youtube_comments")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    comments = df["comment"].to_list()
    replies = df["reply"].to_list()
    replies_dict = [{"reply": reply} for reply in replies]
    encoded_comments = embedding_model.encode(comments)

    collection.add(
        ids=[str(i) for i in range(len(comments))],
        embeddings=encoded_comments,
        documents=comments,
        metadatas=replies_dict
    )
