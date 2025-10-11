# import libraries
import googleapiclient
from googleapiclient.discovery import build
import pandas as pd
import re
import ollama
from langchain.schema import HumanMessage
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# possible global variables

# load env file
load_dotenv()

# import sentence embedder from huggingface
# using all-MiniLM-L6-v2 since llama2:7B doesn't have an encoder and this one is light enough for me to run
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# initialize an instance of the database
# persist ensures that the database is saved to the computer so I can reference it in other scripts
database = chromadb.PersistentClient(path="./youtube_comment_database")
# create a collection (group of documents and their embeddings)
collection = database.get_or_create_collection(name="youtube_comments")

# user prompt
prompt = "I love the Civic Si"

# initialize youtube client
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# video id's for videos about the civic si
video_id = ["RrZSuz-e9NY", "gGmdz9tA1Y8", "rGMWjQX5LG8", "nafje4-tv-w", "JcvQC0eYwJA", "DkW0Fr5KGf0", "ezgZCGM-_bg", "f6WAqT6073w", "F8IEZHeycS4", "VY91tZ3m-qU", "wczsTzaIgcE", "1h4MB5K_w1I", "JOp1xZrbuQM", "_e5mIqafwMA", "evTLpZZp6R0", "pUTj3C-Owx8"]
# create a list dictionary that will store the output
output = []

# build function to call youtube api to go through as many pages of comments as possible per youtube video
# video == id of particular video
# output == output list
# comments_to_view == number of comments model will go through per video (not all will have replies)
def getCommentsPerVideo(video, output, comments_to_view=2000):
    # parameter to specify the next page of comments, must be None for the first page
    # each subsequent api call provides to value to load the next page
    nextPageToken=None

    # lets say we want no more than 1000 comments per video, each api call can get up to 100 comments
    # so we will call the api in a 10 iteration loop and exit early if the nextPageToken is not given (means we already went through all comments)
    # realistically most comments won't have a reply, so by iterating through 2000 we will get between 500 and 1000 usable data entries per video
    for i in range(comments_to_view//100):

        apiCall = youtube.commentThreads().list(part=["snippet","replies"], videoId=video, maxResults=100, order="relevance", pageToken=nextPageToken).execute()

        # iterate through the API response to save all comment-reply pairs (ignore comments that don't have any replies)
        # iterate through the comments the api returned
        for j in range(len(apiCall["items"])):
            
            # get comment text
            textOutput = apiCall["items"][j]["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            
            # get count of replies
            replyCount = apiCall["items"][j]["snippet"]["totalReplyCount"]
            
            if replyCount > 0:
                
                # get list of all the returned replies (api usually returns 5 replies)
                replies = apiCall["items"][j]["replies"]["comments"]

                # get the likes per reply
                likes = []
                for reply in replies:
                    likes.append(reply["snippet"]["likeCount"])
                    
                # get index of comment with most likes
                maxIndex = likes.index(max(likes))
                # get reply with most likes
                mostLikedReplyText = replies[maxIndex]["snippet"]["textDisplay"]

                # save comment text and most liked reply text to output list dictionary
                output.append({"comment":textOutput, "reply":mostLikedReplyText})
            
            # update next page token
            next_page_token = apiCall.get("nextPageToken")
            if not next_page_token:
                break
    
    return output


# build function to get comments for all youtube videos specified
# videos == list of video IDs
def fetchYouTubeComments(videos):

    # define output list
    output = []

    # iterate through list of video IDs
    for video in videos:
        
        output = getCommentsPerVideo(video, output, comments_to_view=5000)

    return output

output = fetchYouTubeComments(video_id)

# function for data cleaning
# accepts the output of the fetchYouTubeComments function

def cleanData(output):

    # convert list of dictionaries to dataframe
    output = pd.DataFrame(output)

    # drop any duplicate entries we have
    output.drop_duplicates(inplace=True)

    # in order: remove links, html tags, special characters and punctuation, emojis
    output['comment'] = output['comment'].astype(str) \
        .str.replace(r"http\S+|www\S+|https\S+", "", regex=True) \
        .str.replace(r"<.*?>", "", regex=True) \
        .str.replace(r"[^\w\s]", "", regex=True) \
        .str.replace(r"[\U00010000-\U0010ffff]", "", regex=True)

    output['reply'] = output['reply'].astype(str) \
        .str.replace(r"http\S+|www\S+|https\S+", "", regex=True) \
        .str.replace(r"<.*?>", "", regex=True) \
        .str.replace(r"[^\w\s]", "", regex=True) \
        .str.replace(r"[\U00010000-\U0010ffff]", "", regex=True)
    
    return output

output = cleanData(output)

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

uploadToVectorDB(output)

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

comments, replies = getSemanticSearchResults(prompt)

# function to call LLM and get results
# also perform pre-prompting / prompt engineering
def callLLM(comments, replies):

    # write out the pre-prompt text for the llm
    pre_prompt = """
    You are a bot whose purpose is to reply to YouTube comments about the Honda Civic Si.
    You will be provided with examples of comment and reply pairings. Do not simply regurgitate these replies, use them as inspiration.
    Do not include special characters or usernames in your response. Do not prompt user for more information.
    Use only the information provided in the sample comment reply pairs to create your response.
    Respond by restating the comment and then your reply.
    Here are your samples:
    """

    # add in the comment and reply pairs
    for i in range(len(comments)):
        pre_prompt += "Comment " + str(i + 1) + ": " + comments[i] + "\nReply " + str(i + 1) + ": " + replies[i][0] + "\n\n"

    # add in the comment from user for llm to reply to
    pre_prompt += "\nDraft a reply to this comment: \n" + prompt

    # call llm and get response
    # using generate and not chat because don't need to have a conversation, just a response to the initial prompt
    response = ollama.generate(model='llama2:7B', prompt=pre_prompt)

    return response["response"]

print(callLLM(comments, replies))