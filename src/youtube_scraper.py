from config import youtube

"""
YouTube Comment Collection Script

This module provides functionality to collect YouTube comments and their most liked
replies using the YouTube Data API. It retrieves multiple pages of comments for each
specified video and stores them as structured comment–reply pairs for downstream
cleaning, embedding, and semantic search.

Functions:
-----------
getCommentsPerVideo(video: str, output: list[dict], comments_to_view: int = 2000) -> list[dict]
    - Iteratively fetches comments and replies for a single YouTube video.
    - For each top-level comment with replies, selects the most liked reply.
    - Aggregates all comment–reply pairs into a shared output list.
    - Handles pagination to retrieve up to `comments_to_view` comments per video.

fetchYouTubeComments(videos: list[str]) -> list[dict]
    - Iterates over multiple YouTube video IDs and aggregates all comment–reply pairs.
    - Returns a consolidated list of structured comment–reply dictionaries.

Parameters:
------------
video : str
    The YouTube video ID to fetch comments from.
output : list[dict]
    A shared list to store the retrieved comment–reply dictionaries.
comments_to_view : int, optional (default = 2000)
    The maximum number of comments to process per video (in batches of 100 per API call).

videos : list of str
    A list of YouTube video IDs to collect comments from.

Returns:
---------
list[dict]
    A list of dictionaries, each containing:
        - "comment": the original comment text
        - "reply": the most liked reply text

Dependencies:
--------------
- src.config (for the initialized `youtube` client)
- google-api-python-client

Notes:
-------
- Each API call retrieves up to 100 comments and their replies.
- Only comments with at least one reply are included in the output.
- The YouTube API may return fewer replies depending on video engagement.
"""

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
