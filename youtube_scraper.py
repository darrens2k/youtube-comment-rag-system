from config import youtube

def getCommentsPerVideo(video, output, comments_to_view=2000):
    nextPageToken = None

    for i in range(comments_to_view // 100):
        apiCall = youtube.commentThreads().list(
            part=["snippet", "replies"],
            videoId=video,
            maxResults=100,
            order="relevance",
            pageToken=nextPageToken
        ).execute()

        for j in range(len(apiCall["items"])):
            textOutput = apiCall["items"][j]["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            replyCount = apiCall["items"][j]["snippet"]["totalReplyCount"]

            if replyCount > 0:
                replies = apiCall["items"][j]["replies"]["comments"]
                likes = [reply["snippet"]["likeCount"] for reply in replies]
                maxIndex = likes.index(max(likes))
                mostLikedReplyText = replies[maxIndex]["snippet"]["textDisplay"]
                output.append({"comment": textOutput, "reply": mostLikedReplyText})

            next_page_token = apiCall.get("nextPageToken")
            if not next_page_token:
                break

    return output


def fetchYouTubeComments(videos):
    output = []
    for video in videos:
        output = getCommentsPerVideo(video, output, comments_to_view=5000)
    return output
