import ollama

def callLLM(comments, replies, prompt):
    pre_prompt = """
    Your purpose is to reply to YouTube comments about the Honda Civic Si.
    You will be provided with examples of comment and reply pairings. Do not regurgitate these replies, use them as inspiration.
    Do not prompt user for more information.
    Use only the information provided in the sample comment reply pairs to create your response.
    Respond by restating the comment and then your reply.
    Here are your samples:
    """

    for i in range(len(comments)):
        pre_prompt += f"Comment {i+1}: {comments[i]}\nReply {i+1}: {replies[i][0]}\n\n"

    pre_prompt += "\nDraft a reply to this comment: \n" + prompt

    response = ollama.generate(model='llama2:7B', prompt=pre_prompt)
    return response["response"]
