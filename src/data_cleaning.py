import pandas as pd

"""
Data Cleaning Script

This module provides a data-cleaning function to preprocess YouTube comment data 
retrieved from the YouTube Data API. It converts raw JSON-style output (a list of 
dictionaries) into a clean, structured pandas DataFrame suitable for downstream 
embedding and analysis.

Function:
----------
cleanData(output: list[dict]) -> pandas.DataFrame
    - Converts the input list of dictionaries into a DataFrame.
    - Removes duplicate entries.
    - Cleans the 'comment' and 'reply' text fields by:
        * Removing URLs (http/https/www patterns)
        * Stripping HTML tags
        * Removing special characters and punctuation
        * Removing emojis and other non-standard Unicode characters

Returns:
--------
A cleaned pandas DataFrame containing processed 'comment' and 'reply' text columns.

Dependencies:
-------------
- pandas
"""


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