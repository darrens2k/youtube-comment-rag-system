import pandas as pd

def cleanData(output):
    output = pd.DataFrame(output)
    output.drop_duplicates(inplace=True)

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
