#PURPOSE: extract twitter15 and twitter16 data from the original dataset files and write to json files
#INPUT: original twitter15 and twitter16 dataset files (label.txt, source_tweets.txt, tree/*.txt)
#OUTPUT: json files for train, val, and test data
#NOTES: 
# import build_data() from twitter_extract.py 
# build_data() function call builds the data and writes to json files

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATH_TWITTER15 = ROOT / "data/rumor_detection_acl2017/twitter15"
PATH_TWITTER16 = ROOT / "data/rumor_detection_acl2017/twitter16"

# get the tree for a given id and path
#get_tree(80080680482123777, "twitter15") -> 80080680482123777.txt tree
def get_tree(id, path):
    with open(f"{path}/tree/{id}.txt", "r") as f:
        return f.read()
   

#read each line of the label.txt file and return the tweet_id and label
def get_id_label(line):
    label, tweet_id = line.strip().split(":", 1)
    return tweet_id, label

# take id and obtain the text of the tweet
def get_texts(id, path):
    with open(f"{path}/source_tweets.txt", "r") as f:
        for line in f:
            tweet_id, text = line.strip().split("\t", 1)
            if tweet_id == id:
                return clean_tweet(text)
    return "Tweet not found"

def clean_tweet(text):
    # normalize raw tweet text for vectorizing (kept emojies, might have to change)
    words = []
    for word in text.lower().split():
        if word.startswith("url") or word.startswith("HTTP"):
            words.append("URL")
        elif word.startswith("@"):
    
            words.append("USER")
        elif word == "&amp;":
            words.append("and")
        else:
            words.append(word)
    return " ".join(words)

#need to seperate the data by type and by time (relative).
def temporal_split(data, train_frac=0.8, val_frac=0.1):
    # stratified temporal split: within each label, oldest train_frac -> train,
    # next val_frac -> val, remainder -> test
    labels = ["true", "false", "non-rumor", "unverified"]
    train, val, test = [], [], []

    for label in labels:
        #list of matching rows for this label
        match = [row for row in data if row["label"] == label]
        #sort by id (chronologically)
        match.sort(key=lambda r: int(r["id"]))
        #split by time by calculating cutoffs
        n = len(match)
        cut_train = int(train_frac * n)
        cut_val = cut_train + int(val_frac * n)
        
        #add to split arrays
        train.extend(match[:cut_train])
        val.extend(match[cut_train:cut_val])
        test.extend(match[cut_val:])

    return train, val, test

def write_split(output_dir, year, train, val, test):
    for split, split_file in [(train, "train"), (val, "val"), (test, "test")]:
                with open(f"{output_dir}/{split_file}_{year}.json", "w") as f:
                    json.dump(split, f, indent=2, ensure_ascii=False)


def build_data():
    data_15 = []
    data_16 = []
    output_dir = ROOT / "data/text_data"
    # create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    #pull data from dataset files
    with open(f"{PATH_TWITTER15}/label.txt", "r") as f:
        for line in f:
            tweet_id, label = get_id_label(line)
            text = get_texts(tweet_id, PATH_TWITTER15)
            #tree = get_tree(tweet_id, PATH_TWITTER15)
            data_15.append({"id": tweet_id, "label": label, "text": text})
    with open(f"{PATH_TWITTER16}/label.txt", "r") as f:
        for line in f:
            tweet_id, label = get_id_label(line)
            text = get_texts(tweet_id, PATH_TWITTER16)
            #tree = get_tree(tweet_id, PATH_TWITTER16)
            data_16.append({"id": tweet_id, "label": label, "text": text})

    # stratified temporal 80/10/10 split per year
    for data, year in [(data_15, "15"), (data_16, "16")]:
        train, val, test = temporal_split(data)
        write_split(output_dir, year, train, val, test)


if __name__ == "__main__":
    build_data()