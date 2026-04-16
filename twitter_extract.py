import json
import random

PATH_TWITTER15 = "data/rumor_detection_acl2017/twitter15"
PATH_TWITTER16 = "data/rumor_detection_acl2017/twitter16"
random.seed(256)

# get the tree for a given id and path
#get_tree(80080680482123777, "twitter15") -> 80080680482123777.txt tree
# def get_tree(id, path):
#     with open(f"{path}/tree/{id}.txt", "r") as f:
#         return f.read()
   

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

def write_split(data, out_dir, year, train, val, test):
    #slice data from pairs
    train_data = data[train[0] : train[1]]
    val_data = data[val[0] : val[1]]
    test_data = data[test[0] : test[1]]

    #write to file for train, val, and test
    with open(f"{out_dir}/train_{year}.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(f"{out_dir}/val_{year}.json", "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(f"{out_dir}/test_{year}.json", "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

def build_data():
    data_15 = []
    data_16 = []

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

    # 80/10/10 split after both lists are built (not inside the twitter16 file read)
    for data, out_dir, year in [(data_15, PATH_TWITTER15, "15"), (data_16, PATH_TWITTER16, "16")]:
        random.shuffle(data)
        #calc lengths of train, val, and test
        n = len(data)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        #store ranges as pairs
        i_train = [0, n_train]
        i_val = [n_train, n_train + n_val]
        i_test = [n_train + n_val, n]
        #send to write function
        write_split(data, out_dir, year, i_train, i_val, i_test)


def main():
    build_data()
    

if __name__ == "__main__":
    main()