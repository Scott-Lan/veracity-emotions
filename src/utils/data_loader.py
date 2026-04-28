#PURPOSE: load a split of the data at a time for classifier
#INPUT: split name
#OUTPUT: array of text and corresponding labels
    #([text1, ..., textn], [label1, ..., labeln])
    # can be adjusted to add emotion tags

import json
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
TFIDF_DIM = 300

PATH_TWITTER15 = ROOT / "data/rumor_detection_acl2017/twitter15"
PATH_TWITTER16 = ROOT / "data/rumor_detection_acl2017/twitter16"
TEXT_DATA = ROOT / "data/text_data"

#return text and label arrays from a given split
def load_split(split_name):
    text = []
    label = []
    for year in ["15", "16"]:
        json_path = TEXT_DATA / f"{split_name}_{year}.json"
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        #load up all the text and labels into arrays from each json file
        for row in data:
            text.append(row["text"])
            label.append(row["label"])
    return (text, label)

#used to load a split ending in "_combined.json"
def load_split_combined(split_name):
    text = []
    label = []
    with open(TEXT_DATA / f"{split_name}_combined.json", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        text.append(row["text"])
        label.append(row["label"])
    return (text, label)

#get rows as a list instead of individual values. used for tfidf features and gnn model
def load_split_rows(split_name):
    rows = []
    for year in ["15", "16"]:
        json_path = TEXT_DATA / f"{split_name}_{year}.json"
        with open(json_path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows

#fetch tree data based on id
def get_tree(id, path_dir):
    tree_path = f"{path_dir}/tree/{id}.txt"
    with open(tree_path, encoding="utf-8") as f:
        tree = f.read()
    return tree

#get features for GNN model using tfidf vectorizer.
def tfidf_features():
    cache_path = ROOT / "data/tfidf_features.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    train_rows = load_split_rows("train")
    val_rows = load_split_rows("val")
    test_rows = load_split_rows("test")

    vectorizer = TfidfVectorizer(max_features=TFIDF_DIM, sublinear_tf=True, stop_words="english", min_df=2)
    vectorizer.fit([r["text"] for r in train_rows])

    features = {}
    for rows in (train_rows, val_rows, test_rows):
        ids = [r["id"] for r in rows]
        mat = vectorizer.transform([r["text"] for r in rows]).toarray()
        tensor = torch.from_numpy(mat).float()
        for tid, vec in zip(ids, tensor):
            features[tid] = vec.contiguous()

    torch.save(features, cache_path)
    return features

if __name__ == "__main__":
    print("PATH_TWITTER15:", PATH_TWITTER15)
    print("PATH_TWITTER16:", PATH_TWITTER16)

    #load_split_with_trees("train")
    #print(load_split("train"))
    #print(load_split("val"))
    #print(load_split("test"))