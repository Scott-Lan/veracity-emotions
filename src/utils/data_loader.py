#PURPOSE: load a split of the data at a time for classifier
#INPUT: split name
#OUTPUT: array of text and corresponding labels
    #([text1, ..., textn], [label1, ..., labeln])
    # can be adjusted to add emotion tags

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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

#the same as load_split(), but with trees.
def load_split_with_trees(split_name):
    #load the data with the trees
    text = []
    label = []
    id = []
    tree = []
    for path_dir, year in [(PATH_TWITTER15, "15"), (PATH_TWITTER16, "16")]:
        json_path = f"{path_dir}/{split_name}_{year}.json"
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            text.append(row["text"])
            label.append(row["label"])
            id.append(row["id"])
            tree.append(get_tree(id, path_dir))
        
 
    return (text, label, tree)

#fetch tree data based on id
def get_tree(id, path_dir):
    tree_path = f"{path_dir}/tree/{id}.txt"
    with open(tree_path, encoding="utf-8") as f:
        tree = f.read()
    return tree

if __name__ == "__main__":
    print("PATH_TWITTER15:", PATH_TWITTER15)
    print("PATH_TWITTER16:", PATH_TWITTER16)

    #load_split_with_trees("train")
    #print(load_split("train"))
    #print(load_split("val"))
    #print(load_split("test"))