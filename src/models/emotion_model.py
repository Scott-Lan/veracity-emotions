from pathlib import Path
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import os
from transformers import pipeline

id_to_emotion = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise"
}

ekman_mapping_ids = {
    "anger": [2, 3, 10],
    "disgust": [11],
    "fear": [14, 19],
    "joy": [17, 1, 4, 13, 15, 18, 20, 23, 21, 0, 8, 5],
    "sadness": [25, 9, 12, 16, 24],
    "surprise": [26, 22, 6, 7]
}

sentiment_mapping_ids = {
    "positive": [1, 13, 17, 18, 8, 20, 5, 21, 0, 15, 23, 4],
    "negative": [14, 19, 24, 12, 9, 25, 16, 11, 2, 3, 10],
    "ambiguous": [22, 26, 7, 6]
}


def load_data(root):
    train_path = root / "data/GoEmotions/data/train.tsv"
    test_path = root / "data/GoEmotions/data/test.tsv"
    val_path = root / "data/GoEmotions/data/dev.tsv"
    train = pd.read_csv(train_path, delimiter="\t", header=None, names=["text", "labels", "id"])
    test = pd.read_csv(test_path, delimiter="\t", header=None, names=["text", "labels", "id"])
    val = pd.read_csv(val_path, delimiter="\t", header=None, names=["text", "labels", "id"])

    train['labels'] = train['labels'].apply(lambda x: [int(i) for i in str(x).split(',')])
    test['labels'] = test['labels'].apply(lambda x: [int(i) for i in str(x).split(',')])
    val['labels'] = val['labels'].apply(lambda x: [int(i) for i in str(x).split(',')])

    return train, test, val


def one_hot_encode_labels(train_df, test_df, val_df):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df['labels'])
    y_test = mlb.transform(test_df['labels'])
    y_val = mlb.transform(val_df['labels'])
    return y_train, y_test, y_val, mlb.classes_


def process_text(train_df, test_df, val_df):
    X_train_text = train_df['text']
    X_test_text = test_df['text']
    X_val_text = val_df['text']

    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    X_val = vectorizer.transform(X_val_text)

    return X_train, X_test, X_val, vectorizer


def preprocess(train_df, test_df, val_df):
    y_train, y_test, y_val, labels = one_hot_encode_labels(train_df, test_df, val_df)
    X_train, X_test, X_val, vectorizer = process_text(train_df, test_df, val_df)
    return X_train, y_train, X_test, y_test, X_val, y_val, labels, vectorizer


def lr_clf(X_train, y_train, X_val, y_val):
    best_c = 0
    best_score = 0
    best_clf = None
    for c in [1]: #[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 100, 200, 500, 1000]:
        lr = LogisticRegression(C=c, max_iter=1000, solver='liblinear', class_weight='balanced')
        clf = OneVsRestClassifier(lr)
        clf.fit(X_train, y_train)
        #evaluate on val data using macro F1 score
        score = f1_score(y_val, clf.predict(X_val), average="macro")
        #print(f"C={C}: F1={score:.4f}") #uncomment to print F1 for each C value
        if score > best_score:
            best_c, best_score, best_clf = c, score, clf
    print(f"Best C: {best_c} with F1: {best_score:.4f}")
    return best_clf


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))


def map_to_ekman(labels, mapping_ids):
    id_to_group_idx = {}
    for idx, (group, ids) in enumerate(mapping_ids.items()):
        for eid in ids:
            id_to_group_idx[eid] = idx
            
    mapped_set = set()
    for l in labels:
        if l in id_to_group_idx:
            mapped_set.add(id_to_group_idx[l])
    return list(mapped_set)


def preprocess_ekman(train_df, test_df, val_df, mapping_ids):
    for df in [train_df, test_df, val_df]:
        df['ekman_labels'] = df['labels'].apply(lambda x: map_to_ekman(x, mapping_ids))

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df['ekman_labels'])
    y_test = mlb.transform(test_df['ekman_labels'])
    y_val = mlb.transform(val_df['ekman_labels'])
    
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    return X_train, y_train, X_test, y_test, X_val, y_val, list(mapping_ids.keys()), vectorizer


def create_emotion_features(clf, vectorizer, labels, base_path, dataset_folder, file_name):
    input_file = os.path.join(base_path, dataset_folder, "source_tweets.txt")
    output_path = os.path.join(base_path, dataset_folder, file_name)
    
    ids = []
    texts = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                ids.append(parts[0])
                texts.append(parts[1])

    X_features = vectorizer.transform(texts)
    probs = clf.predict_proba(X_features)

    df = pd.DataFrame(probs, columns=labels)
    df.insert(0, 'id', ids)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    
EKMAN_KEYS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def get_emotion_features():
    ROOT = Path(__file__).resolve().parents[2]
    cache_path = ROOT / "data/emotion_features.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    features = {}
    for dataset in ["twitter15", "twitter16"]:
        csv_path = ROOT / f"data/rumor_detection_acl2017/{dataset}/emotions.csv"
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            vec = torch.tensor([row[k] for k in EKMAN_KEYS], dtype=torch.float32)
            features[str(row["id"])] = vec

    torch.save(features, cache_path)
    return features




def main():
    ROOT = Path(__file__).resolve().parents[2]
    train_df, test_df, val_df = load_data(ROOT)
    X_train, y_train, X_test, y_test, X_val, y_val, labels, vectorizer = preprocess_ekman(train_df, test_df, val_df, ekman_mapping_ids)
    # X_train, y_train, X_test, y_test, X_val, y_val, labels = preprocess_ekman(train_df, test_df, val_df, sentiment_mapping_ids)
    clf = lr_clf(X_train, y_train, X_val, y_val)
    #evaluate(clf, X_test, y_test)

    base_path = ROOT / 'data/rumor_detection_acl2017'
    t15 = 'twitter15'
    t16 = 'twitter16'
    file_name = 'emotions.csv'

    create_emotion_features(clf, vectorizer, labels, base_path, t15, file_name)
    create_emotion_features(clf, vectorizer, labels, base_path, t16, file_name)


if __name__ == "__main__":
    main()
