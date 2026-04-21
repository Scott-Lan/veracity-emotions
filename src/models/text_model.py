#PURPOSE: Training and evalutating a text-based clasiffier on the twitter15/16 datasets, not using tree data.
#INPUT: Twitter Dataset json files for train, val, and test data
#OUTPUT: classification report for val and test data
'''% python3 text_model.py
Best C: 5 with F1: 0.5251
********** Best Validation Set Performance: ***********
              precision    recall  f1-score   support

           0       0.57      0.30      0.39        57
           1       0.43      0.84      0.57        57
           2       0.65      0.26      0.38        57
           3       0.73      0.81      0.77        57

    accuracy                           0.55       228
   macro avg       0.59      0.55      0.53       228
weighted avg       0.59      0.55      0.53       228

***************** Test Set Performance: ***************
              precision    recall  f1-score   support

           0       0.51      0.40      0.45        58
           1       0.31      0.80      0.45        59
           2       0.11      0.02      0.03        60
           3       0.48      0.25      0.33        59

    accuracy                           0.36       236
   macro avg       0.35      0.37      0.31       236
weighted avg       0.35      0.36      0.31       236
'''
#There is a serious drop in T|F performance in test set, and suspiciously high performance with non-rumor in val set

import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import scipy.sparse as sp
import numpy as np


#this is kinda the lazy way to get the root directory but...
ROOT = Path(__file__).resolve().parents[2]

# add the "src" folder to sys.path so utils.data_loader works
sys.path.insert(0, str(ROOT / "src"))

#get data loading functions
import utils.data_loader as dl
import preprocessing.twitter_extract as te

#debugging function to print label counts for each split
def print_label_counts(train_labels, val_labels, test_labels):
    from collections import Counter
    for name, labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        counts = Counter(labels)
        print(f"{name}: {dict(sorted(counts.items()))}")

#assign each label an integer value
def encode_labels(train_labels, val_labels, test_labels):
    # uncomment below to print label counts for each split
    #print_label_counts(train_labels, val_labels, test_labels)

    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(train_labels)
    Y_val = encoder.transform(val_labels)
    Y_test = encoder.transform(test_labels)

    return Y_train, Y_val, Y_test, encoder

#vectorize text using word and character n-grams, with tf-idf weighting
def vectorize_text(train_texts, val_texts, test_texts):
   
    #word and character vectorizers
    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)
    
    # fit on train ONLY
    X_train = sp.hstack((word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)))
    #use same vectorizer for val and test
    X_val = sp.hstack((word_vec.transform(val_texts), char_vec.transform(val_texts)))
    X_test = sp.hstack((word_vec.transform(test_texts), char_vec.transform(test_texts)))

    return X_train, X_val, X_test


def lr_classifier(X_train, Y_train, X_val, Y_val):
    best_C = None
    best_score = 0
    #test a range of C values, need to adjust to tune more hyperparameters
    for C in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 100, 200, 500, 1000]: #big range to test, can be adjusted
        #train using logistic regression and C value
        clf = LogisticRegression(C=C, class_weight="balanced", random_state=123, max_iter=1000)
        #fit with trainign data
        clf.fit(X_train, Y_train)
        #evaluate on val data using macro F1 score
        score = f1_score(Y_val, clf.predict(X_val), average="macro")
        #print(f"C={C}: F1={score:.4f}") #uncomment to print F1 for each C value
        if score > best_score:
            best_C, best_score = C, score
    print(f"Best C: {best_C} with F1: {best_score:.4f}")
    
    #train final model with best C value on all training data
    classifier = LogisticRegression(C=best_C, class_weight="balanced", random_state=123, max_iter=1000)
    classifier.fit(X_train, Y_train)
    return classifier

def evaluate_classifier(clf, X_val, X_test, Y_val, Y_test):
    #evaluate on val and test, print classification report
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)
    print("********** Best Validation Set Performance: ***********")
    print(classification_report(Y_val, val_preds))
    print("***************** Test Set Performance: ***************")
    print(classification_report(Y_test, test_preds))

def main():
    te.build_data()
    
    #load data
    train_data = dl.load_split("train")
    val_data = dl.load_split("val")
    test_data = dl.load_split("test")
    
    #seperate texts and labels
    train_texts = train_data[0]
    train_labels = train_data[1]
    val_texts = val_data[0]
    val_labels = val_data[1]
    test_texts = test_data[0]
    test_labels = test_data[1]
    
    X_train, X_val, X_test = vectorize_text(train_texts, val_texts, test_texts)

    #encode labels to be only integers 
    Y_train, Y_val, Y_test, encoder = encode_labels(train_labels, val_labels, test_labels)
    
    #train and evaluate classifier using logistic regression
    clf = lr_classifier(X_train, Y_train, X_val, Y_val)
    evaluate_classifier(clf, X_val, X_test, Y_val, Y_test)

if __name__ == "__main__":
    main()