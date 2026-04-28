#PURPOSE: BiGCN classifier for rumor detection on Twitter15/16 cascade trees.
#INPUT: Twitter Dataset json files for train, val, and test data, with tree data
#OUTPUT: classification report for val and test data

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, classification_report, confusion_matrix #goat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.tree_parser import parse_tree, annotate_tree
from utils.data_loader import TFIDF_DIM
import utils.data_loader as dl
from models.emotion_model import get_emotion_features

LABELS = {"false": 0, "non-rumor": 1, "true": 2, "unverified": 3}
PATH_15 = ROOT / "data/rumor_detection_acl2017/twitter15"
PATH_16 = ROOT / "data/rumor_detection_acl2017/twitter16"
TEXT_DATA = ROOT / "data/text_data"

# feature dimensions for one node
STRUCT_DIM = 10
EMOTION_KEYS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMOTION_DIM = len(EMOTION_KEYS)
# paper-style per-node text features: TF-IDF of source tweet broadcast to every node
FEAT_DIM    = STRUCT_DIM + TFIDF_DIM + EMOTION_DIM


#BiGCN: two GCN streams (top-down and bottom-up), root features injected at layer 2
class RumorGNN(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=256, num_classes=4, dropout=0.5):
        super().__init__()
        #top-down conv layers (parent -> child)
        self.top_down_conv1 = GCNConv(in_dim, hidden)
        self.top_down_conv2 = GCNConv(hidden + in_dim, hidden)
        #bottom-up conv layers (child -> parent)
        self.bot_up_conv1 = GCNConv(in_dim, hidden)
        self.bot_up_conv2 = GCNConv(hidden + in_dim, hidden)
        #dropout rate for conv layers and head
        self.dropout  = dropout
        #classifier takes top_down + bot_up pooled vectors concatenated
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, bot_up_edge_index, root_feat, text_feat, batch):
        #broadcast per-tree text features to every node and form full per-node features.
        #text is stored once per tree to avoid the per-node duplication that blew up RAM.
        tf = text_feat[batch]                          # (N, TFIDF_DIM)
        #drop only the TF-IDF slice to stop token memorization 
        #struct+emotion are too few/informative to drop
        tf_in = F.dropout(tf, p=self.dropout, training=self.training)
        x_full = torch.cat([x, tf_in], dim=1)          # (N, STRUCT+EMOTION+TFIDF)
        rf = torch.cat([root_feat[batch], tf], dim=1)  # (N, STRUCT+EMOTION+TFIDF)

        #top-down stream
        #relu after each conv, dropout after layer 1, inject root features before layer 2
            #relu prevents negative values from dominating the results. (Rectified Linear Unit)
        x_top_down = F.relu(self.top_down_conv1(x_full, edge_index))
        #dropout "drops" a random subset of activatiosn to prevent overfitting
        x_top_down = F.dropout(x_top_down, p=self.dropout, training=self.training)
        x_top_down = F.relu(self.top_down_conv2(torch.cat([x_top_down, rf], dim=1), edge_index))

        #bottom-up stream

        x_bot_up = F.relu(self.bot_up_conv1(x_full, bot_up_edge_index))
        x_bot_up = F.dropout(x_bot_up, p=self.dropout, training=self.training)
        x_bot_up = F.relu(self.bot_up_conv2(torch.cat([x_bot_up, rf], dim=1), bot_up_edge_index))

        #pool each stream then concatenate before classification
        x = torch.cat([global_mean_pool(x_top_down, batch),
                        global_mean_pool(x_bot_up, batch)], dim=1)
        return self.head(x)

#might move to data_loader.py   
#from data_loader import compile_data 
# convert one cascade tree into 2 Data objects ready for training
def compile_data(tweet_id, label, path_dir, text_vec=None, emotion_vec=None):
    # parse the tree file and annotate every node
    root = parse_tree(tweet_id, path_dir)
    nodes = annotate_tree(root)

    # emotion remains a single root-level vector for now;
    # text is stored once per tree (see text_feat below) and broadcast at forward time
    nodes[0].emotion_vec = emotion_vec

    # get max values for normalization of features
    max_depth = max(n.depth for n in nodes)
    max_time = max(n.time for n in nodes)
    max_fanout = max(n.num_children for n in nodes)

    zeros_emotion = torch.zeros(EMOTION_DIM)

    # per-node compact features: struct + emotion only (no text)
    rows = []
    for node in nodes:
        struct = torch.tensor(node.feature_vector(max_depth, max_time, max_fanout))
        # if text_vec is not None:
        #     text = text_vec
        # else:
        #     text = torch.zeros(TFIDF_DIM)
        if emotion_vec is not None:
            emotion = node.emotion_vec
        else:
            emotion = zeros_emotion
            
        rows.append(torch.cat([struct, emotion]))
    x = torch.stack(rows)  # (N, STRUCT_DIM + EMOTION_DIM)

    #top-down (top_down) edges (parent->child) and bottom-up (bot_up) edges (child->parent)
    index_of = {n: i for i, n in enumerate(nodes)}
    top_down_edges = []
    bot_up_edges = []
    for parent in nodes:
        for child in parent.children:
            i, j = index_of[parent], index_of[child]
            top_down_edges.append([i, j])
            bot_up_edges.append([j, i])
    top_down_edge_index = torch.tensor(top_down_edges, dtype=torch.long).t().contiguous()
    bot_up_edge_index = torch.tensor(bot_up_edges, dtype=torch.long).t().contiguous()

    # one TF-IDF row per tree; broadcast to every node inside RumorGNN.forward
    if text_vec is None:
        text_vec = torch.zeros(TFIDF_DIM)
    text_feat = text_vec.unsqueeze(0)

    # root's(struct+emotion) features; text comes via text_feat at forward time
    root_feat = x[0].unsqueeze(0)  # (1, STRUCT_DIM + EMOTION_DIM)

    y = torch.tensor(LABELS[label], dtype=torch.long)
    return Data(x=x, top_down_edge_index=top_down_edge_index, bot_up_edge_index=bot_up_edge_index,
                root_feat=root_feat, text_feat=text_feat, y=y)


# load a split of the data and return a list of Data objects
#   split_name : "train", "val", or "test"
#   use_text   : include TF-IDF source-tweet features
#   use_emotion: include NRC emotion features
def load_data_list(split_name, use_text=True, use_emotion=True):
    config = (("t" if use_text else "") + ("e" if use_emotion else "")) or "struct"
    cache_path = ROOT / f"data/cache_{split_name}_{config}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    tfidf = dl.tfidf_features() if use_text else None
    emotion = get_emotion_features() if use_emotion else None
    data_list = []
    for path_dir, year in [(PATH_15, "15"), (PATH_16, "16")]:
        json_path = TEXT_DATA / f"{split_name}_{year}.json"
        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            data_list.append(compile_data(
                row["id"], row["label"], path_dir,
                text_vec=tfidf.get(str(row["id"])) if tfidf else None,
                emotion_vec=emotion.get(str(row["id"])) if emotion else None,
            ))
    torch.save(data_list, cache_path)
    return data_list
    
# one pass through the training data, returns the average batch loss
def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.top_down_edge_index, batch.bot_up_edge_index, batch.root_feat, batch.text_feat, batch.batch)
        loss = F.cross_entropy(logits, batch.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# evaluate the model on a loader, return accuracy, macro F1, and raw preds/labels
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    #iterate over all batches in the loader and evaluate
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.top_down_edge_index, batch.bot_up_edge_index, batch.root_feat, batch.text_feat, batch.batch).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels

#run a full train/val/test cycle for one ablation config, print classification reports
def run_config(use_text, use_emotion, device):
    label_names = sorted(LABELS, key=LABELS.get)
    #build config string for display and cache naming
    config = (("t" if use_text else "") + ("e" if use_emotion else "")) or "struct"
    print(f"\n{'='*55}")
    print(f"  CONFIG: {config}")
    print(f"{'='*55}")

    #load splits for this config
    train_data = load_data_list("train", use_text, use_emotion)
    val_data   = load_data_list("val",   use_text, use_emotion)
    test_data  = load_data_list("test",  use_text, use_emotion)
    print(f"  train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    #dataLoader makes graphs into one disjoint mega-graph
    gen = torch.Generator()
    gen.manual_seed(255)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator = gen)
    val_loader   = DataLoader(val_data,   batch_size=64)
    test_loader  = DataLoader(test_data,  batch_size=64)

    #init model, optimizer, and scheduler
    model     = RumorGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #if val f1 doesnt improve for 7 epochs, reduce lr by factor of 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5)

    #track best val f1 and save preds/labels for classification report
    best_val_f1     = 0
    best_state      = None
    best_val_preds  = None
    best_val_labels = None
    no_improve = 0
    print("  training...")
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
        scheduler.step(val_f1)
        if (0 < (val_f1 - best_val_f1) < 0.01):
            no_improve += 1
        if val_f1 > best_val_f1:
            best_val_f1     = val_f1
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_preds  = val_preds
            best_val_labels = val_labels
            no_improve = 0

        if epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | loss={loss:.4f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f}")

        if no_improve > 15:
            break
    #restore best checkpoint and print val + test reports
    model.load_state_dict(best_state)
    test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)

    print(f"\n  *** Best Validation F1: {best_val_f1:.4f} ***")
    print(classification_report(best_val_labels, best_val_preds, target_names=label_names))
    print(f"  *** Test  |  acc={test_acc:.3f}  macro_f1={test_f1:.3f} ***")
    print(classification_report(test_labels, test_preds, target_names=label_names))
    #confusion matrix shows where each true class actually got predicted
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(label_names))))
    print("  test confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12s}" + "".join(f"{n:>11s}" for n in label_names))
    for name, row in zip(label_names, cm):
        print(f"  {name:12s}" + "".join(f"{v:>11d}" for v in row))


def set_seed(seed=255):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for use_text, use_emotion in [
        (False, False),   #struct only
        (True,  False),   #struct + TF-IDF
        (False, True),    #struct + emotion
        (True,  True),    #struct + TF-IDF + emotion
    ]:
        run_config(use_text, use_emotion, device)


if __name__ == "__main__":
    main()


