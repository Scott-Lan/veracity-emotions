#PURPOSE: BiGCN classifier for rumor detection on Twitter15/16 cascade trees.
#INPUT: Twitter Dataset json files for train, val, and test data, with tree data
#OUTPUT: classification report for val and test data

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, classification_report #goat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.tree_parser import parse_tree, annotate_tree

LABELS = {"false": 0, "non-rumor": 1, "true": 2, "unverified": 3}
PATH_15 = ROOT / "data/rumor_detection_acl2017/twitter15"
PATH_16 = ROOT / "data/rumor_detection_acl2017/twitter16"
TEXT_DATA = ROOT / "data/text_data"

# feature dimensions for one node
STRUCT_DIM  = 10                               
BERT_DIM    = 768                                
EMOTION_DIM = 6                                   
FEAT_DIM    = STRUCT_DIM + BERT_DIM + EMOTION_DIM # 784


#BiGCN: two GCN streams (top-down and bottom-up), root features injected at layer 2
class RumorGNN(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=256, num_classes=4, dropout=0.3):
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

    def forward(self, x, edge_index, bot_up_edge_index, root_feat, batch):
        #broadcast root features to every node in the batch for layer 2 injection
        rf = root_feat[batch]

        #top-down stream
        #relu after each conv, dropout after layer 1, inject root features before layer 2
        x_top_down = F.relu(self.top_down_conv1(x, edge_index))
        x_top_down = F.dropout(x_top_down, p=self.dropout, training=self.training)
        x_top_down = F.relu(self.top_down_conv2(torch.cat([x_top_down, rf], dim=1), edge_index))

        #bottom-up stream
        
        x_bot_up = F.relu(self.bot_up_conv1(x, bot_up_edge_index))
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

    #root-only features
    nodes[0].text_vec    = text_vec
    nodes[0].emotion_vec = emotion_vec

    # get max values for normalization of features
    max_depth = max(n.depth        for n in nodes)
    max_time = max(n.time         for n in nodes)
    max_fanout = max(n.num_children for n in nodes)

    #to fill in any empty text and/or emotion vectors with zeros so we can batch them
    zeros_text = torch.zeros(BERT_DIM)
    zeros_emotion = torch.zeros(EMOTION_DIM)
    
    rows = []
    for node in nodes:
        struct = torch.tensor(node.feature_vector(max_depth, max_time, max_fanout))
        text = node.text_vec    if node.text_vec    is not None else zeros_text
        emotion = node.emotion_vec if node.emotion_vec is not None else zeros_emotion
        #concatenate structure, text and emotion fetures into one vector
        rows.append(torch.cat([struct, text, emotion]))
    x = torch.stack(rows)

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

    #save root features separately so the model can inject them at layer 2
    root_feat = x[0].unsqueeze(0)

    y = torch.tensor(LABELS[label], dtype=torch.long)
    return Data(x=x, top_down_edge_index=top_down_edge_index, bot_up_edge_index=bot_up_edge_index,
                root_feat=root_feat, y=y)


# load a split of the data and return a list of PyG Data objects
#   split_name : "train", "val", or "test"
def load_data_from_split(split_name):
    cache_path = ROOT / f"data/cache_{split_name}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    data_list = []
    for path_dir, year in [(PATH_15, "15"), (PATH_16, "16")]:
        json_path = TEXT_DATA / f"{split_name}_{year}.json"
        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)
        #bot_upild one Data object per tweet in this split
        #print(f"loading {split_name} {year}...")
        for i, row in enumerate(rows):
            data_list.append(compile_data(row["id"], row["label"], path_dir))
            #if i % 100 == 0:
            #    print(f"  {split_name} {year}: {i}/{len(rows)}")
    torch.save(data_list, cache_path)
    return data_list
    
# one pass through the training data, returns the average batch loss
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        #features and edges return logits which we compare to labels
        logits = model(batch.x, batch.top_down_edge_index, batch.bot_up_edge_index, batch.root_feat, batch.batch)
        #y = labels
        loss = F.cross_entropy(logits, batch.y)
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
        preds = model(batch.x, batch.top_down_edge_index, batch.bot_up_edge_index, batch.root_feat, batch.batch).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels


def main():
    # use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load all three splits as lists of Data objects
    print("loading splits...")
    train_data = load_data_from_split("train")
    val_data = load_data_from_split("val")
    test_data = load_data_from_split("test")
    print(f"  train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    #dataLoader makes graphs into one disjoint mega-graph
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data,   batch_size=64)
    test_loader = DataLoader(test_data,  batch_size=64)

    # init model, optimizer, and scheduler
    model = RumorGNN().to(device)
    #optimizer using L2 regularization 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #schedular to prevent overfitting.
        # if f1 doesnt improve for 7 epochs, reduce lr by factor of 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )

    # train, track best val F1, save its state for final test eval
    best_val_f1 = 0
    best_state  = None
    print("training...")
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} | loss={loss:.4f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f}")

    # restore best checkpoint and report final test performance
    model.load_state_dict(best_state)
    test_acc, test_f1, preds, labels = evaluate(model, test_loader, device)
    label_names = sorted(LABELS, key=LABELS.get)
    
    #print final results
    print("********** Best Validation F1: ***********")
    print(f"  {best_val_f1:.4f}")
    print("***************** Test Set Performance: ***************")
    print(f"  acc={test_acc:.3f}  macro_f1={test_f1:.3f}")
    print(classification_report(labels, preds, target_names=label_names))


if __name__ == "__main__":
    main()


