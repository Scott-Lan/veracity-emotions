#!/bin/bash
set -e


if ls data/cache* data/tfidf* data/emo* &>/dev/null; then
    echo "  Clearing cache..."
    rm data/cache* data/tfidf* data/emo*
fi

echo ""
echo "Training emotion model..."
python3 src/models/em*

echo ""
echo "Training GNN model..."
python3 src/models/gnn*

echo ""
echo "Done!"
