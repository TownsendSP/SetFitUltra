"""
Usage:
    python predict_document.py /path/to/file.txt \
        --encoder models/final_deep/stage1_encoder \
        --classifier models/final_deep/stage2_classifier.joblib
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        return self.net(x)

def chunk_text(text, tokenizer, chunk_size=500, overlap=50):
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        ct = tokens[i:i + chunk_size]
        if len(ct) < 20:
            continue
        chunks.append(tokenizer.decode(ct, skip_special_tokens=True))
        if i + chunk_size >= len(tokens):
            break
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to plain text file")
    parser.add_argument("--encoder", default="models/final_deep/stage1_encoder")
    parser.add_argument("--classifier", default="models/final_deep/stage2_classifier.joblib")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    text = open(args.file, encoding="utf-8", errors="ignore").read()
    print(f"Read {len(text):,} chars from {args.file}")

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf_data = joblib.load(args.classifier)
    encoder_path = args.encoder
    encoder = SentenceTransformer(encoder_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)

    tag_ids = np.array(clf_data['tag_ids'])
    tag_names = clf_data.get('tag_names', [f"tag_{t}" for t in tag_ids])

    classifier = MultiLabelClassifier(clf_data['input_dim'], clf_data['num_labels']).to(device)
    classifier.load_state_dict(clf_data['model_state'])
    classifier.eval()

    # Chunk -> encode -> predict
    chunks = chunk_text(text, tokenizer)
    print(f"Chunked into {len(chunks)} passages")

    with torch.no_grad():
        emb = encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        probs = torch.sigmoid(classifier(emb)).cpu().numpy()

    # max confidence across all chunks per tag
    doc_probs = probs.max(axis=0)
    top_idx = np.argsort(doc_probs)[::-1][:args.top_k]

    print(f"\nTop {args.top_k} predicted tags:")
    print(f"{'Rank':<6}{'Tag':<40}{'Confidence':<10}")
    print("-" * 56)
    for rank, idx in enumerate(top_idx, 1):
        conf = doc_probs[idx]
        if conf < args.threshold:
            break
        print(f"{rank:<6}{tag_names[idx]:<40}{conf:.4f}")


if __name__ == "__main__":
    main()
