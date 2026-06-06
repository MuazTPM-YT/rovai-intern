import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DOMAINS = ["NAVIGATION", "STUDY", "STORY", "CHAT", "PET", "GAME", "SYSTEM"]
DOMAIN2IDX = {d: i for i, d in enumerate(DOMAINS)}
EMB_DIM = 384
N_DOMAINS = len(DOMAINS)

EPOCHS = 120
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


class SigmoidDomainHead(nn.Module):
    def __init__(self, in_dim=EMB_DIM, out_dim=N_DOMAINS):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def load_csv(path):
    texts, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["domain"])
    return texts, labels


def encode_labels(labels):
    y = np.zeros((len(labels), N_DOMAINS), dtype=np.float32)
    for i, lbl in enumerate(labels):
        y[i, DOMAIN2IDX[lbl]] = 1.0
    return y


def embed_texts(texts, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"  cached: {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    print("  loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  encoding {len(texts)} texts...")
    embs = np.array(model.encode(texts, show_progress_bar=True, batch_size=256), dtype=np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)

    return embs


def train():
    print("=" * 50)
    print("Training Sigmoid Domain Head")
    print("=" * 50)

    train_texts, train_labels = load_csv(DATA_DIR / "train.csv")
    val_texts, val_labels = load_csv(DATA_DIR / "val.csv")
    print(f"train={len(train_texts)}, val={len(val_texts)}")

    print("\nembedding train...")
    train_emb = embed_texts(train_texts, MODELS_DIR / "cache" / "train_embeddings.npy")
    print("embedding val...")
    val_emb = embed_texts(val_texts, MODELS_DIR / "cache" / "val_embeddings.npy")

    train_y = encode_labels(train_labels)
    val_y = encode_labels(val_labels)

    np.save(MODELS_DIR / "cache" / "train_labels.npy", np.array(train_labels))
    np.save(MODELS_DIR / "cache" / "val_labels.npy", np.array(val_labels))

    train_dl = DataLoader(
        TensorDataset(torch.tensor(train_emb), torch.tensor(train_y)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(val_emb), torch.tensor(val_y)),
        batch_size=BATCH_SIZE,
    )

    model = SigmoidDomainHead().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-5)

    print(f"\ndevice={DEVICE}, params={sum(p.numel() for p in model.parameters()):,}")
    print(f"epochs={EPOCHS}, patience={PATIENCE}\n")

    best_loss = float("inf")
    stale = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            t_loss += loss.item() * xb.size(0)
        t_loss /= len(train_dl.dataset)

        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(1) == yb.argmax(1)).sum().item()
                total += xb.size(0)
        v_loss /= len(val_dl.dataset)
        acc = correct / total

        sched.step()

        if ep % 5 == 0 or ep == 1:
            lr_now = optim.param_groups[0]["lr"]
            print(f"  ep {ep:3d}/{EPOCHS} | t_loss={t_loss:.4f} v_loss={v_loss:.4f} acc={acc:.4f} lr={lr_now:.2e}")

        if v_loss < best_loss:
            best_loss = v_loss
            stale = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "sigmoid_domain_head.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"\n  early stop @ ep {ep}")
                break

    print(f"\nbest val loss: {best_loss:.4f}")
    print(f"saved: {MODELS_DIR / 'sigmoid_domain_head.pt'}")


if __name__ == "__main__":
    train()
