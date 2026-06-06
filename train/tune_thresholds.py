import numpy as np
import torch
import yaml

from train_sigmoid_head import DEVICE, DOMAIN2IDX, DOMAINS, MODELS_DIR, SigmoidDomainHead


def pr_f1(probs, labels, idx, t):
    pred = (probs[:, idx] >= t).astype(int)
    true = labels[:, idx].astype(int)
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    print("=" * 50)
    print("Tuning Per-Domain Thresholds")
    print("=" * 50)

    val_emb = np.load(MODELS_DIR / "cache" / "val_embeddings.npy")
    val_str = np.load(MODELS_DIR / "cache" / "val_labels.npy")

    labels = np.zeros((len(val_str), len(DOMAINS)), dtype=np.float32)
    for i, lbl in enumerate(val_str):
        labels[i, DOMAIN2IDX[lbl]] = 1.0

    model = SigmoidDomainHead()
    model.load_state_dict(torch.load(MODELS_DIR / "sigmoid_domain_head.pt", map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(val_emb)).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    thresholds = {}
    print(f"\n{'domain':<14} {'thresh':>7} {'prec':>7} {'rec':>7} {'f1':>7}")
    print("-" * 46)

    for domain in DOMAINS:
        idx = DOMAIN2IDX[domain]
        best_t, best_f1 = 0.5, 0.0

        for t in np.arange(0.05, 0.95, 0.005):
            p, r, f = pr_f1(probs, labels, idx, t)
            if p >= 0.9 and r >= 0.85 and f > best_f1:
                best_f1 = f
                best_t = float(round(t, 3))

        if best_f1 == 0.0:
            for t in np.arange(0.05, 0.95, 0.005):
                p, r, f = pr_f1(probs, labels, idx, t)
                if f > best_f1:
                    best_f1 = f
                    best_t = float(round(t, 3))

        p, r, f = pr_f1(probs, labels, idx, best_t)
        thresholds[domain] = best_t
        print(f"  {domain:<12} {best_t:>7.3f} {p:>7.4f} {r:>7.4f} {f:>7.4f}")

    out = MODELS_DIR / "multi_label_thresholds.yaml"
    with open(out, "w") as fh:
        yaml.dump(thresholds, fh, default_flow_style=False, sort_keys=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
