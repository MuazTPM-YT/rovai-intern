import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from train_sigmoid_head import DOMAIN2IDX, DOMAINS, MODELS_DIR


def main():
    print("=" * 50)
    print("Building K-means Fallback Centroids")
    print("=" * 50)

    train_emb = np.load(MODELS_DIR / "cache" / "train_embeddings.npy")
    train_labels = np.load(MODELS_DIR / "cache" / "train_labels.npy")
    print(f"loaded {len(train_emb)} embeddings")

    train_norm = normalize(train_emb, norm="l2")

    centroids = np.zeros((len(DOMAINS), train_emb.shape[1]), dtype=np.float32)
    for domain in DOMAINS:
        idx = DOMAIN2IDX[domain]
        mask = train_labels == domain
        domain_vecs = train_norm[mask]

        km = KMeans(n_clusters=1, n_init=10, random_state=42)
        km.fit(domain_vecs)
        centroids[idx] = km.cluster_centers_[0]

    centroids = normalize(centroids, norm="l2").astype(np.float32)

    print("\ncentroid quality (cosine sim to domain mean):")
    for i, domain in enumerate(DOMAINS):
        mask = train_labels == domain
        ref = normalize(train_emb[mask].mean(axis=0, keepdims=True), norm="l2")[0]
        sim = np.dot(centroids[i], ref)
        print(f"  {domain:<14} {sim:.4f}")

    out = MODELS_DIR / "intent_anchors.npy"
    np.save(out, centroids)
    print(f"\nsaved: {out} | shape: {centroids.shape}")


if __name__ == "__main__":
    main()
