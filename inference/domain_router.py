from pathlib import Path
import numpy as np
import onnxruntime as ort
import yaml
from sentence_transformers import SentenceTransformer

DOMAINS = ["NAVIGATION", "STUDY", "STORY", "CHAT", "PET", "GAME", "SYSTEM"]


class DomainRouter:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)

        self.session = ort.InferenceSession(
            str(models_dir / "sigmoid_domain_head.onnx"),
            providers=["CPUExecutionProvider"],
        )

        with open(models_dir / "multi_label_thresholds.yaml") as f:
            self.thresholds = yaml.safe_load(f)

        self.centroids = np.load(models_dir / "intent_anchors.npy")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, text):
        return np.array(
            self.embedder.encode([text], show_progress_bar=False),
            dtype=np.float32,
        )

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _kmeans_softmax(self, emb):
        norm = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)
        sims = norm @ self.centroids.T
        temp = 0.1
        shifted = (sims - sims.max(axis=-1, keepdims=True)) / temp
        exp = np.exp(shifted)
        soft = exp / exp.sum(axis=-1, keepdims=True)
        return sims[0], soft[0]

    def route(self, text):
        text = (text or "").strip()
        if not text:
            return {
                "method": "none", "domains": [], "top_domain": "CHAT",
                "top_score": 0.0, "sigmoid_scores": {d: 0.0 for d in DOMAINS},
                "sigmoid_thresholds": self.thresholds,
                "kmeans_scores": None, "kmeans_sims": None,
            }

        emb = self._embed(text)
        logits = self.session.run(None, {"embedding": emb})[0]
        probs = self._sigmoid(logits)[0]

        sig_scores = {d: float(probs[i]) for i, d in enumerate(DOMAINS)}
        sig_thresh = {d: self.thresholds[d] for d in DOMAINS}

        fired = [d for i, d in enumerate(DOMAINS) if probs[i] >= self.thresholds[d]]

        if fired:
            fired.sort(key=lambda d: sig_scores[d], reverse=True)
            return {
                "method": "sigmoid", "domains": fired,
                "top_domain": fired[0], "top_score": sig_scores[fired[0]],
                "sigmoid_scores": sig_scores, "sigmoid_thresholds": sig_thresh,
                "kmeans_scores": None, "kmeans_sims": None,
            }

        raw_sims, soft_probs = self._kmeans_softmax(emb)
        km_scores = {d: float(soft_probs[i]) for i, d in enumerate(DOMAINS)}
        km_sims = {d: float(raw_sims[i]) for i, d in enumerate(DOMAINS)}
        top = DOMAINS[int(np.argmax(soft_probs))]

        return {
            "method": "kmeans", "domains": [top],
            "top_domain": top, "top_score": km_scores[top],
            "sigmoid_scores": sig_scores, "sigmoid_thresholds": sig_thresh,
            "kmeans_scores": km_scores, "kmeans_sims": km_sims,
        }
