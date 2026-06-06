# wini_intent_pkg — Domain Detection

Two-tier domain router for a child-facing companion robot. Routes natural language utterances to one of 7 domains using a sigmoid primary head with K-means cosine-softmax fallback.

## Domains

| Domain | Examples |
|---|---|
| NAVIGATION | "go to the kitchen", "follow me", "stop moving" |
| STUDY | "explain photosynthesis", "quiz me on math" |
| STORY | "tell me a story about a dragon", "what happens next" |
| CHAT | "tell me a joke", "what's your name" |
| PET | "give me a hug", "you're adorable little robot" |
| GAME | "let's play chess", "give me a hint" |
| SYSTEM | "turn up the volume", "restart yourself" |

## Architecture

```
utterance → MiniLM-L6-v2 (384d) → Sigmoid Head (Linear 384→7)
                                     ↓ threshold check
                              fired? → return domain(s)
                              silent? → K-means cosine fallback → return top domain
```

- **Primary**: `Linear(384, 7)` trained with BCE loss, per-domain thresholds tuned for P≥0.9 R≥0.85
- **Fallback**: Cosine similarity to domain centroids with temperature-scaled softmax (T=0.1)
- **Embeddings**: Frozen `all-MiniLM-L6-v2` sentence transformer

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training Pipeline

Run in order:

```bash
python data/generate_dataset.py
python train/train_sigmoid_head.py
python train/tune_thresholds.py
python train/train_kmeans_fallback.py
python train/export_onnx.py
```

## Demo

```bash
python main.py            # interactive mode — type utterances, see routing
python main.py --batch    # run 30 built-in test cases
```

### Interactive Example

```
> let's play chess
  method:  sigmoid
  domain:  GAME (0.7543)

> turn up the volume
  method:  sigmoid
  domain:  SYSTEM (0.9812)
```

## Artifacts

| File | Description |
|---|---|
| `models/sigmoid_domain_head.onnx` | Exported sigmoid head |
| `models/multi_label_thresholds.yaml` | Per-domain decision thresholds |
| `models/intent_anchors.npy` | K-means centroids (7×384) |

## Results

- Validation accuracy: ~97%
- All domains: precision ≥ 0.9, recall ≥ 0.85
- ONNX parity: max diff < 1e-4
- Batch test: 30/30 (100%)

## Assumptions and Limitations

- Utterances are treated as single-domain in training; multi-label firing is supported at inference when multiple domains exceed threshold.
- The dataset is synthetically generated from templates with child-speech augmentation (filler words, politeness markers). Real-world performance may differ with out-of-distribution phrasing.
- K-means fallback uses a single centroid per domain (class-mean on L2-normalized embeddings), which is equivalent to K-means with K=1 per class.
- The sigmoid head is the authority; K-means only activates when no domain exceeds its tuned threshold.

## Project Structure

```
├── data/
│   ├── generate_dataset.py
│   ├── train.csv
│   └── val.csv
├── train/
│   ├── train_sigmoid_head.py
│   ├── tune_thresholds.py
│   ├── train_kmeans_fallback.py
│   └── export_onnx.py
├── inference/
│   └── domain_router.py
├── models/
│   ├── sigmoid_domain_head.onnx
│   ├── multi_label_thresholds.yaml
│   └── intent_anchors.npy
├── main.py
├── requirements.txt
└── README.md
```
