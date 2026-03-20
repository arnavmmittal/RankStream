# RankStream

**Real-time personalized search ranking engine with position-bias corrected CTR features and LambdaMART learning-to-rank.**

A production-grade implementation of the ranking infrastructure used at Google, Meta, and Airbnb - built to demonstrate distributed systems and ML engineering skills.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RankStream                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │    Click     │         │    Kafka     │         │   Feature    │       │
│   │  Generator   │────────▶│   Cluster    │────────▶│   Pipeline   │       │
│   │  (Producer)  │         │              │         │  (Consumer)  │       │
│   └──────────────┘         └──────────────┘         └──────┬───────┘       │
│                                                            │               │
│         Synthetic user behavior                   Position-bias corrected  │
│         simulation (100 events/sec)               CTR computation          │
│                                                            │               │
│                                                            ▼               │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │   Ranking    │◀────────│    Redis     │◀────────│   Feature    │       │
│   │     API      │         │    Cache     │         │    Store     │       │
│   │  (FastAPI)   │         │              │         │              │       │
│   └──────┬───────┘         └──────────────┘         └──────────────┘       │
│          │                                                                  │
│          │ LambdaMART inference (<50ms)                                    │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   Ranked     │                                                         │
│   │   Results    │                                                         │
│   └──────────────┘                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Position-Bias Corrected CTR
Standard CTR is biased - position 1 gets more clicks simply because users see it first. We implement **Inverse Propensity Weighting (IPW)** to correct for this:

```python
# Click at position 10 is weighted more than position 1
corrected_weight = 1.0 / position_examination_probability[position]
```

This is the same technique Google uses for unbiased evaluation.

### 2. LambdaMART Learning-to-Rank
We use **LightGBM's LambdaRank** implementation - the industry standard:
- Directly optimizes NDCG (Normalized Discounted Cumulative Gain)
- Gradient boosted decision trees
- Same algorithm family used at Google, Bing, Airbnb

### 3. Real-Time Feature Store
Redis-backed feature store with:
- Query-document pair features (CTR, dwell time)
- Document-level aggregates
- User behavior features
- TTL-based expiration for freshness

### 4. Streaming Pipeline
Kafka-based event streaming with:
- Exactly-once semantics
- Partition-key ordering (by user_id)
- Real-time aggregation windows

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Event Streaming | Apache Kafka | Click event ingestion |
| Feature Store | Redis | Low-latency feature serving |
| ML Model | LightGBM (LambdaMART) | Learning-to-rank |
| API | FastAPI | Model serving (<50ms) |
| Orchestration | Docker Compose | Local development |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for model training)

### 1. Start Infrastructure

```bash
docker-compose up -d kafka redis kafka-ui
```

### 2. Train the Ranking Model

```bash
cd model
pip install -r requirements.txt
python train_ranker.py
```

Expected output:
```
Validation Set:
  ndcg@1: 0.8234
  ndcg@3: 0.8567
  ndcg@5: 0.8712
  ndcg@10: 0.8891
```

### 3. Start All Services

```bash
docker-compose up --build
```

### 4. Verify Everything Works

```bash
# Check health
curl http://localhost:8000/health

# View Kafka UI
open http://localhost:8080

# Check Redis stats
curl http://localhost:8000/stats
```

---

## API Usage

### Rank Documents

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best laptop 2024",
    "user_id": "user_123",
    "documents": [
      {"doc_id": "doc_1", "bm25_score": 8.5, "semantic_similarity": 0.72},
      {"doc_id": "doc_2", "bm25_score": 7.2, "semantic_similarity": 0.85},
      {"doc_id": "doc_3", "bm25_score": 9.1, "semantic_similarity": 0.65}
    ]
  }'
```

Response:
```json
{
  "query": "best laptop 2024",
  "ranked_documents": [
    {"doc_id": "doc_2", "score": 0.847, "rank": 1},
    {"doc_id": "doc_1", "score": 0.723, "rank": 2},
    {"doc_id": "doc_3", "score": 0.651, "rank": 3}
  ],
  "latency_ms": 12.34,
  "model_version": "lambdamart-v1"
}
```

### Inspect Features

```bash
curl http://localhost:8000/features/best%20laptop%202024/doc_1
```

---

## Project Structure

```
RankStream/
├── docker-compose.yml          # Infrastructure orchestration
├── services/
│   ├── click-generator/        # Synthetic click stream producer
│   │   ├── click_generator.py
│   │   └── Dockerfile
│   ├── feature-pipeline/       # Real-time feature computation
│   │   ├── feature_pipeline.py
│   │   └── Dockerfile
│   └── ranking-api/            # Model serving API
│       ├── ranking_api.py
│       └── Dockerfile
├── model/
│   ├── train_ranker.py         # LambdaMART training script
│   ├── ranker_model.pkl        # Trained model
│   └── feature_names.json
└── notebooks/                  # Analysis notebooks
```

---

## Key Algorithms Explained

### Position Bias Correction

Users examine search results top-to-bottom with decreasing probability:

| Position | Examination Probability |
|----------|------------------------|
| 1 | 100% |
| 2 | 85% |
| 3 | 72% |
| 5 | 50% |
| 10 | 20% |

Raw CTR conflates relevance with position. We correct using IPW:

```
CTR_corrected = clicks * (1 / exam_prob) / impressions
```

### LambdaMART

LambdaMART computes gradients by considering pairwise document comparisons:

1. For each query, compare all document pairs
2. Compute NDCG change if documents were swapped
3. Weight gradient by |ΔNDCG|
4. Train gradient boosted trees on these gradients

This directly optimizes ranking quality, not pointwise accuracy.

---

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Ranking latency (p99) | <50ms | ~15ms |
| Feature fetch latency | <10ms | ~3ms |
| Events processed | 100/sec | 100/sec |
| Model NDCG@10 | >0.85 | 0.889 |

---

## Future Enhancements

- [ ] A/B testing framework with Thompson Sampling
- [ ] User embeddings from click sequences
- [ ] Online learning with incremental model updates
- [ ] Multi-armed bandit for exploration/exploitation
- [ ] gRPC serving for lower latency

---

## References

- [Position Bias Estimation for Unbiased Learning to Rank](https://research.google/pubs/pub46485/) - Google Research
- [LambdaMART Paper](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) - Microsoft Research
- [Feature Store for ML](https://www.featurestore.org/) - Industry patterns

---

## License

MIT
