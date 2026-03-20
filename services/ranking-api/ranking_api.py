"""
Ranking API Service

FastAPI service that:
1. Receives search queries with candidate documents
2. Fetches real-time features from Redis
3. Runs LightGBM model inference
4. Returns ranked results with scores

Latency target: <50ms p99
"""

import os
import time
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import redis
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "ranker_model.pkl"
FEATURE_NAMES_PATH = Path(__file__).parent.parent.parent / "model" / "feature_names.json"

# Global state
redis_client: Optional[redis.Redis] = None
model: Optional[lgb.Booster] = None
feature_names: List[str] = []


# Request/Response models
class Document(BaseModel):
    doc_id: str
    # Optional static features (would come from search index in production)
    bm25_score: float = 0.0
    semantic_similarity: float = 0.0
    doc_quality: float = 0.5
    doc_freshness: float = 0.5
    doc_length: int = 1000


class RankRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    documents: List[Document]
    # A/B test variant (for experimentation)
    experiment_variant: Optional[str] = None


class RankedDocument(BaseModel):
    doc_id: str
    score: float
    rank: int
    features: Dict[str, float]


class RankResponse(BaseModel):
    query: str
    ranked_documents: List[RankedDocument]
    latency_ms: float
    model_version: str
    experiment_variant: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup."""
    global redis_client, model, feature_names

    # Connect to Redis
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
    )
    redis_client.ping()
    print("Redis connected!")

    # Load model
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded!")

        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = json.load(f)
    else:
        print("WARNING: No model found, using random scoring")

    yield

    # Cleanup
    if redis_client:
        redis_client.close()


app = FastAPI(
    title="RankStream Ranking API",
    description="Real-time personalized search ranking with LambdaMART",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_features_from_redis(query: str, doc_ids: List[str], user_id: Optional[str]) -> Dict[str, Dict]:
    """
    Fetch real-time features from Redis feature store.

    In production, this would be optimized with:
    - Connection pooling
    - Batch fetching
    - Local caching for hot queries
    """
    features = {}
    pipe = redis_client.pipeline()

    # Fetch query-document features
    for doc_id in doc_ids:
        qd_key = f"qd:{query}|{doc_id}"
        doc_key = f"doc:{doc_id}"
        pipe.hgetall(qd_key)
        pipe.hgetall(doc_key)

    # Fetch query features
    query_key = f"query:{query}"
    pipe.hgetall(query_key)

    # Fetch user features if provided
    if user_id:
        user_key = f"user:{user_id}:features"
        pipe.hgetall(user_key)

    results = pipe.execute()

    # Parse results
    for i, doc_id in enumerate(doc_ids):
        qd_features = results[i * 2] or {}
        doc_features = results[i * 2 + 1] or {}

        features[doc_id] = {
            'query_doc_ctr': float(qd_features.get(b'ctr', 0) or 0),
            'query_doc_impressions': int(qd_features.get(b'impressions', 0) or 0),
            'avg_dwell_ms': float(qd_features.get(b'avg_dwell_ms', 0) or 0),
            'doc_ctr': float(doc_features.get(b'ctr', 0) or 0),
            'doc_impressions': int(doc_features.get(b'impressions', 0) or 0),
        }

    # Query features
    query_features_raw = results[len(doc_ids) * 2] or {}
    query_popularity = int(query_features_raw.get(b'search_count', 0) or 0)

    # User features
    user_affinity = 0.5  # Default
    if user_id and len(results) > len(doc_ids) * 2 + 1:
        user_features_raw = results[len(doc_ids) * 2 + 1] or {}
        total_clicks = int(user_features_raw.get(b'total_clicks', 0) or 0)
        user_affinity = min(1.0, total_clicks / 100)  # Normalize

    return features, query_popularity, user_affinity


def build_feature_matrix(
    documents: List[Document],
    redis_features: Dict[str, Dict],
    query_popularity: float,
    user_affinity: float,
) -> np.ndarray:
    """
    Build feature matrix for model inference.

    Feature order must match training!
    """
    feature_matrix = []

    for i, doc in enumerate(documents):
        redis_feat = redis_features.get(doc.doc_id, {})

        # Build feature vector (must match training order)
        features = [
            np.log1p(query_popularity),           # query_popularity
            3.0,                                   # query_length (placeholder)
            doc.doc_quality,                       # doc_quality
            doc.doc_freshness,                     # doc_freshness
            np.log1p(doc.doc_length),             # doc_length_log
            doc.bm25_score,                        # bm25_score
            doc.semantic_similarity,               # semantic_similarity
            redis_feat.get('query_doc_ctr', 0),   # historical_ctr
            np.log1p(redis_feat.get('avg_dwell_ms', 0)),  # avg_dwell_time_log
            np.log1p(redis_feat.get('doc_impressions', 0)),  # num_impressions_log
            user_affinity,                         # user_affinity
            1.0 / (i + 1),                        # position_inverse
        ]

        feature_matrix.append(features)

    return np.array(feature_matrix, dtype=np.float32)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    try:
        redis_client.ping()
        redis_ok = True
    except:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": redis_ok,
        "model_loaded": model is not None,
    }


@app.post("/rank", response_model=RankResponse)
async def rank_documents(request: RankRequest):
    """
    Rank documents for a search query.

    This is the main endpoint called by the search service.
    """
    start_time = time.perf_counter()

    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="No documents to rank")

    doc_ids = [doc.doc_id for doc in request.documents]

    # Fetch features from Redis
    redis_features, query_popularity, user_affinity = fetch_features_from_redis(
        request.query,
        doc_ids,
        request.user_id
    )

    # Build feature matrix
    feature_matrix = build_feature_matrix(
        request.documents,
        redis_features,
        query_popularity,
        user_affinity,
    )

    # Run model inference
    if model is not None:
        scores = model.predict(feature_matrix)
    else:
        # Fallback: random scores
        scores = np.random.rand(len(request.documents))

    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]

    # Build response
    ranked_docs = []
    for rank, idx in enumerate(sorted_indices, 1):
        doc = request.documents[idx]
        redis_feat = redis_features.get(doc.doc_id, {})

        ranked_docs.append(RankedDocument(
            doc_id=doc.doc_id,
            score=float(scores[idx]),
            rank=rank,
            features={
                'bm25_score': doc.bm25_score,
                'semantic_similarity': doc.semantic_similarity,
                'historical_ctr': redis_feat.get('query_doc_ctr', 0),
                'doc_impressions': redis_feat.get('doc_impressions', 0),
            }
        ))

    latency_ms = (time.perf_counter() - start_time) * 1000

    return RankResponse(
        query=request.query,
        ranked_documents=ranked_docs,
        latency_ms=round(latency_ms, 2),
        model_version="lambdamart-v1" if model else "random",
        experiment_variant=request.experiment_variant,
    )


@app.get("/features/{query}/{doc_id}")
async def get_features(query: str, doc_id: str):
    """Debug endpoint to inspect features for a query-doc pair."""
    qd_key = f"qd:{query}|{doc_id}"
    doc_key = f"doc:{doc_id}"

    qd_features = redis_client.hgetall(qd_key)
    doc_features = redis_client.hgetall(doc_key)

    return {
        "query_doc_features": {
            k.decode(): v.decode() for k, v in qd_features.items()
        } if qd_features else None,
        "doc_features": {
            k.decode(): v.decode() for k, v in doc_features.items()
        } if doc_features else None,
    }


@app.get("/stats")
async def get_stats():
    """Get feature store statistics."""
    info = redis_client.info()
    dbsize = redis_client.dbsize()

    return {
        "redis_keys": dbsize,
        "redis_memory_used": info.get('used_memory_human'),
        "redis_connected_clients": info.get('connected_clients'),
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
