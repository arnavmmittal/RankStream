"""
Learning-to-Rank Model Training

Trains a LightGBM LambdaMART model for search ranking.

LambdaMART is the industry standard for search ranking:
- Used at Google, Bing, Airbnb, LinkedIn
- Optimizes directly for ranking metrics (NDCG)
- Handles position bias through feature engineering

This script:
1. Generates synthetic training data
2. Engineers ranking features
3. Trains LambdaMART model
4. Evaluates with NDCG@K
5. Exports to ONNX for serving
"""

import json
import random
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import pickle
from pathlib import Path

# Reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_QUERIES = 5000
DOCS_PER_QUERY = 10
MODEL_OUTPUT_PATH = Path(__file__).parent / "ranker_model.pkl"
FEATURE_NAMES_PATH = Path(__file__).parent / "feature_names.json"


def generate_synthetic_training_data(
    num_queries: int = NUM_QUERIES,
    docs_per_query: int = DOCS_PER_QUERY
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for ranking.

    Returns:
        X: Feature matrix (num_queries * docs_per_query, num_features)
        y: Relevance labels (0-4 scale, like Google's rating scale)
        groups: Query group sizes for LambdaRank
    """
    print(f"Generating {num_queries:,} queries with {docs_per_query} docs each...")

    features = []
    labels = []
    groups = []

    for q in range(num_queries):
        query_features = []
        query_labels = []

        # Query-level features (same for all docs in query)
        query_popularity = np.random.exponential(1.0)  # Power law
        query_length = np.random.randint(1, 8)

        for d in range(docs_per_query):
            # Document features
            doc_quality = np.random.beta(2, 5)  # Most docs are low quality
            doc_freshness = np.random.exponential(0.5)  # Decay over time
            doc_length = np.random.lognormal(6, 1)  # Word count

            # Query-document interaction features
            bm25_score = np.random.exponential(5.0)  # Text match score
            semantic_similarity = np.random.beta(2, 3)  # Embedding similarity

            # Historical engagement features (from feature store)
            historical_ctr = np.random.beta(1, 10)  # Most docs have low CTR
            avg_dwell_time = np.random.exponential(15000)  # milliseconds
            num_impressions = np.random.exponential(100)

            # User-document affinity (personalization)
            user_affinity = np.random.beta(2, 5)

            # Position in current ranking (feature, not label)
            current_position = d + 1

            feature_vector = [
                query_popularity,
                query_length,
                doc_quality,
                doc_freshness,
                np.log1p(doc_length),  # Log transform
                bm25_score,
                semantic_similarity,
                historical_ctr,
                np.log1p(avg_dwell_time),
                np.log1p(num_impressions),
                user_affinity,
                1.0 / current_position,  # Position feature (inverse)
            ]

            # Generate relevance label based on features
            # This simulates ground truth from human ratings or clicks
            relevance_score = (
                0.3 * bm25_score / 10 +
                0.25 * semantic_similarity +
                0.2 * doc_quality +
                0.15 * historical_ctr * 10 +
                0.1 * user_affinity +
                np.random.normal(0, 0.1)  # Noise
            )

            # Convert to 0-4 scale (Google-style relevance grades)
            # 0: Not relevant, 1: Slightly, 2: Moderately, 3: Highly, 4: Perfect
            label = int(np.clip(relevance_score * 4, 0, 4))

            query_features.append(feature_vector)
            query_labels.append(label)

        features.extend(query_features)
        labels.extend(query_labels)
        groups.append(docs_per_query)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    groups = np.array(groups, dtype=np.int32)

    print(f"Generated {len(X):,} training examples")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, groups


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    This is THE metric for ranking quality - what Google optimizes.
    """
    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1][:k]
    y_true_sorted = y_true[order]

    # DCG
    gains = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(gains)) + 2)
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_gains = 2 ** y_true[ideal_order] - 1
    ideal_dcg = np.sum(ideal_gains / discounts[:len(ideal_gains)])

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def evaluate_model(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray
) -> Dict[str, float]:
    """Evaluate ranking model with NDCG@K metrics."""
    predictions = model.predict(X)

    ndcg_scores = {f'ndcg@{k}': [] for k in [1, 3, 5, 10]}

    idx = 0
    for group_size in groups:
        group_y = y[idx:idx + group_size]
        group_pred = predictions[idx:idx + group_size]

        for k in [1, 3, 5, 10]:
            if len(group_y) >= k:
                ndcg = compute_ndcg(group_y, group_pred, k)
                ndcg_scores[f'ndcg@{k}'].append(ndcg)

        idx += group_size

    return {k: np.mean(v) for k, v in ndcg_scores.items()}


def train_lambdamart(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
) -> lgb.Booster:
    """
    Train LambdaMART ranking model.

    LambdaMART uses gradient boosted trees optimized for ranking.
    The "lambda" refers to the gradient computation that considers
    pairwise document comparisons within each query.
    """
    print("\nTraining LambdaMART model...")

    # LightGBM ranking dataset
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=groups_val,
        reference=train_data,
    )

    # LambdaMART parameters
    params = {
        'objective': 'lambdarank',  # LambdaMART objective
        'metric': 'ndcg',           # Optimize for NDCG
        'ndcg_eval_at': [1, 3, 5, 10],
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
    }

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def main():
    print("=" * 60)
    print("RankStream Model Training")
    print("Algorithm: LambdaMART (LightGBM)")
    print("=" * 60)

    # Generate data
    X, y, groups = generate_synthetic_training_data()

    # Split by queries (not individual examples)
    num_queries = len(groups)
    train_queries = int(num_queries * 0.8)

    train_size = sum(groups[:train_queries])

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    groups_train = groups[:train_queries]
    groups_val = groups[train_queries:]

    print(f"\nTrain: {len(X_train):,} examples, {len(groups_train):,} queries")
    print(f"Val: {len(X_val):,} examples, {len(groups_val):,} queries")

    # Train model
    model = train_lambdamart(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val
    )

    # Evaluate
    print("\n" + "=" * 40)
    print("Evaluation Results")
    print("=" * 40)

    train_metrics = evaluate_model(model, X_train, y_train, groups_train)
    val_metrics = evaluate_model(model, X_val, y_val, groups_val)

    print("\nTraining Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nValidation Set:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Feature importance
    print("\n" + "=" * 40)
    print("Feature Importance")
    print("=" * 40)

    feature_names = [
        'query_popularity',
        'query_length',
        'doc_quality',
        'doc_freshness',
        'doc_length_log',
        'bm25_score',
        'semantic_similarity',
        'historical_ctr',
        'avg_dwell_time_log',
        'num_impressions_log',
        'user_affinity',
        'position_inverse',
    ]

    importance = model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(importance)[::-1]

    for idx in sorted_idx:
        print(f"  {feature_names[idx]}: {importance[idx]:.1f}")

    # Save model
    print(f"\nSaving model to {MODEL_OUTPUT_PATH}")
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Save feature names
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_names, f)

    print("\nTraining complete!")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    return model


if __name__ == '__main__':
    main()
