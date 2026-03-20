"""
Real-Time Feature Pipeline

Consumes click events from Kafka and computes ranking features:
1. Position-bias corrected CTR per (query, document) pair
2. Document-level aggregate CTR
3. User behavior features (clicks per session, avg dwell time)
4. Query popularity metrics

These features are stored in Redis for low-latency serving.

Key Concepts Demonstrated:
- Position bias correction (critical for fair evaluation)
- Streaming aggregations with decay
- Feature store pattern used at Google/Meta
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from kafka import KafkaConsumer
import redis

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9094')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
TOPIC_NAME = 'click-events'
CONSUMER_GROUP = 'feature-pipeline'

# Position bias coefficients (learned from historical data)
# These represent P(examine | position) - used to correct CTR
# In production, you'd learn these from randomization experiments
POSITION_BIAS = {
    1: 1.0,
    2: 0.85,
    3: 0.72,
    4: 0.60,
    5: 0.50,
    6: 0.42,
    7: 0.35,
    8: 0.29,
    9: 0.24,
    10: 0.20,
}

# Feature TTLs (seconds)
TTL_QUERY_DOC_FEATURES = 3600 * 24  # 24 hours
TTL_DOC_FEATURES = 3600 * 24 * 7     # 7 days
TTL_USER_FEATURES = 3600 * 24        # 24 hours
TTL_QUERY_FEATURES = 3600 * 24       # 24 hours


class FeaturePipeline:
    """
    Streaming feature computation engine.

    Implements the feature store pattern:
    - Raw events → Aggregations → Feature vectors → Redis
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.events_processed = 0
        self.last_log_time = time.time()

        # In-memory accumulators (flushed to Redis periodically)
        # In production, you'd use Flink state or Redis Streams
        self.query_doc_impressions = defaultdict(int)
        self.query_doc_clicks = defaultdict(int)
        self.query_doc_dwell_sum = defaultdict(float)

        self.doc_impressions = defaultdict(int)
        self.doc_clicks = defaultdict(int)

        self.query_count = defaultdict(int)

    def process_event(self, event: Dict[str, Any]):
        """Process a single click or impression event."""
        event_type = event.get('event_type')

        if event_type == 'impression':
            self._process_impression(event)
        elif event_type == 'click':
            self._process_click(event)

        self.events_processed += 1

        # Periodic flush and logging
        if self.events_processed % 100 == 0:
            self._flush_to_redis()

        if time.time() - self.last_log_time > 10:
            print(f"Processed {self.events_processed:,} events")
            self.last_log_time = time.time()

    def _process_impression(self, event: Dict[str, Any]):
        """Process impression event - user saw search results."""
        query = event['query']
        results = event.get('results', [])

        self.query_count[query] += 1

        for result in results:
            doc_id = result['doc_id']
            position = result['position']

            # Key for (query, doc) pair
            qd_key = f"{query}|{doc_id}"

            # Increment impressions with position bias weighting
            # This gives us "examination-adjusted" impressions
            position_weight = POSITION_BIAS.get(position, 0.1)
            self.query_doc_impressions[qd_key] += position_weight
            self.doc_impressions[doc_id] += position_weight

    def _process_click(self, event: Dict[str, Any]):
        """
        Process click event.

        Key insight: We weight clicks by 1/position_bias to correct for
        position effects. This gives us an unbiased estimate of document
        relevance (what Google calls "counterfactual" estimation).
        """
        query = event['query']
        doc_id = event['doc_id']
        position = event['position']
        dwell_time_ms = event.get('dwell_time_ms', 0)

        qd_key = f"{query}|{doc_id}"

        # Position-bias corrected click weight
        # Intuition: a click at position 10 is more meaningful than at position 1
        position_weight = POSITION_BIAS.get(position, 0.1)
        corrected_click_weight = 1.0 / position_weight  # Inverse propensity weighting

        # But we cap it to avoid extreme weights
        corrected_click_weight = min(corrected_click_weight, 5.0)

        self.query_doc_clicks[qd_key] += corrected_click_weight
        self.doc_clicks[doc_id] += corrected_click_weight

        # Dwell time is a satisfaction signal
        self.query_doc_dwell_sum[qd_key] += dwell_time_ms

        # Store user features
        self._update_user_features(event)

    def _update_user_features(self, event: Dict[str, Any]):
        """Update per-user features in Redis."""
        user_id = event['user_id']
        dwell_time_ms = event.get('dwell_time_ms', 0)

        user_key = f"user:{user_id}:features"

        # Use Redis hash for user features
        pipe = self.redis.pipeline()

        # Increment click count
        pipe.hincrby(user_key, 'total_clicks', 1)

        # Update average dwell time (using running average approximation)
        pipe.hincrbyfloat(user_key, 'dwell_sum', dwell_time_ms)

        # Update last active timestamp
        pipe.hset(user_key, 'last_active', datetime.utcnow().isoformat())

        # Set TTL
        pipe.expire(user_key, TTL_USER_FEATURES)

        pipe.execute()

    def _flush_to_redis(self):
        """Flush accumulated features to Redis."""
        pipe = self.redis.pipeline()

        # Flush query-document features
        for qd_key, impressions in self.query_doc_impressions.items():
            clicks = self.query_doc_clicks.get(qd_key, 0)
            dwell_sum = self.query_doc_dwell_sum.get(qd_key, 0)

            redis_key = f"qd:{qd_key}"

            # Compute CTR (position-bias corrected)
            ctr = clicks / max(impressions, 1)

            # Average dwell time
            avg_dwell = dwell_sum / max(clicks, 1) if clicks > 0 else 0

            # Store as hash
            pipe.hset(redis_key, mapping={
                'impressions': impressions,
                'clicks': clicks,
                'ctr': round(ctr, 4),
                'avg_dwell_ms': round(avg_dwell, 2),
                'updated_at': datetime.utcnow().isoformat(),
            })
            pipe.expire(redis_key, TTL_QUERY_DOC_FEATURES)

        # Flush document-level features
        for doc_id, impressions in self.doc_impressions.items():
            clicks = self.doc_clicks.get(doc_id, 0)

            redis_key = f"doc:{doc_id}"
            ctr = clicks / max(impressions, 1)

            pipe.hset(redis_key, mapping={
                'impressions': impressions,
                'clicks': clicks,
                'ctr': round(ctr, 4),
            })
            pipe.expire(redis_key, TTL_DOC_FEATURES)

        # Flush query popularity
        for query, count in self.query_count.items():
            redis_key = f"query:{query}"
            pipe.hincrby(redis_key, 'search_count', count)
            pipe.expire(redis_key, TTL_QUERY_FEATURES)

        pipe.execute()

        # Clear accumulators
        self.query_doc_impressions.clear()
        self.query_doc_clicks.clear()
        self.query_doc_dwell_sum.clear()
        self.doc_impressions.clear()
        self.doc_clicks.clear()
        self.query_count.clear()

    def get_features_for_ranking(self, query: str, doc_ids: list) -> Dict[str, Dict]:
        """
        Fetch features for ranking a set of documents.
        This is what the ranking API calls.
        """
        features = {}

        pipe = self.redis.pipeline()

        for doc_id in doc_ids:
            qd_key = f"qd:{query}|{doc_id}"
            doc_key = f"doc:{doc_id}"
            pipe.hgetall(qd_key)
            pipe.hgetall(doc_key)

        results = pipe.execute()

        for i, doc_id in enumerate(doc_ids):
            qd_features = results[i * 2]
            doc_features = results[i * 2 + 1]

            features[doc_id] = {
                'query_doc_ctr': float(qd_features.get(b'ctr', 0)),
                'query_doc_impressions': int(qd_features.get(b'impressions', 0)),
                'avg_dwell_ms': float(qd_features.get(b'avg_dwell_ms', 0)),
                'doc_ctr': float(doc_features.get(b'ctr', 0)),
                'doc_impressions': int(doc_features.get(b'impressions', 0)),
            }

        return features


def create_consumer() -> KafkaConsumer:
    """Create Kafka consumer with retry logic."""
    max_retries = 30

    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
            )
            print(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return consumer
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Waiting for Kafka... ({e})")
            time.sleep(2)

    raise Exception("Could not connect to Kafka")


def create_redis_client() -> redis.Redis:
    """Create Redis client with retry logic."""
    max_retries = 30

    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=False,  # Keep bytes for performance
            )
            client.ping()
            print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return client
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Waiting for Redis... ({e})")
            time.sleep(2)

    raise Exception("Could not connect to Redis")


def main():
    print("=" * 60)
    print("RankStream Feature Pipeline")
    print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print("=" * 60)

    redis_client = create_redis_client()
    consumer = create_consumer()
    pipeline = FeaturePipeline(redis_client)

    print("Starting feature computation...")

    try:
        for message in consumer:
            event = message.value
            pipeline.process_event(event)

    except KeyboardInterrupt:
        print(f"\nShutting down. Processed {pipeline.events_processed:,} events")
    finally:
        consumer.close()


if __name__ == '__main__':
    main()
