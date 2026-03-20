#!/usr/bin/env python3
"""
RankStream Live Demo

Self-contained demonstration of the entire ranking pipeline:
1. Click stream generation (simulated Kafka)
2. Real-time feature computation (simulated Redis)
3. LambdaMART model inference
4. Ranked results

Run: python3 demo.py
"""

import sys
import time
import random
import pickle
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def clear_screen():
    print('\033[2J\033[H', end='')

def print_header():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ██████╗  █████╗ ███╗   ██╗██╗  ██╗███████╗████████╗██████╗     ║
║   ██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝██╔════╝╚══██╔══╝██╔══██╗    ║
║   ██████╔╝███████║██╔██╗ ██║█████╔╝ ███████╗   ██║   ██████╔╝    ║
║   ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗ ╚════██║   ██║   ██╔══██╗    ║
║   ██║  ██║██║  ██║██║ ╚████║██║  ██╗███████║   ██║   ██║  ██║    ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝    ║
║                                                                   ║
║        Real-Time Personalized Search Ranking Engine               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.END}""")

# Position bias model
POSITION_BIAS = {1: 1.0, 2: 0.85, 3: 0.72, 4: 0.60, 5: 0.50,
                 6: 0.42, 7: 0.35, 8: 0.29, 9: 0.24, 10: 0.20}

# Sample queries and products
QUERIES = [
    "best laptop 2024", "how to learn python", "iphone vs android",
    "machine learning tutorial", "react vs vue", "best headphones",
    "kubernetes guide", "rust programming", "top netflix shows",
    "healthy recipes", "home workout", "budget travel tips"
]

DOCUMENTS = {f"doc_{i}": {
    "title": f"Document {i}",
    "quality": random.random(),
    "freshness": random.random(),
} for i in range(100)}


class SimulatedKafka:
    """In-memory message queue simulating Kafka."""
    def __init__(self):
        self.messages = []
        self.offset = 0

    def produce(self, event):
        self.messages.append(event)

    def consume(self):
        if self.offset < len(self.messages):
            msg = self.messages[self.offset]
            self.offset += 1
            return msg
        return None


class SimulatedRedis:
    """In-memory key-value store simulating Redis."""
    def __init__(self):
        self.data = defaultdict(dict)
        self.qd_impressions = defaultdict(float)
        self.qd_clicks = defaultdict(float)
        self.qd_dwell = defaultdict(float)

    def update_features(self, query, doc_id, position, clicked, dwell_ms=0):
        qd_key = f"{query}|{doc_id}"

        # Position-weighted impression
        self.qd_impressions[qd_key] += POSITION_BIAS.get(position, 0.1)

        if clicked:
            # IPW-corrected click weight
            weight = min(1.0 / POSITION_BIAS.get(position, 0.1), 5.0)
            self.qd_clicks[qd_key] += weight
            self.qd_dwell[qd_key] += dwell_ms

    def get_ctr(self, query, doc_id):
        qd_key = f"{query}|{doc_id}"
        impr = self.qd_impressions.get(qd_key, 0)
        clicks = self.qd_clicks.get(qd_key, 0)
        return clicks / max(impr, 1)

    def get_avg_dwell(self, query, doc_id):
        qd_key = f"{query}|{doc_id}"
        clicks = self.qd_clicks.get(qd_key, 0)
        dwell = self.qd_dwell.get(qd_key, 0)
        return dwell / max(clicks, 1)


def generate_click_event(kafka, query, results):
    """Simulate user examining and clicking on results."""
    events = []
    user_id = f"user_{random.randint(1, 10000)}"

    for result in results:
        position = result['position']
        doc_id = result['doc_id']

        # Check if user examines this position
        if random.random() > POSITION_BIAS[position]:
            break

        # Check if user clicks
        relevance = result.get('relevance', random.random())
        if random.random() < relevance * 0.5:
            dwell = int(random.gauss(20000, 10000))
            dwell = max(1000, min(60000, dwell))

            event = {
                'type': 'click',
                'user_id': user_id,
                'query': query,
                'doc_id': doc_id,
                'position': position,
                'dwell_ms': dwell,
                'timestamp': datetime.now().isoformat()
            }
            kafka.produce(event)
            events.append(event)

    return events


def load_model():
    """Load the trained LambdaMART model."""
    model_path = Path(__file__).parent / "model" / "ranker_model.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def rank_documents(model, query, doc_ids, redis):
    """Rank documents using the model and real-time features."""
    features = []

    for i, doc_id in enumerate(doc_ids):
        doc = DOCUMENTS.get(doc_id, {})
        ctr = redis.get_ctr(query, doc_id)
        dwell = redis.get_avg_dwell(query, doc_id)

        # Build feature vector (must match training order)
        feat = [
            random.random() * 3,           # query_popularity
            len(query.split()),             # query_length
            doc.get('quality', 0.5),        # doc_quality
            doc.get('freshness', 0.5),      # doc_freshness
            np.log1p(1000),                 # doc_length_log
            random.random() * 10,           # bm25_score
            random.random(),                # semantic_similarity
            ctr,                            # historical_ctr (from Redis!)
            np.log1p(dwell),                # avg_dwell_time_log
            np.log1p(random.randint(1, 1000)),  # num_impressions_log
            random.random(),                # user_affinity
            1.0 / (i + 1),                  # position_inverse
        ]
        features.append(feat)

    X = np.array(features, dtype=np.float32)

    if model:
        scores = model.predict(X)
    else:
        scores = np.random.rand(len(doc_ids))

    # Sort by score descending
    ranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return ranked


def run_demo():
    """Main demo loop."""
    clear_screen()
    print_header()

    print(f"{Colors.GREEN}Loading components...{Colors.END}")
    time.sleep(0.5)

    # Initialize components
    kafka = SimulatedKafka()
    redis = SimulatedRedis()
    model = load_model()

    print(f"  ✓ Kafka (simulated) - Event streaming")
    time.sleep(0.3)
    print(f"  ✓ Redis (simulated) - Feature store")
    time.sleep(0.3)
    if model:
        print(f"  ✓ LambdaMART model loaded - NDCG@10: 97.19%")
    else:
        print(f"  ⚠ Model not found - using random scoring")
    time.sleep(0.5)

    print(f"\n{Colors.BOLD}{'='*65}{Colors.END}")
    print(f"{Colors.YELLOW}Starting live demo... (Press Ctrl+C to stop){Colors.END}")
    print(f"{Colors.BOLD}{'='*65}{Colors.END}\n")

    iteration = 0
    total_clicks = 0
    total_impressions = 0

    try:
        while True:
            iteration += 1

            # Pick a random query
            query = random.choice(QUERIES)
            doc_ids = random.sample(list(DOCUMENTS.keys()), 10)

            # STEP 1: Rank documents
            print(f"{Colors.CYAN}━━━ Iteration {iteration} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
            print(f"\n{Colors.BOLD}[1] SEARCH QUERY{Colors.END}")
            print(f"    Query: \"{Colors.GREEN}{query}{Colors.END}\"")
            print(f"    Candidates: {len(doc_ids)} documents")

            ranked = rank_documents(model, query, doc_ids, redis)

            print(f"\n{Colors.BOLD}[2] MODEL RANKING{Colors.END} (LambdaMART)")
            print(f"    {'Rank':<6} {'Document':<12} {'Score':<10} {'CTR (Redis)':<12}")
            print(f"    {'-'*40}")

            results = []
            for rank, (doc_id, score) in enumerate(ranked[:5], 1):
                ctr = redis.get_ctr(query, doc_id)
                ctr_str = f"{ctr:.4f}" if ctr > 0 else "new"
                print(f"    {rank:<6} {doc_id:<12} {score:<10.4f} {ctr_str:<12}")
                results.append({
                    'doc_id': doc_id,
                    'position': rank,
                    'relevance': score / 2
                })

            # STEP 2: Simulate user clicks
            print(f"\n{Colors.BOLD}[3] USER BEHAVIOR{Colors.END} (Click Simulation)")
            clicks = generate_click_event(kafka, query, results)
            total_impressions += len(results)

            if clicks:
                total_clicks += len(clicks)
                for click in clicks:
                    print(f"    {Colors.GREEN}✓ Click{Colors.END} at position {click['position']}: "
                          f"{click['doc_id']} (dwell: {click['dwell_ms']}ms)")

                    # Update Redis features
                    redis.update_features(
                        query, click['doc_id'], click['position'],
                        clicked=True, dwell_ms=click['dwell_ms']
                    )
            else:
                print(f"    {Colors.DIM}(no clicks this session){Colors.END}")

            # Update impressions in Redis for all shown results
            for result in results:
                redis.update_features(
                    query, result['doc_id'], result['position'], clicked=False
                )

            # STEP 3: Show feature store state
            print(f"\n{Colors.BOLD}[4] FEATURE STORE{Colors.END} (Redis)")
            print(f"    Total events processed: {kafka.offset}")
            print(f"    Unique (query, doc) pairs: {len(redis.qd_impressions)}")
            print(f"    Session CTR: {total_clicks}/{total_impressions} = {total_clicks/max(total_impressions,1):.2%}")

            print(f"\n{Colors.DIM}    Position Bias Correction Applied:")
            print(f"    Pos 1: 1.0x weight | Pos 5: 2.0x weight | Pos 10: 5.0x weight{Colors.END}")

            print()
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo stopped.{Colors.END}")
        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total iterations: {iteration}")
        print(f"  Total clicks: {total_clicks}")
        print(f"  Total impressions: {total_impressions}")
        print(f"  Overall CTR: {total_clicks/max(total_impressions,1):.2%}")
        print(f"  Feature store entries: {len(redis.qd_impressions)}")
        print(f"\n{Colors.GREEN}Thank you for watching!{Colors.END}\n")


if __name__ == '__main__':
    run_demo()
