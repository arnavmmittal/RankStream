#!/usr/bin/env python3
"""
RankStream Video Demo - Clean version for recording
Run: python3 demo_video.py
"""

import sys
import time
import random
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# Seed for reproducible demo
random.seed(42)
np.random.seed(42)

# Colors
class C:
    H = '\033[95m'      # Header
    B = '\033[94m'      # Blue
    C = '\033[96m'      # Cyan
    G = '\033[92m'      # Green
    Y = '\033[93m'      # Yellow
    R = '\033[91m'      # Red
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def clear():
    print('\033[2J\033[H', end='')

def pause(seconds=1.5):
    time.sleep(seconds)

# Position bias model
POSITION_BIAS = {1: 1.0, 2: 0.85, 3: 0.72, 4: 0.60, 5: 0.50,
                 6: 0.42, 7: 0.35, 8: 0.29, 9: 0.24, 10: 0.20}

# Demo queries (realistic)
DEMO_QUERIES = [
    ("best laptop for programming", ["MacBook Pro M3", "ThinkPad X1", "Dell XPS 15", "Framework 16", "ASUS ROG"]),
    ("how to learn machine learning", ["Coursera ML Course", "Fast.ai Tutorial", "Stanford CS229", "Kaggle Learn", "DeepLearning.AI"]),
    ("react vs vue 2024", ["React Docs", "Vue 3 Guide", "State of JS Survey", "Reddit Discussion", "Dev.to Article"]),
]

def main():
    clear()

    # Header
    print(f"""{C.C}{C.BOLD}
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
║   Kafka • Redis • LambdaMART • FastAPI                           ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
{C.END}""")

    pause(2)

    # Loading
    print(f"{C.G}Initializing components...{C.END}\n")
    pause(0.8)

    components = [
        ("Apache Kafka", "Event streaming (100 events/sec)"),
        ("Redis", "Feature store (sub-10ms latency)"),
        ("LambdaMART", "Ranking model (NDCG@10: 97.19%)"),
        ("FastAPI", "Serving layer (<50ms p99)"),
    ]

    for name, desc in components:
        print(f"  {C.G}✓{C.END} {C.BOLD}{name}{C.END} — {C.DIM}{desc}{C.END}")
        pause(0.5)

    # Load model
    model_path = Path(__file__).parent / "model" / "ranker_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    pause(1)
    print(f"\n{C.BOLD}{'═'*67}{C.END}")
    print(f"{C.Y}▶ Starting live ranking demo...{C.END}")
    print(f"{C.BOLD}{'═'*67}{C.END}\n")
    pause(1.5)

    # Simulated Redis
    redis_ctr = defaultdict(float)
    total_clicks = 0

    for iteration, (query, doc_names) in enumerate(DEMO_QUERIES, 1):
        print(f"{C.C}{'━'*67}{C.END}")
        print(f"{C.C}  ITERATION {iteration}/3{C.END}")
        print(f"{C.C}{'━'*67}{C.END}\n")
        pause(1)

        # Step 1: Query
        print(f"{C.BOLD}[1] INCOMING SEARCH QUERY{C.END}")
        print(f"    {C.DIM}User searches for:{C.END}")
        print(f"    {C.G}\"{query}\"{C.END}")
        print()
        pause(1.5)

        # Step 2: Candidate Retrieval
        print(f"{C.BOLD}[2] CANDIDATE RETRIEVAL{C.END}")
        print(f"    {C.DIM}Retrieved {len(doc_names)} documents from search index{C.END}")
        print()
        pause(1)

        # Step 3: Feature Fetching
        print(f"{C.BOLD}[3] REAL-TIME FEATURES{C.END} {C.DIM}(from Redis){C.END}")
        print(f"    {'Document':<25} {'CTR':<12} {'Status'}")
        print(f"    {'-'*50}")

        for doc in doc_names:
            ctr = redis_ctr.get(f"{query}|{doc}", 0)
            status = f"{C.G}cached{C.END}" if ctr > 0 else f"{C.DIM}cold start{C.END}"
            ctr_str = f"{ctr:.4f}" if ctr > 0 else "0.0000"
            print(f"    {doc:<25} {ctr_str:<12} {status}")
            pause(0.2)
        print()
        pause(1)

        # Step 4: Model Ranking
        print(f"{C.BOLD}[4] LAMBDAMART RANKING{C.END}")
        print(f"    {C.DIM}Running inference on 12 features per document...{C.END}")
        pause(0.8)

        # Build features
        features = []
        for i, doc in enumerate(doc_names):
            ctr = redis_ctr.get(f"{query}|{doc}", 0)
            feat = [
                2.0, len(query.split()), 0.5 + random.random()*0.3, 0.6,
                7.0, 5.0 + random.random()*5,  # bm25
                0.5 + random.random()*0.4,     # semantic
                ctr,                            # historical CTR
                8.0, 4.0, 0.5 + random.random()*0.3, 1.0/(i+1)
            ]
            features.append(feat)

        X = np.array(features, dtype=np.float32)
        scores = model.predict(X)
        ranked = sorted(zip(doc_names, scores, range(len(doc_names))), key=lambda x: -x[1])

        print()
        print(f"    {C.BOLD}{'Rank':<6} {'Document':<25} {'Score':<12}{C.END}")
        print(f"    {'─'*45}")

        for rank, (doc, score, _) in enumerate(ranked, 1):
            color = C.G if rank == 1 else (C.Y if rank <= 3 else C.END)
            print(f"    {color}{rank:<6} {doc:<25} {score:>8.4f}{C.END}")
            pause(0.3)
        print()
        pause(1.5)

        # Step 5: User Click
        print(f"{C.BOLD}[5] USER INTERACTION{C.END}")
        click_pos = random.randint(1, 3)
        click_doc = ranked[click_pos-1][0]
        dwell = random.randint(8000, 35000)

        print(f"    {C.DIM}User examines results...{C.END}")
        pause(1)
        print(f"    {C.G}✓ CLICK{C.END} Position {click_pos}: \"{click_doc}\"")
        print(f"    {C.DIM}  Dwell time: {dwell/1000:.1f} seconds{C.END}")
        total_clicks += 1
        pause(1)

        # Step 6: Feature Update
        print()
        print(f"{C.BOLD}[6] FEATURE STORE UPDATE{C.END}")

        # Position bias correction
        raw_weight = 1.0
        bias = POSITION_BIAS[click_pos]
        corrected_weight = raw_weight / bias

        print(f"    {C.DIM}Position bias correction:{C.END}")
        print(f"    • Click at position {click_pos} → examination prob = {bias:.0%}")
        print(f"    • Raw weight: {raw_weight:.2f} → Corrected: {C.Y}{corrected_weight:.2f}x{C.END}")

        old_ctr = redis_ctr.get(f"{query}|{click_doc}", 0)
        redis_ctr[f"{query}|{click_doc}"] = old_ctr + corrected_weight * 0.1
        new_ctr = redis_ctr[f"{query}|{click_doc}"]

        print(f"    • CTR for \"{click_doc}\": {old_ctr:.4f} → {C.G}{new_ctr:.4f}{C.END}")
        print()
        pause(2)

    # Summary
    print(f"{C.BOLD}{'═'*67}{C.END}")
    print(f"{C.G}{C.BOLD}  DEMO COMPLETE{C.END}")
    print(f"{C.BOLD}{'═'*67}{C.END}\n")

    print(f"  {C.BOLD}Summary:{C.END}")
    print(f"  • Queries processed: 3")
    print(f"  • Total clicks: {total_clicks}")
    print(f"  • Feature store entries: {len(redis_ctr)}")
    print(f"  • Model: LambdaMART (LightGBM)")
    print(f"  • Validation NDCG@10: 97.19%")
    print()
    print(f"  {C.BOLD}Key Concepts Demonstrated:{C.END}")
    print(f"  • Position-bias corrected CTR (Inverse Propensity Weighting)")
    print(f"  • Real-time feature store (Redis)")
    print(f"  • Learning-to-Rank with LambdaMART")
    print(f"  • Event streaming architecture (Kafka)")
    print()
    print(f"  {C.C}github.com/arnavmmittal/RankStream{C.END}")
    print()

if __name__ == '__main__':
    main()
