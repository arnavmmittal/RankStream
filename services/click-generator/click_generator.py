"""
Click Stream Generator - Synthetic User Behavior Simulation

Generates realistic click events for a search ranking system:
- Users search for queries
- Results are shown at positions 1-10
- Users click based on position bias + relevance
- Dwell time correlates with satisfaction

This simulates the kind of data Google/Meta collects at scale.
"""

import json
import random
import time
import os
from datetime import datetime
from typing import Dict, Any
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9094')
EVENTS_PER_SECOND = int(os.getenv('EVENTS_PER_SECOND', '50'))
TOPIC_NAME = 'click-events'

# Simulated search queries (realistic distribution)
QUERY_TEMPLATES = [
    "best {product} 2024",
    "how to {action}",
    "{product} reviews",
    "{product} vs {product2}",
    "buy {product} online",
    "{topic} tutorial",
    "what is {topic}",
    "{location} restaurants",
    "{product} price",
    "top 10 {category}",
]

PRODUCTS = ['laptop', 'phone', 'headphones', 'camera', 'tablet', 'watch', 'speaker', 'monitor', 'keyboard', 'mouse']
ACTIONS = ['cook pasta', 'lose weight', 'learn python', 'invest money', 'build muscle', 'meditate', 'sleep better']
TOPICS = ['machine learning', 'blockchain', 'kubernetes', 'react', 'typescript', 'rust', 'golang']
CATEGORIES = ['movies', 'books', 'games', 'restaurants', 'hotels', 'stocks', 'podcasts']
LOCATIONS = ['new york', 'san francisco', 'seattle', 'austin', 'boston', 'chicago', 'denver']

# Position bias model (probability of examining position given user looked at previous)
# Based on real click data research - position 1 gets ~30% CTR, drops exponentially
POSITION_EXAMINATION_PROB = {
    1: 1.0,    # Always see position 1
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

# Simulated document corpus (doc_id -> relevance tier)
# In production, this would come from your search index
NUM_DOCUMENTS = 10000
DOCUMENT_RELEVANCE = {f"doc_{i}": random.choice(['high', 'medium', 'low']) for i in range(NUM_DOCUMENTS)}

# Click probability given examination and relevance
CLICK_PROB_GIVEN_EXAM = {
    'high': 0.65,
    'medium': 0.35,
    'low': 0.10,
}

# Dwell time distribution (milliseconds) - longer = more satisfied
DWELL_TIME_PARAMS = {
    'high': (15000, 45000),      # 15-45 seconds for relevant content
    'medium': (5000, 20000),     # 5-20 seconds
    'low': (1000, 5000),         # 1-5 seconds (quick bounce)
}


def generate_query() -> str:
    """Generate a realistic search query."""
    template = random.choice(QUERY_TEMPLATES)

    return template.format(
        product=random.choice(PRODUCTS),
        product2=random.choice(PRODUCTS),
        action=random.choice(ACTIONS),
        topic=random.choice(TOPICS),
        category=random.choice(CATEGORIES),
        location=random.choice(LOCATIONS),
    )


def generate_search_results(query: str, num_results: int = 10) -> list:
    """
    Generate search results for a query.
    Simulates a ranking system that's decent but not perfect.
    """
    # Sample documents with bias toward higher relevance at top
    docs = []
    for position in range(1, num_results + 1):
        # Higher positions more likely to have relevant docs (simulating existing ranker)
        if position <= 3:
            relevance_weights = {'high': 0.5, 'medium': 0.35, 'low': 0.15}
        elif position <= 6:
            relevance_weights = {'high': 0.25, 'medium': 0.45, 'low': 0.30}
        else:
            relevance_weights = {'high': 0.10, 'medium': 0.35, 'low': 0.55}

        # Pick a document with appropriate relevance
        target_relevance = random.choices(
            list(relevance_weights.keys()),
            weights=list(relevance_weights.values())
        )[0]

        # Find a doc with that relevance
        matching_docs = [d for d, r in DOCUMENT_RELEVANCE.items() if r == target_relevance]
        doc_id = random.choice(matching_docs)

        docs.append({
            'doc_id': doc_id,
            'position': position,
            'relevance': target_relevance,  # Ground truth (wouldn't have in production)
        })

    return docs


def simulate_user_session() -> list:
    """
    Simulate a complete user search session.
    Returns list of click events.
    """
    user_id = f"user_{random.randint(1, 50000)}"
    session_id = f"session_{fake.uuid4()[:8]}"
    query = generate_query()
    results = generate_search_results(query)
    timestamp = datetime.utcnow().isoformat()

    events = []

    # Impression event (user saw results)
    impression_event = {
        'event_type': 'impression',
        'user_id': user_id,
        'session_id': session_id,
        'query': query,
        'results': [{'doc_id': r['doc_id'], 'position': r['position']} for r in results],
        'timestamp': timestamp,
    }
    events.append(impression_event)

    # Simulate user examining and clicking
    for result in results:
        position = result['position']
        relevance = result['relevance']

        # Does user examine this position?
        if random.random() > POSITION_EXAMINATION_PROB[position]:
            break  # User stopped scrolling

        # Does user click given examination?
        if random.random() < CLICK_PROB_GIVEN_EXAM[relevance]:
            # Generate dwell time
            dwell_min, dwell_max = DWELL_TIME_PARAMS[relevance]
            dwell_time_ms = int(random.gauss(
                (dwell_min + dwell_max) / 2,
                (dwell_max - dwell_min) / 4
            ))
            dwell_time_ms = max(500, min(120000, dwell_time_ms))  # Clamp

            click_event = {
                'event_type': 'click',
                'user_id': user_id,
                'session_id': session_id,
                'query': query,
                'doc_id': result['doc_id'],
                'position': position,
                'dwell_time_ms': dwell_time_ms,
                'timestamp': timestamp,
            }
            events.append(click_event)

            # 30% chance user is satisfied and stops searching
            if dwell_time_ms > 20000 and random.random() < 0.3:
                break

    return events


def create_producer() -> KafkaProducer:
    """Create Kafka producer with retry logic."""
    max_retries = 30
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Ensure durability
                retries=3,
                batch_size=16384,
                linger_ms=10,  # Small batching for lower latency
            )
            print(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Waiting for Kafka... ({e})")
            time.sleep(2)

    raise Exception("Could not connect to Kafka")


def main():
    print("=" * 60)
    print("RankStream Click Generator")
    print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Events/sec: {EVENTS_PER_SECOND}")
    print("=" * 60)

    producer = create_producer()

    events_sent = 0
    start_time = time.time()

    try:
        while True:
            # Generate a session
            events = simulate_user_session()

            for event in events:
                # Use user_id as partition key for ordering guarantees
                key = event['user_id']
                producer.send(TOPIC_NAME, key=key, value=event)
                events_sent += 1

            # Rate limiting
            elapsed = time.time() - start_time
            expected_events = elapsed * EVENTS_PER_SECOND
            if events_sent > expected_events:
                sleep_time = (events_sent - expected_events) / EVENTS_PER_SECOND
                time.sleep(min(sleep_time, 0.1))

            # Progress logging
            if events_sent % 1000 == 0:
                rate = events_sent / elapsed
                print(f"Sent {events_sent:,} events ({rate:.1f}/sec)")

    except KeyboardInterrupt:
        print(f"\nShutting down. Total events sent: {events_sent:,}")
    finally:
        producer.flush()
        producer.close()


if __name__ == '__main__':
    main()
