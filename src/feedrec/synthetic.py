from pathlib import Path

import numpy as np
import pandas as pd


TOPICS = {
    "technology": "software python cloud engineering computer science robotics systems",
    "sport": "fitness football running training team match marathon health",
    "travel": "travel city mountains beach route journey hotel adventure",
    "finance": "economy investment budget business savings markets banking money",
    "culture": "cinema music theatre literature painting exhibition book art",
    "food": "cooking recipe restaurant coffee cuisine ingredients baking dinner",
}


def generate_dataset(output: Path, users: int = 120, posts: int = 360,
                     days: int = 30, seed: int = 42) -> dict:
    if users < 8 or posts < 30 or days < 10:
        raise ValueError("Use at least 8 users, 30 posts and 10 days")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    start = pd.Timestamp("2025-01-01")
    user_rows = []
    preferences = {}
    for user_id in range(1, users + 1):
        preferences[user_id] = rng.dirichlet(np.full(len(topics), 0.22))
        user_rows.append({"user_id": user_id, "age": int(rng.integers(18, 65)),
                          "region": str(rng.choice(["north", "south", "central"])),
                          "device": str(rng.choice(["mobile", "desktop"]))})
    post_rows = []
    quality = {}
    for post_id in range(1, posts + 1):
        category = topics[(post_id - 1) % len(topics)]
        created_day = -5 if post_id <= int(posts * 0.7) else int(rng.integers(0, days - 2))
        quality[post_id] = float(rng.uniform(0.7, 1.3))
        post_rows.append({"post_id": post_id, "category": category,
                          "title": f"{category.title()}: story {post_id}",
                          "text": f"{TOPICS[category]} {TOPICS[category]} story {post_id}",
                          "created_at": start + pd.Timedelta(days=created_day)})
    posts_df = pd.DataFrame(post_rows)
    interactions = []
    seen = {user_id: set() for user_id in preferences}
    for day in range(days):
        day_start = start + pd.Timedelta(days=day)
        available = posts_df[posts_df.created_at <= day_start]
        for user_id in preferences:
            candidates = available[~available.post_id.isin(seen[user_id])]
            count = min(6, len(candidates))
            if not count:
                continue
            sampled = candidates.iloc[rng.choice(len(candidates), count, replace=False)]
            for row in sampled.itertuples(index=False):
                affinity = preferences[user_id][topics.index(row.category)]
                probability = min(0.92, 0.025 + 1.35 * affinity * quality[row.post_id])
                timestamp = day_start + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
                interactions.append({"event_id": len(interactions) + 1,
                                     "user_id": user_id, "post_id": row.post_id,
                                     "clicked": int(rng.random() < probability),
                                     "timestamp": timestamp})
                seen[user_id].add(row.post_id)
    user_df = pd.DataFrame(user_rows)
    events = pd.DataFrame(interactions).sort_values(["timestamp", "event_id"])
    for name, frame in {"users": user_df, "posts": posts_df, "interactions": events}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    return {"users": len(user_df), "posts": len(posts_df), "interactions": len(events),
            "seed": seed, "cutoff": str(start + pd.Timedelta(days=int(days * 0.8)))}
