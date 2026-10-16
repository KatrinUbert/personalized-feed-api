from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURES = ["text_similarity", "category_affinity", "user_clicks_log", "item_ctr",
            "item_views_log", "item_age_days", "age", "category", "region", "device"]
CATEGORICAL = ["category", "region", "device"]


def chronological_split(events: pd.DataFrame, cutoff: pd.Timestamp):
    ordered = events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    return ordered[ordered.timestamp < cutoff].copy(), ordered[ordered.timestamp >= cutoff].copy()


class FeatureBuilder:
    def __init__(self, users: pd.DataFrame, posts: pd.DataFrame):
        self.users = users.set_index("user_id").to_dict("index")
        self.posts = posts.set_index("post_id").to_dict("index")
        self.post_order = posts.post_id.to_list()
        self.post_positions = {post_id: i for i, post_id in enumerate(self.post_order)}
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=1, ngram_range=(1, 2))
        self.text_matrix = self.vectorizer.fit_transform(posts.text).toarray().astype(np.float32)
        self.user_vectors = {}
        self.user_clicks = Counter()
        self.category_clicks = defaultdict(Counter)
        self.item_views = Counter()
        self.item_clicks = Counter()
        self.seen = defaultdict(set)

    def feature_row(self, user_id: int, post_id: int, timestamp: pd.Timestamp) -> dict:
        post = self.posts[post_id]
        user = self.users.get(user_id, {"age": 35, "region": "unknown", "device": "unknown"})
        vector = self.user_vectors.get(user_id)
        similarity = 0.0
        if vector is not None:
            norm = float(np.linalg.norm(vector))
            if norm:
                similarity = float(np.dot(vector / norm, self.text_matrix[self.post_positions[post_id]]))
        clicks = self.user_clicks[user_id]
        return {"text_similarity": similarity,
                "category_affinity": self.category_clicks[user_id][post["category"]] / max(1, clicks),
                "user_clicks_log": float(np.log1p(clicks)),
                "item_ctr": self.popularity(post_id),
                "item_views_log": float(np.log1p(self.item_views[post_id])),
                "item_age_days": max(0, (timestamp - post["created_at"]).total_seconds() / 86400),
                "age": user["age"], "category": post["category"],
                "region": user["region"], "device": user["device"]}

    def popularity(self, post_id: int) -> float:
        return (self.item_clicks[post_id] + 1) / (self.item_views[post_id] + 5)

    def observe(self, user_id: int, post_id: int, clicked: int):
        self.seen[user_id].add(post_id)
        self.item_views[post_id] += 1
        if clicked:
            self.item_clicks[post_id] += 1
            self.user_clicks[user_id] += 1
            self.category_clicks[user_id][self.posts[post_id]["category"]] += 1
            vector = self.text_matrix[self.post_positions[post_id]]
            self.user_vectors[user_id] = self.user_vectors.get(user_id, np.zeros_like(vector)) + vector

    def training_features(self, events: pd.DataFrame):
        rows, labels = [], []
        for timestamp, group in events.sort_values(["timestamp", "event_id"]).groupby("timestamp", sort=True):
            for event in group.itertuples(index=False):
                rows.append(self.feature_row(event.user_id, event.post_id, timestamp))
                labels.append(event.clicked)
            for event in group.itertuples(index=False):
                self.observe(event.user_id, event.post_id, event.clicked)
        return pd.DataFrame(rows, columns=FEATURES), np.asarray(labels)

    def candidates(self, user_id: int, as_of: pd.Timestamp) -> list[int]:
        seen = self.seen.get(user_id, set())
        return sorted(post_id for post_id, post in self.posts.items()
                      if post_id not in seen and post["created_at"] <= as_of)
