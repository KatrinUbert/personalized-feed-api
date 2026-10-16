import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feedrec.features import FEATURES, FeatureBuilder


class Ranker:
    def __init__(self, model, features: FeatureBuilder, as_of: pd.Timestamp):
        self.model = model
        self.features = features
        self.as_of = as_of

    def recommend(self, user_id: int, k: int = 10, baseline: bool = False) -> dict:
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        candidates = self.features.candidates(user_id, self.as_of)
        cold_start = self.features.user_clicks[user_id] == 0
        use_popularity = baseline or cold_start
        if not candidates:
            return {"user_id": user_id, "as_of": self.as_of.isoformat(), "strategy": "empty", "items": []}
        if use_popularity:
            scores = np.array([self.features.popularity(post_id) for post_id in candidates])
        else:
            frame = pd.DataFrame([self.features.feature_row(user_id, post_id, self.as_of)
                                  for post_id in candidates], columns=FEATURES)
            scores = self.model.predict_proba(frame)[:, 1]
        ranked = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0]))[:k]
        items = [{"post_id": post_id, "title": self.features.posts[post_id]["title"],
                  "category": self.features.posts[post_id]["category"], "score": round(float(score), 6)}
                 for post_id, score in ranked]
        return {"user_id": user_id, "as_of": self.as_of.isoformat(),
                "strategy": "popularity" if use_popularity else "catboost", "items": items}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_ranker(ranker: Ranker, directory: Path, manifest: dict):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ranker.joblib"
    joblib.dump(ranker, path, compress=3)
    manifest["artifact_sha256"] = sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_ranker(directory: Path) -> tuple[Ranker, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "ranker.joblib"
    if sha256(path) != manifest["artifact_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    return joblib.load(path), manifest
