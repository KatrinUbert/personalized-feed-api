import hashlib
import json
import platform
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from feedrec.features import CATEGORICAL, FEATURES, FeatureBuilder, chronological_split
from feedrec.ranking import Ranker, save_ranker
from feedrec.storage import read_dataset


def evaluate_ranking(ranker: Ranker, holdout: pd.DataFrame, k: int) -> dict:
    values = {"catboost": [], "popularity": []}
    hit_rates = {"catboost": [], "popularity": []}
    candidate_counts, relevant_counts = [], []
    positives = holdout[holdout.clicked == 1]
    for user_id, group in positives.groupby("user_id", sort=True):
        candidates = set(ranker.features.candidates(int(user_id), ranker.as_of))
        relevant = set(group.post_id) & candidates
        if not relevant:
            continue
        candidate_counts.append(len(candidates))
        relevant_counts.append(len(relevant))
        for strategy in values:
            response = ranker.recommend(int(user_id), k, baseline=strategy == "popularity")
            recommended = {item["post_id"] for item in response["items"]}
            hits = len(recommended & relevant)
            values[strategy].append(hits / len(relevant))
            hit_rates[strategy].append(float(hits > 0))
    if not candidate_counts:
        raise ValueError("No eligible held-out positives; use a larger dataset or earlier cutoff")
    return {"k": k, "evaluated_users": len(candidate_counts),
            "mean_candidates": float(np.mean(candidate_counts)),
            "mean_relevant_posts": float(np.mean(relevant_counts)),
            "catboost": {"recall_at_k": float(np.mean(values["catboost"])),
                         "hit_rate_at_k": float(np.mean(hit_rates["catboost"]))},
            "popularity": {"recall_at_k": float(np.mean(values["popularity"])),
                           "hit_rate_at_k": float(np.mean(hit_rates["popularity"]))}}


def train(cutoff: str, artifact_dir: Path, url: str | None = None,
          seed: int = 42, iterations: int = 180, k: int = 10) -> dict:
    users, posts, events = read_dataset(url)
    cutoff_time = pd.Timestamp(cutoff)
    train_events, holdout = chronological_split(events, cutoff_time)
    if train_events.empty or holdout.empty:
        raise ValueError("Cutoff must leave non-empty train and holdout sets")
    eligible_posts = posts[posts.created_at < cutoff_time].sort_values("post_id").copy()
    if not set(train_events.post_id).issubset(set(eligible_posts.post_id)):
        raise ValueError("Training events contain a post unavailable before the cutoff")
    if train_events.clicked.nunique() != 2:
        raise ValueError("Training data must contain both clicked and unclicked impressions")
    features = FeatureBuilder(users, eligible_posts)
    x_train, y_train = features.training_features(train_events)
    model = CatBoostClassifier(iterations=iterations, depth=5, learning_rate=0.07,
                               loss_function="Logloss", random_seed=seed,
                               cat_features=CATEGORICAL, thread_count=2,
                               verbose=False, allow_writing_files=False)
    model.fit(x_train, y_train)
    ranker = Ranker(model, features, cutoff_time)
    metrics = evaluate_ranking(ranker, holdout, k)
    eligible_holdout = holdout[holdout.post_id.isin(features.posts)]
    test_frame = pd.DataFrame([features.feature_row(row.user_id, row.post_id, cutoff_time)
                               for row in eligible_holdout.itertuples(index=False)], columns=FEATURES)
    if not test_frame.empty and eligible_holdout.clicked.nunique() == 2:
        predictions = model.predict_proba(test_frame)[:, 1]
        metrics["impression_roc_auc"] = float(roc_auc_score(eligible_holdout.clicked, predictions))
        metrics["impression_average_precision"] = float(average_precision_score(eligible_holdout.clicked, predictions))
    fingerprint = hashlib.sha256()
    for frame in [users.sort_values("user_id"), posts.sort_values("post_id"), events.sort_values("event_id")]:
        fingerprint.update(frame.to_csv(index=False).encode("utf-8"))
    manifest = {"schema_version": 1, "model": "CatBoostClassifier", "seed": seed,
                "iterations": iterations, "cutoff": cutoff_time.isoformat(),
                "train_events": len(train_events), "holdout_events": len(holdout),
                "catalog_at_cutoff": len(eligible_posts),
                "dataset_sha256": fingerprint.hexdigest(), "features": FEATURES,
                "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                             "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__}, "metrics": metrics}
    save_ranker(ranker, artifact_dir, manifest)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(artifact_dir / "feature_importance.csv", index=False)
    return manifest
