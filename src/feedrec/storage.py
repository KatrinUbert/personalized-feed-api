import os
from pathlib import Path

import pandas as pd
from sqlalchemy import (Column, DateTime, ForeignKey, Integer, MetaData,
                        String, Table, Text, create_engine, select)


metadata = MetaData()
users_table = Table("users", metadata, Column("user_id", Integer, primary_key=True),
                    Column("age", Integer, nullable=False),
                    Column("region", String(40), nullable=False),
                    Column("device", String(40), nullable=False))
posts_table = Table("posts", metadata, Column("post_id", Integer, primary_key=True),
                    Column("category", String(60), nullable=False),
                    Column("title", Text, nullable=False), Column("text", Text, nullable=False),
                    Column("created_at", DateTime, nullable=False))
events_table = Table("interactions", metadata,
                     Column("event_id", Integer, primary_key=True),
                     Column("user_id", ForeignKey("users.user_id"), nullable=False, index=True),
                     Column("post_id", ForeignKey("posts.post_id"), nullable=False),
                     Column("clicked", Integer, nullable=False),
                     Column("timestamp", DateTime, nullable=False, index=True))


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///data/feed.db")


def get_engine(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def import_dataset(directory: Path, url: str | None = None) -> dict:
    engine = get_engine(url)
    metadata.create_all(engine)
    frames = {
        "users": pd.read_csv(directory / "users.csv"),
        "posts": pd.read_csv(directory / "posts.csv", parse_dates=["created_at"]),
        "interactions": pd.read_csv(directory / "interactions.csv", parse_dates=["timestamp"]),
    }
    with engine.begin() as connection:
        for table in [events_table, posts_table, users_table]:
            connection.execute(table.delete())
        for name, frame in frames.items():
            frame.to_sql(name, connection, if_exists="append", index=False, chunksize=1000)
    engine.dispose()
    return {name: len(frame) for name, frame in frames.items()}


def read_dataset(url: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = get_engine(url)
    with engine.connect() as connection:
        users = pd.read_sql(select(users_table), connection)
        posts = pd.read_sql(select(posts_table), connection, parse_dates=["created_at"])
        events = pd.read_sql(select(events_table), connection, parse_dates=["timestamp"])
    engine.dispose()
    return users, posts, events
