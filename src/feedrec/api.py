import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel

from feedrec.ranking import load_ranker


class RecommendationItem(BaseModel):
    post_id: int
    title: str
    category: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    as_of: str
    strategy: str
    items: list[RecommendationItem]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    directory = artifact_dir or Path(os.environ.get("ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranker, app.state.manifest = load_ranker(directory)
        yield

    app = FastAPI(title="Personalized Feed API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.manifest["model"],
                "snapshot": app.state.manifest["cutoff"],
                "catalog_size": app.state.manifest["catalog_at_cutoff"]}

    @app.get("/recommendations", response_model=RecommendationResponse)
    def recommendations(user_id: int = Query(ge=1), k: int = Query(default=10, ge=1, le=100)):
        return app.state.ranker.recommend(user_id, k)

    return app


app = create_app()
