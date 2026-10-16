import argparse
import json
from pathlib import Path

from feedrec.ranking import load_ranker
from feedrec.storage import import_dataset
from feedrec.synthetic import generate_dataset
from feedrec.training import train


def main():
    parser = argparse.ArgumentParser(prog="feedrec")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", type=Path, default=Path("data"))
    generate.add_argument("--users", type=int, default=120)
    generate.add_argument("--posts", type=int, default=360)
    generate.add_argument("--days", type=int, default=30)
    generate.add_argument("--seed", type=int, default=42)
    ingest = commands.add_parser("import")
    ingest.add_argument("--input", type=Path, default=Path("data"))
    fit = commands.add_parser("train")
    fit.add_argument("--cutoff", default="2025-01-25")
    fit.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    fit.add_argument("--iterations", type=int, default=180)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--k", type=int, default=10)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_dataset(args.output, args.users, args.posts, args.days, args.seed)
    elif args.command == "import":
        result = import_dataset(args.input)
    elif args.command == "train":
        result = train(args.cutoff, args.artifacts, iterations=args.iterations, seed=args.seed, k=args.k)
    elif args.command == "recommend":
        ranker, _ = load_ranker(args.artifacts)
        result = ranker.recommend(args.user_id, args.k)
    else:
        import uvicorn
        uvicorn.run("feedrec.api:app", host=args.host, port=args.port)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))
