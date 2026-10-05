"""Local trace browser for the quantization-interpretability evals and token KL dumps.

Run from webui/:  uv run server.py [--raw-dir DIR] [--dump-dir DIR] [--results-dir DIR] [--port 8765] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from data import DumpStore, RawStore, TokenDecoder

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _split(conditions: str) -> list[str]:
    return [c.strip() for c in conditions.split(",") if c.strip()]


def not_found(what: str) -> JSONResponse:
    return JSONResponse({"error": f"{what} not found"}, status_code=404)


def create_app(raw_dir: Path, dump_dir: Path, results_dir: Path) -> FastAPI:
    raw = RawStore(raw_dir, results_dir)
    decoder = TokenDecoder(REPO / ".env")
    dumps = DumpStore(dump_dir, decoder, REPO / "configs" / "data" / "kl_set.yaml")
    decoder.start()  # load the tokenizer in the background so the first token view does not wait

    app = FastAPI(title="qi-webui", docs_url=None, redoc_url=None)  # the Swagger page would need a CDN
    app.add_middleware(GZipMiddleware, minimum_size=4096)

    @app.middleware("http")
    async def no_cache(request, call_next):  # the static files change while developing; always revalidate
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.get("/api/conditions")
    def conditions(refresh: int = 0):
        return raw.conditions(refresh=bool(refresh))

    @app.get("/api/items")
    def items(task: str, conditions: str = "", filter_: str | None = Query(None, alias="filter"), q: str = "",
              status: str = "all", compare: str | None = None, subject: str | None = None, sort: str = "doc"):
        return raw.items(task, _split(conditions), filt=filter_, q=q, status=status, compare=compare,
                         subject=subject or None, sort=sort)

    @app.get("/api/item")
    def item(task: str, doc_id: str, conditions: str = "", filter_: str | None = Query(None, alias="filter")):
        out = raw.item(task, doc_id, _split(conditions), filt=filter_)
        return out if out is not None else not_found(f"{task} item {doc_id}")

    @app.get("/api/dumps")
    def dump_list():
        return dumps.list()

    @app.get("/api/dump/docs")
    def dump_docs(name: str, sort: str = "mean_kl", limit: int = 200, offset: int = 0, source: int | None = None):
        out = dumps.docs(name, sort=sort, limit=limit, offset=offset, source=source)
        return out if out is not None else not_found(f"dump {name}")

    @app.get("/api/dump/doc")
    def dump_doc(name: str, doc_id: int, source: int = 0, start: int = 0, limit: int = 2048):
        out = dumps.doc(name, doc_id, source=source, start=start, limit=limit)
        return out if out is not None else not_found(f"document {source}:{doc_id} in dump {name}")

    app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=Path("~/qi_archive/boxA/lm_eval_raw").expanduser(),
                    help="lm-eval output root: <condition>/<task>/<model_dir>/samples_*.jsonl")
    ap.add_argument("--dump-dir", type=Path, default=REPO / "results" / "phase1" / "token_dump",
                    help="directory with <cond>_<set>.parquet token dumps")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results",
                    help="repo results directory (fallback source of headline accuracies)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn
    app = create_app(args.raw_dir.expanduser(), args.dump_dir.expanduser(), args.results_dir.expanduser())
    print(f"qi-webui on http://{args.host}:{args.port}  (raw: {args.raw_dir}, dumps: {args.dump_dir})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
