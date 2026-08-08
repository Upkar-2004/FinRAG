"""
Lightweight, append-only observability for the pipeline's two
Groq-cost-bearing stages: retrieval (local, free, but still worth timing
since it's user-facing latency) and generation (real dollars AND real
latency -- Groq bills per token, and generate_answer() can loop through
several tool-use rounds, each one a separate billed call).

Why a flat JSONL file instead of a database: the project already logs
data this exact way (data/eval/eval_set.jsonl) -- one JSON object per
line. Trivial to append to from multiple processes (the demo UI, an eval
script) without any locking/transaction machinery, and trivial to load
later with `pandas.read_json(path, lines=True)` for analysis. No schema
to migrate if a new field shows up in some records and not others.

Not persisted to git (data/ is gitignored) -- this is a runtime record of
YOUR actual usage, not a reproducible artifact someone else's clone needs.
"""

import json
import time
from contextlib import contextmanager

import config

TELEMETRY_PATH = config.DATA_DIR / "telemetry.jsonl"

# On-demand Groq pricing in USD per 1M tokens, verified 2026-08-03.
# Keep pricing keyed by model so historical telemetry is not silently
# recalculated with the current default model's rates.
GROQ_PRICING_USD_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


def estimate_cost_usd(
    prompt_tokens: int, completion_tokens: int, *, model: str | None = None
) -> float | None:
    """Estimate on-demand token cost, or return None for an unknown model."""
    model = model or config.GROQ_MODEL
    pricing = GROQ_PRICING_USD_PER_M_TOKENS.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def _append(record: dict) -> None:
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@contextmanager
def timed(stage: str, **fields):
    """Time a block and append one record to TELEMETRY_PATH when it exits
    -- including when it raises, so a slow call that then fails still
    shows up in the log instead of vanishing (a stuck-then-failing call is
    exactly the kind of thing worth being able to see later).

    Yields the record dict itself, still open for the caller to enrich
    from inside the `with` block -- e.g. add token counts once they're
    known, partway through:

        with telemetry.timed("retrieval", question=question) as rec:
            chunks = retrieve_with_expansion(question, k=config.TOP_K)
            rec["num_results"] = len(chunks)
        # latency_ms and the record are written to disk here, on exit
    """
    record = {"stage": stage, "timestamp": time.time(), **fields}
    start = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record["succeeded"] = False
        record["error_type"] = type(exc).__name__
        raise
    finally:
        record.setdefault("succeeded", True)
        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        _append(record)
