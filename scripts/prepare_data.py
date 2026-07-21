"""
Day 0 - Data preparation.

Run once after cloning:  python scripts/prepare_data.py

What it does, in order:
  1. Makes sure we have the FinanceBench source (clones it if missing).
  2. Reads the 150 labeled questions, keeps only the ones about OUR 5 docs.
  3. Writes those to data/eval/eval_set.jsonl  -> our held-out test set.
  4. Copies the 5 source PDFs into data/pdfs/    -> the corpus to index.

Why a script instead of committing the files? Reproducibility ("clean clone
-> one command -> ready") and we avoid redistributing the filings ourselves.
"""

import json
import shutil
import subprocess
import sys

# Import our central config. This works because you run the script from the
# repo root (python scripts/prepare_data.py), so the root is on the path.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402

FINANCEBENCH_REPO = "https://github.com/patronus-ai/financebench.git"


def ensure_financebench_source() -> None:
    """Clone the FinanceBench repo (shallow) if we don't already have it."""
    if config.FINANCEBENCH_SRC.exists():
        print(f"[ok] FinanceBench source already present at {config.FINANCEBENCH_SRC}")
        return
    print(f"[..] Cloning FinanceBench into {config.FINANCEBENCH_SRC} ...")
    config.FINANCEBENCH_SRC.parent.mkdir(parents=True, exist_ok=True)
    # --depth 1 = only the latest commit, much faster/smaller than full history.
    subprocess.run(
        ["git", "clone", "--depth", "1", FINANCEBENCH_REPO, str(config.FINANCEBENCH_SRC)],
        check=True,
    )
    print("[ok] Clone complete.")


def build_eval_set() -> list[dict]:
    """Filter the 150 questions down to our corpus and normalise the fields."""
    src = config.FINANCEBENCH_SRC / "data" / "financebench_open_source.jsonl"
    rows = [json.loads(line) for line in src.open()]

    kept = []
    for r in rows:
        if r["doc_name"] not in config.CORPUS_DOCS:
            continue
        # Each question lists one or more "evidence" entries with the gold page.
        # We take the first evidence page as the retrieval ground truth.
        evidence = r.get("evidence") or [{}]
        gold_page = evidence[0].get("evidence_page_num")
        kept.append(
            {
                "id": r["financebench_id"],
                "company": r["company"],
                "doc_name": r["doc_name"],
                "question": r["question"],
                "answer": r["answer"],
                "question_type": r["question_type"],
                "gold_page": gold_page,       # 0-indexed page in the PDF
            }
        )
    return kept


def copy_pdfs() -> list[str]:
    """Copy the corpus PDFs out of the source repo into data/pdfs/."""
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    src_pdf_dir = config.FINANCEBENCH_SRC / "pdfs"
    copied, missing = [], []
    for doc_name in config.CORPUS_DOCS:
        src = src_pdf_dir / f"{doc_name}.pdf"
        if src.exists():
            shutil.copy2(src, config.PDF_DIR / src.name)
            copied.append(src.name)
        else:
            missing.append(src.name)
    if missing:
        print(f"[!!] Missing PDFs (check names): {missing}")
    return copied


def main() -> None:
    ensure_financebench_source()

    eval_set = build_eval_set()
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVAL_DIR / "eval_set.jsonl"
    with out.open("w") as f:
        for row in eval_set:
            f.write(json.dumps(row) + "\n")

    copied = copy_pdfs()

    # Summary so you can see it worked at a glance.
    print("\n" + "=" * 60)
    print("DATA PREP COMPLETE")
    print("=" * 60)
    print(f"PDFs copied      : {len(copied)}  -> {config.PDF_DIR}")
    print(f"Eval questions   : {len(eval_set)}  -> {out}")
    by_type: dict[str, int] = {}
    for r in eval_set:
        by_type[r["question_type"]] = by_type.get(r["question_type"], 0) + 1
    print(f"Question types   : {by_type}")
    print("\nPer document:")
    for doc, label in config.CORPUS_DOCS.items():
        n = sum(1 for r in eval_set if r["doc_name"] == doc)
        print(f"  {n:2d}  {doc:28s} {label}")


if __name__ == "__main__":
    main()
