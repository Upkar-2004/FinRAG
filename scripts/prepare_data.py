"""
Prepare the evaluation set and PDFs for retrieval evaluation.
Performing the ETL: Extract, Transform, Load
"""

import json
import shutil
import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT)) 

import config  

FINANCEBENCH_REPO = "https://github.com/patronus-ai/financebench.git"
FINANCEBENCH_REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"



def get_git_revision(repo_dir: Path) -> str:
    """Return the commit currently checked out in a git repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check = True,
        capture_output = True,
        text = True
    )
    return result.stdout.strip() 




def ensure_financebench_source() -> None:
    """Ensure that the pinned FinanceBench source revision is available."""
    source_dir = config.FINANCEBENCH_SRC

    if not source_dir.exists():
        print(f"[..] Cloning FinanceBench source into {source_dir}...")
        source_dir.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["git", "clone", FINANCEBENCH_REPO, str(source_dir)],
            check = True,
        )

        subprocess.run(
            ["git", "-C", str(source_dir), "checkout", FINANCEBENCH_REVISION],
            check = True,
        )

    if not (source_dir / ".git").exists():
        raise RuntimeError(
            f"Expected a Git repository at {source_dir}, but none was found."
        )

    actual_revision = get_git_revision(source_dir)

    if actual_revision != FINANCEBENCH_REVISION:
        raise RuntimeError(
            "FinanceBench revision mismatch.\n"
            f"Expected: {FINANCEBENCH_REVISION}\n"
            f"Found:    {actual_revision}"
        )

    print(
        f"[ok] FinanceBench source is ready at revision "
        f"{actual_revision[:12]}."
    )




def build_eval_set() -> list[dict]:
    """Build the evaluation set from the FinanceBench source JSONL file, filtering to only include documents in the corpus."""

    src = config.FINANCEBENCH_SRC / "data" / "financebench_open_source.jsonl"
    rows = [json.loads(line) for line in src.open()]

    eval_rows = []

    for r in rows:

        if r["doc_name"] not in config.CORPUS_DOCS:
            continue

        evidence = r.get("evidence") or []
        gold_pages = sorted(
            {e["evidence_page_num"] for e in evidence if e.get("evidence_page_num") is not None}
        )

        eval_rows.append(
            {
                "id": r["financebench_id"],
                "company": r["company"],
                "doc_name": r["doc_name"],
                "question": r["question"],
                "answer": r["answer"],
                "question_type": r["question_type"],
                "gold_pages": gold_pages,  # all evidence page_index values (0-based)
                "gold_page": gold_pages[0] if gold_pages else None,  # first page, back-compat only
            }
        )
    return eval_rows



def copy_pdfs() -> list[str]:
    """Copy the corpus PDFs out of the source repo into data/pdfs/."""

    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    src_pdf_dir = config.FINANCEBENCH_SRC / "pdfs"

    copied: list[str] = []
    missing: list[str] = []

    for doc_name in config.CORPUS_DOCS:
        src = src_pdf_dir / f"{doc_name}.pdf"
        if src.is_file():
            shutil.copy2(src, config.PDF_DIR / src.name)
            copied.append(src.name)
        else:
            missing.append(src.name)

    if missing:
        raise FileNotFoundError(
            "The following configured PDFs are missing from FinanceBench: "
            f"{', '.join(missing)}"
    )
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


    print("\n" + "=" * 60)
    print("DATA PREP COMPLETE")
    print("=" * 60)
    print(f"PDFs copied      : {len(copied)}  -> {config.PDF_DIR}")
    print(f"Eval questions   : {len(eval_set)}  -> {out}")

    by_type: dict[str, int] = {}

    for r in eval_set:
        by_type[r["question_type"]] = by_type.get(r["question_type"], 0) + 1

    print(f"Question types   : {by_type}")
    multi_page = sum(1 for r in eval_set if len(r["gold_pages"]) > 1)
    print(f"Multi-page gold  : {multi_page}/{len(eval_set)} questions cite evidence on >1 page")
    print("\nPer document:")

    for doc, label in config.CORPUS_DOCS.items():
        n = sum(1 for r in eval_set if r["doc_name"] == doc)
        print(f"  {n:2d}  {doc:28s} {label}")


if __name__ == "__main__":
    main()
