"""Deterministic financial-vocabulary expansion for dense retrieval."""

_DASH_TRANSLATION = str.maketrans({"-": " ", "–": " ", "—": " "})

EXPANSIONS: dict[str, tuple[str, ...]] = {
    "quick ratio": (
        "cash and cash equivalents",
        "short-term investments",
        "accounts receivable",
        "total current liabilities",
    ),
    "operating margin": (
        "operating income",
        "operating expenses",
        "net revenue",
        "net sales",
    ),
    "gross margin": (
        "gross profit",
        "cost of goods sold",
        "cost of revenue",
        "net sales",
        "net revenue",
    ),
    "effective tax rate": (
        "income tax expense",
        "provision for income taxes",
        "income before income taxes",
        "pretax income",
    ),
    "ebitda": (
        "operating income",
        "depreciation and amortization",
    ),
    "ebitda margin": (
        "net revenue",
        "net sales",
    ),
    "capex": (
        "capital expenditures",
        "statement of cash flows",
    ),
    "capital expenditures": (
        "capital expenditures",
        "statement of cash flows",
    ),
}


def _normalize_for_matching(text: str) -> str:
    """Normalize case, dash variants, and whitespace for term matching."""
    return " ".join(text.casefold().translate(_DASH_TRANSLATION).split())


def _contains_term(normalized_text: str, normalized_term: str) -> bool:
    """Match a complete term rather than a substring of a longer word."""
    return f" {normalized_term} " in f" {normalized_text} "


def expand_query(question: str) -> str:
    """Append unique filing vocabulary for financial concepts in a question."""
    normalized_question = _normalize_for_matching(question)
    additions: list[str] = []
    seen_phrases: set[str] = set()

    for term, phrases in EXPANSIONS.items():
        normalized_term = _normalize_for_matching(term)
        if not _contains_term(normalized_question, normalized_term):
            continue

        for phrase in phrases:
            normalized_phrase = _normalize_for_matching(phrase)
            if normalized_phrase in seen_phrases:
                continue
            if _contains_term(normalized_question, normalized_phrase):
                continue

            additions.append(phrase)
            seen_phrases.add(normalized_phrase)

    if not additions:
        return question

    return f"{question} {' '.join(additions)}"
