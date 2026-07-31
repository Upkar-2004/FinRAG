"""
Query expansion: a cheap, zero-Groq-cost proxy for HyDE (flagged in
Section 17 of the learning report as the next genuinely different lever
-- every technique tried before operated on the document side; this is
the first thing that touches the QUERY itself).

Instead of asking an LLM to generate a hypothetical answer passage per
query (real per-query API cost, and quota was constrained the day this
was built), this hand-maps a handful of financial metric names to the
RAW LINE-ITEM VOCABULARY their standard definitions are built from, and
appends that vocabulary to any query mentioning the metric -- directly
targeting the vocabulary/register mismatch diagnosed repeatedly
(Section 15, 17): abstract metric names like "quick ratio" never appear
in filings; only the raw numbers they're computed from do.

Validated first as a 2-question manual check: AMD's quick-ratio gold page
went from NOT in the top-20 at all to rank 5, using only the standard
textbook quick-ratio formula's terms -- not anything read off the actual
gold filing text, so this is representative of what a real LLM-driven
HyDE pass would plausibly generate on its own, not an answer-leaked test.

Deliberately narrow and manually maintained -- NOT a general solution.
Only covers metrics actually repeated across this project's 29-question
eval set. A real production system would use HyDE (LLM-generated per
query, generalizes to any question) instead of hand-listed keywords.
"""

EXPANSIONS = {
    "quick ratio": "cash and cash equivalents short-term investments accounts receivable total current liabilities",
    "operating margin": "operating income operating expenses net revenue net sales",
    "gross margin": "gross profit cost of goods sold cost of revenue net sales net revenue",
    "effective tax rate": "income tax expense provision for income taxes income before income taxes pretax income",
    "ebitda": "operating income depreciation and amortization capital expenditures net revenue statement of cash flows",
}


def expand_query(question: str) -> str:
    """Append raw line-item vocabulary for any financial metric named in
    `question`. Case-insensitive substring match against EXPANSIONS'
    keys; appends every match found -- a question could reference more
    than one metric. Returns `question` unchanged if nothing matches."""
    lower = question.lower()
    additions = [expansion for term, expansion in EXPANSIONS.items() if term in lower]
    if not additions:
        return question
    return question + " " + " ".join(additions)
