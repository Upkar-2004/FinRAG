"""
Stage 4 — Generation: question + retrieved chunks -> grounded answer with
citations, via Groq's chat completions API (OpenAI-compatible tool-use).

This is the missing last step of the pipeline. Everything built so far
(ingest -> index -> retrieve) only ever produces a ranked list of source
passages; nothing has turned that into an actual answer yet. This module
closes that loop.

It also carries a calculator TOOL, pulled forward from the original Day-3
roadmap slot. Justification (see docs/finrag_learning_report.md, Section 15):
three separate retrieval-only fixes (chunk-size ablation, doc-scoped
filtering, table serialization) all measured zero Recall@5 improvement,
and digging into why showed that some gold answers (e.g. AMD's FY22 quick
ratio) are DERIVED numbers that never appear as text anywhere in the
source filing -- they're computed from raw line items an analyst would
pull off the balance sheet. No amount of better retrieval fixes that,
because the "fact" being asked for was never a fact in the text to begin
with. And having the LLM just do the arithmetic itself in free text isn't
reliable either -- language models generate text by predicting plausible
next tokens, not by executing arithmetic, so they're well known to make
computational errors even when they've picked out the right numbers. A
tool gives the model a way to hand the actual computation off to real,
deterministic Python code instead of "guessing" a plausible-looking
number.

Security note: the calculator tool's argument comes from the model, and
the model's output is effectively untrusted input (it could, in principle,
be steered by adversarial content hidden in a retrieved chunk -- a prompt
injection). So this does NOT call Python's eval() on it. calculate() below
only walks a parsed expression tree and permits numeric constants and
+ - * / -- nothing else (no names, no attribute access, no calls) can
execute.
"""

import ast
import json
import operator
import os

from dotenv import load_dotenv
from groq import BadRequestError, Groq

import config

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about company 10-K filings.

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- Every claim must be followed by a citation in the form (Company, page N), using the page numbers given in the context.
- If the question requires a computed value, use the most specific tool available: `percent_change` for any year-over-year or "how did X change" question, `ratio` for a single-period ratio or margin (quick ratio, operating margin, EBITDA margin), or `calculate` for anything else. Never do arithmetic yourself -- you are not reliable at it, the tools are.
- If the context does NOT contain the actual numeric values a computation needs, do not call a tool at all. State plainly that you don't have enough information. Never pass a description or label (e.g. "cash + short-term investments") as a tool argument -- only real numbers taken from the context.
"""

# OpenAI-compatible function-calling schemas. Groq's chat completions API
# accepts this same "tools" shape (see console.groq.com/docs/tool-use).
#
# Three tools, not one, so the FORMULA for a percent change or a ratio is
# fixed in Python rather than re-derived from text by the model every
# time (e.g. calculate() alone requires the model to parenthesize its own
# expression correctly -- "a + b + c / d" silently divides only c by d,
# per normal precedence).
#
# numerator/denominator/old/new are typed as STRINGS, not JSON numbers --
# found out why the hard way. A first version typed them as "number", and
# a live run showed the model wanting to combine several line items in
# one argument, e.g. numerator="4835 + 1020 + 4126". That isn't valid
# JSON for a "number" field, so Groq's own schema validation rejected the
# whole generation server-side (400 tool_use_failed) before our code ever
# saw it. Accepting a string instead -- "a plain number OR a small
# expression" -- matches what the model actually wants to send, and
# percent_change()/ratio() below route it through the SAME restricted
# evaluator calculate() uses (_resolve -> calculate() when it isn't a
# bare number), so the "not eval()" security property still holds; it's
# just applied one level lower than originally designed.
PERCENT_CHANGE_TOOL = {
    "type": "function",
    "function": {
        "name": "percent_change",
        "description": (
            "Compute the percent change from an old value to a new value: "
            "(new - old) / old. Use this for any 'how did X change "
            "year-over-year' question -- margin change, tax rate change, "
            "revenue change, segment growth -- instead of writing the "
            "formula yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old": {
                    "type": "string",
                    "description": "The earlier/base value as a number or expression, e.g. '2535' or the FY2021 figure",
                },
                "new": {
                    "type": "string",
                    "description": "The later/comparison value as a number or expression, e.g. '4835' or the FY2022 figure",
                },
            },
            "required": ["old", "new"],
        },
    },
}

RATIO_TOOL = {
    "type": "function",
    "function": {
        "name": "ratio",
        "description": (
            "Compute numerator / denominator. Use this for a single-period "
            "ratio or margin (quick ratio, operating margin, gross margin, "
            "EBITDA margin) instead of writing the division yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "numerator": {
                    "type": "string",
                    "description": "A number or expression, e.g. '4835 + 1020 + 4126' for (cash + short-term investments + accounts receivable)",
                },
                "denominator": {
                    "type": "string",
                    "description": "A number or expression, e.g. '6369' for revenue or total current liabilities",
                },
            },
            "required": ["numerator", "denominator"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression using +, -, *, /, **, % "
            "and parentheses. Use this as a fallback for a computation that "
            "isn't a percent change or a simple ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A basic arithmetic expression, e.g. '(4835 + 1020 + 4126) / 9583'",
                }
            },
            "required": ["expression"],
        },
    },
}

TOOLS = [PERCENT_CHANGE_TOOL, RATIO_TOOL, CALCULATOR_TOOL]

# Only these node types are legal inside a `calculate` expression. Anything
# else (names, function calls, attribute access, subscripts, ...) raises
# instead of silently doing something unexpected -- see module docstring.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Evaluate a restricted arithmetic expression. Not eval() -- see module docstring."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


def _resolve(value: str) -> float:
    """Turn a percent_change/ratio argument into a float. The model may
    send a bare number ('4835') or a small expression ('4835 + 1020 +
    4126') -- see the "why strings" comment above PERCENT_CHANGE_TOOL.
    Either way this goes through calculate()'s restricted evaluator, not
    a plain float() cast, so a smuggled expression still can't do
    anything beyond +-*/**%."""
    try:
        return float(value)
    except ValueError:
        return calculate(value)


def percent_change(old: str, new: str) -> float:
    """(new - old) / old, as a fraction (0.10 == a 10% increase)."""
    old, new = _resolve(old), _resolve(new)
    if old == 0:
        raise ValueError("percent_change: old value is 0, cannot divide by zero")
    return (new - old) / old


def ratio(numerator: str, denominator: str) -> float:
    """numerator / denominator -- for a single-period ratio or margin."""
    numerator, denominator = _resolve(numerator), _resolve(denominator)
    if denominator == 0:
        raise ValueError("ratio: denominator is 0, cannot divide by zero")
    return numerator / denominator


# Maps a tool name (as the model requests it) to the function that actually
# runs it and how to pull its arguments out of the model's parsed JSON.
_TOOL_FUNCTIONS = {
    "calculate": lambda args: calculate(args["expression"]),
    "percent_change": lambda args: percent_change(args["old"], args["new"]),
    "ratio": lambda args: ratio(args["numerator"], args["denominator"]),
}


def _format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a labeled context block the model can cite from."""
    blocks = []
    for c in chunks:
        meta = c["metadata"]
        label = f"[{meta['company']}, page {meta['page_number']}]"
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n".join(blocks)


MAX_TOOL_ATTEMPTS = 3


def _call_with_tools(messages: list[dict]): #Self-correction loop for tool calls
    """First API call, with tools enabled -- retried on malformed tool-call
    generations.

    Empirically observed, NOT documented by Groq: llama-3.3-70b-versatile
    occasionally emits a broken <function=NAME>{...}</function> wrapper --
    a stray character where '>' should be -- specifically when a tool
    argument value is a multi-term expression string (e.g.
    numerator="4835 + 1020 + 4126") rather than a bare number. Groq's API
    then rejects the whole generation server-side with a 400
    'tool_use_failed' before we ever see a parseable response.
    Retrying the identical request doesn't reliably help here -- the
    trigger looked repeatable across live runs, not just sampling noise --
    so each retry also appends explicit corrective feedback steering the
    model toward pre-computing sums as a single number instead.
    """
    last_error = None
    for attempt in range(1, MAX_TOOL_ATTEMPTS + 1):
        try:
            return _client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        except BadRequestError as e:
            last_error = e
            print(f"[tool call attempt {attempt} failed: {e}]")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last tool call was malformed and rejected. Retry with "
                        "SIMPLE arguments: if a value requires adding multiple line "
                        "items, call `calculate` first to get a single total, then "
                        "pass that single number to `ratio` or `percent_change` -- "
                        "do not put a multi-term expression directly in numerator/"
                        "denominator/old/new."
                    ),
                }
            )
    raise last_error


MAX_TOOL_ROUNDS = 6
# Was 4; raised after live traces showed that was too tight. Root cause:
# _call_with_tools' own malformed-<function=...>-wrapper recovery (see its
# docstring) sometimes only resolves on internal retry attempt 2 or 3, and
# its corrective message explicitly tells the model to DECOMPOSE the
# calculation (compute a sum with `calculate` first, THEN pass the single
# result to `ratio`) -- turning a 1-call ratio(expr, expr) into a 3-4 call
# chain (calculate numerator, calculate denominator, ratio, final text),
# right after a round was already spent recovering. Reproduced live on the
# AMD quick-ratio question across 6 identical trials: a clean run finished
# in 3 rounds, but a run that hit the wrapper bug needed 5 -- the old
# MAX_TOOL_ROUNDS=4 cut it off one round early, degrading to the "couldn't
# finish" message even though the model would have reached a real answer
# immediately after. 6 leaves headroom for that worst case plus one
# redundant self-check call (also observed live) without letting a truly
# stuck model loop forever.


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Answer `question` grounded only in `chunks`, dispatching to whichever
    tool(s) the model requests for any computation. Returns
    {"answer": str, "used_tool": bool}.

    Loops -- not a fixed two-step sequence -- because some questions need
    MULTIPLE separate tool calls before a final answer (e.g. a ratio needs
    the numerator summed, then the denominator summed, then divided: three
    calls, not one). An earlier fixed-two-round version offered tools only
    on the FIRST call, then made a second, tool-less call expecting a final
    answer -- but if the model still wanted another tool call at that point,
    it had no way to make one, and it just wrote the tool-call syntax out as
    literal text instead (a real, observed failure: "<function=calculate>
    {...}</function>" leaking straight into the user-facing answer). Now
    every round offers tools via _call_with_tools(), and the loop keeps
    executing tool_calls and re-asking until the model returns a response
    with no tool_calls at all -- a genuine final answer. Bounded by
    MAX_TOOL_ROUNDS so a model stuck in a tool-call loop degrades to an
    honest message instead of hanging.

    Guardrail: if the best chunk's similarity is below
    config.MIN_RELEVANCE_SCORE, refuses without calling the LLM at all --
    both a correctness guardrail (catches clearly out-of-scope queries)
    and a quota-saving one (no wasted API call on a doomed question). See
    config.py for how the threshold was calibrated, and its documented
    limitation (can't catch "right structure, wrong company" queries).
    Assumes `chunks` came from a cosine-similarity retriever (retrieve()
    or retrieve_with_expansion()) -- BM25/hybrid/reranked scores are on
    different scales and would make this comparison meaningless.
    """
    if not chunks or max(c["similarity"] for c in chunks) < config.MIN_RELEVANCE_SCORE:
        return {
            "answer": (
                "I don't have enough relevant information in these filings to answer "
                "that question. It may be outside the scope of the documents available."
            ),
            "used_tool": False,
        }

    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    used_tool = False
    for _ in range(MAX_TOOL_ROUNDS):
        response = _call_with_tools(messages)
        message = response.choices[0].message

        if not message.tool_calls:
            return {"answer": message.content, "used_tool": used_tool}

        used_tool = True
        messages.append(
            {
                # NOT message.content -- Groq's server sometimes echoes the
                # raw "<function=...>" wrapper text into content even when
                # tool_calls parsed successfully (empirically observed; see
                # _call_with_tools docstring for the related wrapper-parsing
                # issue). Forwarding that raw text here got imitated by the
                # model on the next turn instead of a real answer -- the
                # tool call itself was correct, only the replayed text was
                # polluted. content=None is the standard shape for a
                # tool-call-only assistant turn anyway.
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = _TOOL_FUNCTIONS[tc.function.name](args)
            except Exception as e:
                result = f"Error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return {
        "answer": (
            "I wasn't able to finish this calculation after several steps. "
            "Please try rephrasing the question or breaking it into simpler parts."
        ),
        "used_tool": used_tool,
    }


if __name__ == "__main__":
    from .retrieve import retrieve_with_expansion as retrieve

    question = (
        "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22? "
        "Quick ratio is defined as (cash and cash equivalents + short-term investments + accounts "
        "receivable) / total current liabilities."
    )
    chunks = retrieve(question, k=5)
    result = generate_answer(question, chunks)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"[used calculator tool: {result['used_tool']}]")
