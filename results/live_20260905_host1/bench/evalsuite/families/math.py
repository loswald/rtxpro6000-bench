"""
families/math.py - reasoning / maths.

Three sub-families, all public and ungated, fetched with common.fetch() and sha256-recorded:

  aime25        AIME 2025 I + II, all 30 problems (math-ai/aime25 test.jsonl; ids 0-14 = I, 15-29 = II,
                cross-checked against the opencompass/AIME2025 mirror).  Integer answers 0-999.
  hmmt25        HMMT February 2025, all 30 problems (MathArena/hmmt_feb_2025; the repo is parquet-only, so the
                rows are fetched as JSON through the public datasets-server endpoint).  Closed-form answers
                (fractions, radicals, \\pi, 2^{25}\\cdot 26!, a root pair).
  math500_hard  a seeded prefix (20 default / 60 full) of a shuffled pool of MATH-500 level-5 problems (HuggingFaceH4/MATH-500)
                restricted to answers the programmatic grader can check exactly (numbers, fractions, radicals,
                \\pi expressions, degrees, complex numbers, ordered tuples, root sets) - no intervals, matrices
                or symbolic expressions in free variables.

              The default 20 are the first 20 of the full 60 (shuffle-then-prefix), so the profiles nest.

Prompt: the problem + "Put your final answer within \\boxed{}."  Scoring: last \\boxed{} (brace-matched), else
the expression after the last "answer is"/"final answer:" phrase, else unparsed.  Two \\boxed{} with different
values joined by nothing but a disjunction ("either \\boxed{A} or \\boxed{B}") are a simultaneous hedge and
score 'unparsed'; any prose between them (a self-correction: "actually", "wait", "I made an error") means the
model revised itself and the last box wins as usual, and two boxes holding the SAME value in two notations
("\\boxed{0.5} or \\boxed{\\frac{1}{2}}") are one committed answer, never a hedge.  The candidate and the gold
are canonicalised (delimiters, \\left/\\right, spacing, \\dfrac, \\text units, degrees, $/%, thousands
separators, leading zeros, x= prefixes, unit powers '\\text{ cm}^2', '\\pmod{1000}' residue tags,
'\\text{answer: }' labels, ...) and compared as strings, then component-wise (tuples ordered,
sets/\\pm-pairs unordered) with a sympy-free safe numeric evaluator (\\frac, \\sqrt[n], \\binom, !, ^, \\pi, i)
at 1e-9 relative tolerance (1e-12 for a bare decimal candidate).  Approximations of exact closed forms are
counted wrong (MathArena convention).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
from typing import Callable, Optional

import common
from common import DEFAULT_SEED, ShortPool, Verdict
from families import _base

NAME = "math"
DESCRIPTION = "AIME 2025 I+II (30), HMMT Feb 2025 (30), MATH-500 level-5 sample (20/60); \\boxed{} + canonical/numeric equivalence"
SUBFAMILIES = ["aime25", "hmmt25", "math500_hard"]
PRIORITY = 30
DEFAULT_MAX_TOKENS = {"default": 4096, "reasoning": 8192}
ITEM_TIME_FALLBACK_S = 120.0
NOTES = [
    "math: exact-equivalence grading (canonical string, then numeric at 1e-9 rel., 1e-12 for bare decimals); decimal approximations of closed forms count as wrong",
    "math/hmmt25: MathArena/hmmt_feb_2025 is cc-by-nc-sa-4.0 and parquet-only; rows fetched via the public datasets-server JSON endpoint",
    "math/math500_hard: level-5 pool filtered to programmatically gradable answers, then shuffled once and cut to n (the default 20 are a prefix of the full 60)",
    "math: two differing \\boxed{} joined only by a disjunction ('either A or B') score 'unparsed'; with any prose between them (a self-correction) the last box wins",
]

PROMPT_SUFFIX = "\n\nPut your final answer within \\boxed{}."

AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
HMMT_REPO = "MathArena/hmmt_feb_2025"
HMMT_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=MathArena%2Fhmmt_feb_2025"
                 "&config=default&split=train&offset=0&length=100")
MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_FILE = "test.jsonl"

N_AIME = 30
N_HMMT = 30
N_MATH500 = {"default": 20, "full": 60}


# ======================================================================================
# canonicalisation
# ======================================================================================

_UNIT_WORDS = {
    "degree", "degrees", "deg", "cent", "cents", "dollar", "dollars", "unit", "units", "percent", "cm", "mm", "km",
    "m", "inch", "inches", "in", "feet", "foot", "ft", "meter", "meters", "metre", "metres", "mile", "miles", "yard",
    "yards", "minute", "minutes", "hour", "hours", "second", "seconds", "sq", "square", "cubic", "ways", "mph",
    "year", "years", "day", "days", "people", "students", "cents.",
}
# NB: "or" is deliberately NOT a unit word - dropping it would collapse the hedge "3 \text{ or } 5" into "35".
# the optional trailing power is captured so that a *unit* group takes its exponent with it:
# '70 \text{ cm}^2' -> '70' and not '70^2' (= 4900).  A non-unit group keeps the exponent ('\text{x}^2').
_TEXT_CMD = re.compile(r"\\(?:text|textbf|textit|textrm|textnormal|textsf|texttt|mathrm|mathbf|mathit|mbox|operatorname)"
                       r"\s*\{([^{}]*)\}(\s*\^\s*(?:\{\s*\d+\s*\}|\d+))?")
_SPACING = re.compile(r"\\(?:left|right|displaystyle|quad|qquad|,|;|:|!| )")
_DEGREES = re.compile(r"\^\s*(?:\\circ|\{\\circ\}|\{\\?circ\})|\\degree\b|\\deg\b")
_TRAIL_UNITS = re.compile(r"(?i)(?:degrees?|cents?|dollars?|units?|percent|sq\.?units?|squareunits?)$")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_LEAD_ASSIGN = re.compile(r"^[a-zA-Z](?:_\{?\w+\}?)?(?:=|\\in)(?=.)")   # 'x=5', 'x\in(-1,1)'
# a trailing modulus annotation on an AIME-style residue: '70\pmod{1000}', '70(mod1000)', '70\bmod1000'.
# Applied to the whitespace-free string.  Dropping it can only ever LOSE a match ('1070\pmod{1000}' stays
# 1070 and still fails against a gold of 70), so it cannot manufacture a false positive.
_MODULO = re.compile(r"(?:\\pmod|\\bmod|\\mod|\(mod)\s*\{?[^{}()]*\}?\)?$")
# an answer label the model left inside the box: '\text{answer: } 70', 'Final answer = 70'
_LEAD_LABEL = re.compile(r"(?i)^(?:the)?(?:final)?answer(?:is)?[:=]?(?=.)")
_PLAIN_LITERAL = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")                  # a bare integer / decimal candidate
_UNICODE = {"\u2212": "-", "\u00d7": "\\times", "\u22c5": "\\cdot", "\u221a": "\\sqrt", "\u03c0": "\\pi",
            "\u00b0": "^\\circ", "\u221e": "\\infty", "\u2044": "/", "\u00b1": "\\pm", "\u2013": "-", "\u2010": "-",
            "\u00bd": "\\frac{1}{2}", "\u00bc": "\\frac{1}{4}", "\u00be": "\\frac{3}{4}", "\u2019": "'", "\u00a0": " "}


def _text_repl(m: "re.Match") -> str:
    inner = m.group(1).strip()
    power = m.group(2) or ""
    words = [w.strip(".,") for w in inner.lower().split()]
    if words == ["and"]:            # '\frac{..}{2} \text{ and } \frac{..}{2}' -> a comma-separated list
        return ","
    if words and all(w in _UNIT_WORDS for w in words):
        return ""                   # drop the unit AND its exponent: 'cm}^2' is part of the unit
    return inner + power


def canon(s) -> str:
    """Canonical string form of a LaTeX/plain maths answer (never raises)."""
    if s is None:
        return ""
    s = str(s)
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    s = common.strip_math_delims(s)
    s = s.strip()
    # outer \boxed{} left by a model that nests boxes, e.g. \boxed{\boxed{7}}
    b = common.last_boxed(s)
    if b is not None and b.strip():
        s = b.strip()
    s = _TEXT_CMD.sub(_text_repl, s)
    s = _SPACING.sub("", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cfrac", "\\frac")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = _DEGREES.sub("", s)
    s = s.replace("{,}", ",")
    # thousands separators: only 'd,ddd' with NO whitespace after the comma ('6, 300' is a list) and never
    # inside a tuple/set/interval ('(6, 300)' must not become 6300) - hence before the whitespace removal
    if not s.lstrip().startswith(("(", "[", "{")):
        s = _THOUSANDS.sub("", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    s = _TRAIL_UNITS.sub("", s)
    s = re.sub(r"[.;,:]+$", "", s)
    s = _MODULO.sub("", s)
    s = re.sub(r"[.;,:]+$", "", s)
    s = _LEAD_LABEL.sub("", s)
    s = s.lstrip("=")
    s = _LEAD_ASSIGN.sub("", s)
    # bare-argument \frac / \sqrt -> braced
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    # number forms: 007 -> 7, 7.0 -> 7, +7 -> 7
    s = re.sub(r"(?<![\d.])0+(?=\d)", "", s)
    s = re.sub(r"(?<=\d)\.0+(?![\d])", "", s)
    s = re.sub(r"^\+", "", s)
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\^\{(\d+|[a-zA-Z])\}", r"^\1", s)
    return s


# ======================================================================================
# sympy-free numeric evaluation
# ======================================================================================

_MACRO = re.compile(r"\\(frac|sqrt|binom|dbinom|tbinom)")


def _group(s: str, i: int) -> Optional[tuple[str, int]]:
    """A LaTeX argument at s[i:]: a brace-matched {...} or one token. -> (content, next index)."""
    if i >= len(s):
        return None
    if s[i] == "{":
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
        return None
    m = re.match(r"\\[a-zA-Z]+|\d|[a-zA-Z]", s[i:])
    if m:
        return m.group(0), i + m.end()
    return None


def _expand_macros(s: str) -> Optional[str]:
    out: list[str] = []
    i = 0
    while True:
        m = _MACRO.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        name, j = m.group(1), m.end()
        if name == "sqrt":
            n = None
            if j < len(s) and s[j] == "[":
                k = s.find("]", j)
                if k < 0:
                    return None
                n, j = s[j + 1:k], k + 1
            g = _group(s, j)
            if g is None:
                return None
            a = _expand_macros(g[0])
            if a is None:
                return None
            if n:
                nn = _expand_macros(n)
                if nn is None:
                    return None
                out.append(f"(({a})**(1/({nn})))")
            else:
                out.append(f"(({a})**0.5)")
            i = g[1]
        else:
            g1 = _group(s, j)
            if g1 is None:
                return None
            g2 = _group(s, g1[1])
            if g2 is None:
                return None
            a, b = _expand_macros(g1[0]), _expand_macros(g2[0])
            if a is None or b is None:
                return None
            out.append(f"(({a})/({b}))" if name == "frac" else f"comb(({a}),({b}))")
            i = g2[1]
    return "".join(out)


_ALLOWED_IDENTS = {"pi", "comb", "factorial"}


def to_python(c: str) -> Optional[str]:
    """Canonical string -> a Python arithmetic expression (or None when not purely numeric)."""
    if not c or "\\infty" in c:
        return None
    s = re.sub(r"(\d)\\frac", r"\1+\\frac", c)          # mixed numbers 1\frac{4}{5}
    s = _expand_macros(s)
    if s is None:
        return None
    s = s.replace("\\pi", "(pi)").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d+)!", r"factorial(\1)", s)
    if "\\" in s or "!" in s:
        return None
    idents = set(re.findall(r"[a-zA-Z_]+", s))
    has_i = "i" in idents
    idents.discard("i")
    if idents - _ALLOWED_IDENTS:
        return None
    # implicit multiplication
    s = re.sub(r"(?<=[\d)])(?=\()", "*", s)
    s = re.sub(r"(?<=\))(?=\d)", "*", s)
    s = re.sub(r"(?<=[\d)])(?=[a-zA-Z])", "*", s)
    if has_i:
        s = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "(1j)", s)
    return s


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        raise ValueError(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("unary")
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Pow):
            if isinstance(b, complex) or abs(b) > 4096 or (isinstance(a, int) and abs(a) > 10 ** 6 and abs(b) > 64):
                raise ValueError("pow")
            return a ** b
        raise ValueError("binop")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_ev(x) for x in node.args]
        if node.func.id == "factorial" and len(args) == 1:
            n = args[0]
            if isinstance(n, float) and n.is_integer():
                n = int(n)
            if not isinstance(n, int) or n < 0 or n > 3000:
                raise ValueError("factorial")
            return math.factorial(n)
        if node.func.id == "comb" and len(args) == 2:
            n, k = args
            if isinstance(n, float) and n.is_integer():
                n = int(n)
            if isinstance(k, float) and k.is_integer():
                k = int(k)
            if not (isinstance(n, int) and isinstance(k, int)) or n < 0 or k < 0 or n > 5000:
                raise ValueError("comb")
            return math.comb(n, k)
    raise ValueError(type(node).__name__)


def to_number(c: str):
    """Numeric value (int | float | complex) of a canonical string, or None."""
    expr = to_python(c)
    if expr is None or len(expr) > 2000:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        v = _ev(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError, MemoryError):
        return None
    if isinstance(v, complex):
        if abs(v.imag) <= 1e-12 * max(1.0, abs(v.real)):
            v = v.real
        else:
            return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        if v.is_integer() and abs(v) < 1e15:
            return int(v)
    return v


def num_equal(a, b, rel: float = 1e-9) -> bool:
    """Numeric equality.  1e-9 relative is far above double-precision error for any closed form we evaluate
    (sqrt/pi/frac combinations agree to ~1e-15) yet rejects every decimal approximation shorter than 10 s.f.
    and every distinct fraction with denominators below ~3e4 (Farey neighbours differ by >= 1/(q q'))."""
    if a is None or b is None:
        return False
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        return abs(complex(a) - complex(b)) <= rel * max(1.0, abs(complex(b)))
    except (OverflowError, TypeError):
        return False


# ======================================================================================
# structure: tuples, sets, \pm pairs
# ======================================================================================

def _split_top(s: str, sep: str = ",") -> list[str]:
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _unwrap(s: str) -> tuple[str, str]:
    """-> (kind, inner) with kind in bare | tuple | set | interval."""
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        depth = 0
        for k, ch in enumerate(s):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if depth == 0 and k < len(s) - 1:
                return "bare", s
        inner = s[1:-1]
        if s[0] == "{" and s[-1] == "}":
            return "set", inner
        if (s[0], s[-1]) in (("(", ")"), ("[", "]")):
            if len(_split_top(inner)) >= 2:
                return "tuple", inner
            return "bare", inner if s[0] == "(" else s
        return "interval", s
    return "bare", s


def _expand_pm(parts: list[str]) -> tuple[list[str], bool]:
    out, changed = [], False
    for p in parts:
        if "\\pm" in p:
            out.append(p.replace("\\pm", "+", 1))
            out.append(p.replace("\\pm", "-", 1))
            changed = True
        elif "\\mp" in p:
            out.append(p.replace("\\mp", "-", 1))
            out.append(p.replace("\\mp", "+", 1))
            changed = True
        else:
            out.append(p)
    return out, changed


def components(c: str) -> tuple[str, list[str], bool]:
    """(kind, parts, ordered) of a canonical string."""
    kind, inner = _unwrap(c)
    if kind == "interval":
        return kind, [c], True
    parts = [_rhs(p) for p in _split_top(inner)]
    parts, pm = _expand_pm(parts)
    ordered = kind == "tuple" and not pm
    return kind, parts, ordered


def _rhs(p: str) -> str:
    """'AB=5', '\\frac{7}{2}=3.5', 'x=\\frac{-1+\\sqrt{17}}{2}' -> the last non-empty top-level '=' segment."""
    if "=" not in p:
        return p
    segs = [x for x in _split_top(p, "=") if x]
    return segs[-1] if segs else p


def _atom_equal(a: str, b: str) -> Optional[str]:
    """a = candidate atom, b = gold atom.  A bare decimal candidate must match the gold to ~double precision:
    '0.5' == \\frac{1}{2} and '-0.96' == -\\frac{24}{25}, but '2.44949', '0.649975' or '69.9999999' are
    approximations and count as wrong (MathArena convention)."""
    if a == b:
        return "string"
    rel = 1e-9
    if _PLAIN_LITERAL.match(a):
        rel = 1e-12
        gv = to_number(b)
        if ("\\sqrt" in b or "\\pi" in b) and not isinstance(gv, int):
            return None     # a decimal for an irrational closed form is an approximation however many digits it has
    if num_equal(to_number(a), to_number(b), rel):
        return "numeric"
    return None


def equivalent(candidate: str, gold: str) -> tuple[bool, str]:
    """(correct, how) with how in string | numeric | tuple | set | mismatch | empty."""
    c, g = canon(candidate), canon(gold)
    if not c:
        return False, "empty"
    if c == g:
        return True, "string"
    gk, gparts, gordered = components(g)
    ck, cparts, _ = components(c)
    if gk == "interval" or ck == "interval":
        return False, "mismatch"
    if len(gparts) == 1 and len(cparts) == 1:
        how = _atom_equal(cparts[0], gparts[0])
        return (True, how) if how else (False, "mismatch")
    if len(gparts) != len(cparts):
        return False, "mismatch"
    if gordered:
        if all(_atom_equal(x, y) for x, y in zip(cparts, gparts)):
            return True, "tuple"
        return False, "mismatch"
    left = list(cparts)
    for y in gparts:
        hit = next((i for i, x in enumerate(left) if _atom_equal(x, y)), None)
        if hit is None:
            return False, "mismatch"
        left.pop(hit)
    return True, "set"


def gradable(gold: str) -> bool:
    """True when every component of the gold answer evaluates numerically (exact programmatic grading)."""
    c = canon(gold)
    if not c:
        return False
    kind, parts, _ = components(c)
    if kind == "interval":
        return False
    return all(to_number(p) is not None for p in parts)


# ======================================================================================
# extraction + scoring
# ======================================================================================

_PHRASE_CUT = re.compile(r"(?i)\s*[,;]?\s+(?:so|since|because|which|and|as|hence|therefore|thus|i\s+hope|where)\b.*$")

# ---- two boxes: self-correction (last wins) vs simultaneous hedge (unparsed) -------------------
# The convention stays "the last \boxed{} wins", because a model that revises itself
# ("\boxed{71} ... actually \boxed{70}") means the later value.  The one case where that convention
# turns a refusal to commit into a coin flip is a SIMULTANEOUS hedge - "either \boxed{A} or \boxed{B}" -
# where the two boxes are joined by nothing but a disjunction.  Those are reported 'unparsed'.
# The test is deliberately narrow: the text BETWEEN the last two boxes must consist only of disjunction
# words and punctuation, and be short.  Any prose in the gap (a self-correction cue such as "actually",
# "wait", "I made an error", but equally "but rechecking ... gives") means the later box supersedes the
# earlier one, so the last box wins exactly as before.
_BOXED_CMD = re.compile(r"\\(?:boxed|fbox)\b\s*")
_HEDGE_WORD = r"(?:or|and/or|alternatively|possibly|perhaps|maybe|either|else|equivalently)"
_HEDGE_GAP = re.compile(rf"(?i)^[\s,;/&.]*(?:{_HEDGE_WORD}[\s,;/&.]*)+$")
_HEDGE_GAP_MAX = 40


def _boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """[(start of the \\boxed command, end of the box, content)] - the spans behind common.find_boxed().

    Mirrors common.find_boxed() exactly; extract() cross-checks the contents against it and falls back to
    the plain last-box path if the two ever disagree, so this copy can never change what is extracted."""
    out: list[tuple[int, int, str]] = []
    for m in _BOXED_CMD.finditer(text):
        pos = m.end()
        if pos >= len(text):
            continue
        if text[pos] == "{":
            depth = 0
            i = pos
            while i < len(text):
                ch = text[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append((m.start(), i + 1, text[pos + 1:i]))
                        break
                i += 1
        else:
            bm = re.match(r"[^\s$\\]+", text[pos:])
            if bm:
                out.append((m.start(), pos + bm.end(), bm.group(0)))
    return out


def _same_value(a: str, b: str) -> bool:
    """True when two \\boxed{} contents denote the same answer.  Restating one answer in two notations
    ('\\boxed{0.5} or \\boxed{\\frac{1}{2}}', '\\boxed{\\dfrac{1}{4}} or equivalently \\boxed{0.25}') is a
    single committed answer, not a hedge, so it must keep scoring as it did before the hedge rule existed.
    Order-independent, because equivalent() applies the decimal-approximation rule to its first argument
    only (an exact form and a decimal approximation of it are NOT the same value and stay a hedge)."""
    if canon(a) == canon(b):
        return True
    if len(a) > 400 or len(b) > 400:
        return False
    return equivalent(a, b)[0] or equivalent(b, a)[0]


def hedged_boxes(text: str) -> Optional[tuple[str, str, str]]:
    """(first, second, gap) when the last two \\boxed{} are a simultaneous hedge, else None."""
    spans = _boxed_spans(text)
    if [c for _, _, c in spans] != common.find_boxed(text) or len(spans) < 2:
        return None
    (_, end1, c1), (start2, _, c2) = spans[-2], spans[-1]
    if not c1.strip() or not c2.strip() or _same_value(c1, c2):
        return None
    gap = text[end1:start2]
    if len(gap) <= _HEDGE_GAP_MAX and _HEDGE_GAP.match(gap):
        return c1.strip(), c2.strip(), gap
    return None


def _trim_phrase(c: str) -> str:
    """Tidy an 'answer is ...' candidate: prefer the last $...$ group, cut at sentence/clause ends."""
    c = re.sub(r"^[\s:=*]+", "", c).strip()          # 'answer is: **70**' -> '70**'
    dollars = re.findall(r"\$([^$]+)\$", c)
    if dollars:
        return dollars[-1].strip()
    c = re.split(r"\.\s+(?=[A-Z])", c, 1)[0]
    c = _PHRASE_CUT.sub("", c)
    return c.strip().strip("*").strip().strip("$").strip()


def extract(text: str) -> tuple[Optional[str], str]:
    text = text or ""
    if hedged_boxes(text) is not None:
        return None, "ambiguous_boxed"
    cand, method = common.extract_final_answer(text, allow_last_integer=False)
    if cand is not None and method == "phrase":
        cand = _trim_phrase(cand) or None
    return cand, method


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    expected = str(item.get("answer", ""))
    cand, method = extract(response_text)
    if cand is None:
        if method == "ambiguous_boxed":
            hb = hedged_boxes(response_text or "")
            return Verdict.unparsed(canon(expected),
                                    {"method": method, "boxes": [hb[0][:100], hb[1][:100]] if hb else []},
                                    ["ambiguous_boxed"])
        return Verdict.unparsed(canon(expected), {"method": method}, ["no_boxed", "no_answer"])
    flags = [] if method == "boxed" else ["no_boxed"]
    if len(cand) > 400:
        return Verdict(False, extracted=cand[:200], expected=canon(expected),
                       detail={"method": method, "match": "long_candidate"}, flags=flags + ["long_candidate"])
    ok, how = equivalent(cand, expected)
    return Verdict(ok, extracted=cand[:200], expected=canon(expected),
                   detail={"method": method, "match": how, "canonical": canon(cand)}, flags=flags)


def mock_response(item: dict):
    """Oracle text for mock_server.py: a plausible preamble (with an unboxed decoy number) ending in \\boxed{answer}."""
    return ("Let me set this up carefully. A quick estimate suggests something near 999, but working it "
            f"through exactly gives the final answer.\n\n\\boxed{{{item['answer']}}}")


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    if not scored:
        return {}
    unparsed = sum(1 for r in scored if r.get("status") == "unparsed")
    hedged = sum(1 for r in scored if (r.get("detail") or {}).get("method") == "ambiguous_boxed")
    boxed = sum(1 for r in scored if (r.get("detail") or {}).get("method") == "boxed")
    matches: dict[str, int] = {}
    for r in scored:
        if r.get("correct"):
            k = (r.get("detail") or {}).get("match") or "?"
            matches[k] = matches.get(k, 0) + 1
    return {"unparsed_rate": round(unparsed / len(scored), 4), "boxed_rate": round(boxed / len(scored), 4),
            "ambiguous_boxed": hedged, "match_methods": dict(sorted(matches.items()))}


# ======================================================================================
# prepare
# ======================================================================================

def _raw_dir(data_dir: str, repo: str) -> str:
    return os.path.join(data_dir, "raw", repo.replace("/", "_"))


def _revision(data_dir: str, repo: str, refresh: bool, log: Callable[[str], None]) -> Optional[str]:
    """Pinned commit sha of a HF dataset repo (cached api.json); None when the API is unreachable."""
    try:
        info = common.fetch_json(f"{common.HF_ENDPOINT}/api/datasets/{repo}", os.path.join(_raw_dir(data_dir, repo), "api.json"),
                                 retries=2, timeout=30, refresh=refresh, log=log)
        return info.get("sha") if isinstance(info, dict) else None
    except Exception as e:  # informational only
        log(f"   math: revision lookup for {repo} failed ({e}); recording revision=None")
        return None


def _make_item(sub: str, iid: str, problem: str, answer: str, meta: dict) -> dict:
    return {"id": iid, "family": NAME, "subfamily": sub, "order": 0,
            "messages": [{"role": "user", "content": problem.strip() + PROMPT_SUFFIX}],
            "answer": answer, "meta": meta}


def _assign_order(items: list[dict], seed: int, sub: str) -> None:
    rng = common.seeded_rng(seed, NAME, sub)
    perm = list(range(len(items)))
    rng.shuffle(perm)
    for pos, idx in enumerate(perm):
        items[idx]["order"] = pos


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, n_math500: Optional[int] = None,
            hmmt: bool = True, **opts) -> dict:
    n_math500 = int(n_math500) if n_math500 is not None else N_MATH500.get(profile, N_MATH500["default"])
    sources: dict = {}
    counts: dict = {}
    pools: dict = {}
    notes: list[str] = list(NOTES)
    items: list[dict] = []

    # ---- aime25 -------------------------------------------------------------------------
    url = common.hf_url(AIME_REPO, AIME_FILE)
    dest = os.path.join(_raw_dir(data_dir, AIME_REPO), AIME_FILE)
    f = common.fetch(url, dest, refresh=refresh, log=log)
    rows = common.read_jsonl(dest)
    rows.sort(key=lambda r: int(r["id"]))
    sources[AIME_REPO] = {"url": url, "sha256": f["sha256"], "bytes": f["bytes"], "revision": _revision(data_dir, AIME_REPO, refresh, log)}
    pools["aime25"] = len(rows)
    if len(rows) < N_AIME:
        if not allow_short:
            raise ShortPool(f"aime25: {len(rows)} rows < {N_AIME}")
        notes.append(f"aime25: only {len(rows)} rows")
    aime: list[dict] = []
    for r in rows[:N_AIME]:
        k = int(r["id"])
        contest, num = ("I", k + 1) if k < 15 else ("II", k - 14)
        ans = r["answer"]
        ans = str(int(ans)) if isinstance(ans, (int, float)) or re.fullmatch(r"\s*\d+\s*", str(ans)) else str(ans).strip()
        aime.append(_make_item("aime25", f"aime25-{contest}-{num}", r["problem"], ans,
                               {"source": AIME_REPO, "source_id": str(r["id"]), "contest": f"AIME 2025 {contest}", "problem": num}))
    _assign_order(aime, seed, "aime25")
    counts["aime25"] = len(aime)
    items += aime

    # ---- hmmt25 -------------------------------------------------------------------------
    hmmt_items: list[dict] = []
    if hmmt:
        dest = os.path.join(_raw_dir(data_dir, HMMT_REPO), "rows.json")
        try:
            data = common.fetch_json(HMMT_ROWS_URL, dest, refresh=refresh, log=log)
        except Exception as e:
            if not allow_short:
                raise
            notes.append(f"hmmt25 skipped: could not fetch {HMMT_ROWS_URL}: {e}")
            data = None
        if data is not None:
            hrows = [x["row"] for x in data.get("rows", []) if isinstance(x, dict) and "row" in x]
            hrows.sort(key=lambda r: int(r["problem_idx"]))
            with open(dest, "rb") as fh:
                hsha = common.sha256_bytes(fh.read())
            sources[HMMT_REPO] = {"url": HMMT_ROWS_URL, "sha256": hsha, "bytes": os.path.getsize(dest),
                                  "revision": _revision(data_dir, HMMT_REPO, refresh, log), "license": "cc-by-nc-sa-4.0"}
            pools["hmmt25"] = len(hrows)
            if len(hrows) < N_HMMT:
                if not allow_short:
                    raise ShortPool(f"hmmt25: {len(hrows)} rows < {N_HMMT}")
                notes.append(f"hmmt25: only {len(hrows)} rows")
            for r in hrows[:N_HMMT]:
                idx = int(r["problem_idx"])
                hmmt_items.append(_make_item("hmmt25", f"hmmt25-feb-{idx}", r["problem"], str(r["answer"]).strip(),
                                             {"source": HMMT_REPO, "source_id": str(idx), "contest": "HMMT February 2025",
                                              "problem_type": list(r.get("problem_type") or [])}))
            _assign_order(hmmt_items, seed, "hmmt25")
    else:
        notes.append("hmmt25 disabled by --opt math.hmmt=false")
    counts["hmmt25"] = len(hmmt_items)
    items += hmmt_items

    # ---- math500_hard -------------------------------------------------------------------
    url = common.hf_url(MATH500_REPO, MATH500_FILE)
    dest = os.path.join(_raw_dir(data_dir, MATH500_REPO), MATH500_FILE)
    f = common.fetch(url, dest, refresh=refresh, log=log)
    mrows = common.read_jsonl(dest)
    sources[MATH500_REPO] = {"url": url, "sha256": f["sha256"], "bytes": f["bytes"], "revision": _revision(data_dir, MATH500_REPO, refresh, log)}
    level5 = [r for r in mrows if int(r.get("level", 0)) == 5]
    pool = [r for r in level5 if gradable(str(r["answer"]))]
    pool.sort(key=lambda r: r["unique_id"])
    pools["math500_hard"] = len(pool)
    notes.append(f"math500_hard: {len(level5)} level-5 problems, {len(pool)} with programmatically gradable answers")
    if len(pool) < n_math500:
        if not allow_short:
            raise ShortPool(f"math500_hard: gradable level-5 pool {len(pool)} < {n_math500}")
        notes.append(f"math500_hard: pool {len(pool)} < requested {n_math500}")
    # shuffle-then-prefix, NOT rng.sample(pool, k): random.sample() switches algorithm with k, so the
    # default 20 would not be a subset of the full-profile 60.  A single shuffle of the (deterministically
    # sorted) pool makes every n a prefix of every larger n, which is what the README promises about
    # smaller runs being nested subsets.  The prefix also fixes `order`, so --limit nesting agrees.
    rng = common.seeded_rng(seed, NAME, "math500_hard")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked = shuffled[:min(n_math500, len(pool))]
    m500: list[dict] = []
    for pos, r in enumerate(picked):
        slug = re.sub(r"[^a-z0-9]+", "-", r["unique_id"].lower().replace("test/", "").replace(".json", "")).strip("-")
        it = _make_item("math500_hard", f"math500-hard-{slug}", r["problem"], str(r["answer"]).strip(),
                        {"source": MATH500_REPO, "source_id": r["unique_id"], "subject": r.get("subject"), "level": 5})
        it["order"] = pos
        m500.append(it)
    counts["math500_hard"] = len(m500)
    items += m500

    # ---- write --------------------------------------------------------------------------
    for it in items:
        if not equivalent(it["answer"], it["answer"])[0]:  # the gold must grade against itself
            raise RuntimeError(f"math: gold answer of {it['id']} does not canonicalise: {it['answer']!r}")
    items.sort(key=lambda it: (SUBFAMILIES.index(it["subfamily"]), it["order"], it["id"]))
    rel = f"items/{NAME}.jsonl"
    common.write_jsonl(_base.items_path(NAME, data_dir), items)
    return {"file": rel, "counts": counts, "pools": pools, "sources": sources, "notes": notes,
            "manifest_extra": {"math_profile": {"profile": profile, "n_math500": n_math500}}}
