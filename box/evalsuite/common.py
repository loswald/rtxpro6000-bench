"""
common.py - shared core of the Sqwish Labs quality eval suite.

Everything that more than one entry point or family needs lives here:

  * constants, seeded RNG, hashing, jsonl I/O, atomic writes
  * Wilson interval and exact McNemar (core/stats of the design spec)
  * response normalisation: <think>/reasoning stripping, harmony-leak handling,
    answer-from-reasoning fallback, degenerate-loop detection (same rule as box/quality20.py)
  * answer extraction helpers: \\boxed{} with brace matching, "answer is" phrases,
    last integer, bounded multiple-choice letter cascade
  * Verdict / ItemOutcome / RunContext - the objects that cross the family plugin boundary
  * fetch(): urllib download with .part files, sha256, retries (used by prepare_data / families)
  * ChatClient: aiohttp OpenAI-compatible chat client with round-robin (sticky) base URLs,
    a request-level concurrency semaphore, timeouts, retries with back-off, token accounting,
    /v1/models + /version probing and /tokenize

Python 3.12, standard library + aiohttp only.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Optional

try:  # aiohttp is only needed by the runner and the mock; prepare_data must work without it
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

# --------------------------------------------------------------------------------------
# constants and small utilities
# --------------------------------------------------------------------------------------

DEFAULT_SEED = 20260903
EVALSUITE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(EVALSUITE_DIR, "data")
USER_AGENT = "sqwish-evalsuite/1.0"
SCHEMA_VERSION = 1

# item/run statuses (design spec 5.0)
STATUSES = ("correct", "wrong", "unparsed", "truncated", "empty", "error", "cancelled", "skipped")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def seeded_rng(seed: int, *parts: Any) -> random.Random:
    """The suite's only source of randomness: sha256 of 'seed:part:part' -> 64-bit seed.
    Never Python's salted hash()."""
    key = ":".join([str(seed)] + [str(p) for p in parts])
    return random.Random(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"))


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Sorted keys, one object per line, atomic (deterministic bytes for identical rows)."""
    atomic_write_text(path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def percentile(values: list[float], q: float) -> Optional[float]:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[k]


def mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def parse_kv_list(spec: Optional[str], cast: Callable[[str], Any] = str) -> dict:
    """'a=1,b=2' -> {'a': cast('1'), 'b': cast('2')}"""
    out: dict = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = cast(v.strip())
    return out


def json_or_str(s: str) -> Any:
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


class ShortPool(Exception):
    """Raised by a family's prepare() when a source pool is smaller than requested."""


def evalsuite_git_sha() -> Optional[str]:
    """HEAD of the repository containing evalsuite/, or None (no git, not a repo)."""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", EVALSUITE_DIR, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def load_manifest(data_dir: str) -> Optional[dict]:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_digest(manifest: dict) -> str:
    """sha256 of the manifest object without its own manifest_sha256 field (sorted keys)."""
    m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return sha256_json(m)


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval. n == 0 -> [0, 1]."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------------------
# the objects that cross the family plugin boundary
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class Verdict:
    """What a family scorer returns for one item.

    correct   - the headline boolean
    status    - correct | wrong | unparsed (no answer could be extracted)
    score     - 0..1 partial credit (needles found / 4, turns passed / turns, ...); defaults to correct
    extracted - what the scorer pulled out of the response (for audit)
    expected  - the ground truth as the scorer compared it
    detail    - free-form scorer detail (reason codes, per-test results, ...)
    flags     - short strings appended to the item record's flags (e.g. 'no_boxed')
    """
    correct: bool
    status: str = ""
    score: Optional[float] = None
    extracted: Any = None
    expected: Any = None
    detail: dict = dataclasses.field(default_factory=dict)
    flags: list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not self.status:
            self.status = "correct" if self.correct else "wrong"
        if self.score is None:
            self.score = 1.0 if self.correct else 0.0

    @classmethod
    def unparsed(cls, expected=None, detail=None, flags=None) -> "Verdict":
        return cls(False, "unparsed", 0.0, None, expected, detail or {}, list(flags or []))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ItemOutcome:
    """Returned by a family's optional run_item() (multi-request items such as BFCL multi-turn).
    The runner fills in the timing; the family reports what it observed."""
    verdict: Verdict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    retries: int = 0
    finish_reason: Optional[str] = None
    content: str = ""
    reasoning: str = ""
    flags: list = dataclasses.field(default_factory=list)
    error: Optional[str] = None          # set -> status 'error' (verdict ignored)
    base_url: Optional[str] = None
    latency_s: float = 0.0
    extra: dict = dataclasses.field(default_factory=dict)   # copied into the item record


@dataclasses.dataclass
class RunContext:
    """Handed to build_messages() / score() / run_item() / prepare_run()."""
    family: str
    data_dir: str
    cfg: Any = None                       # the run configuration (argparse Namespace)
    client: Any = None                    # ChatClient (None in --dry-run and in prepare_data)
    max_tokens: int = 2048
    reasoning: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = DEFAULT_SEED
    profile: str = "default"
    opts: dict = dataclasses.field(default_factory=dict)   # --family-opt values for this family
    extra_body: dict = dataclasses.field(default_factory=dict)
    log: Callable[[str], None] = print
    deadline: Optional[float] = None      # time.monotonic() deadline for new requests

    def opt(self, key: str, default: Any = None) -> Any:
        return self.opts.get(key, default)


# --------------------------------------------------------------------------------------
# response normalisation (design spec 5.0)
# --------------------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<(think|thinking|reason|reasoning|thought)>(.*?)</\1>\s*", re.S | re.I)
_BOT_BLOCK = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*", re.S)
_THINK_OPEN = re.compile(r"<(think|thinking)>", re.I)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", re.S)
_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", re.S)
_HARMONY_TOKEN = re.compile(r"<\|[a-z_]+\|>(?:assistant)?")


@dataclasses.dataclass
class Normalized:
    visible: str                 # what the scorers see
    reasoning: str               # everything that was hidden reasoning
    flags: list                  # unclosed_think, harmony_leak, answer_from_reasoning, degenerate
    status: str                  # ok | truncated | empty
    finish_reason: Optional[str]


def message_text(content: Any) -> str:
    """OpenAI content may be a string or a list of {'type': 'text', 'text': ...} parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def repetition_stats(text: str) -> tuple[int, float]:
    """(max repeated 6-gram count, distinct-token ratio) over \\S+ tokens - as box/quality20.py."""
    toks = re.findall(r"\S+", text)
    if len(toks) < 12:
        return 0, 1.0
    grams = collections.Counter(tuple(toks[i:i + 6]) for i in range(len(toks) - 5))
    return max(grams.values()), len(set(toks)) / len(toks)


def is_degenerate(text: str, finish_reason: Optional[str]) -> bool:
    """A loop: hit the token cap and (same 6-gram >= 4 times or vocabulary collapsed) in the last 2000 chars."""
    if finish_reason != "length":
        return False
    top, distinct = repetition_stats(text[-2000:])
    return top >= 4 or distinct < 0.35


def normalize_response(message: dict, finish_reason: Optional[str]) -> Normalized:
    """Split an assistant message into visible text and reasoning; classify empties.

    1. content + reasoning field ('reasoning' on current vLLM, 'reasoning_content' on older builds)
    2. closed <think>...</think> (and siblings) blocks -> reasoning
    3. an unclosed <think> swallows the rest -> reasoning, flag unclosed_think
    4. harmony leak (<|channel|>analysis ...) -> reasoning, keep the final channel, flag harmony_leak
    5. strip
    6. empty visible: finish 'length' -> truncated; else reasoning present -> visible = reasoning tail,
       flag answer_from_reasoning; else empty
    7. degenerate detection on content + reasoning when finish_reason == 'length'
    """
    message = message or {}
    content = message_text(message.get("content"))
    reasoning = message_text(message.get("reasoning") or message.get("reasoning_content") or "")
    flags: list[str] = []
    hidden: list[str] = [reasoning] if reasoning else []

    def _take2(m):
        hidden.append(m.group(2))
        return ""

    def _take1(m):
        hidden.append(m.group(1))
        return ""

    content = _THINK_BLOCK.sub(_take2, content)
    content = _BOT_BLOCK.sub(_take1, content)

    m = _THINK_OPEN.search(content)
    if m:
        hidden.append(content[m.end():])
        content = content[:m.start()]
        flags.append("unclosed_think")

    if "<|channel|>" in content or "<|message|>" in content:
        for am in _HARMONY_ANALYSIS.finditer(content):
            hidden.append(am.group(1))
        fm = _HARMONY_FINAL.search(content)
        if fm:
            content = fm.group(1)
        else:
            content = _HARMONY_ANALYSIS.sub("", content)
        content = _HARMONY_TOKEN.sub("", content)
        flags.append("harmony_leak")

    visible = content.strip()
    reasoning_all = "\n".join(h for h in hidden if h)
    status = "ok"
    if not visible:
        if finish_reason == "length":
            status = "truncated"
        elif reasoning_all.strip():
            visible = reasoning_all.strip()[-4000:]
            flags.append("answer_from_reasoning")
        else:
            status = "empty"
    if is_degenerate(content + reasoning_all, finish_reason):
        flags.append("degenerate")
    return Normalized(visible, reasoning_all, flags, status, finish_reason)


# --------------------------------------------------------------------------------------
# answer extraction helpers
# --------------------------------------------------------------------------------------

_BOXED_CMD = re.compile(r"\\(?:boxed|fbox)\b\s*")


def find_boxed(text: str) -> list[str]:
    """All \\boxed{...} / \\fbox{...} contents with brace matching (nested braces, \\{ \\} escapes).
    The bare form '\\boxed 5' is accepted. Unterminated boxes are ignored."""
    out: list[str] = []
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
                        out.append(text[pos + 1:i])
                        break
                i += 1
        else:
            bm = re.match(r"[^\s$\\]+", text[pos:])
            if bm:
                out.append(bm.group(0))
    return out


def last_boxed(text: str) -> Optional[str]:
    boxes = find_boxed(text)
    return boxes[-1] if boxes else None


_ANSWER_PHRASE = re.compile(r"(?i)(?:final answer|answer)\s*(?:is|:)\s*")


def extract_answer_phrase(text: str) -> Optional[str]:
    """Text after the last 'answer is' / 'final answer:' up to end of line, $ and * stripped."""
    last = None
    for m in _ANSWER_PHRASE.finditer(text):
        last = m
    if last is None:
        return None
    rest = text[last.end():].split("\n", 1)[0]
    rest = rest.strip().strip("*").strip()
    rest = rest.strip("$").strip()
    rest = re.sub(r"[.。]+$", "", rest).strip()
    return rest or None


def last_integer(text: str) -> Optional[str]:
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return ints[-1] if ints else None


def extract_final_answer(visible: str, allow_last_integer: bool = False) -> tuple[Optional[str], str]:
    """(candidate, method) with method in {boxed, phrase, last_integer, none}.

    boxed -> 'answer is' phrase -> (optionally) last integer in the candidate / last 300 chars."""
    b = last_boxed(visible)
    if b is not None and b.strip():
        return b.strip(), "boxed"
    p = extract_answer_phrase(visible)
    if p:
        if allow_last_integer:
            li = last_integer(p)
            if li is not None:
                return li, "phrase"
        return p, "phrase"
    if allow_last_integer:
        li = last_integer(visible[-300:])
        if li is not None:
            return li, "last_integer"
    return None, "none"


def strip_math_delims(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    s = s.strip("$").strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    return s.strip()


_LETTER_STEPS = [
    re.compile(r"\\boxed\{\s*\(?\**([A-J])\**\)?\s*\}"),
    re.compile(r"(?i:answer is):?\s*\(?\**([A-J])\**\)?"),
    re.compile(r"(?i:answer):?\s*\(?\**([A-J])\**\)?"),
]
_LETTER_LINE = re.compile(r"^\s*\(?\**([A-J])\**\)?\s*[.:)]?\s*$")
_LETTER_LAST = re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)", re.S)


def extract_letter(visible: str, n_options: int = 10) -> tuple[Optional[str], str]:
    """Bounded multiple-choice letter cascade (design spec 5.5). Returns (letter|None, method)."""
    valid = set("ABCDEFGHIJ"[:max(1, min(10, n_options))])

    def _last_valid(rx, text):
        found = None
        for m in rx.finditer(text):
            if m.group(1) in valid:
                found = m.group(1)
        return found

    for i, rx in enumerate(_LETTER_STEPS):
        hit = _last_valid(rx, visible)
        if hit:
            return hit, ("boxed", "answer_is", "answer")[i]
    lines = [ln for ln in visible.splitlines() if ln.strip()][-3:]
    for ln in reversed(lines):
        m = _LETTER_LINE.match(ln)
        if m and m.group(1) in valid:
            return m.group(1), "bare_line"
    m = _LETTER_LAST.search(visible[-300:])
    if m and m.group(1) in valid:
        return m.group(1), "last_letter"
    return None, "none"


# --------------------------------------------------------------------------------------
# fetching (prepare_data and family prepare())
# --------------------------------------------------------------------------------------

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def hf_url(repo: str, filename: str, revision: str = "main", repo_type: str = "dataset") -> str:
    prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[repo_type]
    return f"{HF_ENDPOINT}/{prefix}{repo}/resolve/{revision}/{filename}"


def _headers_for(url: str) -> dict:
    h = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    tok = os.environ.get("HF_TOKEN")
    if tok and ("huggingface.co" in url or "datasets-server" in url):
        h["Authorization"] = f"Bearer {tok}"
    return h


def fetch(url: str, dest: str, expected_sha256: Optional[str] = None, retries: int = 5,
          timeout: float = 120.0, refresh: bool = False, log: Callable[[str], None] = print) -> dict:
    """Download url -> dest (via dest.part), verify/record sha256, reuse the cache when present.
    Returns {'path', 'sha256', 'bytes', 'cached'}."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    sha_path = dest + ".sha256"
    if os.path.exists(dest) and not refresh:
        digest = open(sha_path).read().strip() if os.path.exists(sha_path) else sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            return {"path": dest, "sha256": digest, "bytes": os.path.getsize(dest), "cached": True}
        log(f"[fetch] cached {dest} has sha256 {digest[:12]}.. != expected {expected_sha256[:12]}..; refetching")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers_for(url))
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest + ".part", "wb") as out:
                h = hashlib.sha256()
                n = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            digest = h.hexdigest()
            if expected_sha256 and digest != expected_sha256:
                os.remove(dest + ".part")
                raise IOError(f"sha256 mismatch for {url}: got {digest}, expected {expected_sha256}")
            os.replace(dest + ".part", dest)
            with open(sha_path, "w") as f:
                f.write(digest + "\n")
            return {"path": dest, "sha256": digest, "bytes": n, "cached": False}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code not in (408, 429):
                break  # a hard client error will not fix itself
            wait = 3 * (2 ** attempt)
            log(f"[fetch] attempt {attempt + 1}/{retries} failed for {url}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise IOError(f"could not fetch {url}: {last_err}")


def fetch_json(url: str, dest: str, **kw) -> Any:
    fetch(url, dest, **kw)
    with open(dest, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# aiohttp chat client
# --------------------------------------------------------------------------------------

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclasses.dataclass
class ChatResult:
    ok: bool
    message: dict = dataclasses.field(default_factory=dict)
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    base_url: str = ""
    retries: int = 0
    http_status: Optional[int] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None     # http | connection | timeout | malformed | context_length | cancelled | closed
    raw: Optional[dict] = None

    @property
    def content(self) -> str:
        return message_text(self.message.get("content")) if self.message else ""

    @property
    def reasoning(self) -> str:
        if not self.message:
            return ""
        return message_text(self.message.get("reasoning") or self.message.get("reasoning_content") or "")


class ChatClient:
    """OpenAI-compatible chat client over several vLLM replicas.

    * one aiohttp session per base URL, TCPConnector(limit=concurrency)
    * asyncio.Semaphore(concurrency) over REQUESTS (what 'concurrency 64' means to the server)
    * sticky routing: chat(route_key=i) pins to urls[i % n] (prefix-cache locality for chains);
      each retry moves to the next URL
    * retries on connection errors, timeouts, HTTP 408/409/429/5xx and malformed 200 bodies,
      back-off 2, 8, 20 s (+U(0,1)) by default; other 4xx are terminal
    * token accounting from `usage` (vLLM counts reasoning tokens in completion_tokens)
    """

    def __init__(self, base_urls: list[str], model: str, concurrency: int = 64, request_timeout: float = 600.0,
                 retries: int = 3, backoff: tuple = (2.0, 8.0, 20.0), connect_timeout: float = 10.0,
                 log: Callable[[str], None] = None, rng_seed: int = 0):
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for ChatClient (pip install aiohttp)")
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.retries = max(0, int(retries))
        self.backoff = tuple(float(b) for b in backoff) or (2.0,)
        self.log = log or (lambda s: print(s, file=sys.stderr))
        self._rng = random.Random(rng_seed)
        self._sem: Optional[asyncio.Semaphore] = None
        self._sessions: dict[str, Any] = {}
        self.totals = collections.Counter()          # requests, retries, errors, prompt_tokens, completion_tokens
        self.per_url: dict[str, collections.Counter] = {u: collections.Counter() for u in self.base_urls}
        self.max_model_len: Optional[int] = None
        self.served_models: list[str] = []
        self.versions: dict[str, Any] = {}
        self._closed = False

    # -- lifecycle ---------------------------------------------------------------------
    async def open(self) -> None:
        self._sem = asyncio.Semaphore(self.concurrency)
        for u in self.base_urls:
            if u not in self._sessions:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout, sock_connect=self.connect_timeout,
                                                sock_read=self.request_timeout)
                connector = aiohttp.TCPConnector(limit=self.concurrency)
                self._sessions[u] = aiohttp.ClientSession(timeout=timeout, connector=connector,
                                                          headers={"User-Agent": USER_AGENT,
                                                                   "Content-Type": "application/json"})

    async def close(self) -> None:
        self._closed = True
        for s in self._sessions.values():
            await s.close()
        self._sessions.clear()
        await asyncio.sleep(0.05)  # let the connectors finish closing

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _session(self, url: str):
        if url not in self._sessions:
            raise RuntimeError("ChatClient.open() was not awaited")
        return self._sessions[url]

    # -- probing -----------------------------------------------------------------------
    async def _get_json(self, url: str, path: str, timeout: float = 5.0) -> Optional[Any]:
        try:
            async with self._session(url).get(url + path, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status != 200:
                    return None
                return await r.json(content_type=None)
        except Exception:
            return None

    async def probe(self, attempts: int = 3, timeout: float = 5.0) -> dict:
        """GET /health (ignored on failure), /v1/models (attempts x timeout), /version.
        Drops dead URLs from self.base_urls. Returns a summary dict."""
        alive, dead, models = [], [], set()
        for u in list(self.base_urls):
            await self._get_json(u, "/health", timeout)
            data = None
            for _ in range(attempts):
                data = await self._get_json(u, "/v1/models", timeout)
                if data is not None:
                    break
                await asyncio.sleep(1.0)
            if data is None:
                dead.append(u)
                continue
            alive.append(u)
            for m in (data.get("data") or []):
                if isinstance(m, dict) and "id" in m:
                    models.add(m["id"])
            ver = await self._get_json(u, "/version", timeout)
            if ver:
                self.versions[u] = ver.get("version", ver)
        self.base_urls = alive
        self.served_models = sorted(models)
        return {"alive": alive, "dead": dead, "served_models": self.served_models, "versions": self.versions}

    async def tokenize(self, prompt: Optional[str] = None, messages: Optional[list] = None,
                       add_generation_prompt: bool = True, url: Optional[str] = None) -> Optional[dict]:
        """POST /tokenize (vLLM, server root). None when unsupported (404/405) or unreachable."""
        if not self.base_urls:
            return None
        u = url or self.base_urls[0]
        body: dict = {"model": self.model}
        if messages is not None:
            body["messages"] = messages
            body["add_generation_prompt"] = add_generation_prompt
        else:
            body["prompt"] = prompt or ""
        try:
            async with self._session(u).post(u + "/tokenize", json=body) as r:
                if r.status in (404, 405):
                    return None
                if r.status != 200:
                    return None
                data = await r.json(content_type=None)
        except Exception:
            return None
        if isinstance(data, dict) and data.get("max_model_len"):
            self.max_model_len = int(data["max_model_len"])
        return data if isinstance(data, dict) else None

    # -- chat --------------------------------------------------------------------------
    @staticmethod
    def _error_kind(status: int, body: str) -> str:
        low = body.lower()
        if any(k in low for k in ("context length", "max_model_len", "maximum context", "context_length", "too long")):
            return "context_length"
        return "http"

    async def chat(self, messages: list[dict], *, route_key: int = 0, max_tokens: int = 2048, temperature: float = 0.0,
                   top_p: float = 1.0, seed: Optional[int] = DEFAULT_SEED, tools: Optional[list] = None,
                   tool_choice: Optional[Any] = None, extra_body: Optional[dict] = None,
                   model: Optional[str] = None) -> ChatResult:
        if not self.base_urls:
            return ChatResult(False, error="no live base URL", error_kind="closed")
        body: dict = {"model": model or self.model, "messages": messages, "max_tokens": int(max_tokens),
                      "temperature": float(temperature), "top_p": float(top_p), "stream": False}
        if seed is not None:
            body["seed"] = int(seed)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if extra_body:
            body.update(extra_body)

        n_urls = len(self.base_urls)
        attempts = self.retries + 1
        last: Optional[ChatResult] = None
        t_all = time.monotonic()
        for attempt in range(attempts):
            url = self.base_urls[(route_key + attempt) % n_urls]
            if attempt > 0:
                self.totals["retries"] += 1
                self.per_url[url]["retries"] += 1
                wait = self.backoff[min(attempt - 1, len(self.backoff) - 1)] + self._rng.random()
                await asyncio.sleep(wait)
            if self._closed:
                return ChatResult(False, error="client closed", error_kind="closed", retries=attempt)
            self.totals["requests"] += 1
            self.per_url[url]["requests"] += 1
            t0 = time.monotonic()
            retryable = False
            try:
                async with self._sem:
                    async with self._session(url).post(url + "/v1/chat/completions", json=body) as r:
                        status = r.status
                        text = await r.text()
                lat = time.monotonic() - t0
                if status != 200:
                    retryable = status in RETRY_STATUS
                    kind = self._error_kind(status, text)
                    last = ChatResult(False, http_status=status, error=f"HTTP {status}: {text[:300]}",
                                      error_kind=kind, base_url=url, latency_s=lat, retries=attempt)
                    if kind == "context_length":
                        retryable = False
                else:
                    try:
                        data = json.loads(text)
                        choice = data["choices"][0]
                        msg = choice.get("message") or {}
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        retryable = True
                        last = ChatResult(False, http_status=200, error=f"malformed body: {e}: {text[:300]}",
                                          error_kind="malformed", base_url=url, latency_s=lat, retries=attempt)
                    else:
                        usage = data.get("usage") or {}
                        pt = int(usage.get("prompt_tokens") or 0)
                        ct = int(usage.get("completion_tokens") or 0)
                        self.totals["prompt_tokens"] += pt
                        self.totals["completion_tokens"] += ct
                        self.per_url[url]["completion_tokens"] += ct
                        return ChatResult(True, message=msg, finish_reason=choice.get("finish_reason"),
                                          prompt_tokens=pt, completion_tokens=ct, latency_s=lat, base_url=url,
                                          retries=attempt, http_status=200, raw=data)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                retryable = True
                last = ChatResult(False, error=f"timeout after {self.request_timeout:.0f}s", error_kind="timeout",
                                  base_url=url, latency_s=time.monotonic() - t0, retries=attempt)
            except aiohttp.ClientError as e:
                retryable = True
                last = ChatResult(False, error=f"{type(e).__name__}: {e}", error_kind="connection",
                                  base_url=url, latency_s=time.monotonic() - t0, retries=attempt)
            except Exception as e:  # pragma: no cover - defensive
                retryable = False
                last = ChatResult(False, error=f"{type(e).__name__}: {e}", error_kind="exception",
                                  base_url=url, latency_s=time.monotonic() - t0, retries=attempt)
            self.per_url[url]["errors"] += 1
            if not retryable:
                break
        self.totals["errors"] += 1
        assert last is not None
        last.latency_s = time.monotonic() - t_all
        return last

    def summary(self) -> dict:
        return {"totals": dict(self.totals), "per_url": {u: dict(c) for u, c in self.per_url.items()},
                "served_models": self.served_models, "versions": self.versions, "max_model_len": self.max_model_len}
