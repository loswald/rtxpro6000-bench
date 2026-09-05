#!/usr/bin/env python3
"""
mock_server.py - aiohttp OpenAI-compatible mock so the whole pipeline runs WITHOUT the node.

  python3 evalsuite/mock_server.py --port 9000 [--model m] [--items-dir evalsuite/data/items] [--lookup answers.json]
      --mode oracle|noisy|canned|echo|garbage [--accuracy 0.7]
      [--think none|inline|field|unclosed] [--reasoning-field reasoning|reasoning_content]
      [--fail-rate 0.0 --fail-mode first|always] [--latency-ms 0] [--jitter-ms 0] [--slow-every 0 --slow-ms 20000]
      [--truncate-rate 0.0] [--seed 7]

Faults: --fail-rate p returns HTTP 503 with probability p - by default only on the FIRST sighting of a prompt
(so a runner with retries >= 1 must end with n_error == 0), or on every request with --fail-mode always;
--slow-every N sleeps --slow-ms on every N-th request (request timeouts); --truncate-rate p cuts the answer in
half with finish_reason="length"; --latency-ms/--jitter-ms shape the response time (time-budget tests).

Answers come from a LOOKUP KEYED ON A MARKER IN THE PROMPT:

  * --items-dir  every item in data/items/*.jsonl is indexed twice: by the sha1 of its whitespace-normalised
                 last user message (exact match on the rendered prompt) and by a marker - item["mock_marker"]
                 if present, else the first 240 normalised chars of its last user message - searched as a
                 substring of the whole rendered prompt (so 5-shot exemplars, system prompts and suffixes
                 added by a family's build_messages() do not break the match).
                 The response is item["mock"] if present, else families.<family>.mock_response(item), else the
                 default oracle  "Let me think.\\nThe final answer is \\boxed{<answer>}."
  * --lookup     a JSON object {marker: response} or list of {"marker":..., "response":...}; a response is a
                 string or {"content", "reasoning", "finish_reason", "tool_calls": [{"name", "arguments"}]}.
  * no match     echo of the last 200 chars of the last user message.
  * tool results a last message with role "tool", or a user message starting with '[{"role": "tool"'
                 (BFCL prompt mode), is answered with "Done." (ends the multi-turn step loop).

Modes: oracle (correct), noisy (correct with probability --accuracy, seeded per item, else a plausible wrong
answer: +1 / next letter / altered string), canned ("I am not sure." everywhere -> exercises unparsed),
garbage (reversed prompt tail + random unicode), echo.  --think inline wraps a decoy "<think>...\\boxed{999}
...</think>" around the answer (the decoy must never be scored), field puts it into the reasoning field,
unclosed returns "<think>..." without a closing tag and finish_reason="length".

Endpoints: GET /health, GET /v1/models, GET /version, POST /tokenize, GET /stats, POST /v1/chat/completions.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiohttp import web  # noqa: E402

import common  # noqa: E402

DECOY = ("Let me think about this carefully. A tempting answer would be \\boxed{999} or (Z), and the access code "
         "might be ZZZZ9999... but let me reconsider before answering.")
MIN_MARKER = 24


def norm(s: str) -> str:
    return " ".join((s or "").split())


def as_spec(resp: Any, answer: Any = None) -> Optional[dict]:
    if resp is None:
        return None
    if isinstance(resp, str):
        spec = {"content": resp}
    elif isinstance(resp, dict):
        spec = dict(resp)
    else:
        spec = {"content": str(resp)}
    if answer is not None and "answer" not in spec:
        spec["answer"] = str(answer)
    return spec


class Mock:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.rng = random.Random(a.seed)
        self.stats: collections.Counter = collections.Counter()
        self.in_flight = 0
        self.max_in_flight = 0
        self.counter = 0
        self.exact: dict[str, tuple[str, dict]] = {}
        self.markers: list[tuple[str, str, dict]] = []   # (normalised marker, key, spec)
        self.seen: set[str] = set()                      # request signatures (for --fail-mode first)
        self._family_mods: dict[str, Any] = {}
        if a.lookup:
            self.load_lookup(a.lookup)
        if a.items_dir and os.path.isdir(a.items_dir):
            self.load_items(a.items_dir)
        self.markers.sort(key=lambda t: -len(t[0]))

    # -- index -------------------------------------------------------------------------
    def load_lookup(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.items() if isinstance(data, dict) else ((e["marker"], e["response"]) for e in data)
        for marker, resp in entries:
            spec = as_spec(resp)
            if spec is not None:
                self.markers.append((norm(marker), f"lookup:{marker[:40]}", spec))

    def family_mod(self, name: Optional[str]):
        if not name:
            return None
        if name not in self._family_mods:
            try:
                import families as famreg
                self._family_mods[name] = famreg.load(name)
            except Exception:
                self._family_mods[name] = None
        return self._family_mods[name]

    def spec_for_item(self, row: dict) -> Optional[dict]:
        answer = row.get("answer")
        if "mock" in row:
            return as_spec(row["mock"], answer)
        mod = self.family_mod(row.get("family"))
        if mod is not None:
            try:
                r = mod.mock_response(row)
            except Exception:
                r = None
            if r is not None:
                return as_spec(r, answer)
        if answer is not None and isinstance(answer, (str, int, float)):
            return {"content": f"Let me think.\nThe final answer is \\boxed{{{answer}}}.", "answer": str(answer)}
        return None

    def load_items(self, items_dir: str) -> None:
        for path in sorted(glob.glob(os.path.join(items_dir, "*.jsonl"))):
            for row in common.read_jsonl(path):
                if "id" not in row:
                    continue
                spec = self.spec_for_item(row)
                if spec is None:
                    continue
                last_user = ""
                for m in reversed(row.get("messages") or []):
                    if m.get("role") == "user":
                        last_user = norm(common.message_text(m.get("content")))
                        break
                if last_user:
                    self.exact[hashlib.sha1(last_user.encode("utf-8")).hexdigest()] = (row["id"], spec)
                marker = norm(row.get("mock_marker") or last_user[:240])
                if len(marker) >= MIN_MARKER:
                    self.markers.append((marker, row["id"], spec))

    # -- matching ----------------------------------------------------------------------
    def match(self, messages: list[dict]) -> tuple[Optional[dict], str, str]:
        """(spec, key, how)"""
        if not messages:
            return None, "empty", "none"
        last = messages[-1]
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = norm(common.message_text(m.get("content")))
                break
        if last.get("role") == "tool" or last_user.startswith('[{"role": "tool"') or last_user.startswith("[{'role': 'tool'"):
            return {"content": "Done."}, "tool_results", "tool_results"
        h = hashlib.sha1(last_user.encode("utf-8")).hexdigest()
        if h in self.exact:
            key, spec = self.exact[h]
            return spec, key, "exact"
        all_text = norm(" ".join(common.message_text(m.get("content")) for m in messages))
        for marker, key, spec in self.markers:
            if marker in all_text:
                return spec, key, "marker"
        return None, h[:12], "none"

    # -- rendering ---------------------------------------------------------------------
    @staticmethod
    def perturb(spec: dict, content: str) -> str:
        ans = spec.get("answer")
        if ans is None:
            return "I am not sure."
        s = str(ans)
        if re.fullmatch(r"-?\d+", s):
            wrong = str(int(s) + 1)
        elif re.fullmatch(r"[A-J]", s):
            wrong = "ABCD"[( "ABCD".index(s) + 1) % 4] if s in "ABCD" else chr((ord(s) - 65 + 1) % 10 + 65)
        else:
            wrong = s[::-1] if len(s) > 1 and s[::-1] != s else s + "x"
        return content.replace(s, wrong) if s in content else "I am not sure."

    def render(self, spec: Optional[dict], messages: list[dict], key: str) -> tuple[dict, str, str]:
        """-> (message, finish_reason, reasoning_text_for_usage)"""
        a = self.a
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = common.message_text(m.get("content"))
                break
        item_rng = random.Random(f"{a.seed}:{key}")
        content: Optional[str] = None
        tool_calls = None
        fin = "stop"
        reasoning_from_spec = None
        if a.mode == "canned":
            content = "I am not sure."
        elif a.mode == "garbage":
            # no ASCII letters, digits, boxes, fences or codes: nothing any scorer could latch on to
            content = "".join(chr(item_rng.randint(0x3040, 0x30FF)) if item_rng.random() > 0.15 else " " for _ in range(300))
        elif a.mode == "echo" or spec is None:
            content = last_user[-200:]
            self.stats["echoed"] += 1
        else:
            content = spec.get("content")
            tool_calls = spec.get("tool_calls")
            fin = spec.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
            reasoning_from_spec = spec.get("reasoning")
            if a.mode == "noisy" and item_rng.random() > a.accuracy:
                content = self.perturb(spec, content or "")
                tool_calls = None
                fin = "stop"
                self.stats["perturbed"] += 1
        message: dict = {"role": "assistant", "content": content}
        reasoning_text = reasoning_from_spec or DECOY
        if a.think == "inline":
            message["content"] = f"<think>{reasoning_text}</think>\n{content or ''}"
        elif a.think == "field":
            message[a.reasoning_field] = reasoning_text
        elif a.think == "unclosed":
            message["content"] = f"<think>{reasoning_text} {content or ''}"
            fin = "length"
        elif reasoning_from_spec:
            message[a.reasoning_field] = reasoning_from_spec
        if a.truncate_rate and self.rng.random() < a.truncate_rate and message.get("content"):
            c = message["content"]
            message["content"] = c[: max(1, len(c) // 2)]
            fin = "length"
            self.stats["truncated"] += 1
        if tool_calls:
            message["tool_calls"] = [{"id": f"call_{key[:8]}_{i}", "type": "function",
                                      "function": {"name": tc["name"],
                                                   "arguments": tc["arguments"] if isinstance(tc.get("arguments"), str)
                                                   else json.dumps(tc.get("arguments") or {})}}
                                     for i, tc in enumerate(tool_calls)]
            if message.get("content") in ("", None):
                message["content"] = None
        return message, fin, (reasoning_text if a.think != "none" or reasoning_from_spec else "")

    # -- handlers ----------------------------------------------------------------------
    async def chat(self, request: web.Request) -> web.Response:
        a = self.a
        self.counter += 1
        n = self.counter
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.stats["requests"] += 1
        try:
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, status=400)
            messages = body.get("messages") or []
            await asyncio.sleep(0)  # yield once so concurrent arrivals are visible in max_in_flight
            if a.fail_rate:
                sig = hashlib.sha1(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
                first_sighting = sig not in self.seen
                self.seen.add(sig)
                if (a.fail_mode == "always" or first_sighting) and self.rng.random() < a.fail_rate:
                    self.stats["errors_injected"] += 1
                    return web.json_response({"error": {"message": "injected failure", "type": "server_error"}}, status=503)
            delay = a.latency_ms / 1000.0 + (self.rng.uniform(0, a.jitter_ms / 1000.0) if a.jitter_ms else 0.0)
            if a.slow_every and n % a.slow_every == 0:
                delay += a.slow_ms / 1000.0
                self.stats["slowed"] += 1
            if delay > 0:
                await asyncio.sleep(delay)
            spec, key, how = self.match(messages)
            self.stats[f"match_{how}"] += 1
            message, fin, reasoning_text = self.render(spec, messages, key)
            prompt_text = "".join(common.message_text(m.get("content")) for m in messages)
            prompt_tokens = len(prompt_text) // 4 + 6 * len(messages)
            completion_tokens = int(len(((message.get("content") or "") + " " + reasoning_text).split()) * 1.3) + 1
            resp = {"id": f"chatcmpl-mock-{n}", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model") or a.model,
                    "choices": [{"index": 0, "message": message, "finish_reason": fin, "logprobs": None}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens}}
            return web.json_response(resp)
        finally:
            self.in_flight -= 1

    async def tokenize(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if body.get("messages") is not None:
            msgs = body["messages"]
            text = "".join(common.message_text(m.get("content")) for m in msgs)
            count = len(text) // 4 + 6 * len(msgs)
        else:
            count = len(body.get("prompt") or "") // 4
        return web.json_response({"count": count, "max_model_len": self.a.max_model_len, "tokens": [], "token_strs": []})

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.a.model, "object": "model", "owned_by": "mock"}]})

    async def version(self, request: web.Request) -> web.Response:
        return web.json_response({"version": "mock-0.1"})

    async def health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def get_stats(self, request: web.Request) -> web.Response:
        return web.json_response({"requests": self.stats["requests"], "in_flight": self.in_flight,
                                  "max_in_flight": self.max_in_flight, "errors_injected": self.stats["errors_injected"],
                                  "stats": dict(self.stats), "markers": len(self.markers), "exact": len(self.exact),
                                  "mode": self.a.mode, "think": self.a.think})


def build_app(a: argparse.Namespace) -> tuple[web.Application, Mock]:
    mock = Mock(a)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.get("/health", mock.health), web.get("/v1/models", mock.models), web.get("/version", mock.version),
                    web.post("/tokenize", mock.tokenize), web.get("/stats", mock.get_stats),
                    web.post("/v1/chat/completions", mock.chat)])
    return app, mock


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenAI-compatible mock server for evalsuite",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--model", default="m")
    p.add_argument("--items-dir", default=os.path.join(common.DEFAULT_DATA_DIR, "items"))
    p.add_argument("--lookup", default=None, help="JSON file: {marker: response} or [{marker, response}]")
    p.add_argument("--mode", choices=["oracle", "noisy", "canned", "echo", "garbage"], default="oracle")
    p.add_argument("--accuracy", type=float, default=0.7, help="noisy mode: probability of the correct answer")
    p.add_argument("--think", choices=["none", "inline", "field", "unclosed"], default="none")
    p.add_argument("--reasoning-field", choices=["reasoning", "reasoning_content"], default="reasoning")
    p.add_argument("--fail-rate", type=float, default=0.0, help="probability of an HTTP 503 (see --fail-mode)")
    p.add_argument("--fail-mode", choices=["first", "always"], default="first",
                   help="first: only the first sighting of a prompt can fail (retries succeed); always: every request independently")
    p.add_argument("--latency-ms", type=float, default=0.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--slow-every", type=int, default=0, help="every N-th request sleeps --slow-ms")
    p.add_argument("--slow-ms", type=float, default=20000.0)
    p.add_argument("--truncate-rate", type=float, default=0.0, help="probability of cutting the answer with finish_reason=length")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    app, mock = build_app(a)
    print(f"mock: http://{a.host}:{a.port} model={a.model} mode={a.mode} think={a.think} "
          f"markers={len(mock.markers)} exact={len(mock.exact)} items_dir={a.items_dir}", file=sys.stderr, flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
