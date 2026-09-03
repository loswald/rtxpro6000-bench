#!/usr/bin/env python3
"""
rr_proxy.py — tiny stdlib round-robin HTTP proxy for the x4 replica cells.

    python3 rr_proxy.py --port 8080 --backends 127.0.0.1:8000,127.0.0.1:8001,...

* Every request (POST /v1/completions, /v1/chat/completions, GET /v1/models, ...) is
  forwarded to the next backend in round-robin order; SSE / chunked responses are
  streamed through as they arrive (re-chunked, Connection: close).
* GET /health returns 200 only when EVERY backend returns 200.
* GET /proxy_stats returns per-backend request / error counters.
* Backend connect/read failures -> 502 JSON error (counted); a backend dying mid-stream
  closes the client connection without the chunked terminator (client sees an error).

This is a single Python process and *will* become the bottleneck at very high
aggregate token rates.  sweep.sh therefore defaults to per-port fan-out for the x4
cells and only uses this proxy with --via-proxy; the proxy exists mainly so the
correctness gates (lm-eval local-completions) and ad-hoc clients see one endpoint.
"""
import argparse
import http.client
import itertools
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
              "upgrade", "proxy-authorization", "proxy-authenticate", "content-length", "host"}


class State:
    def __init__(self, backends, logf, timeout, connect_timeout):
        self.backends = backends
        self.cycle = itertools.cycle(range(len(backends)))
        self.lock = threading.Lock()
        self.counts = [0] * len(backends)
        self.errors = [0] * len(backends)
        self.inflight = [0] * len(backends)
        self.logf = logf
        self.timeout = timeout                  # read timeout (long: a 2048-token completion at C=256 can take minutes)
        self.connect_timeout = connect_timeout  # short: a dead backend must fail fast, not after `timeout`

    def pick(self):
        with self.lock:
            i = next(self.cycle)
            self.counts[i] += 1
            self.inflight[i] += 1
        return i

    def done(self, i, error=False):
        with self.lock:
            self.inflight[i] -= 1
            if error:
                self.errors[i] += 1

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        if self.logf:
            try:
                with open(self.logf, "a") as f:
                    f.write(line)
            except OSError:
                pass
        else:
            sys.stderr.write(line)


STATE = None  # set in main()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence per-request stderr noise
        pass

    # ---- helpers -----------------------------------------------------------------
    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _health(self):
        bad = []
        for host, port in STATE.backends:
            try:
                c = http.client.HTTPConnection(host, port, timeout=5)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                if r.status != 200:
                    bad.append(f"{host}:{port}={r.status}")
                c.close()
            except (OSError, http.client.HTTPException) as e:
                bad.append(f"{host}:{port}={e!r}")
        if bad:
            self._send_json(503, {"status": "degraded", "unhealthy": bad})
        else:
            self._send_json(200, {"status": "ok", "backends": len(STATE.backends),
                                  "requests": STATE.counts, "errors": STATE.errors})

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is not None:
            n = int(length)
            return self.rfile.read(n) if n > 0 else b""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        return None

    def _forward(self):
        try:
            body = self._read_body()
        except (ValueError, OSError) as e:
            return self._send_json(400, {"error": f"bad request body: {e}"})
        idx = STATE.pick()
        host, port = STATE.backends[idx]
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Host"] = f"{host}:{port}"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(host, port, timeout=STATE.connect_timeout)
        error = False
        try:
            conn.connect()                              # fails fast on a dead backend
            conn.sock.settimeout(STATE.timeout)         # then allow long generations
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            STATE.log(f"backend {host}:{port} error on {self.command} {self.path}: {e!r}")
            self._send_json(502, {"error": f"backend {host}:{port} unreachable: {e!r}"})
            conn.close()
            STATE.done(idx, error=True)
            return
        try:
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP:
                    continue
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read1(65536)  # returns whatever is available -> streams SSE promptly
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            if resp.status >= 500:
                error = True
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream; nothing to do
        except (OSError, http.client.HTTPException) as e:
            error = True
            STATE.log(f"backend {host}:{port} failed mid-stream on {self.path}: {e!r}")
        finally:
            conn.close()
            self.close_connection = True
            STATE.done(idx, error=error)

    # ---- verbs -----------------------------------------------------------------------
    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self._health()
        if self.path.split("?")[0] == "/proxy_stats":
            return self._send_json(200, {"backends": [f"{h}:{p}" for h, p in STATE.backends],
                                         "requests": STATE.counts, "errors": STATE.errors,
                                         "inflight": STATE.inflight})
        return self._forward()

    def do_POST(self):
        return self._forward()

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_POST


def main():
    global STATE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backends", required=True, help="comma-separated host:port list")
    ap.add_argument("--log", default="", help="append log file (default stderr)")
    ap.add_argument("--timeout", type=float, default=3600.0, help="backend read timeout in seconds")
    ap.add_argument("--connect-timeout", type=float, default=10.0, help="backend connect timeout in seconds")
    a = ap.parse_args()
    backends = []
    for b in a.backends.split(","):
        b = b.strip()
        if not b:
            continue
        h, _, p = b.rpartition(":")
        backends.append((h or "127.0.0.1", int(p)))
    if not backends:
        sys.exit("no backends")
    STATE = State(backends, a.log, a.timeout, a.connect_timeout)
    srv = ThreadingHTTPServer((a.host, a.port), Proxy)
    srv.daemon_threads = True

    def _stop(signum, _frame):
        STATE.log(f"rr_proxy got signal {signum}; shutting down")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    STATE.log(f"rr_proxy listening on {a.host}:{a.port} -> {backends}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.log(f"rr_proxy exiting; requests={STATE.counts} errors={STATE.errors}")


if __name__ == "__main__":
    main()
