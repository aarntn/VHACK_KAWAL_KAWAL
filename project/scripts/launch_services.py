"""Start all services with one command — backend + frontend.

Usage (from repo root):
    python project/scripts/launch_services.py

Services started:
  :8002  Mock MCP external-signal server
  :8000  Fraud detection API
  :8001  Wallet gateway API
  :5173  React frontend (Vite dev server)

Flags:
  --no-mcp          Skip MCP mock server
  --no-frontend     Skip Vite dev server
  --reload          Enable uvicorn hot-reload (dev only)
  --fraud-port N    Override fraud API port   (default 8000)
  --wallet-port N   Override wallet API port  (default 8001)
  --mcp-port N      Override MCP port         (default 8002)
  --frontend-port N Override frontend port    (default 5173)
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _kill_port_win32(port: int) -> bool:
    """Kill the process holding a port on Windows. Returns True if cleared."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = int(parts[-1])
                print(f"  Port {port} held by PID {pid} — killing…", file=sys.stderr)
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )
                time.sleep(0.8)
                return True
    except Exception as exc:
        print(f"  Warning: could not auto-free port {port}: {exc}", file=sys.stderr)
    return False


def preflight_ports(host: str, ports: List[int]) -> None:
    """Detect ports already in use and free them on Windows, or abort with clear message."""
    blocked = [p for p in ports if _port_in_use(host, p)]
    if not blocked:
        return

    print(f"\nWARNING: Port(s) already in use: {blocked}", file=sys.stderr)

    if sys.platform == "win32":
        print("  Auto-freeing occupied ports on Windows…", file=sys.stderr)
        for port in blocked:
            _kill_port_win32(port)
        time.sleep(1.0)
        still_blocked = [p for p in blocked if _port_in_use(host, p)]
        if still_blocked:
            print(
                f"\nERROR: Could not free port(s) {still_blocked}.\n"
                "  In PowerShell run:  Get-Process -Name python | Stop-Process -Force\n"
                "  Then retry the launcher.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  All ports cleared. Starting services…\n", file=sys.stderr)
    else:
        cmds = "  &&  ".join(f"kill $(lsof -ti:{p})" for p in blocked)
        print(
            f"\nERROR: Port(s) {blocked} are occupied.\n"
            f"  Run:  {cmds}\n  Then retry.",
            file=sys.stderr,
        )
        sys.exit(1)


def _npm() -> str:
    """Return the correct npm executable name for the current platform."""
    if sys.platform == "win32":
        npm = shutil.which("npm.cmd") or shutil.which("npm")
    else:
        npm = shutil.which("npm")
    if not npm:
        raise FileNotFoundError(
            "npm not found on PATH. Install Node.js from https://nodejs.org"
        )
    return npm


def resolve_runtime_stack() -> tuple[str, str]:
    if _has_module("uvloop") and _has_module("httptools"):
        return "uvloop", "httptools"
    return "auto", "auto"


def recommended_workers() -> int:
    cpu = multiprocessing.cpu_count()
    return max(2, min(4, (cpu + 1) // 2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Start fraud API, wallet gateway, MCP server, and React frontend."
    )
    p.add_argument("--host",           default="127.0.0.1")
    p.add_argument("--fraud-port",     type=int, default=8000)
    p.add_argument("--wallet-port",    type=int, default=8001)
    p.add_argument("--mcp-port",       type=int, default=8002)
    p.add_argument("--frontend-port",  type=int, default=5173)
    p.add_argument("--reload",         action="store_true",
                   help="Enable uvicorn reload (dev mode, forces 1 worker)")
    p.add_argument("--no-mcp",         action="store_true",
                   help="Skip the mock MCP server")
    p.add_argument("--no-frontend",    action="store_true",
                   help="Skip the Vite dev server")
    p.add_argument("--workload-profile", choices=["balanced", "cpu", "io"],
                   default="balanced")
    p.add_argument("--fraud-workers",  type=int, default=None)
    p.add_argument("--wallet-workers", type=int, default=None)
    p.add_argument("--upstream-timeout",     type=float, default=0.8)
    p.add_argument("--upstream-max-retries", type=int,   default=1)
    p.add_argument("--upstream-backoff-ms",  type=float, default=25.0)
    return p.parse_args()


# ── uvicorn command builder ────────────────────────────────────────────────────

def uvicorn_cmd(module: str, host: str, port: int, reload: bool, workers: int) -> List[str]:
    loop_impl, http_impl = resolve_runtime_stack()
    cmd = [
        sys.executable, "-m", "uvicorn", module,
        "--host", host, "--port", str(port),
        "--loop", loop_impl,
        "--http", http_impl,
    ]
    if reload:
        # --workers is incompatible with --reload in uvicorn; reload implies single process
        cmd.append("--reload")
    else:
        cmd += ["--workers", str(max(1, workers))]
    return cmd


# ── process lifecycle ─────────────────────────────────────────────────────────

def terminate_all(procs: List[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 5
    for p in procs:
        if p.poll() is None:
            rem = deadline - time.time()
            if rem > 0:
                try:
                    p.wait(timeout=rem)
                except subprocess.TimeoutExpired:
                    p.kill()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.fraud_port == args.wallet_port:
        print("Error: --fraud-port and --wallet-port must differ.", file=sys.stderr)
        return 2

    # worker counts
    derived = recommended_workers()
    fraud_w  = args.fraud_workers  if args.fraud_workers  is not None else derived
    wallet_w = args.wallet_workers if args.wallet_workers is not None else derived

    # On Windows, multi-worker spawn can be unreliable under load; cap at 2
    if sys.platform == "win32":
        fraud_w  = min(fraud_w,  2)
        wallet_w = min(wallet_w, 2)

    if sys.platform == "win32" and (fraud_w > 1 or wallet_w > 1):
        print(f"INFO: Multi-worker mode on Windows (fraud={fraud_w}, wallet={wallet_w}). "
              "Uses spawn — stable for this app.", file=sys.stderr)

    # ── pre-flight port check ─────────────────────────────────────────────────
    ports_to_check = [args.fraud_port, args.wallet_port]
    if not args.no_mcp:
        ports_to_check.append(args.mcp_port)
    preflight_ports(args.host, ports_to_check)

    # URLs
    mcp_url    = f"http://{args.host}:{args.mcp_port}"
    fraud_url  = f"http://{args.host}:{args.fraud_port}"
    wallet_url = f"http://{args.host}:{args.wallet_port}"
    frontend_url = f"http://localhost:{args.frontend_port}"

    # environments
    fraud_env = os.environ.copy()
    fraud_env["FRAUD_MCP_URL"]     = mcp_url
    fraud_env["FRAUD_MCP_ENABLED"] = "false" if args.no_mcp else "true"
    # Default demo key so operator-gated endpoints work in local dev.
    # Override via env: FRAUD_OPERATOR_API_KEY=your-key python launch_services.py
    if not fraud_env.get("FRAUD_OPERATOR_API_KEY"):
        fraud_env["FRAUD_OPERATOR_API_KEY"] = "vhack-demo-operator-2026"
        fraud_env.setdefault("FRAUD_OPERATOR_AUTH_MODE", "required")
    # Activate Redis-compatible in-process store when no external Redis is configured.
    # Set REDIS_URL=redis://your-host:6379 to use a real Redis instead.
    if not fraud_env.get("REDIS_URL"):
        fraud_env["REDIS_URL"] = "fakeredis://local"

    wallet_env = os.environ.copy()
    wallet_env["FRAUD_ENGINE_URL"]          = f"{fraud_url}/score_transaction"
    wallet_env["UPSTREAM_TIMEOUT_SECONDS"]  = str(args.upstream_timeout)
    wallet_env["UPSTREAM_MAX_RETRIES"]      = str(max(0, args.upstream_max_retries))
    wallet_env["UPSTREAM_BACKOFF_MS"]       = str(max(0.0, args.upstream_backoff_ms))
    wallet_env["UPSTREAM_BACKOFF_MAX_MS"]   = str(max(args.upstream_backoff_ms * 3, args.upstream_backoff_ms))
    wallet_env["MAX_INFLIGHT_REQUESTS"]     = wallet_env.get("MAX_INFLIGHT_REQUESTS", "250")

    # commands
    mcp_cmd    = uvicorn_cmd("project.scripts.mock_mcp_server:app",
                             args.host, args.mcp_port, False, 1)
    fraud_cmd  = uvicorn_cmd("project.app.hybrid_fraud_api:app",
                             args.host, args.fraud_port, args.reload, fraud_w)
    wallet_cmd = uvicorn_cmd("project.app.wallet_gateway_api:app",
                             args.host, args.wallet_port, args.reload, wallet_w)

    # frontend: `npm run dev -- --port XXXX`
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    npm_exe = _npm() if not args.no_frontend else None
    frontend_cmd = (
        [npm_exe, "run", "dev", "--", "--port", str(args.frontend_port)]
        if npm_exe else None
    )

    # ── preflight summary ─────────────────────────────────────────────────
    loop_label = ("uvloop+httptools"
                  if _has_module("uvloop") and _has_module("httptools")
                  else "auto")
    print()
    print("=" * 58)
    print("  VHack Fraud Detection — launching all services")
    print("=" * 58)
    print(f"  MCP server  : {mcp_url}       {'[skipped]' if args.no_mcp else ''}")
    print(f"  Fraud API   : {fraud_url}")
    print(f"  Wallet API  : {wallet_url}")
    print(f"  Frontend    : {frontend_url}   {'[skipped]' if args.no_frontend else ''}")
    print(f"  Workers     : fraud={fraud_w}  wallet={wallet_w}  loop={loop_label}")
    print("=" * 58)
    print()

    processes: List[subprocess.Popen] = []
    labels:    List[str] = []
    critical:  List[bool] = []   # True = whole stack should stop if this dies

    try:
        # 1. MCP
        if not args.no_mcp:
            processes.append(subprocess.Popen(mcp_cmd, stderr=subprocess.PIPE))
            labels.append("MCP server")
            critical.append(False)   # MCP failing alone doesn't kill the demo
            time.sleep(0.4)

        # 2. Fraud API  (critical — UI is unusable without it)
        processes.append(subprocess.Popen(fraud_cmd, env=fraud_env, stderr=subprocess.PIPE))
        labels.append("Fraud API")
        critical.append(True)
        time.sleep(1.5)

        # 3. Wallet
        processes.append(subprocess.Popen(wallet_cmd, env=wallet_env, stderr=subprocess.PIPE))
        labels.append("Wallet API")
        critical.append(True)
        time.sleep(0.4)

        # 4. Frontend
        if frontend_cmd is not None:
            processes.append(subprocess.Popen(
                frontend_cmd,
                cwd=str(frontend_dir),
                # inherit stdio so Vite's coloured output shows in terminal
            ))
            labels.append("Frontend")
            critical.append(True)

        print("All services started. Press Ctrl+C to stop everything.\n")
        print("  Open in browser:")
        print(f"    Fraud ring graph  -> {frontend_url}/rings")
        print(f"    Dashboard         -> {frontend_url}")
        print(f"    Fraud API docs    -> {fraud_url}/docs")
        print()

        # monitor — exit if any critical child dies; warn on non-critical
        while True:
            time.sleep(0.5)
            for proc, label, is_critical in zip(processes, labels, critical):
                if proc.poll() is not None:
                    stderr_tail = ""
                    if proc.stderr:
                        try:
                            raw = proc.stderr.read(4096)
                            stderr_tail = raw.decode(errors="replace").strip()
                        except Exception:
                            pass
                    if is_critical:
                        print(f"\n[{label}] exited unexpectedly (code {proc.returncode}).",
                              file=sys.stderr)
                        if stderr_tail:
                            print(f"  Last output:\n    " +
                                  "\n    ".join(stderr_tail.splitlines()[-10:]),
                                  file=sys.stderr)
                        terminate_all(processes)
                        return 1
                    else:
                        print(f"\n[{label}] stopped (code {proc.returncode}) — "
                              "non-critical, continuing.", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nCtrl+C — shutting down all services…")
        terminate_all(processes)
        print("Done.")
        return 0
    except FileNotFoundError as exc:
        print(f"Startup failed — executable not found: {exc}", file=sys.stderr)
        terminate_all(processes)
        return 1
    except Exception as exc:
        print(f"Launcher error: {exc}", file=sys.stderr)
        terminate_all(processes)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
