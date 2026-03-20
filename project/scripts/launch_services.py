"""Launch both fraud engine and wallet gateway with one command.

Usage:
    python project/scripts/launch_services.py
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import time
from typing import List


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_runtime_stack() -> tuple[str, str]:
    """Resolve safest uvicorn loop/http pair for the current environment."""
    if _has_module("uvloop") and _has_module("httptools"):
        return "uvloop", "httptools"
    return "auto", "auto"


def recommended_workers(workload_profile: str) -> int:
    cpu_count = multiprocessing.cpu_count()
    profile = workload_profile.strip().lower()
    if profile == "cpu":
        return max(2, min(4, cpu_count // 2))
    if profile == "io":
        return max(2, min(4, cpu_count))
    return max(2, min(4, (cpu_count + 1) // 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start fraud API and wallet gateway together."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for both services")
    parser.add_argument("--fraud-port", type=int, default=8000, help="Port for fraud API")
    parser.add_argument("--wallet-port", type=int, default=8001, help="Port for wallet gateway")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode for local development only",
    )
    parser.add_argument(
        "--workload-profile",
        choices=["balanced", "cpu", "io"],
        default="balanced",
        help="Workload profile used to derive default worker counts (2-4 range)",
    )
    parser.add_argument(
        "--fraud-workers",
        type=int,
        default=None,
        help="Uvicorn worker processes for fraud API (default derived from workload profile)",
    )
    parser.add_argument(
        "--wallet-workers",
        type=int,
        default=None,
        help="Uvicorn worker processes for wallet API (default derived from workload profile)",
    )
    parser.add_argument(
        "--allow-windows-multi-worker",
        action="store_true",
        help=(
            "Allow >1 worker per service on Windows. "
            "Use only if you understand the instability risks."
        ),
    )
    parser.add_argument("--upstream-timeout", type=float, default=0.8)
    parser.add_argument("--upstream-max-retries", type=int, default=1)
    parser.add_argument("--upstream-backoff-ms", type=float, default=25.0)
    return parser.parse_args()


def build_uvicorn_cmd(
    module_path: str,
    host: str,
    port: int,
    reload_enabled: bool,
    workers: int,
) -> List[str]:
    loop_impl, http_impl = resolve_runtime_stack()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        module_path,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(max(1, workers)),
        "--loop",
        loop_impl,
        "--http",
        http_impl,
    ]
    if reload_enabled:
        cmd.append("--reload")
    return cmd


def terminate_processes(processes: List[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 5
    for process in processes:
        if process.poll() is None:
            remaining = deadline - time.time()
            if remaining > 0:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    process.kill()


def main() -> int:
    args = parse_args()

    if args.fraud_port == args.wallet_port:
        print("Error: fraud-port and wallet-port must be different.", file=sys.stderr)
        return 2

    reload_enabled = args.reload
    derived_workers = recommended_workers(args.workload_profile)
    fraud_workers = args.fraud_workers if args.fraud_workers is not None else derived_workers
    wallet_workers = args.wallet_workers if args.wallet_workers is not None else derived_workers

    is_windows = sys.platform == "win32"
    if is_windows and (fraud_workers > 1 or wallet_workers > 1):
        if args.allow_windows_multi_worker:
            print(
                "WARNING: Running with >1 worker on Windows can be unstable. "
                "Proceeding because --allow-windows-multi-worker was provided.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: Multi-worker mode on Windows is unstable. "
                "Downgrading to 1 worker per service. "
                "For multi-worker or load testing, use WSL2 or Linux.",
                file=sys.stderr,
            )
            fraud_workers = 1
            wallet_workers = 1

    fraud_url = f"http://{args.host}:{args.fraud_port}/score_transaction"
    wallet_env = os.environ.copy()
    wallet_env["FRAUD_ENGINE_URL"] = fraud_url
    wallet_env["UPSTREAM_TIMEOUT_SECONDS"] = str(args.upstream_timeout)
    wallet_env["UPSTREAM_MAX_RETRIES"] = str(max(0, args.upstream_max_retries))
    wallet_env["UPSTREAM_BACKOFF_MS"] = str(max(0.0, args.upstream_backoff_ms))
    wallet_env["UPSTREAM_BACKOFF_MAX_MS"] = str(max(args.upstream_backoff_ms * 3.0, args.upstream_backoff_ms))
    wallet_env["MAX_INFLIGHT_REQUESTS"] = wallet_env.get("MAX_INFLIGHT_REQUESTS", "250")

    fraud_module = "project.app.hybrid_fraud_api:app"
    wallet_module = "project.app.wallet_gateway_api:app"

    fraud_cmd = build_uvicorn_cmd(
        fraud_module, args.host, args.fraud_port, reload_enabled, fraud_workers
    )
    wallet_cmd = build_uvicorn_cmd(
        wallet_module, args.host, args.wallet_port, reload_enabled, wallet_workers
    )

    effective_cwd = os.getcwd()
    transport_runtime = (
        "uvloop+httptools"
        if _has_module("uvloop") and _has_module("httptools")
        else "auto(loop/http)"
    )

    print("Starting services...")
    print("Preflight:")
    print("- CWD:", effective_cwd)
    print("- Fraud module:", fraud_module)
    print("- Wallet module:", wallet_module)
    print(f"- Startup context: cwd={effective_cwd}, fraud_module={fraud_module}, wallet_module={wallet_module}")
    print(f"- Worker tuning: profile={args.workload_profile}, fraud_workers={fraud_workers}, wallet_workers={wallet_workers}")
    print("- Fraud API:", " ".join(fraud_cmd))
    print("- Wallet API:", " ".join(wallet_cmd))
    print("- Wallet -> Fraud URL:", fraud_url)
    print(f"- Runtime stack (loop/http): {transport_runtime}")

    processes: List[subprocess.Popen] = []

    try:
        processes.append(subprocess.Popen(fraud_cmd))
        time.sleep(1.0)
        processes.append(subprocess.Popen(wallet_cmd, env=wallet_env))

        print("\nServices are running:")
        print(f"  Fraud API  : http://{args.host}:{args.fraud_port}")
        print(f"  Wallet API : http://{args.host}:{args.wallet_port}")
        print("Press Ctrl+C to stop both services.")

        while True:
            time.sleep(0.5)
            for process in processes:
                if process.poll() is not None:
                    print(
                        f"A service exited unexpectedly with code {process.returncode}.",
                        file=sys.stderr,
                    )
                    terminate_processes(processes)
                    return 1
    except KeyboardInterrupt:
        print("\nShutting down services...")
        terminate_processes(processes)
        return 0
    except FileNotFoundError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        terminate_processes(processes)
        return 1
    except Exception as exc:
        print(f"Launcher error: {exc}", file=sys.stderr)
        terminate_processes(processes)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
