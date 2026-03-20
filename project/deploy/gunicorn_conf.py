import multiprocessing
import os


WORKLOAD_PROFILE = os.getenv("WORKLOAD_PROFILE", "balanced").strip().lower()

def _recommended_workers(cpu_count: int, workload_profile: str) -> int:
    """Pick a conservative worker range (2-4) and allow profile-aware tuning."""
    profile = workload_profile if workload_profile in {"cpu", "io", "balanced"} else "balanced"
    if profile == "cpu":
        baseline = max(2, min(4, cpu_count // 2))
    elif profile == "io":
        baseline = max(2, min(4, cpu_count))
    else:
        baseline = max(2, min(4, (cpu_count + 1) // 2))
    return baseline


bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("GUNICORN_WORKERS", _recommended_workers(multiprocessing.cpu_count(), WORKLOAD_PROFILE)))
worker_class = "project.deploy.uvicorn_worker.UvloopHttptoolsWorker"
threads = int(os.getenv("GUNICORN_THREADS", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
