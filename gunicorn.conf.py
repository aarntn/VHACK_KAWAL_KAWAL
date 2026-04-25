"""
Gunicorn configuration for Linux / cloud deployment.

Usage:
    gunicorn project.app.hybrid_fraud_api:app -c gunicorn.conf.py

Environment overrides:
    GUNICORN_WORKERS     — number of worker processes (default: 2 * CPU + 1, capped at 8)
    GUNICORN_TIMEOUT     — worker timeout in seconds   (default: 30)
    GUNICORN_BIND        — bind address                (default: 0.0.0.0:8000)
"""
import multiprocessing
import os

# ── workers ───────────────────────────────────────────────────────────────────
_cpus = multiprocessing.cpu_count()
workers = int(os.getenv("GUNICORN_WORKERS", min(8, 2 * _cpus + 1)))
worker_class = "uvicorn.workers.UvicornWorker"

# ── networking ────────────────────────────────────────────────────────────────
bind    = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
backlog = 1024

# ── timeouts ─────────────────────────────────────────────────────────────────
timeout        = int(os.getenv("GUNICORN_TIMEOUT", 30))
graceful_timeout = 20
keepalive      = 5

# ── logging ──────────────────────────────────────────────────────────────────
accesslog  = "-"
errorlog   = "-"
loglevel   = os.getenv("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── process ──────────────────────────────────────────────────────────────────
preload_app = True   # load model artifacts once, share across workers
max_requests             = 2000
max_requests_jitter      = 200
worker_connections       = 1000
