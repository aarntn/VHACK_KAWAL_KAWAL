from uvicorn.workers import UvicornWorker


class UvloopHttptoolsWorker(UvicornWorker):
    """Gunicorn worker with explicit uvloop + httptools runtime."""

    CONFIG_KWARGS = {"loop": "uvloop", "http": "httptools"}
