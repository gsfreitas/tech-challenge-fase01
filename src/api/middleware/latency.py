"""
Middleware de latência para medir o tempo de processamento de cada request
"""

import logging
import time

from fastapi import Request

logger = logging.getLogger("api.middleware.latency_middleware")

async def latency_middleware(request: Request, call_next):
    """
    mede o tempo de processamento + log em json
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request_complete",
        extra = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "client_ip": request.client.host if request.client else "unknown",
        }
    )

    return response