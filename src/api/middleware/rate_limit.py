import time
from collections import defaultdict, deque
from fastapi import Request
from fastapi.responses import JSONResponse

from api.core.config import REQUESTS_LIMIT, WINDOW_SECONDS

request_history = defaultdict(deque)

async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    history = request_history[client_ip]

    # Limpa requisições antigas
    while history and now - history[0] > WINDOW_SECONDS:
        history.popleft()

    # Verifica limite
    if len(history) >= REQUESTS_LIMIT:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Limite de {REQUESTS_LIMIT} requisições excedido",
                "retry_after": int(WINDOW_SECONDS - (now - history[0]))
            }
        )

    history.append(now)

    response = await call_next(request)

    # Headers informativos
    response.headers["X-RateLimit-Remaining"] = str(REQUESTS_LIMIT - len(history))
    response.headers["X-RateLimit-Reset"] = str(WINDOW_SECONDS)

    return response