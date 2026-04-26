import logging
import json
import sys
from datetime import datetime, timezone

_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "sstack_info", "thread", "threadName",
        "taskName",
    }
)

class JsonFormatter(logging.Formatter):
    """
    formato que emite cada log como uma linha json
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")[:-3] + "|",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        
        # campos extras
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                log_data[key] = value

        # excessoes
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dump(log_data, default=str, ensure_ascii=False)

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura logging estruturado JSON globalmente.

    Args:
        level: nível mínimo dos logs. Padrão INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    root_logger.addHandler(handler)

    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers = []
        lib_logger.propagate = True