import json
import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mcp import types as mcp_types
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext


class DocsStructuredLoggingMiddleware(Middleware):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
        log_path: str = "logs/request_log.jsonl",
    ):
        """
        Args:
            logger: Logger instance to use. If None, creates a logger named 'fastmcp.request_logger'
            log_level: Log level for messages (default: INFO)
        """
        if logger is None:
            self.logger = logging.getLogger("fastmcp.request_logger")
            self.logger.setLevel(log_level)
            self.logger.propagate = False  # prevent duplicate logs

            # Make sure log directory exists
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=32 * 1024 * 1024,  # 32 MB
                backupCount=5,  # number of rotated files to keep
                encoding="utf-8",
            )

            # Formatter that just outputs the message (already JSON)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.log_level = log_level

    async def on_call_tool(
        self,
        context: MiddlewareContext[mcp_types.CallToolRequestParams],
        call_next: CallNext[mcp_types.CallToolRequestParams, mcp_types.CallToolResult],
    ) -> mcp_types.CallToolResult:

        entry = {
            "timestamp": context.timestamp.isoformat(),
            "level": logging.getLevelName(self.log_level),
            "tool_name": context.message.name,
            "tool_arguments": context.message.arguments,
        }

        try:
            result = await call_next(context)

            self.logger.log(self.log_level, json.dumps(entry))

            return result
        except Exception:
            entry["level"] = logging.getLevelName(logging.ERROR)
            entry["result"] = traceback.format_exc()
            self.logger.log(logging.ERROR, json.dumps(entry))
            raise
