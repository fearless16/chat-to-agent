"""Structured logging with structlog — JSON and human-readable output."""

from __future__ import annotations

from typing import Any

import structlog


class Logger:
    """Thin wrapper around a structlog bound logger.

    Usage::

        log = Logger(service_name="my-app")
        log.info("hello", user="admin")
        task_log = log.with_task("task-abc")
        task_log.info("processing")
    """

    def __init__(
        self,
        service_name: str = "ai-orchestrator",
        log_level: str = "INFO",
        json_output: bool = True,
    ) -> None:
        self._service_name = service_name
        self._log_level = log_level.upper()

        # Configure structlog once (idempotent after first call per process).
        self._configure_structlog(json_output)

        # Build the initial bound logger with service context.
        self._logger: structlog.stdlib.BoundLogger = structlog.get_logger(
            service=service_name,
        )

    # ------------------------------------------------------------------
    # Public log-level helpers
    # ------------------------------------------------------------------

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an info-level message."""
        self._logger.info(msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        """Log a warning-level message."""
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an error-level message."""
        self._logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log a debug-level message."""
        self._logger.debug(msg, **kwargs)

    # ------------------------------------------------------------------
    # Context helpers — return *new* logger instances
    # ------------------------------------------------------------------

    def bind(self, **kwargs: Any) -> Logger:
        """Return a new ``Logger`` with *kwargs* bound as extra context."""
        dup = Logger.__new__(Logger)
        dup._service_name = self._service_name
        dup._log_level = self._log_level
        dup._logger = self._logger.bind(**kwargs)
        return dup

    def with_task(self, task_id: str) -> Logger:
        """Return a new logger bound with the given *task_id*."""
        return self.bind(task_id=task_id)

    def with_agent(self, agent_id: str) -> Logger:
        """Return a new logger bound with the given *agent_id*."""
        return self.bind(agent_id=agent_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_structlog(json_output: bool) -> None:
        """Set up the structlog processor pipeline once.

        This is safe to call multiple times (structlog does not guard against
        reconfiguration internally, so we use a module-level guard).
        """
        if getattr(Logger, "_configured", False):
            return

        timestamper = structlog.processors.TimeStamper(fmt="iso")

        renderer = (
            structlog.processors.JSONRenderer()
            if json_output
            else structlog.dev.ConsoleRenderer()
        )

        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ]

        structlog.configure(
            processors=shared_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        Logger._configured = True


Logger._configured = False  #: Module-level guard for one-shot configuration.
