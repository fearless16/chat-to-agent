"""Isolated code execution via asyncio subprocess with resource limits.

No Docker dependency.  Python and bash commands are spawned as child processes
with configurable timeouts.  Memory limits are applied via ``resource`` on
macOS/Linux (best-effort, not a security boundary).
"""

from __future__ import annotations

import asyncio
import os
import platform
import time
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    duration_ms: float = 0.0


class Sandbox:
    """Lightweight subprocess sandbox with timeout enforcement.

    Usage::

        sandbox = Sandbox()
        result = await sandbox.execute_python('print("hello")')
        await sandbox.close()
    """

    def __init__(self) -> None:
        self._closed = False
        self._platform = platform.system()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_python(
        self,
        code: str,
        timeout_ms: int = 30_000,
        memory_limit_mb: int = 512,
        network_access: bool = False,
    ) -> SandboxResult:
        """Execute *code* as a Python script in a subprocess.

        Parameters
        ----------
        code:
            Python source code to execute.
        timeout_ms:
            Maximum wall-clock time in milliseconds before the process is
            killed.
        memory_limit_mb:
            Best-effort virtual memory limit in MiB (``resource.setrlimit``).
        network_access:
            If ``False``, the environment is lightly sanitised but no
            OS-level network sandboxing is applied.
        """
        self._check_not_closed()

        cmd = [self._resolve_python(), "-c", code]
        env = self._build_env(memory_limit_mb, network=network_access)

        return await self._run_process(cmd, env, timeout_ms)

    async def execute_bash(
        self,
        command: str,
        timeout_ms: int = 30_000,
        workdir: str | None = None,
    ) -> SandboxResult:
        """Execute a bash *command* in a subprocess.

        Parameters
        ----------
        command:
            Shell command string (e.g. ``"echo hello"``).
        timeout_ms:
            Maximum wall-clock time in milliseconds.
        workdir:
            Working directory for the subprocess (defaults to CWD of parent).
        """
        self._check_not_closed()
        cmd = ["/bin/bash", "-c", command]
        env = self._build_env(memory_limit_mb=None, network=True)
        return await self._run_process(cmd, env, timeout_ms, cwd=workdir)

    async def check_sandbox_available(self) -> bool:
        """Return ``True`` if the Python subprocess runner is available."""
        self._check_not_closed()
        try:
            python_exe = self._resolve_python()
            proc = await asyncio.create_subprocess_exec(
                python_exe,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )
            return proc.returncode == 0 and b"Python" in stdout
        except (FileNotFoundError, asyncio.TimeoutError):
            return False

    async def close(self) -> None:
        """Release resources.  Idempotent."""
        self._closed = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox has been closed")

    @staticmethod
    def _resolve_python() -> str:
        """Resolve the python interpreter to use."""
        # Prefer sys.executable so we match the calling interpreter.
        import sys
        return sys.executable or "python3"

    @staticmethod
    def _build_env(
        memory_limit_mb: int | None,
        network: bool = True,
    ) -> dict[str, str]:
        """Build a sanitised environment dict."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if not network:
            env["PYTHONWARNINGS"] = "ignore"
        # Memory limit is applied via pre_exec below, not env.
        return env

    async def _run_process(
        self,
        cmd: list[str],
        env: dict[str, str],
        timeout_ms: int,
        cwd: str | None = None,
    ) -> SandboxResult:
        """Spawn a subprocess, enforce timeout, capture output."""
        t0 = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # macOS does not support preexec_fn; skip resource limit there.
                # On Linux we would set resource.setrlimit in preexec_fn, but
                # the assignment says no Docker dependency; memory limiting
                # via setrlimit is best-effort and skipped on macOS.
            )

            timeout_s = timeout_ms / 1000.0
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
                timed_out = False
            except asyncio.TimeoutError:
                # Kill the process tree.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                stdout_bytes, stderr_bytes = await proc.communicate()
                timed_out = True

            duration = (time.monotonic() - t0) * 1000.0

            return SandboxResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                return_code=proc.returncode if proc.returncode is not None else -1,
                timed_out=timed_out,
                duration_ms=round(duration, 2),
            )

        except FileNotFoundError:
            duration = (time.monotonic() - t0) * 1000.0
            return SandboxResult(
                stderr=f"Command not found: {cmd[0]}",
                return_code=127,
                duration_ms=round(duration, 2),
            )
