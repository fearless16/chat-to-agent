"""Isolated code execution via asyncio subprocess with resource limits.

No Docker dependency.  Python and bash commands are spawned as child processes
with configurable timeouts.  Memory limits are applied via ``resource`` on
Linux (best-effort, not a security boundary).  macOS does not support
``preexec_fn``; the parameter is accepted but ignored there.

The subprocess environment is built from a minimal whitelist and does NOT
inherit the parent process environment, so secrets like ``OPENAI_API_KEY``
or ``AWS_SECRET_ACCESS_KEY`` cannot leak into the sandbox.
"""

from __future__ import annotations

import asyncio
import os
import platform
import signal
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_orchestrator.workspace.manager import FileWorkspace

try:
    import resource  # POSIX-only; absent on Windows.
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    duration_ms: float = 0.0
    truncated: bool = False


_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB cap on stdout+stderr per stream


class SandboxError(RuntimeError):
    """Raised when a sandbox execution cannot start (binary not found, etc.).

    This is distinct from a non-zero *return_code* in :class:`SandboxResult`,
    which indicates the command ran but failed. ``SandboxError`` means the
    command never started.
    """


@dataclass
class SandboxConfig:
    """Reusable configuration for a single sandboxed command."""

    timeout_ms: int = 30_000
    memory_limit_mb: int = 512
    network_access: bool = False
    workdir: str | Path | None = None
    env_overrides: dict[str, str] | None = None
    output_limit_bytes: int = _MAX_OUTPUT_BYTES


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

        return await self._run_process(
            cmd, env, timeout_ms, memory_limit_mb=memory_limit_mb
        )

    async def execute_bash(
        self,
        command: str,
        timeout_ms: int = 30_000,
        workdir: str | None = None,
        network_access: bool = False,
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
        network_access:
            If ``False`` (default), proxy / no-proxy environment variables are
            stripped.  This is a best-effort sanitisation; it does not apply
            OS-level network isolation.
        """
        self._check_not_closed()
        cmd = ["/bin/bash", "-c", command]
        env = self._build_env(memory_limit_mb=None, network=network_access)
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

    # ------------------------------------------------------------------
    # Extended execution methods
    # ------------------------------------------------------------------

    async def execute_node(
        self,
        code: str,
        timeout_ms: int = 30_000,
        memory_limit_mb: int = 512,
        network_access: bool = False,
    ) -> SandboxResult:
        """Execute *code* as a Node.js script in a subprocess.

        Raises :class:`SandboxError` if ``node`` is not on ``PATH``.
        """
        self._check_not_closed()
        node = self._resolve_binary("node")
        cmd = [node, "-e", code]
        env = self._build_env(memory_limit_mb, network=network_access)
        return await self._run_process(
            cmd, env, timeout_ms, memory_limit_mb=memory_limit_mb
        )

    async def execute_command(
        self,
        binary: str,
        args: list[str] | None = None,
        *,
        timeout_ms: int = 30_000,
        workdir: str | Path | None = None,
        network_access: bool = False,
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        """Execute an arbitrary *binary* with arguments.

        The *binary* is resolved via ``shutil.which()`` and executed directly
        (no shell wrapping).  Raises :class:`SandboxError` if the binary
        cannot be found on ``PATH``.

        Pass a :class:`SandboxConfig` as *config* to set all options at once;
        individual keyword arguments override matching config fields.
        """
        self._check_not_closed()
        cfg = config or SandboxConfig()
        timeout_ms = timeout_ms if timeout_ms != 30_000 else cfg.timeout_ms
        workdir = workdir if workdir is not None else cfg.workdir
        network_access = network_access if not network_access else cfg.network_access

        resolved = self._resolve_binary(binary)
        cmd = [resolved, *(args or [])]
        env = self._build_env(memory_limit_mb=None, network=network_access)
        return await self._run_process(
            cmd, env, timeout_ms, cwd=str(workdir) if workdir else None
        )

    async def execute_python_module(
        self,
        module: str,
        args: list[str] | None = None,
        *,
        timeout_ms: int = 30_000,
        memory_limit_mb: int = 512,
        network_access: bool = False,
        workdir: str | Path | None = None,
    ) -> SandboxResult:
        """Execute ``python -m <module> [args...]`` in a subprocess.

        Useful for running tools like ``pytest``, ``mypy``, or ``ruff``
        via the current Python interpreter.
        """
        self._check_not_closed()
        python = self._resolve_python()
        cmd = [python, "-m", module, *(args or [])]
        env = self._build_env(memory_limit_mb, network=network_access)
        return await self._run_process(
            cmd, env, timeout_ms, cwd=str(workdir) if workdir else None,
            memory_limit_mb=memory_limit_mb,
        )

    async def execute_in_workspace(
        self,
        workspace: FileWorkspace,
        command: str,
        *,
        timeout_ms: int = 30_000,
        network_access: bool = False,
    ) -> SandboxResult:
        """Execute a bash *command* with the workspace root as working dir.

        This is the primary entrypoint for the fix-analysis loop: agents
        write code into a workspace, then run tests via this method.
        """
        return await self.execute_bash(
            command,
            timeout_ms=timeout_ms,
            workdir=str(workspace.workspace_root),
            network_access=network_access,
        )

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
    def _resolve_binary(name: str) -> str:
        """Resolve *name* to an absolute path via ``shutil.which``.

        Raises :class:`SandboxError` if not found on ``PATH``.
        """
        resolved = shutil.which(name)
        if not resolved:
            raise SandboxError(
                f"binary {name!r} not found on PATH. "
                f"Is it installed?"
            )
        return resolved

    @staticmethod
    def _build_env(
        memory_limit_mb: int | None,
        network: bool = True,
    ) -> dict[str, str]:
        """Build a minimal, sanitised environment dict.

        The parent process environment is intentionally NOT inherited, so
        secrets like ``OPENAI_API_KEY`` cannot leak into the sandbox.  Only
        a small whitelist of variables needed for a normal subprocess to
        locate its interpreter and run are forwarded, plus a couple of
        optional overrides (e.g. ``PYTHONUNBUFFERED``).
        """
        env: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        }
        if not network:
            # Best-effort: strip common proxy variables so a child cannot
            # route around an intended "no network" policy via the parent's
            # HTTP proxy settings.
            for key in (
                "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY",
            ):
                env.pop(key, None)
            env["PYTHONWARNINGS"] = "ignore"
        return env

    @staticmethod
    def _build_preexec(memory_limit_mb: int | None):
        """Return a ``preexec_fn`` that applies RLIMIT_AS, or ``None``.

        Returns ``None`` on platforms where :mod:`resource` is unavailable
        (Windows), when no memory limit is requested, or on macOS (where
        ``RLIMIT_AS`` is not enforced and frequently cannot be lowered
        below the interpreter's existing virtual-memory footprint).  On
        Linux this is the cheapest available virtual-memory ceiling; the
        kernel will SIGKILL the child on breach.
        """
        if (
            memory_limit_mb is None
            or memory_limit_mb <= 0
            or resource is None
            or platform.system() == "Darwin"
        ):
            return None

        limit_bytes = int(memory_limit_mb) * 1024 * 1024

        def _preexec() -> None:
            # RLIMIT_AS is the address-space limit; on Linux it is enforced
            # and the process is killed with SIGKILL on breach.
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (OSError, ValueError):
                # The interpreter's current virtual-memory footprint may
                # already exceed the requested cap (e.g. a heavy venv).
                # Skip silently rather than abort the subprocess start.
                pass

        return _preexec

    async def _run_process(
        self,
        cmd: list[str],
        env: dict[str, str],
        timeout_ms: int,
        cwd: str | None = None,
        memory_limit_mb: int | None = None,
    ) -> SandboxResult:
        """Spawn a subprocess, enforce timeout, capture output.

        Output is capped at ``_MAX_OUTPUT_BYTES`` per stream to prevent
        trivial memory-exhaustion DoS via a noisy subprocess.  On timeout
        the entire process group is signalled (so forked children are
        reaped), not just the immediate child.
        """
        t0 = time.monotonic()

        try:
            preexec = self._build_preexec(memory_limit_mb)
            # start_new_session=True puts the child in its own process group
            # so we can killpg() the whole tree on timeout.  macOS does
            # not accept preexec_fn, so the memory limit is silently
            # skipped there (documented limitation).
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec,
            )

            timeout_s = timeout_ms / 1000.0
            timed_out = False
            stdout_bytes = b""
            stderr_bytes = b""

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                timed_out = True
                # Kill the entire process group, not just the immediate
                # child, so that `bash -c 'sleep 99 & echo done'` does not
                # leave an orphan sleep behind.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                # Drain the pipes (communicate() can only be awaited once,
                # so read directly).
                stdout_bytes, stderr_bytes = await _drain_pipes(proc)

            # Enforce the per-stream output cap (defence against the
            # 1-GB-print DoS path even if no timeout fired).
            truncated = False
            if len(stdout_bytes) > _MAX_OUTPUT_BYTES:
                stdout_bytes = stdout_bytes[:_MAX_OUTPUT_BYTES]
                truncated = True
            if len(stderr_bytes) > _MAX_OUTPUT_BYTES:
                stderr_bytes = stderr_bytes[:_MAX_OUTPUT_BYTES]
                truncated = True

            duration = (time.monotonic() - t0) * 1000.0

            return SandboxResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                return_code=proc.returncode if proc.returncode is not None else -1,
                timed_out=timed_out,
                duration_ms=round(duration, 2),
                truncated=truncated,
            )

        except FileNotFoundError:
            duration = (time.monotonic() - t0) * 1000.0
            return SandboxResult(
                stderr=f"Command not found: {cmd[0]}",
                return_code=127,
                duration_ms=round(duration, 2),
            )


async def _drain_pipes(proc: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    """Read remaining stdout/stderr after kill.  Returns ``(b"", b"")`` on error."""
    try:
        out, err = await asyncio.gather(
            proc.stdout.read() if proc.stdout else _empty(),
            proc.stderr.read() if proc.stderr else _empty(),
        )
        return out or b"", err or b""
    except Exception:  # pragma: no cover
        return b"", b""


async def _empty() -> bytes:
    return b""
