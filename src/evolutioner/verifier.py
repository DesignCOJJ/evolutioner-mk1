"""Deterministic Verifier Sandbox (RLVR) — spec pillar 2.

Runs model-emitted Python inside a bubblewrap jail (unshares every
namespace, read-only /usr, writable scratch dir only). Falls back to a
resource-limited subprocess when bwrap is unavailable. Outputs or error
messages are returned so the harness can inject them back into the
reasoning context for real-time self-correction.
"""

from __future__ import annotations

import ast
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_BWRAP_ARGV = [
    "bwrap",
    "--unshare-all",
    "--die-with-parent",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind", "/lib", "/lib",
    "--ro-bind", "/lib64", "/lib64",
    "--proc", "/proc",
    "--dev", "/dev",
    "--tmpfs", "/tmp",
    "--clearenv",
    "--setenv", "PATH", "/usr/bin:/bin",
    "--setenv", "HOME", "/tmp",
    "--setenv", "PYTHONHASHSEED", "0",
]


@dataclass
class VerificationResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    sandbox: str = "bwrap"
    outputs: List[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.ok:
            return (self.stdout or "(no output)").strip()[:2000]
        return ((self.stderr or self.stdout) or f"exit={self.returncode}").strip()[:2000]


class SandboxVerifier:
    """Executes Python snippets and returns standard outputs / errors."""

    def __init__(self, timeout_s: float = 5.0, use_bwrap: Optional[bool] = None) -> None:
        self.timeout_s = timeout_s
        self.bwrap_path = shutil.which("bwrap")
        if use_bwrap is None:
            self.use_bwrap = self._probe_bwrap()
        else:
            self.use_bwrap = bool(use_bwrap and self.bwrap_path)

    def _probe_bwrap(self) -> bool:
        """Probe once at init so a broken kernel setup falls back cleanly."""
        if not self.bwrap_path:
            return False
        try:
            probe = subprocess.run(
                [self.bwrap_path, "--unshare-all", "--ro-bind", "/usr", "/usr",
                 "--clearenv", "--dev", "/dev", "/bin/true"],
                capture_output=True, timeout=5.0,
            )
            return probe.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _augment_last_expr(code: str) -> str:
        """REPL semantics: make a trailing bare expression print its value.

        `result = 37 * 43\nresult` prints nothing in a script; we append a
        print(repr(...)) for the final expression so the verifier output is
        visible to the model (RLVR feedback needs the value).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            try:
                expr_src = ast.unparse(tree.body[-1].value)
                return code.rstrip() + f"\nprint(repr({expr_src}))\n"
            except Exception:
                return code
        return code

    def verify(self, code: str) -> VerificationResult:
        code = (code or "").strip()
        if not code:
            return VerificationResult(ok=False, stderr="empty snippet", sandbox="none")
        code = self._augment_last_expr(code)
        with tempfile.TemporaryDirectory(prefix="evm1-") as scratch:
            script = Path(scratch) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            if self.use_bwrap:
                argv = _BWRAP_ARGV + [
                    "--bind", scratch, "/tmp/evm1",
                    "--chdir", "/tmp/evm1",
                    sys.executable, "/tmp/evm1/snippet.py",
                ]
                sandbox = "bwrap"
            else:
                argv = [sys.executable, str(script)]
                sandbox = "subprocess"

            preexec = self._limited if sandbox == "subprocess" else None
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True,
                    timeout=self.timeout_s, preexec_fn=preexec, cwd=scratch,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    ok=False, stderr=f"timeout after {self.timeout_s}s",
                    sandbox=sandbox,
                )
            except OSError as exc:
                return VerificationResult(ok=False, stderr=str(exc), sandbox=sandbox)

            res = VerificationResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout[-4000:],
                stderr=proc.stderr[-4000:],
                returncode=proc.returncode,
                sandbox=sandbox,
            )
            res.outputs = [line for line in res.stdout.splitlines() if line.strip()]
            return res

    @staticmethod
    def _limited() -> None:  # pragma: no cover - runs in child
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
