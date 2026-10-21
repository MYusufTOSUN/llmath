"""Run model-written Python (PoT strategy) in a separate, restricted process.

This is a pilot-level guard, not real isolation: the code is checked against
an import/name allowlist, then run in a fresh interpreter with a time limit,
an empty working directory and an environment that carries no API keys.
Full isolation (a network-less Docker container) is planned for later.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Computation-only modules: none of them reads files, opens sockets or starts processes.
ALLOWED_MODULES = {"math", "cmath", "fractions", "decimal", "itertools", "functools", "statistics",
                   "sympy", "collections", "datetime", "operator", "heapq", "bisect", "string", "re"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
                "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
                "memoryview", "lambdify"}
# Only the builtin open() reaches files; as an attribute it is SymPy's Interval.open.
BANNED_ATTRIBUTES = BANNED_NAMES - {"open"}
# These evaluate strings as code when given one (e.g. sympify("__import__('os')...")).
STRING_EVALUATORS = {"sympify", "S", "parse_expr", "parse_latex"}


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool = False
    blocked: bool = False

    @property
    def answer(self) -> str | None:
        lines = [line.strip() for line in self.stdout.splitlines() if line.strip()]
        return lines[-1] if lines else None


def check_code(code: str) -> list[str]:
    """Reasons the code is refused; an empty list means it may run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"sözdizimi hatası: {exc.msg} (satır {exc.lineno})"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module.split(".")[0] not in ALLOWED_MODULES:
                problems.append(f"izin verilmeyen modül: {module}")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            problems.append(f"izin verilmeyen ad: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BANNED_ATTRIBUTES):
            problems.append(f"izin verilmeyen öznitelik: {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in STRING_EVALUATORS and any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in node.args
            ):
                problems.append(f"{name}() metin argümanıyla kullanılamaz")
    return sorted(set(problems))


def run_code(code: str, timeout: float = 15.0) -> ExecResult:
    problems = check_code(code)
    if problems:
        return ExecResult(ok=False, stdout="", stderr="; ".join(problems), blocked=True)
    with tempfile.TemporaryDirectory(prefix="llmath_") as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp, env=child_env(tmp), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(ok=False, stdout=_text(exc.stdout), stderr="zaman aşımı",
                              timed_out=True)
    return ExecResult(ok=proc.returncode == 0, stdout=proc.stdout, stderr=tidy_traceback(proc.stderr)[-2000:])


_FRAME = re.compile(r'File "([^"]+)"')


def _short_path(path: str) -> str:
    parts = re.split(r"[\\/]", path)
    for marker in ("site-packages", "Lib", "lib"):
        if marker in parts:
            return "/".join(parts[len(parts) - parts[::-1].index(marker):])
    return parts[-1]


def tidy_traceback(text: str) -> str:
    """A traceback without local paths, which name the user: library files as sympy/..., the program as solution.py.

    Error messages are shown to models (refinement, LLM judge), so they must not leak the machine's paths.
    """
    return _FRAME.sub(lambda m: f'File "{_short_path(m.group(1))}"', text)


def child_env(tmp: str) -> dict[str, str]:
    """The only environment the program sees: no API keys, no user paths."""
    return {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": tmp,
        "TMP": tmp,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
