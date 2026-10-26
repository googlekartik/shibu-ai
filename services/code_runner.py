"""Safe Python runner for the user's OWN code, on their own PC.

Security model (documented, not hidden):
- Only a small allow-list of imports is permitted: math, random, statistics, datetime.
- Banned: os, sys, subprocess, open, eval, exec, __dunder__, file/network access.
- Code runs in a separate subprocess with a 3-second timeout, so a loop
  cannot hang the chatbot server.
- Never run code you did not write / do not trust — this is for learning,
  calculations and data analysis, not for untrusted scripts.
"""
import ast
import subprocess
import sys
import tempfile
import os

ALLOWED_IMPORTS = {"math", "random", "statistics", "datetime"}
BANNED_NAMES = {"os", "sys", "subprocess", "open", "eval", "exec", "compile",
                 "__import__", "input", "globals", "locals"}


def run_python_code(code: str):
    code = code.strip().strip("`")
    if not code:
        return {"success": False, "output": "", "error": "No code supplied."}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"success": False, "output": "", "error": f"Syntax error: {exc}"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    return {"success": False, "output": "",
                            "error": f"Import '{name}' is not allowed here. Allowed: {', '.join(sorted(ALLOWED_IMPORTS))}."}
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return {"success": False, "output": "",
                    "error": f"'{node.id}' is not allowed in the safe runner."}
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return {"success": False, "output": "", "error": "Dunder attributes are not allowed."}
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=3)
        finally:
            os.unlink(path)
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Code timed out after 3 seconds (possible infinite loop)."}
    except Exception as exc:
        return {"success": False, "output": "", "error": str(exc)}
    if proc.returncode != 0:
        return {"success": False, "output": proc.stdout, "error": proc.stderr.strip() or "Code failed."}
    return {"success": True, "output": proc.stdout, "error": ""}
