"""Read-only repository inventory. Python 3.10+, standard library only.

Usage: python scan_repository.py LOCAL_DIRECTORY_OR_GITHUB_URL [--ref TAG]
       [--output report.json] [--max-file-bytes 2000000]
Findings are static candidates with file/line evidence, not completeness verdicts.
Remote repositories are shallow-cloned into an automatically removed temp folder.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import tokenize
from urllib.parse import urlparse


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh"}
PATH_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".jsonl", ".txt", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
CATEGORY_PATTERNS = {
    "training": r"(^|[_\-.])(train|training|finetune|sft|grpo)([_\-.]|$)",
    "inference": r"(^|[_\-.])(infer|inference|predict|generate|demo)([_\-.]|$)",
    "evaluation": r"(^|[_\-.])(eval|evaluate|evaluation|test|benchmark|metrics)([_\-.]|$)",
    "preprocessing": r"(^|[_\-.])(preprocess|preprocessing|prepare|convert|build_dataset|extract)([_\-.]|$)",
    "model_definitions": r"(^|[_\-.])(model|models|network|networks|architecture)([_\-.]|$)",
}


def git(root, *arguments):
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


@contextmanager
def repository(source, ref=None):
    if source.startswith(("https://", "http://")):
        url = urlparse(source)
        parts = url.path.strip("/").split("/")
        if url.scheme != "https" or url.hostname != "github.com" or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use an HTTPS GitHub repository URL, e.g. https://github.com/owner/repo")
        if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in parts):
            raise ValueError("Use the repository root URL. Select a branch or tag with --ref.")
        with tempfile.TemporaryDirectory(prefix="paper-repro-audit-") as folder:
            root = Path(folder) / "repository"
            command = ["git", "clone", "--depth", "1", "--no-tags"]
            if ref:
                command.extend(["--branch", ref])
            command.extend(["--", source, str(root)])
            environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, env=environment)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "git clone failed")
            yield root
    else:
        if ref:
            raise ValueError("--ref applies to GitHub URLs. For a local directory, select the revision before scanning.")
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Local repository directory does not exist")
        yield root


def snapshot(root, source):
    result = {"source": source, "checked_at": datetime.now(timezone.utc).isoformat(), "commit_sha": None}
    try:
        result["commit_sha"] = git(root, "rev-parse", "HEAD")
        result["working_tree_dirty"] = bool(git(root, "status", "--porcelain", "--untracked-files=normal"))
        result["repository_subdirectory"] = git(root, "rev-parse", "--show-prefix")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["git_error"] = str(error)
    return result


def collect_files(root):
    files, skipped = [], []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError as error:
            skipped.append({"file": directory.relative_to(root).as_posix(), "reason": str(error)})
            continue
        for path in children:
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                skipped.append({"file": relative, "reason": "symbolic link or junction"})
            elif path.is_dir():
                if path.name in SKIP_DIRS:
                    skipped.append({"file": relative, "reason": "excluded directory"})
                else:
                    pending.append(path)
            elif path.is_file():
                files.append(path)
    return sorted(files), skipped


def read_python(root, files, max_bytes):
    parsed, errors = {}, []
    for path in files:
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if path.stat().st_size > max_bytes:
                errors.append({"file": relative, "reason": "file size limit"})
                continue
            with tokenize.open(path) as stream:
                source = stream.read()
            parsed[relative] = (source, ast.parse(source, filename=relative))
        except (OSError, SyntaxError, UnicodeError, LookupError) as error:
            errors.append({"file": relative, "reason": str(error)})
    return parsed, errors


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def literal_paths(root, relative, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in {"open", "load", "load_file", "load_dataset", "read_csv", "read_json", "read_parquet", "from_pretrained", "Path"}:
            continue
        arguments = list(node.args[:1]) + [item.value for item in node.keywords if item.arg in {"path", "file", "filename", "data_files", "pretrained_model_name_or_path"}]
        for argument in arguments:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            value = argument.value
            if not value or "\n" in value or value.startswith(("http://", "https://", "s3://", "hf://")):
                continue
            item = {"file": relative, "line": node.lineno, "call": name, "value": value}
            if leaf in {"from_pretrained", "load_dataset"} and Path(value).suffix.lower() not in PATH_SUFFIXES:
                item["status"] = "model_or_dataset_identifier_or_local_path"
            elif Path(value).suffix.lower() not in PATH_SUFFIXES and leaf != "Path":
                continue
            elif Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
                item["status"] = "external_absolute_path"
            else:
                candidates = [root / value, root / Path(relative).parent / value]
                inside = [path for path in candidates if path.resolve().is_relative_to(root)]
                item["status"] = "present" if any(path.exists() for path in inside) else "missing_candidate"
                item["resolution"] = "repository root and source directory; runtime working directory may differ"
                if leaf == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) and any(mode in node.args[1].value for mode in "wax"):
                    item["status"] = "output_path"
                if leaf == "open" and any(keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str) and any(mode in keyword.value.value for mode in "wax") for keyword in node.keywords):
                    item["status"] = "output_path"
            findings.append(item)
    return findings


def placeholders(relative, source, tree):
    findings = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and re.search(r"\b(TODO|FIXME|XXX)\b", token.string, re.I):
            findings.append({"file": relative, "line": token.start[0], "kind": "todo_comment", "text": token.string.strip()})
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc:
            exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if dotted(exception).rsplit(".", 1)[-1] == "NotImplementedError":
                findings.append({"file": relative, "line": node.lineno, "kind": "not_implemented"})
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if not body or all(isinstance(statement, ast.Pass) or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is Ellipsis) for statement in body):
                findings.append({"file": relative, "line": node.lineno, "kind": "empty_function_candidate", "symbol": node.name})
    return findings


def scan(root, source, max_bytes):
    files, skipped = collect_files(root)
    parsed, parse_errors = read_python(root, files, max_bytes)
    categories = {name: [] for name in CATEGORY_PATTERNS}
    categories.update({"dependencies": [], "configuration": [], "weights": [], "weight_references": []})
    for path in files:
        relative = path.relative_to(root).as_posix()
        for category, pattern in CATEGORY_PATTERNS.items():
            if path.suffix.lower() in {".py", ".sh", ".yaml", ".yml"} and re.search(pattern, relative.lower().replace("/", "_")):
                categories[category].append({"file": relative, "basis": "filename_candidate"})
        if path.name.lower().startswith(("requirements", "environment")) or path.name.lower() in {"pyproject.toml", "setup.py", "setup.cfg", "pipfile", "poetry.lock", "uv.lock", "dockerfile"}:
            categories["dependencies"].append({"file": relative})
        if path.suffix.lower() in {".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"}:
            categories["configuration"].append({"file": relative})
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}:
            categories["weights"].append({"file": relative, "basis": "extension_candidate"})
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                if path.stat().st_size > max_bytes:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if re.search(r"from_pretrained|huggingface\.co/|\bcheckpoint\b|\.safetensors\b|\.ckpt\b", line, re.I):
                        categories["weight_references"].append({"file": relative, "line": number, "text": line.strip()[:400]})
            except OSError as error:
                skipped.append({"file": relative, "reason": str(error)})
    path_findings, placeholder_findings = [], []
    for relative, (text, tree) in parsed.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for category in ("training", "inference", "evaluation", "preprocessing"):
                    if re.search(CATEGORY_PATTERNS[category], node.name.lower()):
                        categories[category].append({"file": relative, "line": node.lineno, "symbol": node.name, "basis": "function_name_candidate"})
            elif isinstance(node, ast.ClassDef) and re.search(r"model|network|transformer|encoder|decoder|classifier", node.name, re.I):
                categories["model_definitions"].append({"file": relative, "line": node.lineno, "symbol": node.name, "basis": "class_name_candidate"})
        path_findings.extend(literal_paths(root, relative, tree))
        placeholder_findings.extend(placeholders(relative, text, tree))
    return {
        "schema_version": 1, "analysis": "repository_inventory", "repository": snapshot(root, source),
        "coverage": {"file_count": len(files), "python_parsed": len(parsed), "max_file_bytes": max_bytes, "skipped": skipped, "parse_errors": parse_errors},
        "categories": categories, "path_references": path_findings, "placeholders": placeholder_findings,
        "limitations": ["Filename matches identify candidates; inspect actual implementations and callers.", "Missing path candidates may be generated, downloaded, or relative to another runtime directory.", "Abstract methods and TODO comments may be intentional.", "Git LFS objects and submodules are not fetched. No repository code is imported or executed."],
    }


def emit(result, output=None):
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Local directory or HTTPS GitHub repository URL")
    parser.add_argument("--ref", help="Remote branch or tag")
    parser.add_argument("--output", help="Write UTF-8 JSON to this file")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.max_file_bytes <= 0:
        parser.error("--max-file-bytes must be positive")
    try:
        with repository(args.source, args.ref) as root:
            result = scan(root, args.source, args.max_file_bytes)
        emit(result, args.output)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        emit({"schema_version": 1, "error": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
