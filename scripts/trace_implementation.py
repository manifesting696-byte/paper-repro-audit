"""Trace static Python implementation paths without importing repository code.

Usage: python trace_implementation.py LOCAL_DIRECTORY_OR_GITHUB_URL
       --entry train.py [--entry train.py::main] --target GRPOTrainer
       [--target compute_reward] [--output trace.json] [--ref TAG]

Direct imports, aliases, simple instances, self/cls methods and callback references
are supported. A candidate path is evidence for manual review, not runtime proof.
"""

import argparse
import ast
from collections import deque
from pathlib import Path
import subprocess

from scan_repository import collect_files, dotted, emit, read_python, repository, snapshot


class Graph:
    def __init__(self, parsed):
        self.parsed = parsed
        self.nodes = {}
        self.edges = []
        self.unresolved = []
        self.modules = {}
        self.file_modules = {}
        self.scopes = {}
        self.bindings = {}
        self.parents = {}
        self.edge_keys = set()
        for file in parsed:
            parts = list(Path(file).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts)
            self.file_modules[file] = module
            self.modules[module] = file
        for file, (_, tree) in parsed.items():
            self.add_scope(file, "<module>", tree, None, "module")
        for identifier, node in list(self.scopes.items()):
            self.collect_bindings(identifier, node)

    def add_scope(self, file, qualified, node, parent, kind):
        identifier = f"{file}::{qualified}"
        self.nodes[identifier] = {"id": identifier, "file": file, "line": getattr(node, "lineno", 1), "name": qualified, "kind": kind}
        self.scopes[identifier] = node
        self.bindings[identifier] = {}
        self.parents[identifier] = parent
        for statement in self.local_nodes(node):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_name = statement.name if qualified == "<module>" else qualified + "." + statement.name
                child_kind = "class" if isinstance(statement, ast.ClassDef) else "function"
                child = self.add_scope(file, child_name, statement, identifier, child_kind)
                self.bindings[identifier][statement.name] = child
        return identifier

    def local_nodes(self, node):
        """Visit a scope body while keeping nested definitions separate."""
        pending = list(reversed(node.body))
        while pending:
            current = pending.pop()
            yield current
            if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                pending.extend(reversed(list(ast.iter_child_nodes(current))))

    def imported(self, module, name=None):
        if module in self.modules:
            file = self.modules[module]
            if name:
                symbol = f"{file}::{name}"
                if symbol in self.nodes:
                    return symbol
                child_module = f"{module}.{name}".strip(".")
                if child_module in self.modules:
                    return f"{self.modules[child_module]}::<module>"
            else:
                return f"{file}::<module>"
        return "external:" + ".".join(part for part in [module, name] if part)

    def collect_bindings(self, scope, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            parameters += [argument for argument in (node.args.vararg, node.args.kwarg) if argument]
            for parameter in parameters:
                self.bindings[scope][parameter.arg] = None
        file = self.nodes[scope]["file"]
        module = self.file_modules[file]
        package = module if Path(file).stem == "__init__" else module.rpartition(".")[0]
        for statement in self.local_nodes(node):
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    imported_module = alias.name if alias.asname else alias.name.split(".")[0]
                    self.bindings[scope][alias.asname or imported_module] = self.imported(imported_module)
            elif isinstance(statement, ast.ImportFrom):
                base = statement.module or ""
                if statement.level:
                    components = package.split(".") if package else []
                    keep = len(components) - statement.level + 1
                    base = ".".join(components[:max(keep, 0)] + ([base] if base else []))
                for alias in statement.names:
                    if alias.name != "*":
                        self.bindings[scope][alias.asname or alias.name] = self.imported(base, alias.name)

    def lookup(self, scope, name):
        current = scope
        while current:
            if name in self.bindings[current]:
                return self.bindings[current][name]
            current = self.parents[current]
        return None

    def resolve(self, scope, expression):
        if isinstance(expression, ast.Name):
            if expression.id in {"self", "cls"}:
                parent = self.parents[scope]
                if parent and self.nodes[parent]["kind"] == "class":
                    return parent
            return self.lookup(scope, expression.id)
        if isinstance(expression, ast.Attribute):
            value = self.resolve(scope, expression.value)
            if not value:
                return None
            if value.startswith("external:"):
                return value + "." + expression.attr
            if value.endswith("::<module>"):
                module = self.file_modules[self.nodes[value]["file"]]
                binding = self.lookup(value, expression.attr)
                return binding or self.imported(module, expression.attr)
            candidate = value + "." + expression.attr
            return candidate if candidate in self.nodes else None
        return None

    def add_edge(self, caller, callee, line, expression, kind="call", confidence="static"):
        if callee.startswith("external:"):
            self.nodes.setdefault(callee, {"id": callee, "name": callee.removeprefix("external:"), "kind": "external", "file": None, "line": None})
        key = (caller, callee, line, kind)
        if key not in self.edge_keys:
            self.edge_keys.add(key)
            self.edges.append({"from": caller, "to": callee, "file": self.nodes[caller]["file"], "line": line, "expression": expression, "kind": kind, "confidence": confidence})

    def build(self):
        # Resolve simple aliases and instances to a fixed point across scopes.
        # Assignments with several possible values stay explicitly ambiguous.
        for _ in range(4):
            changed = False
            for scope, node in self.scopes.items():
                assignments = {}
                for statement in self.local_nodes(node):
                    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        value = statement.value
                        expression = value.func if isinstance(value, ast.Call) else value
                        resolved = self.resolve(scope, expression)
                        if isinstance(value, ast.Call) and resolved and self.nodes.get(resolved, {}).get("kind") != "class":
                            resolved = None
                        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                        for target in targets:
                            if isinstance(target, ast.Name):
                                assignments.setdefault(target.id, set()).add(resolved)
                for name, values in assignments.items():
                    value = next(iter(values)) if len(values) == 1 else None
                    if self.bindings[scope].get(name) != value:
                        self.bindings[scope][name] = value
                        changed = True
            if not changed:
                break
        for scope, node in list(self.scopes.items()):
            if self.nodes[scope]["kind"] == "class":
                init = scope + ".__init__"
                if init in self.nodes:
                    self.add_edge(scope, init, self.nodes[init]["line"], "__init__", "constructor")
            for statement in self.local_nodes(node):
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Decorators run in the enclosing scope. A registry may later
                    # invoke the decorated object, so retain a candidate edge.
                    target = self.bindings[scope].get(statement.name)
                    for decorator in statement.decorator_list:
                        for call in ast.walk(decorator):
                            if isinstance(call, ast.Call):
                                self.call(scope, call)
                        if target:
                            self.add_edge(scope, target, decorator.lineno, ast.unparse(decorator), "decorator_registration_candidate", "candidate")
                elif isinstance(statement, ast.Call):
                    self.call(scope, statement)
                elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                    for alias in statement.names:
                        name = alias.asname or alias.name.split(".")[0]
                        imported = self.lookup(scope, name)
                        if imported and not imported.startswith("external:"):
                            file = self.nodes[imported]["file"]
                            self.add_edge(scope, f"{file}::<module>", statement.lineno, ast.unparse(statement), "import")
        return self

    def call(self, scope, node):
        expression = ast.unparse(node.func)
        callee = self.resolve(scope, node.func)
        if callee:
            self.add_edge(scope, callee, node.lineno, expression)
        else:
            self.unresolved.append({"file": self.nodes[scope]["file"], "line": node.lineno, "scope": scope, "expression": expression, "reason": "built-in, dynamic dispatch, inherited method, factory return, or unresolved binding"})
        arguments = list(node.args) + [keyword.value for keyword in node.keywords]
        for argument in arguments:
            for reference in ast.walk(argument):
                if isinstance(reference, (ast.Name, ast.Attribute)):
                    target = self.resolve(scope, reference)
                    if target in self.nodes and self.nodes[target]["kind"] == "function":
                        self.add_edge(scope, target, node.lineno, ast.unparse(reference), "callback_argument_candidate", "candidate")


def paths_from(graph, entries):
    adjacency = {}
    for edge in graph.edges:
        adjacency.setdefault(edge["from"], []).append(edge)
    # Prefer paths consisting entirely of resolved static edges.
    found = {}
    for allow_candidates in (False, True):
        pending = deque((entry, []) for entry in entries)
        visited = set(entries)
        while pending:
            current, path = pending.popleft()
            found.setdefault(current, path)
            for edge in adjacency.get(current, []):
                target = edge["to"]
                if (allow_candidates or edge["confidence"] == "static") and target not in visited:
                    visited.add(target)
                    pending.append((target, path + [edge]))
    return found


def trace(root, source, entries, targets, max_bytes):
    files, skipped = collect_files(root)
    parsed, errors = read_python(root, files, max_bytes)
    graph = Graph(parsed).build()
    resolved_entries = []
    for entry in entries:
        file, separator, name = entry.replace("\\", "/").partition("::")
        identifier = f"{file}::{name if separator else '<module>'}"
        if identifier not in graph.nodes:
            raise ValueError(f"Entry not parsed or not found: {entry}. Use a repository-relative Python file or file.py::qualified_name")
        resolved_entries.append(identifier)
    paths = paths_from(graph, resolved_entries)
    findings = []
    for target in targets:
        matches = [identifier for identifier, node in graph.nodes.items() if identifier == target or node["name"] == target or node["name"].rsplit(".", 1)[-1] == target]
        evidence = []
        for identifier in matches:
            path = paths.get(identifier)
            status = "no_static_path_found"
            if path is not None:
                status = "candidate_path_found" if any(edge["confidence"] == "candidate" for edge in path) else "static_path_found"
            evidence.append({"symbol": identifier, "status": status, "path": path})
        findings.append({"query": target, "matches": evidence, "status": "symbol_not_resolved" if not matches else "see_matches"})
    return {
        "schema_version": 1, "analysis": "static_implementation_paths", "repository": snapshot(root, source),
        "coverage": {"python_parsed": len(parsed), "max_file_bytes": max_bytes, "skipped": skipped, "parse_errors": errors},
        "entries": resolved_entries, "targets": findings,
        "nodes": sorted(graph.nodes.values(), key=lambda node: node["id"]), "edges": graph.edges,
        "unresolved_calls": graph.unresolved,
        "limitations": ["Static paths include conditional branches; actual execution and branch selection require verification.", "Callback arguments and decorated registrations provide candidate edges only; registration does not establish invocation.", "Dynamic imports, reflection, inheritance, factory returns, monkey-patching, and external library internals are unresolved.", "No path found means analysis could not establish a path. It does not establish unused code.", "Name bindings are scope-level approximations. Parameter values and runtime configuration are not evaluated.", "Git LFS objects and submodules are not fetched. No repository code is imported or executed."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--entry", action="append", required=True, help="Repository-relative file.py or file.py::qualified_name; repeatable")
    parser.add_argument("--target", action="append", required=True, help="Symbol name or exact file.py::qualified_name; repeatable")
    parser.add_argument("--ref", help="Remote branch or tag")
    parser.add_argument("--output")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.max_file_bytes <= 0:
        parser.error("--max-file-bytes must be positive")
    try:
        with repository(args.source, args.ref) as root:
            result = trace(root, args.source, args.entry, args.target, args.max_file_bytes)
        emit(result, args.output)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        emit({"schema_version": 1, "error": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
