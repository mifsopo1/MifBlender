"""The ModuleScript dependency graph, read from script sources on the MCP side.

A require resolves when its argument is a static path: `script`, `.Parent`, `.Name`, `["Name"]`, `:WaitForChild("Name")`,
`:FindFirstChild("Name")`, `game.Service`, `game:GetService("Service")`, or a local assigned one of those once. Anything
else - a number (an asset id), a variable assigned twice, a call result - is listed as unresolved with its text, never
guessed.
"""
import re

REQUIRE = re.compile(r"\brequire\s*\(")
STEP = re.compile(r"""\s*(?:\.\s*([A-Za-z_]\w*)|\[\s*(["'])(.*?)\2\s*\]|:\s*(WaitForChild|FindFirstChild|GetService)\s*\(\s*(["'])(.*?)\5[^)]*\))""")
LOCAL = re.compile(r"(?m)^\s*local\s+([A-Za-z_]\w*)\s*=\s*([^\n;]+)")
COMMENT = re.compile(r"--\[(=*)\[.*?\]\1\]|--[^\n]*", re.S)


def _balanced(src, start):
    """The text inside require( ... ) from start (just after the paren), or None."""
    depth, i = 1, start
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[start:i]
        i += 1
    return None


def resolve_expr(expr, script_path, locals_=None, depth=0):
    """A static path for expr as a list of names from game, or None."""
    expr = expr.strip()
    m = re.match(r"([A-Za-z_]\w*)", expr)
    if not m or depth > 5:
        return None
    head, rest = m.group(1), expr[m.end():]
    if head == "script":
        path = list(script_path)
    elif head == "game":
        path = []
    elif locals_ and head in locals_ and locals_[head] is not None:
        base = resolve_expr(locals_[head], script_path, locals_, depth + 1)
        if base is None:
            return None
        path = list(base)
    else:
        return None
    pos = 0
    while pos < len(rest):
        s = STEP.match(rest, pos)
        if not s:
            return None if rest[pos:].strip() else path
        name = s.group(1) or s.group(3) or s.group(6)
        if s.group(1) == "Parent":
            if not path:
                return None
            path = path[:-1]
        else:
            path = path + [name]
        pos = s.end()
    return path


def graph(sources):
    """sources: [{path: [...], className, source}]. Returns edges, unresolved requires and cycles among modules."""
    by_path = {tuple(s["path"]): s for s in sources}
    edges, unresolved = [], []
    for s in sources:
        src = COMMENT.sub("", s.get("source") or "")
        counts = {}
        for m in LOCAL.finditer(src):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        locals_ = {m.group(1): (m.group(2) if counts[m.group(1)] == 1 else None) for m in LOCAL.finditer(src)}
        for m in REQUIRE.finditer(src):
            arg = _balanced(src, m.end())
            line = src.count("\n", 0, m.start()) + 1
            target = resolve_expr(arg or "", s["path"], locals_) if arg is not None else None
            if target is None:
                unresolved.append({"from": ".".join(s["path"]), "line": line, "expr": (arg or "")[:120]})
                continue
            known = tuple(target) in by_path
            edges.append({"from": ".".join(s["path"]), "to": ".".join(target), "line": line, "exists": known,
                          "isModule": known and by_path[tuple(target)].get("className") == "ModuleScript"})
    adj = {}
    for e in edges:
        if e["exists"]:
            adj.setdefault(e["from"], []).append(e["to"])
    cycles, seen = [], set()

    def dfs(node, stack, onstack):
        for nxt in adj.get(node, []):
            if nxt in onstack:
                cyc = stack[stack.index(nxt):] + [nxt]
                key = tuple(sorted(cyc[:-1]))
                if key not in seen:
                    seen.add(key)
                    cycles.append(cyc)
            elif len(stack) < 200:
                dfs(nxt, stack + [nxt], onstack | {nxt})
    for n in list(adj):
        dfs(n, [n], {n})
    return {"edges": edges, "unresolved": unresolved, "cycles": cycles}
