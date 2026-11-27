"""Does a Luau source compile? Asked before a script write (spec: "Extend rbx_script_write: refuse source that does not
compile, naming the line"; found on a game where a real line break inside a string literal killed a whole client in play,
reported only in the play log as "Malformed string").

Two checkers, the better one that is present:
- a Luau analyzer on this machine (luau-analyze, found as rbx_script_analyze finds it: MIF_LUAU_ANALYZE or PATH), whose
  SyntaxError lines are Luau's own parser speaking. Only SyntaxError counts here: type and lint findings are
  rbx_script_analyze's business and never refuse a write.
- otherwise MifBridge's own lexer pass, which knows the token rules only, so it catches what a lexer can: a string
  literal broken by a raw line break (Luau: "Malformed string"), and a long string, a long comment or an interpolated
  string that never ends. It does not parse statements; an unbalanced `end` needs the analyzer. The answer says which
  checker ran (by), so a pass is never read as more than it is.
Nothing is installed or downloaded here.
"""
import json
import os
import shutil
import subprocess
import tempfile

ANALYZE_SECONDS = 60


def _err(line, col, message, by="lexer"):
    return {"line": line, "column": col, "message": message, "by": by}


class _Lex:
    def __init__(self, src):
        self.s, self.n, self.i, self.line, self.bol = src, len(src), 0, 1, 0

    def col(self, i=None):
        return (self.i if i is None else i) - self.bol + 1

    def newline(self, at):
        self.line += 1
        self.bol = at + 1

    def long_level(self, i):
        """At a '[': the level of a long bracket opening there ([[ = 0, [=[ = 1), or None."""
        j = i + 1
        while j < self.n and self.s[j] == "=":
            j += 1
        return j - i - 1 if j < self.n and self.s[j] == "[" else None

    def long_body(self, i, level, what):
        """Skip a long string or comment whose opening bracket starts at i; returns an error or None."""
        line, col = self.line, self.col(i)
        close = "]" + "=" * level + "]"
        j = i + level + 2
        end = self.s.find(close, j)
        stop = self.n if end < 0 else end + len(close)
        for k in range(j, stop):
            if self.s[k] == "\n":
                self.newline(k)
        self.i = stop
        if end < 0:
            return _err(line, col, "Unfinished long %s (no closing %s)" % (what, close))
        return None

    def short_string(self, q):
        """At a quote: skip the string; a raw line break inside it is Luau's 'Malformed string'."""
        line, col = self.line, self.col()
        i = self.i + 1
        while i < self.n:
            c = self.s[i]
            if c == q:
                self.i = i + 1
                return None
            if c == "\\":
                if i + 1 < self.n and self.s[i + 1] == "\r" and i + 2 < self.n and self.s[i + 2] == "\n":
                    self.newline(i + 2)
                    i += 3
                    continue
                if i + 1 < self.n and self.s[i + 1] in "\n\r":
                    self.newline(i + 1)
                    i += 2
                    continue
                if i + 1 < self.n and self.s[i + 1] == "z":
                    i += 2
                    while i < self.n and self.s[i] in " \t\r\n\f\v":
                        if self.s[i] == "\n":
                            self.newline(i)
                        i += 1
                    continue
                i += 2
                continue
            if c in "\r\n":
                self.i = i
                return _err(line, col, "Malformed string; did you forget to finish it? (a line break inside a %s-quoted "
                                       "string)" % ("double" if q == '"' else "single"))
            i += 1
        self.i = i
        return _err(line, col, "Malformed string; did you forget to finish it? (the source ends inside it)")

    def interp(self):
        """At a backtick: an interpolated string; {expr} inside is code, scanned by the same rules."""
        line, col = self.line, self.col()
        i = self.i + 1
        while i < self.n:
            c = self.s[i]
            if c == "`":
                self.i = i + 1
                return None
            if c == "\\":
                if i + 1 < self.n and self.s[i + 1] == "\n":
                    self.newline(i + 1)
                i += 2
                continue
            if c in "\r\n":
                self.i = i
                return _err(line, col, "Malformed interpolated string; did you forget to add a '`'?")
            if c == "{":
                self.i = i + 1
                e = self.code(stop_brace=True)
                if e:
                    return e
                if self.i >= self.n:
                    return _err(line, col, "Malformed interpolated string; the source ends inside {}")
                i = self.i + 1          # past the closing }
                continue
            i += 1
        self.i = i
        return _err(line, col, "Malformed interpolated string; the source ends inside it")

    def code(self, stop_brace=False):
        depth = 0
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\n":
                self.newline(self.i)
                self.i += 1
            elif c == "-" and self.s.startswith("--", self.i):
                if self.i + 2 < self.n and self.s[self.i + 2] == "[":
                    lv = self.long_level(self.i + 2)
                    if lv is not None:
                        e = self.long_body(self.i + 2, lv, "comment")
                        if e:
                            return e
                        continue
                nl = self.s.find("\n", self.i)
                self.i = self.n if nl < 0 else nl
            elif c == "[":
                lv = self.long_level(self.i)
                if lv is not None:
                    e = self.long_body(self.i, lv, "string")
                    if e:
                        return e
                else:
                    self.i += 1
            elif c in "\"'":
                e = self.short_string(c)
                if e:
                    return e
            elif c == "`":
                e = self.interp()
                if e:
                    return e
            elif stop_brace and c == "{":
                depth += 1
                self.i += 1
            elif stop_brace and c == "}":
                if depth == 0:
                    return None
                depth -= 1
                self.i += 1
            else:
                self.i += 1
        return None


def lex_check(source):
    """The first token-level error in source, or None."""
    return _Lex(source).code()


def find_analyzer():
    from mif_roblox import domain_scripts as S
    return S.find_analyzer()


def analyzer_check(source, cmd):
    """(error or None, ran): Luau's own parser through the analyzer, SyntaxError lines only. ran False when the analyzer
    could not run (then the caller falls back to the lexer)."""
    from mif_roblox import domain_scripts as S
    tmp = tempfile.mkdtemp(prefix="mif_luau_syntax_")
    try:
        with open(os.path.join(tmp, "s.luau"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(source)
        with open(os.path.join(tmp, ".luaurc"), "w", encoding="utf-8") as fh:
            json.dump({"languageMode": "nocheck"}, fh)
        try:
            p = subprocess.run(cmd + ["s.luau"], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=ANALYZE_SECONDS,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            return None, False
        diags, _raw = S.parse_diagnostics((p.stdout or "") + "\n" + (p.stderr or ""), {"s.luau": "s"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    syn = [d for d in diags if d["kind"] == "SyntaxError"]
    if not syn:
        return None, True
    first = min(syn, key=lambda d: (d["line"], d["column"]))
    out = _err(first["line"], first["column"], first["message"], by=os.path.basename(cmd[-1]))
    if len(syn) > 1:
        out["more"] = [{"line": d["line"], "column": d["column"], "message": d["message"]} for d in syn if d is not first][:10]
    return out, True


def check(source):
    """{"ok": bool, "by": checker, "error": {line, column, message, by} or None}. The analyzer when it runs, else the
    lexer pass."""
    cmd = find_analyzer()
    if cmd:
        err, ran = analyzer_check(source, cmd)
        if ran:
            return {"ok": err is None, "by": os.path.basename(cmd[-1]), "error": err}
    err = lex_check(source)
    return {"ok": err is None, "by": "lexer", "error": err,
            "note": "no Luau analyzer on this machine: only token errors (a line break inside a string, an unfinished long "
                    "string or comment) are caught; luau-analyze on PATH or MIF_LUAU_ANALYZE checks the whole grammar"}


def refusal(path, source):
    """None when source compiles (as far as the checker present can tell), else the refusal of a write of it."""
    r = check(source)
    if r["ok"]:
        return None
    e = r["error"]
    return {"ok": False, "ran": False, "refusedRule": "syntax", "target": path, "line": e["line"], "column": e["column"],
            "message": e["message"], "checkedBy": r["by"], **({"more": e["more"]} if e.get("more") else {}),
            "error": "%s does not compile: line %d, column %d: %s. Nothing was written; allow_syntax_errors true writes "
                     "a deliberately broken script" % (path, e["line"], e["column"], e["message"])}
