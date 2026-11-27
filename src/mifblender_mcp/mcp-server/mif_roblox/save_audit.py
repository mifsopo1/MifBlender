"""A static check of a game's save layer: does it fall back to memory when DataStore fails?

Found on two games' save layers: the first DataStore call at server start was wrapped in a pcall and, when it failed,
the failure path set a memory flag or put a table where the store was, and carried on. The server then ran on RAM for
its whole life: loads came back empty, a leaderboard write put a zero over a real score, and ProcessReceipt answered
PurchaseGranted for a grant that was only held in memory. The right shape is the save template's (rbx_script_scaffold
kind save): the stand-in answers only under RunService:IsStudio(), a failed load kicks, a failed save answers false and
a receipt waits for its save.

What it flags, by script, lines and why:
  stand-in-outside-studio  a table-backed store is chosen where RunService:IsStudio() can be false. A stand-in is an
                           object whose DataStore methods (GetAsync, SetAsync, UpdateAsync, IncrementAsync, RemoveAsync,
                           GetSortedAsync, GetDataStore, GetOrderedDataStore) are defined in the scripts and call no
                           real store, a module that returns one, or a require whose path names a mock, fake, stub,
                           stand-in or in-memory store. It is chosen where its value is assigned, returned, passed or
                           constructed (.new).
  memory-fallback          after a pcall (or xpcall, or a function it calls) around a DataStore call, the failure path
                           - `if not ok`, the else of `if ok`, the code after an `if ok ... return end`, `ok and x or y`,
                           xpcall's handler - sets a memory flag, puts a table or a stand-in where the store was, or
                           returns one from the store's getter; or a flag or the store variable itself is read as a
                           switch whose branch for "DataStore failed" reads or writes an in-memory table and does not
                           refuse (error, Kick, NotProcessedYet).
A flag is a memory flag when its name says so (memory, mem, ram, fallback, mock, fake, stub, offline, local store or
mode, temp store, sandbox, stand-in) and it is set true, or when it is read as that switch.

What counts as Studio only: a branch whose condition requires RunService:IsStudio() (or a local or a small function
holding it, or game.PlaceId == 0, which a published place never has), an else after `if not IsStudio()`, and the rest of
a block after `if not IsStudio() then return ... end`. The save template's seam - the stand-in only when IsStudio() and
Workspace.MifScratch.DataStoreStandIn is there - passes; so does MifBridge's stand-in module itself, which defines a
stand-in and chooses none.

What it cannot see (static analysis, by tokens and blocks, not a Luau type checker):
  - values that travel through tables, other modules' return values or upvalues set in another script are followed
    only by name: a store handed in by a function argument, or a flag set through a setter, is not traced; a stand-in
    factory's call passed straight to another call is judged in the callee only by its own names;
  - a factory (a function that builds a stand-in in a local and returns it, or returns another factory's call) is
    judged where it is called, outside test scripts (a path naming a self-test or a Test); its own return is not a
    choice; a Studio flag a caller can override (`if opts.studio ~= nil then opts.studio else RunService:IsStudio()`)
    is read as IsStudio();
  - a stand-in whose methods are made at run time (setmetatable with an __index function, loadstring) is not one it
    knows; neither is a wrapper library (a profile or DataStore2-style module) whose own fallback is inside code the
    place does not hold;
  - it does not run the code: whether a branch is reachable, whether a retry loop ends, and whether a refusal is shown
    to the player are not proven. A clean answer means none of these shapes was found, not that saves are safe - the
    play test with rbx_play_probe's fault_store (a DataStore that fails on cue) is that proof.
NEVER RAISE from check(): a source that does not tokenize is reported under unreadable, never dropped.
"""
import os
import re

from mif_roblox import test_switches as TS

DS_METHODS = ("GetAsync", "SetAsync", "UpdateAsync", "IncrementAsync", "RemoveAsync", "GetSortedAsync")
DS_GETTERS = ("GetDataStore", "GetOrderedDataStore", "GetGlobalDataStore")
DS_CALLS = DS_METHODS + DS_GETTERS + ("ListKeysAsync", "GetVersionAsync", "ListVersionsAsync")
MEMORY_NAME = re.compile(r"(?i)(memory|inmem|(?<![a-z])mem(?![a-z])|mem(?=[A-Z_])|(?<![a-z])ram(?![a-z])|fallback|mock|fake"
                         r"|stub|offline|local_?(only|store|data|save|mode)|temp_?(store|data)|sandbox|stand_?in)")
STANDIN_PATH = re.compile(r"(?i)(mock|fake|stub|stand_?in|in_?memory|memory_?data_?store|memory_?store(?!service)"
                          r"|offline_?store|local_?data_?store|table_?data_?store)")
KEYWORDS = {"and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in", "local", "nil", "not",
            "or", "repeat", "return", "then", "true", "until", "while"}
OPS = ("...", "..=", "//=", "..", "==", "~=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "^=", "//", "::", "->")
REFUSAL = ("error", "Kick", "NotProcessedYet")
FIX = ("retry the call with a wait; if it still fails, refuse: kick a join that cannot load, answer false from a save "
       "and NotProcessedYet from ProcessReceipt, and say so in the output. A stand-in belongs under "
       "RunService:IsStudio() only")


class Tok:
    __slots__ = ("kind", "text", "line", "q")

    def __init__(self, kind, text, line, q=None):
        self.kind, self.text, self.line, self.q = kind, text, line, q   # q: a string's quote character

    def __repr__(self):
        return "%s:%r@%d" % (self.kind, self.text, self.line)


# ---------------------------------------------------------------- tokens
def _long_level(s, i):
    j = i + 1
    while j < len(s) and s[j] == "=":
        j += 1
    return j - i - 1 if j < len(s) and s[j] == "[" else None


def tokenize(src):
    """[Tok]: names, keywords, numbers, strings (their value, quotes off) and operators; comments dropped. An unfinished
    string or long bracket ends at the end of the source (the Luau syntax check is luau_syntax's business)."""
    s, n, i, line, out = src, len(src), 0, 1, []
    while i < n:
        c = s[i]
        if c == "\n":
            line += 1
            i += 1
        elif c in " \t\r\f\v":
            i += 1
        elif s.startswith("--", i):
            if i + 2 < n and s[i + 2] == "[":
                lv = _long_level(s, i + 2)
                if lv is not None:
                    close = "]" + "=" * lv + "]"
                    e = s.find(close, i + 4 + lv)
                    e = n if e < 0 else e + len(close)
                    line += s.count("\n", i, e)
                    i = e
                    continue
            e = s.find("\n", i)
            i = n if e < 0 else e
        elif c == "[" and _long_level(s, i) is not None:
            lv = _long_level(s, i)
            close = "]" + "=" * lv + "]"
            e = s.find(close, i + 2 + lv)
            e2 = n if e < 0 else e + len(close)
            out.append(Tok("str", s[i + 2 + lv:e if e >= 0 else n], line))
            line += s.count("\n", i, e2)
            i = e2
        elif c in "\"'`":
            j, buf = i + 1, []
            while j < n and s[j] != c:
                if s[j] == "\\" and j + 1 < n:
                    if s[j + 1] == "\n":
                        line += 1
                    buf.append(s[j + 1])
                    j += 2
                    continue
                if s[j] == "\n":
                    if c != "`":
                        break
                    line += 1
                buf.append(s[j])
                j += 1
            out.append(Tok("str", "".join(buf), line, c))
            i = j + 1
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            w = s[i:j]
            out.append(Tok("kw" if w in KEYWORDS else "name", w, line))
            i = j
        elif c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] in "._" or (s[j] in "+-" and s[j - 1] in "eEpP")):
                j += 1
            out.append(Tok("num", s[i:j], line))
            i = j
        else:
            for op in OPS:
                if s.startswith(op, i):
                    out.append(Tok("op", op, line))
                    i += len(op)
                    break
            else:
                out.append(Tok("op", c, line))
                i += 1
    return out


# ---------------------------------------------------------------- blocks
class Branch:
    __slots__ = ("cond", "body", "prior")

    def __init__(self, cond, body, prior):
        self.cond, self.body, self.prior = cond, body, prior   # cond (start, end) or None for else; body [start, end)


class Node:
    __slots__ = ("kind", "start", "end", "branches", "body_start", "awaiting", "cond_start", "name", "owner")

    def __init__(self, kind, start):
        self.kind, self.start, self.end, self.branches = kind, start, None, []
        self.body_start, self.awaiting, self.cond_start, self.name, self.owner = start + 1, False, None, None, None


def _stmt_if(toks, i):
    if i == 0:
        return True
    p = toks[i - 1]
    if p.kind in ("name", "num", "str"):
        return True
    if p.kind == "op":
        return p.text in (")", "]", "}", ";")
    return p.text in ("end", "then", "do", "else", "repeat", "break", "true", "false", "nil")


def blocks(toks):
    """(nodes, balanced): every function, if (with its branches), loop, do and repeat block, by token index."""
    stack, nodes = [], []
    for i, t in enumerate(toks):
        if t.kind != "kw":
            continue
        w = t.text
        top = stack[-1] if stack else None
        if w == "function":
            nd = Node("function", i)
            stack.append(nd)
            nodes.append(nd)
        elif w == "if":
            if _stmt_if(toks, i):
                nd = Node("if", i)
                nd.cond_start = i + 1
                stack.append(nd)
                nodes.append(nd)
            else:
                stack.append(Node("ifexpr", i))
        elif w == "then":
            if top is not None and top.kind == "if" and top.cond_start is not None:
                prior = [b.cond for b in top.branches if b.cond]
                top.branches.append(Branch((top.cond_start, i), [i + 1, None], prior))
                top.cond_start = None
        elif w == "elseif":
            if top is not None and top.kind == "if" and top.branches:
                top.branches[-1].body[1] = i
                top.cond_start = i + 1
        elif w == "else":
            if top is not None and top.kind == "ifexpr":
                stack.pop()
            elif top is not None and top.kind == "if" and top.branches:
                top.branches[-1].body[1] = i
                top.branches.append(Branch(None, [i + 1, None], [b.cond for b in top.branches if b.cond]))
        elif w in ("while", "for"):
            nd = Node("loop", i)
            nd.awaiting = True
            stack.append(nd)
            nodes.append(nd)
        elif w == "do":
            if top is not None and top.kind == "loop" and top.awaiting:
                top.awaiting = False
                top.body_start = i + 1
            else:
                nd = Node("do", i)
                stack.append(nd)
                nodes.append(nd)
        elif w == "repeat":
            nd = Node("repeat", i)
            stack.append(nd)
            nodes.append(nd)
        elif w in ("end", "until"):
            while stack and stack[-1].kind == "ifexpr":
                stack.pop()
            if not stack:
                continue
            want = "repeat" if w == "until" else None
            if (want and stack[-1].kind != want) or (not want and stack[-1].kind == "repeat"):
                continue
            nd = stack.pop()
            nd.end = i
            if nd.kind == "if" and nd.branches:
                nd.branches[-1].body[1] = i
    balanced = not [s for s in stack if s.kind != "ifexpr"]
    for nd in stack:
        if nd.kind != "ifexpr":
            nd.end = len(toks)
            if nd.kind == "if" and nd.branches and nd.branches[-1].body[1] is None:
                nd.branches[-1].body[1] = len(toks)
    for nd in nodes:
        if nd.kind == "if":
            nd.branches = [b for b in nd.branches if b.body[1] is not None]
    return nodes, balanced


# ---------------------------------------------------------------- one script
class Script:
    def __init__(self, path, source, class_name=None):
        self.path = path
        self.source = source or ""
        self.class_name = class_name
        self.toks = tokenize(self.source)
        self.nodes, self.balanced = blocks(self.toks)
        self.lines = self.source.split("\n")
        for nd in self.nodes:
            if nd.kind == "function":
                nd.name, nd.owner = self._fn_name(nd.start)
        self.aliases = self._studio_aliases()
        self.early = self._early_returns()

    # -- token helpers
    def t(self, i):
        return self.toks[i] if 0 <= i < len(self.toks) else None

    def is_(self, i, text, kind=None):
        t = self.t(i)
        return t is not None and t.text == text and (kind is None or t.kind == kind)

    def text(self, a, b):
        return " ".join(t.text for t in self.toks[a:b])

    def line_text(self, line):
        return self.lines[line - 1].strip()[:160] if 0 < line <= len(self.lines) else ""

    def chain_back(self, j):
        """The name chain (a.b:c) ending at token j: (start index, text) or (None, None)."""
        if not (self.t(j) and self.t(j).kind == "name"):
            return None, None
        k = j
        while k - 2 >= 0 and self.t(k - 1).text in (".", ":") and self.t(k - 1).kind == "op" and self.t(k - 2).kind == "name":
            k -= 2
        return k, "".join(x.text for x in self.toks[k:j + 1])

    def chain_fwd(self, i):
        """The name chain starting at token i: (index after it, text)."""
        if not (self.t(i) and self.t(i).kind == "name"):
            return i, None
        j = i + 1
        while self.is_(j, ".", "op") and self.t(j + 1) is not None and self.t(j + 1).kind == "name":
            j += 2
        return j, "".join(x.text for x in self.toks[i:j])

    def _fn_name(self, i):
        """function a.b:c( -> ("a.b:c", "a.b"); an anonymous function -> (None, None)."""
        j, chain = self.chain_fwd(i + 1)
        if chain is None:
            return None, None
        if self.is_(j, ":", "op") and self.t(j + 1) is not None and self.t(j + 1).kind == "name":
            return chain + ":" + self.t(j + 1).text, chain
        if "." in chain:
            return chain, chain.rsplit(".", 1)[0]
        return chain, None

    def match_paren(self, i):
        """Index of the bracket closing the one at i."""
        pairs = {"(": ")", "[": "]", "{": "}"}
        o, c, d = self.t(i).text, pairs[self.t(i).text], 0
        for k in range(i, len(self.toks)):
            tk = self.toks[k]
            if tk.kind == "op" and tk.text == o:
                d += 1
            elif tk.kind == "op" and tk.text == c:
                d -= 1
                if d == 0:
                    return k
        return len(self.toks)

    # -- containment
    def containers(self, i):
        """[(node, branch or None)] holding token i, outermost first."""
        out = []
        for nd in self.nodes:
            if nd.end is None or not nd.start <= i <= nd.end:
                continue
            if nd.kind == "if":
                for b in nd.branches:
                    if b.body[0] <= i < b.body[1]:
                        out.append((nd, b))
                        break
                else:
                    out.append((nd, None))
            else:
                out.append((nd, None))
        out.sort(key=lambda x: (x[0].start, -(x[0].end or 0)))
        return out

    def container_end(self, i):
        """The end of the innermost block holding token i (a branch body, a block, or the chunk)."""
        best = len(self.toks)
        for nd, b in self.containers(i):
            if nd.start == i:
                continue
            e = b.body[1] if b else nd.end
            best = min(best, e)
        return best

    def enclosing_function(self, i):
        fs = [nd for nd, _b in self.containers(i) if nd.kind == "function" and nd.start != i]
        return fs[-1] if fs else None

    def in_functions(self, i, fns):
        return any(f.start < i <= f.end for f in fns)

    # -- conditions
    def split_top(self, a, b, word):
        parts, d, s = [], 0, a
        for k in range(a, b):
            tk = self.toks[k]
            if tk.kind == "op" and tk.text in "([{":
                d += 1
            elif tk.kind == "op" and tk.text in ")]}":
                d -= 1
            elif d == 0 and tk.kind == "kw" and tk.text == word:
                parts.append((s, k))
                s = k + 1
        parts.append((s, b))
        return parts

    def strip(self, a, b):
        while b - a >= 2 and self.is_(a, "(", "op") and self.match_paren(a) == b - 1:
            a, b = a + 1, b - 1
        return a, b

    def atom_studio(self, a, b):
        """+1 when (a, b) requires Studio, -1 when it requires a live server, else 0."""
        a, b = self.strip(a, b)
        neg = False
        while a < b and self.is_(a, "not", "kw"):
            neg = not neg
            a, b = self.strip(a + 1, b)
        if a >= b:
            return 0
        texts = [x.text for x in self.toks[a:b]]
        pos = None
        if "PlaceId" in texts:
            j = texts.index("PlaceId")
            if j + 2 < len(texts) and texts[j + 2] == "0" and texts[j + 1] in ("==", "~="):
                pos = texts[j + 1] == "=="
        if pos is None:
            if any(x in ("==", "~=", "or", "and") for x in texts):
                return 0
            has = "IsStudio" in texts or (texts[0] in self.aliases and all(x in ("(", ")") for x in texts[1:]))
            if not has:
                return 0
            pos = True
        return (1 if pos else -1) * (-1 if neg else 1)

    def cond_implies_studio(self, a, b):
        return all(any(self.atom_studio(x, y) == 1 for x, y in self.split_top(da, db, "and"))
                   for da, db in self.split_top(a, b, "or"))

    def false_implies_studio(self, a, b):
        for da, db in self.split_top(a, b, "or"):
            conj = self.split_top(da, db, "and")
            if len(conj) == 1 and self.atom_studio(*conj[0]) == -1:
                return True
        return False

    def branch_studio(self, br):
        if br.cond and self.cond_implies_studio(*br.cond):
            return True
        return any(self.false_implies_studio(*c) for c in br.prior)

    def _studio_aliases(self):
        al = set()
        for k, tk in enumerate(self.toks):
            if tk.text == "IsStudio" and tk.kind == "name":
                # local X = RunService:IsStudio()   /   X = game:GetService("RunService"):IsStudio()
                j = k
                text = self.line_text(tk.line)
                override = re.match(r"\s*(local\s+)?([A-Za-z_]\w*)\s*=\s*if\s+([\w.]+)\s*~=\s*nil\s+then\s+\3\s+else\s+"
                                    r"(game\s*:\s*GetService\s*\(\s*[\"']RunService[\"']\s*\)|\w+)\s*:\s*IsStudio\s*\(\s*\)\s*$", text)
                if override:
                    al.add(override.group(2))
                    continue
                while j > 0 and self.toks[j - 1].line == tk.line and not self.is_(j - 1, "=", "op"):
                    if self.toks[j - 1].text in ("or", "and", "not", "if", "return"):
                        j = -1
                        break
                    j -= 1
                if j > 1 and self.is_(j - 1, "=", "op"):
                    _s, chain = self.chain_back(j - 2)
                    if chain:
                        al.add(chain)
        for nd in self.nodes:
            if nd.kind == "function" and nd.name and nd.end is not None and nd.end - nd.start < 25:
                body = [x.text for x in self.toks[nd.start:nd.end]]
                if "IsStudio" in body and "return" in body and not any(x in ("or", "not", "==", "~=") for x in body):
                    al.add(nd.name)
        return al

    def _returns_in(self, a, b):
        fns = [nd for nd in self.nodes if nd.kind == "function" and a <= nd.start < b]
        return any(self.toks[k].kind == "kw" and self.toks[k].text == "return"
                   or (self.toks[k].text == "error" and self.is_(k + 1, "(", "op"))
                   for k in range(a, b) if not self.in_functions(k, fns))

    def _early_returns(self):
        """[(start, end)] token ranges that run only in Studio because a guard before them returned on a live server."""
        out = []
        for nd in self.nodes:
            if nd.kind != "if" or len(nd.branches) != 1 or nd.branches[0].cond is None:
                continue
            br = nd.branches[0]
            if self.false_implies_studio(*br.cond) and self._returns_in(br.body[0], br.body[1]):
                out.append((nd.end + 1, self.container_end(nd.start)))
        return out

    def studio_only(self, i):
        if any(a <= i < b for a, b in self.early):
            return True
        return any(b is not None and self.branch_studio(b) for _nd, b in self.containers(i))

    # -- what the script names
    def ds_call_at(self, k):
        tk = self.toks[k]
        return (tk.kind == "name" and tk.text in DS_CALLS and k > 0 and self.toks[k - 1].text in (":", ".")
                and self.toks[k - 1].kind == "op")

    def is_definition(self, k):
        """Token k (a member name) is the name of a function being defined: function a.b:k("""
        s, _c = self.chain_back(k)
        return s is not None and s > 0 and self.is_(s - 1, "function", "kw")

    def uses_datastore(self):
        return any(t.kind == "name" and t.text in ("DataStoreService",) + DS_GETTERS for t in self.toks) or \
            any(t.kind == "str" and t.text == "DataStoreService" for t in self.toks)


# ---------------------------------------------------------------- the check over a set of scripts
def _standins(sc):
    """{owner chain: {"line", "methods"}} of objects whose DataStore methods are defined here and call no real store;
    and the table constructors that define them."""
    owners = {}
    for nd in sc.nodes:
        if nd.kind != "function" or not nd.name or not nd.owner:
            continue
        meth = re.split(r"[.:]", nd.name)[-1]
        if meth in DS_METHODS + DS_GETTERS:
            owners.setdefault(nd.owner, {"line": sc.toks[nd.start].line, "methods": [], "bodies": []})
            owners[nd.owner]["methods"].append(meth)
            owners[nd.owner]["bodies"].append(nd)
    # a.GetAsync = function ... ; { GetAsync = function ... }
    for k, tk in enumerate(sc.toks):
        if tk.kind == "name" and tk.text in DS_METHODS + DS_GETTERS and sc.is_(k + 1, "=", "op") \
                and sc.is_(k + 2, "function", "kw"):
            fn = next((nd for nd in sc.nodes if nd.kind == "function" and nd.start == k + 2), None)
            owner = None
            if k >= 2 and sc.is_(k - 1, ".", "op"):
                _s, owner = sc.chain_back(k - 2)
            else:
                owner = _constructor_owner(sc, k)
            if owner and fn is not None:
                owners.setdefault(owner, {"line": tk.line, "methods": [], "bodies": []})
                owners[owner]["methods"].append(tk.text)
                owners[owner]["bodies"].append(fn)
    real = {}
    for owner, o in owners.items():
        calls = False
        for fn in o["bodies"]:
            for k in range(fn.start, fn.end or fn.start):
                if sc.ds_call_at(k) and not sc.is_definition(k):
                    recv = sc.t(k - 2)
                    if not (recv is not None and recv.text == "self"):
                        calls = True
                        break
                if sc.toks[k].text == "DataStoreService":
                    calls = True
                    break
            if calls:
                break
        if not calls:
            real[owner] = o
    return real


def _constructor_owner(sc, k):
    """For a field at token k inside a table constructor: the name the table is assigned to, "<return>" when it is
    returned, else None."""
    d = 0
    for j in range(k - 1, -1, -1):
        tk = sc.toks[j]
        if tk.kind == "op" and tk.text in ")]}":
            d += 1
        elif tk.kind == "op" and tk.text in "([{":
            if d == 0:
                if tk.text != "{":
                    return None
                p = sc.t(j - 1)
                if p is not None and p.kind == "op" and p.text == "=":
                    return sc.chain_back(j - 2)[1]
                if p is not None and p.kind == "kw" and p.text == "return":
                    return "<return@%d>" % j
                return None
            d -= 1
    return None


def _chain_at(sc, i, chain):
    parts = re.split(r"([.:])", chain)
    if i + len(parts) > len(sc.toks):
        return False
    return all(sc.toks[i + n].text == p for n, p in enumerate(parts))


def _value_sites(sc, owner, internals, top_returns):
    """Token indices where the stand-in owner is chosen as a value: assigned, returned, passed or constructed."""
    sites = []
    parts = len(re.split(r"([.:])", owner))
    base = re.split(r"[.:]", owner)[0]
    for i, tk in enumerate(sc.toks):
        if tk.kind != "name" or not _chain_at(sc, i, owner):
            continue
        if i > 0 and sc.toks[i - 1].text in (".", ":") and sc.toks[i - 1].kind == "op":
            continue
        if sc.in_functions(i, internals):
            continue
        p = sc.t(i - 1)
        if p is None or p.text not in ("=", "return", "or", "and", "(", ",", "{") or p.kind not in ("op", "kw"):
            continue
        if p.text == "=":
            s, chain = sc.chain_back(i - 2)
            if chain and re.split(r"[.:]", chain)[0] == base:
                continue
        if p.text == "return" and i in top_returns:
            continue
        nxt = i + parts
        if sc.is_(nxt, ".", "op") or sc.is_(nxt, ":", "op"):
            m = sc.t(nxt + 1)
            if not (m is not None and m.text.lower() in ("new", "create", "open", "get") and sc.is_(nxt + 2, "(", "op")):
                continue
        sites.append(i)
    return sites


def _memory_tables(sc):
    out = set()
    for k, tk in enumerate(sc.toks):
        if tk.kind == "op" and tk.text == "=" and sc.is_(k + 1, "{", "op") and sc.is_(k + 2, "}", "op"):
            _s, chain = sc.chain_back(k - 1)
            if chain:
                out.add(chain.split(".")[-1])
    return out


def _memory_body(sc, a, b, tables):
    """(line of the in-memory read or write, refused) in tokens [a, b)."""
    hit = None
    for k in range(a, b):
        tk = sc.toks[k]
        if tk.kind == "name" and tk.text in REFUSAL:
            return None
        if hit is None and tk.kind == "name" and tk.text in tables and sc.is_(k + 1, "[", "op"):
            hit = tk.line
    return hit


def _switches(sc, tables):
    """{(name, value): line} - a name read as a branch condition whose branch, taken when the name has that value
    (True or False), reads or writes an in-memory table and does not refuse."""
    out = {}
    for nd in sc.nodes:
        if nd.kind != "if":
            continue
        for bi, br in enumerate(nd.branches):
            line = _memory_body(sc, br.body[0], br.body[1], tables)
            if line is None:
                continue
            reads = []
            if br.cond:
                ors = sc.split_top(br.cond[0], br.cond[1], "or")
                if len(ors) == 1:
                    for x, y in sc.split_top(ors[0][0], ors[0][1], "and"):
                        x, y = sc.strip(x, y)
                        val = True
                        if sc.is_(x, "not", "kw"):
                            val, x = False, x + 1
                        e, chain = sc.chain_fwd(x)
                        if chain and e == y:
                            reads.append((chain, val))
            # an else (or a later branch) is taken when an earlier single-name condition was false
            for c in br.prior[-1:] if br.cond is None else []:
                ors = sc.split_top(c[0], c[1], "or")
                x, y = sc.strip(*c)
                val = False
                if sc.is_(x, "not", "kw"):
                    val, x = True, x + 1
                e, chain = sc.chain_fwd(x)
                if chain and e == y and len(ors) == 1 and len(nd.branches) == 2:
                    reads.append((chain, val))
            for r in reads:
                out.setdefault(r, line)
    return out


def _store_vars(sc):
    out = set()
    for k, tk in enumerate(sc.toks):
        if tk.kind == "name" and tk.text in DS_GETTERS:
            j = k
            while j > 0 and sc.toks[j - 1].line == tk.line and not sc.is_(j - 1, "=", "op"):
                j -= 1
            if j > 1 and sc.is_(j - 1, "=", "op"):
                _s, chain = sc.chain_back(j - 2)
                if chain:
                    out.add(chain)
    return out


def _ds_functions(sc):
    """Names of functions defined here whose bodies call DataStore."""
    out = set()
    for nd in sc.nodes:
        if nd.kind == "function" and nd.name and nd.end is not None:
            if any(sc.ds_call_at(k) and not sc.is_definition(k) for k in range(nd.start, nd.end)):
                out.add(nd.name.replace(":", "."))
                out.add(nd.name.split(".")[-1].split(":")[-1])
    return out


def _pcalls(sc, dsfns):
    """[{at, end, ok, values, handler}] for pcall/xpcall calls whose protected code calls DataStore."""
    out = []
    for k, tk in enumerate(sc.toks):
        if tk.kind != "name" or tk.text not in ("pcall", "xpcall") or not sc.is_(k + 1, "(", "op"):
            continue
        if k > 0 and sc.toks[k - 1].text in (".", ":"):
            continue
        close = sc.match_paren(k + 1)
        args = range(k + 2, close)
        direct = any(sc.ds_call_at(j) and not sc.is_definition(j) for j in args)
        e, first = sc.chain_fwd(k + 2)
        named = first is not None and (first in dsfns or first.split(".")[-1] in dsfns)
        if not (direct or named):
            continue
        handler = None
        if tk.text == "xpcall":
            parts = _top_commas(sc, k + 2, close)
            if len(parts) >= 2 and sc.is_(parts[1][0], "function", "kw"):
                fn = next((nd for nd in sc.nodes if nd.kind == "function" and nd.start == parts[1][0]), None)
                if fn is not None:
                    handler = (fn.start, fn.end)
        names = []
        j = k - 1
        if sc.is_(j, "=", "op"):
            j -= 1
            while j >= 0 and sc.toks[j].kind == "name":
                names.insert(0, sc.toks[j].text)
                if sc.is_(j - 1, ",", "op"):
                    j -= 2
                else:
                    break
        getter = any(sc.toks[j2].text in DS_GETTERS for j2 in args) and not any(
            sc.toks[j2].text in DS_METHODS for j2 in args)
        out.append({"at": k, "end": close, "ok": names[0] if names else None, "values": names[1:], "handler": handler,
                    "getter": getter, "cond_if": _pcall_in_if(sc, k)})
    return out


def _top_commas(sc, a, b):
    parts, d, s = [], 0, a
    for j in range(a, b):
        tk = sc.toks[j]
        if tk.kind == "op" and tk.text in "([{":
            d += 1
        elif tk.kind == "op" and tk.text in ")]}":
            d -= 1
        elif d == 0 and tk.kind == "op" and tk.text == ",":
            parts.append((s, j))
            s = j + 1
    parts.append((s, b))
    return parts


def _pcall_in_if(sc, k):
    """The if node whose first condition is `pcall(...)` or `not pcall(...)` holding token k, with the condition's
    polarity (True: the branch runs when the call succeeded)."""
    for nd in sc.nodes:
        if nd.kind == "if" and nd.branches and nd.branches[0].cond:
            a, b = nd.branches[0].cond
            if a <= k < b:
                x = a
                pol = True
                while sc.is_(x, "not", "kw"):
                    pol, x = not pol, x + 1
                if x == k:
                    return nd, pol
    return None


def _failure_regions(sc, pc):
    """[(start, end, how)] token ranges that run when the protected DataStore call failed."""
    regions = []
    if pc["handler"]:
        regions.append((pc["handler"][0], pc["handler"][1], "xpcall's error handler"))
    ci = pc["cond_if"]
    if ci:
        nd, pol = ci
        for bi, br in enumerate(nd.branches):
            if (bi == 0) != pol:
                regions.append((br.body[0], br.body[1], "the branch taken when the pcall failed"))
    ok = pc["ok"]
    if ok:
        stop = sc.container_end(pc["at"])
        for nd in sc.nodes:
            if nd.kind != "if" or not nd.branches or nd.start <= pc["end"] or nd.start >= stop:
                continue
            first = nd.branches[0]
            if first.cond is None:
                continue
            ors = sc.split_top(first.cond[0], first.cond[1], "or")
            for da, db in ors:
                conj = sc.split_top(da, db, "and")
                if any(_is_name(sc, x, y, ok, neg=True) for x, y in conj) and len(ors) >= 1:
                    regions.append((first.body[0], first.body[1], "`if not %s`" % ok))
                    break
            if len(ors) == 1 and any(_is_name(sc, x, y, ok) for x, y in sc.split_top(ors[0][0], ors[0][1], "and")):
                for br in nd.branches[1:]:
                    regions.append((br.body[0], br.body[1], "the else of `if %s`" % ok))
                if len(nd.branches) == 1 and sc._returns_in(first.body[0], first.body[1]):
                    regions.append((nd.end + 1, sc.container_end(nd.start), "the code after `if %s ... return end`" % ok))
    return regions


def _is_name(sc, a, b, name, neg=False):
    a, b = sc.strip(a, b)
    if neg:
        if not sc.is_(a, "not", "kw"):
            if sc.is_(a + 1, "==", "op") and sc.is_(a + 2, "false", "kw") and b == a + 3 and sc.is_(a, name):
                return True
            return False
        a, b = sc.strip(a + 1, b)
    return b == a + 1 and sc.is_(a, name)


def _assignments(sc, a, b):
    """[(index of '=', lhs chain, rhs start)] in tokens [a, b)."""
    out = []
    for k in range(a, b):
        tk = sc.toks[k]
        if tk.kind == "op" and tk.text == "=":
            _s, chain = sc.chain_back(k - 1)
            if chain:
                out.append((k, chain, k + 1))
    return out


def _last(chain):
    return re.split(r"[.:]", chain)[-1]


def check(sources):
    """sources: [{path (list or dotted str), source, className?}]. -> {scanned, dataStoreScripts, standIns, findings,
    unreadable}. Each finding: {script, kind, line, lines, why, text}."""
    scripts, unreadable = [], []
    for s in sources:
        path = s.get("path")
        path = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        try:
            scripts.append(Script(path, s.get("source") or "", s.get("className")))
        except Exception as exc:  # noqa: BLE001 - a script this cannot read is named, never dropped
            unreadable.append({"script": path, "error": "%s: %s" % (type(exc).__name__, exc)})
    # pass 1: stand-in objects, and modules that return one
    standin_modules, per = {}, {}
    for sc in scripts:
        owners = _standins(sc)
        tops = set()
        for k, tk in enumerate(sc.toks):
            if tk.kind == "kw" and tk.text == "return" and not sc.containers(k):
                tops.add(k + 1)
                e, chain = sc.chain_fwd(k + 1)
                if chain in owners or (sc.is_(k + 1, "{", "op") and any(o.startswith("<return@%d>" % (k + 1)) for o in owners)):
                    standin_modules[sc.path.split(".")[-1]] = sc.path
        per[sc.path] = (owners, tops)
    # pass 1b: factories. A function that builds a stand-in in a local and returns it (`local m = {...} function
    # m:GetAsync ... return m`) constructs it; it chooses nothing, so its own `return m` is not a choice (found
    # 2026-11-26: a game's MemoryStore() factories were flagged on every publish while every caller chose them under
    # IsStudio()). The choice is a call of the factory, judged where it is called, in any script.
    factories, factory_returns = {}, set()
    for sc in scripts:
        owners, tops = per[sc.path]
        internals = [fn for o in owners.values() for fn in o["bodies"]] + [
            nd for nd in sc.nodes if nd.kind == "function" and nd.owner and nd.owner in owners]
        for owner in owners:
            if owner.startswith("<return@") or re.search(r"[.:]", owner):
                continue
            for i in _value_sites(sc, owner, internals, tops):
                if sc.t(i - 1) is None or sc.toks[i - 1].text != "return":
                    continue
                fn = sc.enclosing_function(i)
                if fn is None or not fn.name or fn.end is None:
                    continue
                local = any(sc.toks[k].text == "local" and sc.is_(k + 1, owner) for k in range(fn.start, i))
                if local:
                    factory_returns.add((sc.path, i))
                    factories.setdefault(re.split(r"[.:]", fn.name)[-1], []).append(
                        {"script": sc.path, "name": fn.name, "line": sc.toks[fn.start].line, "fn": fn})
    for _round in range(4):
        grew = False
        for sc in scripts:
            for k, tk in enumerate(sc.toks):
                if tk.kind != "name" or tk.text not in factories or not sc.is_(k + 1, "(", "op"):
                    continue
                fn = sc.enclosing_function(k)
                if fn is None or not fn.name or fn.end is None:
                    continue
                last = re.split(r"[.:]", fn.name)[-1]
                if last in factories or not sc.line_text(tk.line).lstrip().startswith("return"):
                    continue
                factories[last] = [{"script": sc.path, "name": fn.name, "line": sc.toks[fn.start].line, "fn": fn}]
                grew = True
        if not grew:
            break
    findings, standins = [], []

    def add(sc, kind, idx_lines, why):
        # the line that does it: a stand-in's choice (first), a fallback's flag, substitution or memory read (second;
        # the first is the pcall it follows)
        primary = idx_lines[1] if kind == "memory-fallback" and len(idx_lines) > 1 and idx_lines[1] else idx_lines[0]
        lines = sorted({ln for ln in idx_lines if ln})
        findings.append({"script": sc.path, "kind": kind, "line": primary, "lines": lines,
                         "why": why, "text": sc.line_text(primary) if primary else ""})

    for sc in scripts:
        owners, tops = per[sc.path]
        internals = [fn for o in owners.values() for fn in o["bodies"]] + [
            nd for nd in sc.nodes if nd.kind == "function" and nd.owner and nd.owner in owners]
        for owner, o in sorted(owners.items()):
            standins.append({"script": sc.path, "owner": owner, "line": o["line"], "methods": sorted(set(o["methods"]))})
        # -- stand-ins chosen where IsStudio() can be false
        for owner, o in owners.items():
            if owner.startswith("<return@"):
                at = int(owner[8:-1]) - 1
                if at not in [t - 1 for t in tops] and not sc.studio_only(at):
                    add(sc, "stand-in-outside-studio", [sc.toks[at].line, o["line"]],
                        "a table with DataStore methods (%s, defined at line %d) is returned where "
                        "RunService:IsStudio() can be false" % (", ".join(sorted(set(o["methods"]))), o["line"]))
                continue
            for i in _value_sites(sc, owner, internals, tops):
                if (sc.path, i) in factory_returns:
                    continue
                if not sc.studio_only(i):
                    add(sc, "stand-in-outside-studio", [sc.toks[i].line, o["line"]],
                        "%s, a stand-in store (its %s are defined at line %d and call no real DataStore), is chosen "
                        "here where RunService:IsStudio() can be false" % (
                            owner, ", ".join(sorted(set(o["methods"]))), o["line"]))
        # -- a call of a stand-in factory (pass 1b), judged where it is called; a test script's calls are its own
        # stand-ins (TS.is_test_path), not the place's choice, and are left out
        if factories and not TS.is_test_path(sc.path.split(".")):
            for k, tk in enumerate(sc.toks):
                if tk.kind != "name" or tk.text not in factories or not sc.is_(k + 1, "(", "op"):
                    continue
                if sc.t(k - 1) is not None and sc.toks[k - 1].text == "function":
                    continue
                fn = sc.enclosing_function(k)
                if fn is not None and fn.name and re.split(r"[.:]", fn.name)[-1] in factories \
                        and sc.line_text(tk.line).lstrip().startswith("return"):
                    continue                       # a factory's own return line builds the bundle: not a choice
                s0, _chain = sc.chain_back(k)
                before = sc.t(s0 - 1)
                if before is not None and before.kind == "op" and before.text in ("(", ","):
                    continue                       # passed to another call: judged where the callee chooses it
                inside = [f for f in factories[tk.text] if f["script"] == sc.path and f["fn"].start <= k <= f["fn"].end]
                if inside or sc.studio_only(k):
                    continue
                f = factories[tk.text][0]
                add(sc, "stand-in-outside-studio", [tk.line],
                    "%s() builds a stand-in store (%s, line %d) and is called here where RunService:IsStudio() can be "
                    "false" % (f["name"], f["script"], f["line"]))
        # -- a require of a stand-in module
        for k, tk in enumerate(sc.toks):
            if tk.kind != "name" or tk.text != "require" or not sc.is_(k + 1, "(", "op"):
                continue
            close = sc.match_paren(k + 1)
            argtoks = sc.toks[k + 2:close]
            named = next((x.text for x in argtoks if x.kind in ("name", "str") and (
                STANDIN_PATH.search(x.text) or x.text in standin_modules)), None)
            if not named:
                continue
            target = standin_modules.get(named)
            p = sc.t(k - 1)
            alias = None
            if p is not None and p.text == "=" and not (sc.is_(close + 1, ".", "op") or sc.is_(close + 1, ":", "op")):
                s0, alias = sc.chain_back(k - 2)
            if alias:
                for i in _value_sites(sc, alias, internals, tops):
                    if not sc.studio_only(i):
                        add(sc, "stand-in-outside-studio", [sc.toks[i].line, tk.line],
                            "%s holds a stand-in store module (%s, required at line %d) and is chosen here where "
                            "RunService:IsStudio() can be false" % (alias, target or named, tk.line))
            elif not sc.studio_only(k):
                add(sc, "stand-in-outside-studio", [tk.line],
                    "a stand-in store module (%s) is required and used where RunService:IsStudio() can be false"
                    % (target or named))
        # -- a failed DataStore call that carries on in memory
        tables = _memory_tables(sc)
        switches = _switches(sc, tables)
        stores = _store_vars(sc)
        store_last = {_last(c) for c in stores}
        dsfns = _ds_functions(sc)
        flagged_lines = set()
        for pc in _pcalls(sc, dsfns):
            pline = sc.toks[pc["at"]].line
            values = set(pc["values"]) if pc["getter"] else set()
            for a, b, how in _failure_regions(sc, pc):
                fn = sc.enclosing_function(pc["at"])
                getter_fn = fn is not None and any(sc.toks[j].text in DS_GETTERS for j in range(fn.start, fn.end or fn.start))
                for eq, lhs, r in _assignments(sc, a, b):
                    if sc.studio_only(eq):
                        continue
                    rt = sc.t(r)
                    name = _last(lhs)
                    why = None
                    if rt is not None and rt.kind == "kw" and rt.text in ("true", "false", "nil"):
                        val = rt.text == "true"
                        reader = switches.get((lhs, val)) or switches.get((name, val))
                        if rt.text == "nil" and (lhs in stores or name in store_last or name in values):
                            if reader:
                                why = ("the store variable %s is set to nil when the DataStore call at line %d fails, "
                                       "and line %d then reads or writes memory in its place" % (lhs, pline, reader))
                        elif rt.text != "nil" and (MEMORY_NAME.search(name) and val or reader):
                            why = "%s = %s is set when the DataStore call at line %d fails (in %s)%s" % (
                                lhs, rt.text, pline, how, ", and line %d then reads or writes memory instead of the store"
                                % reader if reader else ": its name makes it a memory flag")
                        lines = [pline, rt.line, reader]
                    elif lhs in stores or name in store_last or name in values:
                        if _substitute(sc, r, owners) and not _has_getter_on_line(sc, r):
                            why = ("%s, the store, is given %s in its place when the DataStore call at line %d fails "
                                   "(in %s)" % (lhs, _what(sc, r, owners), pline, how))
                        lines = [pline, rt.line if rt else None]
                    if why and rt is not None and rt.line not in flagged_lines:
                        flagged_lines.add(rt.line)
                        add(sc, "memory-fallback", lines, why)
                if getter_fn:
                    for k in range(a, b):
                        if sc.is_(k, "return", "kw") and _substitute(sc, k + 1, owners) and not sc.studio_only(k):
                            ln = sc.toks[k].line
                            if ln not in flagged_lines:
                                flagged_lines.add(ln)
                                add(sc, "memory-fallback", [pline, ln],
                                    "the store's getter returns %s when the DataStore call at line %d fails (in %s)"
                                    % (_what(sc, k + 1, owners), pline, how))
            ok = pc["ok"]
            if ok:
                reader = switches.get((ok, False))
                if reader and not sc.studio_only(pc["at"]) and reader not in flagged_lines:
                    flagged_lines.add(reader)
                    add(sc, "memory-fallback", [pline, reader],
                        "%s, the result of the DataStore pcall at line %d, is read as a switch: when it is false, "
                        "line %d reads or writes memory instead of the store" % (ok, pline, reader))
                # x = not ok / x = ok, and ok and store or {}
                stop = sc.container_end(pc["at"])
                for k in range(pc["end"] + 1, min(stop, len(sc.toks))):
                    if sc.is_(k, ok) and sc.t(k).kind == "name":
                        p = sc.t(k - 1)
                        neg = p is not None and p.text == "not"
                        eqk = k - 2 if neg else k - 1
                        if sc.is_(eqk, "=", "op") and not sc.is_(k + 1, "and", "kw"):
                            _s, lhs = sc.chain_back(eqk - 1)
                            if lhs and lhs != ok:
                                val = neg  # the flag's value when the call failed
                                reader = switches.get((lhs, val)) or switches.get((_last(lhs), val))
                                if (reader or (MEMORY_NAME.search(_last(lhs)) and val)) and not sc.studio_only(k) \
                                        and sc.t(k).line not in flagged_lines:
                                    flagged_lines.add(sc.t(k).line)
                                    add(sc, "memory-fallback", [pline, sc.t(k).line, reader],
                                        "%s = %s%s: set from the DataStore pcall at line %d%s" % (
                                            lhs, "not " if neg else "", ok, pline,
                                            ", and line %d reads or writes memory when the call failed" % reader
                                            if reader else ", a memory flag by its name"))
                        if sc.is_(k + 1, "and", "kw"):
                            j = k + 2
                            while j < len(sc.toks) and sc.toks[j].line == sc.t(k).line and not sc.is_(j, "or", "kw"):
                                j += 1
                            if sc.is_(j, "or", "kw") and _substitute(sc, j + 1, owners) and not sc.studio_only(j) \
                                    and sc.t(k).line not in flagged_lines:
                                flagged_lines.add(sc.t(k).line)
                                add(sc, "memory-fallback", [pline, sc.t(k).line],
                                    "`%s and ... or %s`: a table stands in for the store when the DataStore call at "
                                    "line %d fails" % (ok, _what(sc, j + 1, owners), pline))
        # the store variable itself read as a switch: `if not store then return memory[key] end`
        for (name, val), reader in switches.items():
            if val is False and (name in stores or _last(name) in store_last) and reader not in flagged_lines:
                at = next((k for k, t in enumerate(sc.toks) if t.line == reader), None)
                if at is not None and sc.studio_only(at):
                    continue
                flagged_lines.add(reader)
                gl = next((t.line for t in sc.toks if t.text in DS_GETTERS), None)
                add(sc, "memory-fallback", [gl, reader],
                    "when the store %s is missing (its GetDataStore failed or never ran), line %d reads or writes "
                    "memory instead of refusing" % (name, reader))
    findings.sort(key=lambda f: (f["script"], f["line"] or 0, f["kind"]))
    findings = _merge(findings)
    for f in findings:
        f["fix"] = FIX
    ds = [sc.path for sc in scripts if sc.uses_datastore()]
    return {"scanned": len(scripts), "dataStoreScripts": ds, "standIns": standins, "findings": findings,
            "unreadable": unreadable, "unbalanced": [sc.path for sc in scripts if not sc.balanced]}


def _substitute(sc, r, owners):
    """Does the expression at token r put a table or a stand-in where a store was?"""
    t = sc.t(r)
    if t is None:
        return False
    if t.kind == "op" and t.text == "{":
        return True
    if t.kind == "name":
        if t.text == "setmetatable" and sc.is_(r + 1, "(", "op"):
            return True
        e, chain = sc.chain_fwd(r)
        if chain is None:
            return False
        if chain in owners or chain.split(".")[0] in owners:
            return True
        if MEMORY_NAME.search(chain) or STANDIN_PATH.search(chain):
            return True
        if sc.is_(e, ":", "op") and sc.t(e + 1) is not None and sc.t(e + 1).text.lower() == "new":
            return bool(MEMORY_NAME.search(chain) or chain in owners)
    return False


def _what(sc, r, owners):
    """A few words for the value at token r."""
    t = sc.t(r)
    if t is not None and t.text == "{":
        return "a table"
    if t is not None and t.text == "setmetatable":
        return "a setmetatable table"
    e, chain = sc.chain_fwd(r)
    if chain:
        if (sc.is_(e, ".", "op") or sc.is_(e, ":", "op")) and sc.t(e + 1) is not None:
            return "%s%s%s()" % (chain, sc.t(e).text, sc.t(e + 1).text)
        return chain
    return "a value"


def _merge(findings):
    """One finding per script and first line; the kinds and reasons of the others folded in."""
    out, at = [], {}
    for f in findings:
        k = (f["script"], f["line"])
        if k in at:
            g = at[k]
            if f["kind"] not in g["kinds"]:
                g["kinds"].append(f["kind"])
            g["lines"] = sorted(set(g["lines"]) | set(f["lines"]))
            if f["why"] not in g["why"]:
                g["why"] += "; also: " + f["why"]
            continue
        f = dict(f, kinds=[f["kind"]])
        at[k] = f
        out.append(f)
    return out


def _has_getter_on_line(sc, r):
    line = sc.t(r).line
    return any(t.line == line and t.text in DS_GETTERS for t in sc.toks[r:r + 40])


# ---------------------------------------------------------------- game passes
# Found on three games' shops: a pass granted on MarketplaceService.PromptGamePassPurchaseFinished alone. The event
# reports the client's prompt closing (wasPurchased), not Roblox's record; UserOwnsGamePassAsync is the record. A pass
# copied into the save and granted from it on join outlives a refund or a revoked pass.
PASS_FIX = ("call MarketplaceService:UserOwnsGamePassAsync(player.UserId, passId) in a pcall and grant only when it "
            "answers true; on every join grant each pass from that answer, never from a flag or list in the save; a "
            "failed ownership read grants nothing and is retried")
OWNS = "UserOwnsGamePassAsync"
PASS_EVENT = "PromptGamePassPurchaseFinished"
GRANT_CALL = re.compile(r"(?i)(grant|give|award|unlock|apply|activate|enable|reward|perk)")
# "Owned" alone is left out: owned items (an inventory) are restored from the save by design
PASS_WORD = re.compile(r"(?i:pass|vip|perk)|[Oo]wns[A-Z_]")
SAVED_WORD = re.compile(r"(?i)(data|profile|save|saved|stored|record|store)")


def _handler_nodes(sc, k):
    """The function nodes an event's :Connect( at token k hands its callback to: an inline function, or a function
    defined in this script under the name passed (a local, a.b or a:b)."""
    if not (sc.is_(k + 1, ":", "op") and sc.t(k + 2) is not None and sc.t(k + 2).text in ("Connect", "Once")
            and sc.is_(k + 3, "(", "op")):
        return []
    a = k + 4
    if sc.is_(a, "function", "kw"):
        return [nd for nd in sc.nodes if nd.kind == "function" and nd.start == a]
    _e, chain = sc.chain_fwd(a)
    if not chain:
        return []
    return _functions_named(sc, chain)


def _functions_named(sc, chain):
    out = [nd for nd in sc.nodes if nd.kind == "function" and nd.name and nd.name.replace(":", ".") == chain.replace(":", ".")]
    if out:
        return out
    # local NAME = function(...)
    for k, tk in enumerate(sc.toks):
        if tk.kind == "op" and tk.text == "=" and sc.is_(k + 1, "function", "kw"):
            _s, lhs = sc.chain_back(k - 1)
            if lhs == chain:
                out += [nd for nd in sc.nodes if nd.kind == "function" and nd.start == k + 1]
    return out


def _reach(sc, roots, depth=4):
    """The roots and every function of this script they call by name, to depth calls deep."""
    seen, todo = [], list(roots)
    while todo and depth >= 0:
        nxt = []
        for fn in todo:
            if fn in seen or fn.end is None:
                continue
            seen.append(fn)
            for k in range(fn.start, fn.end):
                tk = sc.toks[k]
                if tk.kind == "name" and sc.is_(k + 1, "(", "op") and not sc.is_definition(k):
                    s, chain = sc.chain_back(k)
                    if chain and not (s is not None and s > 0 and sc.is_(s - 1, "function", "kw")):
                        nxt += [f for f in _functions_named(sc, chain) if f not in seen]
        todo, depth = nxt, depth - 1
    return seen


def _grants(sc, fns):
    """[(token index, what)] of permanent grants in the functions: `= true` on a target naming a pass, a perk or saved
    data, or a BoolValue's Value; SetAttribute(..., true); table.insert; or a call named grant, give, award, unlock,
    apply, activate, enable, reward or perk."""
    out = []
    for fn in fns:
        for k in range(fn.start, fn.end or fn.start):
            tk = sc.toks[k]
            if tk.kind == "op" and tk.text == "=" and sc.is_(k + 1, "true", "kw"):
                lhs = _lhs_text(sc, k)
                # a debounce or a local flag is not a grant: the target names a pass, a perk or the saved data
                if lhs and (PASS_WORD.search(lhs) or SAVED_WORD.search(lhs) or lhs.endswith(".Value")):
                    out.append((k, "%s = true" % lhs))
            elif tk.kind == "name" and tk.text == "SetAttribute" and sc.is_(k + 1, "(", "op"):
                close = sc.match_paren(k + 1)
                if sc.is_(close - 1, "true", "kw"):
                    out.append((k, "SetAttribute(..., true)"))
            elif tk.kind == "name" and tk.text == "insert" and sc.is_(k - 1, ".", "op") and sc.is_(k - 2, "table"):
                out.append((k, "table.insert"))
            elif tk.kind == "name" and sc.is_(k + 1, "(", "op") and GRANT_CALL.search(tk.text) \
                    and not sc.is_definition(k) and tk.text not in ("SetAttribute",):
                s, _c = sc.chain_back(k)
                if not (s is not None and s > 0 and sc.is_(s - 1, "function", "kw")):
                    out.append((k, "a call to %s" % tk.text))
    out = [g for g in out if not sc.studio_only(g[0])]
    out.sort()
    return out


def _lhs_text(sc, k):
    """The assignment target before the '=' at token k, brackets included (profile.Data.Passes[id])."""
    j = k - 1
    while j >= 0:
        t = sc.toks[j]
        if t.kind == "op" and t.text == "]":
            d = 0
            while j >= 0:
                if sc.toks[j].text == "]":
                    d += 1
                elif sc.toks[j].text == "[":
                    d -= 1
                    if d == 0:
                        break
                j -= 1
            j -= 1
            continue
        if t.kind == "name" and j + 1 < k and sc.toks[j + 1].kind == "name":
            break                                   # two names in a row: the previous statement's end
        if t.kind == "name" or (t.kind == "op" and t.text in (".", ":")):
            j -= 1
            continue
        break
    return "".join(x.text for x in sc.toks[j + 1:k])


def _owns_in(sc, fns):
    return any(sc.toks[k].text == OWNS for fn in fns for k in range(fn.start, fn.end or fn.start))


def _restores(sc, fns):
    """[(token, line of the saved-pass read)]: a branch or loop in the join path that reads a pass from saved data and
    grants inside it."""
    out = []
    for nd in sc.nodes:
        if not any(f.start <= nd.start <= (f.end or 0) for f in fns):
            continue
        if nd.kind == "if":
            for br in nd.branches:
                if not br.cond:
                    continue
                words = [sc.toks[k].text for k in range(br.cond[0], br.cond[1])]
                if any(PASS_WORD.search(w) for w in words) and any(SAVED_WORD.search(w) for w in words):
                    body = Node("function", br.body[0] - 1)
                    body.end = br.body[1]
                    g = _grants(sc, [body])
                    if g:
                        out.append((g[0][0], sc.toks[br.cond[0]].line, g[0][1]))
        elif nd.kind == "loop":
            words = [sc.toks[k].text for k in range(nd.start, nd.body_start)]
            if any(PASS_WORD.search(w) for w in words) and any(SAVED_WORD.search(w) for w in words):
                body = Node("function", nd.body_start - 1)
                body.end = nd.end
                g = _grants(sc, [body])
                if g:
                    out.append((g[0][0], sc.toks[nd.start].line, g[0][1]))
    return out


def check_passes(sources):
    """{scanned, passScripts, findings, unreadable}: game-pass grants with no UserOwnsGamePassAsync read on their path."""
    scripts, unreadable = [], []
    for s in sources:
        path = s.get("path")
        path = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        try:
            scripts.append(Script(path, s.get("source") or "", s.get("className")))
        except Exception as exc:  # noqa: BLE001 - named, never dropped
            unreadable.append({"script": path, "error": "%s: %s" % (type(exc).__name__, exc)})
    findings, pass_scripts = [], []
    for sc in scripts:
        texts = {t.text for t in sc.toks if t.kind == "name"}
        if PASS_EVENT in texts or OWNS in texts:
            pass_scripts.append(sc.path)
        for k, tk in enumerate(sc.toks):
            if tk.kind != "name" or tk.text not in (PASS_EVENT, "PlayerAdded"):
                continue
            roots = _handler_nodes(sc, k)
            if not roots:
                continue
            fns = _reach(sc, roots)
            if _owns_in(sc, fns):
                continue
            hl = tk.line
            if tk.text == PASS_EVENT:
                g = _grants(sc, fns)
                if g:
                    gl = sc.toks[g[0][0]].line
                    findings.append({"script": sc.path, "kind": "pass-granted-unchecked", "line": gl,
                                     "lines": sorted({hl, gl}), "text": sc.line_text(gl),
                                     "why": "the PromptGamePassPurchaseFinished handler connected at line %d grants (%s, "
                                            "line %d) with no UserOwnsGamePassAsync read on its path: the event's "
                                            "wasPurchased is the prompt closing, not Roblox's record" % (hl, g[0][1], gl),
                                     "fix": PASS_FIX})
            else:
                for gi, cl, what in _restores(sc, fns):
                    gl = sc.toks[gi].line
                    findings.append({"script": sc.path, "kind": "pass-restored-from-save", "line": gl,
                                     "lines": sorted({hl, cl, gl}), "text": sc.line_text(gl),
                                     "why": "the join path (PlayerAdded at line %d) grants (%s, line %d) from a pass "
                                            "read from saved data at line %d with no UserOwnsGamePassAsync read: a "
                                            "refunded or revoked pass stays granted" % (hl, what, gl, cl),
                                     "fix": PASS_FIX})
    seen, out = set(), []
    for f in sorted(findings, key=lambda f: (f["script"], f["line"])):
        if (f["script"], f["line"]) not in seen:
            seen.add((f["script"], f["line"]))
            out.append(f)
    return {"scanned": len(scripts), "passScripts": pass_scripts, "findings": out, "unreadable": unreadable}


CANNOT_SEE = ("static: values passed through arguments, other modules' tables or setters are followed by name only",
              "a stand-in made at run time (an __index function, loadstring) or inside a library the place does not hold",
              "nothing is run: a clean answer means none of these shapes was found; rbx_play_probe fault_store (a "
              "DataStore that fails on cue) is the play-test proof")
PASS_CANNOT_SEE = ("a handler is followed into functions this script defines and calls by name (four calls deep); a "
                   "grant inside another module's function is seen only when its name says grant, give, award, unlock, "
                   "apply, activate, enable, reward or perk",
                   "an ownership read anywhere on the handler's path counts, whether or not the grant waits for it",
                   "a join path restoring passes is recognised by a condition or loop naming both a pass and saved data")
CHECKS = ("save", "passes")


def answer(sources, origin, truncated=False, checks=CHECKS):
    """The tool's answer over [{path, source, className?}] read from origin ("studio", "rbxlx", "rbxl", "folder"):
    the save layer (checks "save") and game-pass grants ("passes")."""
    if not sources:
        return {"ok": False, "ran": True, "origin": origin, "scanned": 0, "checks": list(checks),
                "error": "no scripts were read from the %s: nothing was checked, which is not clean" % origin}
    out = {"ran": True, "origin": origin, "truncated": bool(truncated), "checks": list(checks), "cannotSee": []}
    found = []
    if "save" in checks:
        r = check(sources)
        found += r["findings"]
        out.update(scanned=r["scanned"], dataStoreScripts=len(r["dataStoreScripts"]),
                   dataStoreScriptPaths=r["dataStoreScripts"][:100], standIns=r["standIns"][:100])
        out["cannotSee"] += list(CANNOT_SEE)
        if r["unreadable"]:
            out["unreadable"] = r["unreadable"]
        if r["unbalanced"]:
            out["unbalanced"] = r["unbalanced"]
            out["unbalancedNote"] = ("these scripts' blocks do not balance (an end too many or too few): their findings "
                                     "are read on a best-effort block structure; rbx_script_analyze names the syntax "
                                     "error")
        if not r["dataStoreScripts"]:
            out["note"] = "no script names DataStoreService or GetDataStore: there is no save layer here to check"
    if "passes" in checks:
        p = check_passes(sources)
        found += p["findings"]
        out.update(scanned=p["scanned"], passScripts=len(p["passScripts"]), passScriptPaths=p["passScripts"][:100])
        out["cannotSee"] += list(PASS_CANNOT_SEE)
        if p["unreadable"] and "unreadable" not in out:
            out["unreadable"] = p["unreadable"]
        if not p["passScripts"]:
            out["passNote"] = "no script names PromptGamePassPurchaseFinished or UserOwnsGamePassAsync"
    out.update(ok=not found, count=len(found), clean=not found, findings=found)
    if found:
        out["error"] = "%d finding(s): %s" % (len(found), "; ".join(
            "%s line %s (%s)" % (f["script"], f["line"], f["kind"]) for f in found[:8]))
        fixes = []
        if any(f["kind"] in ("memory-fallback", "stand-in-outside-studio") for f in found):
            fixes.append(FIX)
        if any(f["kind"].startswith("pass-") for f in found):
            fixes.append(PASS_FIX)
        out["fix"] = " | ".join(fixes)
    return out


# ---------------------------------------------------------------- reading scripts from a file or a folder
SCRIPT_CLASSES = ("Script", "LocalScript", "ModuleScript")


def sources_from_rbxlx(text, text_classes=()):
    """[{path, className, source}] for every script in an XML place (.rbxlx). With text_classes (TEXT_CLASSES), each
    instance of those classes is answered too, as {path, className, text} with its Text property."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    out = []

    def walk(item, path):
        props = item.find("Properties")
        name = None
        if props is not None:
            for p in props:
                if p.get("name") == "Name" and p.tag in ("string", "ProtectedString"):
                    name = p.text or ""
                    break
        here = path + [name if name is not None else item.get("class", "?")]
        if item.get("class") in SCRIPT_CLASSES and props is not None:
            src = next((p.text or "" for p in props if p.get("name") == "Source"), None)
            if src is not None:
                out.append({"path": here, "className": item.get("class"), "source": src})
        elif item.get("class") in text_classes and props is not None:
            txt = next((p.text or "" for p in props if p.get("name") == "Text" and p.tag == "string"), None)
            if txt is not None:
                out.append({"path": here, "className": item.get("class"), "text": txt})
        for child in item.findall("Item"):
            walk(child, here)
    for item in root.findall("Item"):
        walk(item, [])
    return out


def sources_from_file(path, text_classes=()):
    """(sources, origin) for every script in a saved place file: a binary .rbxl through mif_roblox.rbxl, an XML .rbxlx
    here; origin is "rbxl" or "rbxlx". Nothing is cut short: the whole file is read. Raises on a file that is not a
    readable place (the caller names the reason)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if path.lower().endswith(".rbxl") or blob.startswith(b"<roblox!"):
        from mif_roblox import rbxl
        return rbxl.script_sources(blob, text_classes=text_classes), "rbxl"
    return sources_from_rbxlx(blob.decode("utf-8", "replace"), text_classes=text_classes), "rbxlx"


def instance_paths_from_file(path):
    """[path] of every instance in a saved place file (.rbxl through mif_roblox.rbxl, .rbxlx here), each a list of names
    from the root service. Raises on a file that is not a readable place."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if path.lower().endswith(".rbxl") or blob.startswith(b"<roblox!"):
        from mif_roblox import rbxl
        return rbxl.instance_paths(blob)
    import xml.etree.ElementTree as ET
    out = []

    def walk(item, here):
        props = item.find("Properties")
        name = None
        if props is not None:
            for p in props:
                if p.get("name") == "Name" and p.tag in ("string", "ProtectedString"):
                    name = p.text or ""
                    break
        me = here + [name if name is not None else item.get("class", "?")]
        out.append(me)
        for child in item.findall("Item"):
            walk(child, me)
    for item in ET.fromstring(blob.decode("utf-8", "replace")).findall("Item"):
        walk(item, [])
    return out


def value_from_file(path, inst_path):
    """(found, value, why) of a ValueBase's Value in a saved place: .rbxl through mif_roblox.rbxl, .rbxlx here (bool,
    int, int64, double, float and string Values). Raises on a file that is not a readable place."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if path.lower().endswith(".rbxl") or blob.startswith(b"<roblox!"):
        from mif_roblox import rbxl
        return rbxl.value_at(blob, inst_path)
    import xml.etree.ElementTree as ET
    want = list(inst_path)
    conv = {"bool": lambda t: (t or "").strip() == "true", "int": int, "int64": int, "double": float, "float": float,
            "string": lambda t: t or ""}

    def walk(item, here):
        props = item.find("Properties")
        name, val, kind = None, None, None
        if props is not None:
            for p in props:
                if p.get("name") == "Name" and p.tag in ("string", "ProtectedString"):
                    name = p.text or ""
                elif p.get("name") == "Value":
                    val, kind = p.text, p.tag
        me = here + [name if name is not None else item.get("class", "?")]
        if me == want:
            if kind is None:
                return True, None, "%s at that path has no Value property" % item.get("class")
            if kind not in conv:
                return True, None, "%s's Value is a %s, which this reader does not decode" % (item.get("class"), kind)
            return True, conv[kind](val), None
        for child in item.findall("Item"):
            got = walk(child, me)
            if got is not None:
                return got
        return None
    for item in ET.fromstring(blob.decode("utf-8", "replace")).findall("Item"):
        got = walk(item, [])
        if got is not None:
            return got
    return False, None, "no instance at %s in the file" % ".".join(want)


def sources_from_folder(folder, exts=(".lua", ".luau")):
    """[{path, className, source}] for every .lua/.luau file under folder; a Rojo-style name (x.server.luau,
    x.client.lua, init.luau) gives the class and path."""
    out = []
    for dirpath, _dirs, files in os.walk(folder):
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in exts:
                continue
            cls = "ModuleScript"
            if stem.endswith(".server"):
                cls, stem = "Script", stem[:-7]
            elif stem.endswith(".client"):
                cls, stem = "LocalScript", stem[:-7]
            rel = os.path.relpath(os.path.join(dirpath, f), folder)
            parts = [p for p in os.path.dirname(rel).replace("\\", "/").split("/") if p and p != "."]
            if stem != "init":
                parts.append(stem)
            with open(os.path.join(dirpath, f), encoding="utf-8", errors="replace") as fh:
                out.append({"path": parts or [stem], "className": cls, "source": fh.read(), "file": rel})
    return out


# ---------------------------------------------------------------- Robux prices typed into UI text and scripts
# A price typed into a label or a script string is the list price for every player, while the price a player pays is
# set per region and Roblox's price management will not finish its opt-in while it
# detects hard-coded prices. The same script reader and tokens as the save check, plus the Text of the UI classes below.
TEXT_CLASSES = ("TextLabel", "TextButton")
# R$, the word, and the Robux glyph of Roblox's own fonts (U+E002), also as a script writes it: "\u{E002}", whose
# backslash tokenize() takes off
ROBUX_MARK = re.compile("R\\$|robux|\ue002|u\\{0*E002\\}", re.I)
PRICE_FIX = ("read the price at run time: MarketplaceService:GetProductInfoAsync(id, Enum.InfoType.GamePass or "
             ".Product).PriceInRobux on the player's client, and show that")
PRICE_CANNOT_SEE = ("static: a price put together at run time (a number variable beside a Robux mark, string.format, a "
                    "config value shown as it is) is not a typed price and is not followed",
                    "text other than the Text of %s and script string literals (images, a TextBox, attributes, "
                    "localization tables)" % " and ".join(TEXT_CLASSES),
                    "a shortened price (1.5K) or one written out in words")


def _price_pattern(price):
    forms = {str(price), "{:,}".format(price)}
    return re.compile(r"(?<![\d.,])(?:%s)(?![\d]|[.,]\d)" % "|".join(re.escape(f) for f in sorted(forms)))


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _dotted(path):
    return ".".join(str(p) for p in path) if isinstance(path, (list, tuple)) else str(path)


def typed_prices(items, sources, origin, truncated=False):
    """Every place one of the items' list prices is typed beside a Robux mark or the item's name: in the Text of a
    TextLabel or TextButton, and in a script's string literals (a number literal counts when a string on its line
    carries the Robux mark). items is [{key, prices: [whole Robux], names: [str]}]; sources is what the readers above
    answer, scripts as {path, source, className?} and UI text as {path, text, className}. A price read at run time has no
    number to find and is not a finding. Nothing scanned fails."""
    scripts = [s for s in sources if isinstance(s, dict) and isinstance(s.get("source"), str)]
    texts = [s for s in sources if isinstance(s, dict) and isinstance(s.get("text"), str)]
    out = {"ok": False, "ran": True, "origin": origin, "scanned": {"texts": len(texts), "scripts": len(scripts)},
           "truncated": bool(truncated)}
    if not scripts and not texts:
        out["error"] = "no UI text and no scripts were read from the %s: nothing was checked, which is not clean" % origin
        return out
    want = []
    for it in items:
        names = sorted(n for n in {_squash(n) for n in it.get("names") or []} if len(n) >= 3)
        for price in sorted(set(it.get("prices") or [])):
            want.append((it["key"], price, _price_pattern(price), names))
    out["prices"] = {it["key"]: sorted(set(it.get("prices") or [])) for it in items}
    out["noPrice"] = [it["key"] for it in items if not it.get("prices")]
    if not want:
        out["error"] = "no item has a price to look for: give each item its list price"
        return out
    found = {}

    def hit(path, cls, kind, line, text, key, price, by):
        f = found.setdefault((path, line, price), {"path": path, "className": cls, "kind": kind, "line": line,
                                                   "price": price, "items": [], "by": by, "text": text.strip()[:160],
                                                   "fix": PRICE_FIX})
        if key not in f["items"]:
            f["items"].append(key)
        if by == "robux-mark":
            f["by"] = by

    for t in texts:
        path = _dotted(t["path"])
        in_path = _squash(path)
        for ln, line in enumerate(t["text"].split("\n"), 1):
            mark, flat = ROBUX_MARK.search(line), _squash(line)
            for key, price, pat, names in want:
                if not pat.search(line):
                    continue
                by = ("robux-mark" if mark else "item-name" if any(n in flat for n in names)
                      else "item-name-in-path" if any(n in in_path for n in names) else None)
                if by:
                    hit(path, t.get("className"), "ui-text", ln, line, key, price, by)
    for s in scripts:
        path = _dotted(s["path"])
        by_line = {}
        for tok in tokenize(s["source"]):
            if tok.kind in ("str", "num"):
                by_line.setdefault(tok.line, []).append(tok)
        src_lines = s["source"].split("\n")
        for ln, toks in sorted(by_line.items()):
            strs = [x.text for x in toks if x.kind == "str"]
            if not strs:
                continue
            mark = any(ROBUX_MARK.search(x) for x in strs)
            flat = _squash(" ".join(strs))
            nums = {x.text for x in toks if x.kind == "num"}
            for key, price, pat, names in want:
                typed = any(pat.search(x) for x in strs)
                by = ("robux-mark" if mark and (typed or str(price) in nums)
                      else "item-name" if typed and any(n in flat for n in names) else None)
                if by:
                    hit(path, s.get("className"), "script-literal", ln,
                        src_lines[ln - 1] if 0 < ln <= len(src_lines) else "", key, price, by)
    findings = sorted(found.values(), key=lambda f: (f["path"], f["line"], f["price"]))
    out.update(ok=not findings, clean=not findings, count=len(findings), findings=findings,
               cannotSee=list(PRICE_CANNOT_SEE))
    if findings:
        out["error"] = "%d typed price(s), each the list price for every player whatever their region's price: %s" % (
            len(findings), "; ".join("%s line %s (%s)" % (f["path"], f["line"], f["price"]) for f in findings[:8]))
        out["fix"] = PRICE_FIX
    return out
