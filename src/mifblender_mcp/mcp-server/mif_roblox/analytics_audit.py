"""A static check of a game's analytics events: are they bounded, logged once, and logged after the change they report?

Roblox's AnalyticsService takes an onboarding funnel, named funnels, economy (source and sink) events and custom events
from server scripts. The members and their parameter order below are the 0.748 catalog's (roblox_catalog.json; the
offline test compares this table with it, so a Studio update that moves a parameter fails the test, not a game).

ASSUMED, NOT MEASURED HERE: the limits. LIMITS holds what Roblox's documentation was read to state on 2026-11-22 - 10
funnels, 100 steps a funnel, 100 custom event names, 3 custom fields an event, 10 currency (resource) types, and a
request budget of 120 + 20 x concurrent players a minute - and nothing in this repo has measured any of them. Every
answer carries the numbers it counted against and says they are assumed. Whether the onboarding funnel counts toward
the 10 is not known, so it is not counted.

What it flags, by script, line and why:
  dynamic-event-name        an event, funnel, step, currency, journey or progression name that is built instead of
                            declared: a concatenation, an interpolated string, string.format or tostring, an
                            instance's or player's .Name/.DisplayName/.UserId/.Text, or an argument a client sent
                            (OnServerEvent, OnServerInvoke, Chatted). Each distinct name is a new event, so a name made
                            from player data has no bound.
  per-frame-log             a log call (or a call to a function that logs) inside a function connected to
                            RunService's per-frame signals or bound with BindToRenderStep, or inside a while or repeat
                            loop whose only wait is one frame long.
  unlogged-currency-change  a declared currency's field or index changed (+=, -=, x.C = x.C + n) in a server script by a
                            function that reaches no economy event. A currency is declared by being the currencyType
                            of a LogEconomyEvent call (a literal, or a declared table's value) or by the caller
                            (currencies); it is matched by its exact name. Code that only runs in Studio is skipped.
  logged-before-save        an economy event sent before a DataStore write (UpdateAsync, SetAsync, IncrementAsync) or a
                            call named save... later in the same function: the event reports a change the save has not
                            confirmed.
  funnel-step-missing       a funnel whose literal step numbers leave a gap from 1.
  funnel-step-order         a function that logs a lower step of a funnel after a higher one.
  over-limit                more distinct names than an assumed limit, a funnel step above the step limit, or more
                            custom fields in one table than the limit.
  deprecated-member         a Fire... member the catalog tags Deprecated, in a script that names AnalyticsService.

A name is declared when it is a string literal, an Enum item, or a lookup in a table whose constructor holds only
string literals (EVENTS = { RoundPlayed = "RoundPlayed" }; EVENTS[key] or EVENTS.RoundPlayed). A name that is a
function's parameter is followed one call out: the arguments its callers pass are classified instead. The analytics
template (rbx_script_scaffold kind analytics) passes: every name it sends is a lookup in a declared table.

What it cannot see is in CANNOT_SEE. NEVER RAISE from check(): a source that does not tokenize is reported under
unreadable, never dropped. The bridge never calls a Log member itself; this only reads source text.
"""
import re

from mif_roblox import save_audit as SA

# AnalyticsService's non-deprecated Log members and their parameters, in order (0.748 catalog)
LOG_PARAMS = {
    "LogCustomEvent": ("player", "eventName", "value", "customFields"),
    "LogEconomyEvent": ("player", "flowType", "currencyType", "amount", "endingBalance", "transactionType", "itemSku",
                        "customFields"),
    "LogFunnelStepEvent": ("player", "funnelName", "funnelSessionId", "step", "stepName", "customFields"),
    "LogJourneyEvent": ("player", "journeyName", "nodeName", "journeySessionId", "customFields"),
    "LogOnboardingFunnelStepEvent": ("player", "step", "stepName", "customFields"),
    "LogProgressionCompleteEvent": ("player", "progressionPathName", "level", "levelName", "customFields"),
    "LogProgressionEvent": ("player", "progressionPathName", "status", "level", "levelName", "customFields"),
    "LogProgressionFailEvent": ("player", "progressionPathName", "level", "levelName", "customFields"),
    "LogProgressionStartEvent": ("player", "progressionPathName", "level", "levelName", "customFields"),
}
DEPRECATED = ("FireCustomEvent", "FireEvent", "FireInGameEconomyEvent", "FireLogEvent", "FirePlayerProgressionEvent")
NAME_PARAMS = ("eventName", "funnelName", "currencyType", "stepName", "journeyName", "nodeName", "progressionPathName")
ECONOMY = "LogEconomyEvent"
FUNNELS = ("LogFunnelStepEvent", "LogOnboardingFunnelStepEvent")
ONBOARDING = "<onboarding>"
# RunService's per-frame signals (0.748 catalog), and the calls that make a name out of data
FRAME_SIGNALS = ("Heartbeat", "Stepped", "RenderStepped", "PreRender", "PreAnimation", "PreSimulation", "PostSimulation")
FRAME_SECONDS = 1 / 30.0
BUILDERS = ("format", "tostring", "lower", "upper", "sub", "gsub", "rep", "concat")
DATA_FIELDS = ("Name", "DisplayName", "UserId", "Text")
INPUT_SIGNALS = {"OnServerEvent": 1, "Chatted": 0}   # signal -> the first parameter a client controls
SAVE_WRITES = ("UpdateAsync", "SetAsync", "IncrementAsync")
SAVE_NAME = re.compile(r"^[Ss]ave")
# ASSUMED (module docstring): Roblox's stated limits as read on 2026-11-22, not measured in this repo
LIMITS = {"funnels": 10, "funnelSteps": 100, "customEventNames": 100, "customFields": 3, "currencyTypes": 10,
          "requestsPerMinute": "120 + 20 x concurrent players"}
LIMITS_NOTE = ("assumed: Roblox's documented limits as read on 2026-11-22, not measured by MifBridge; the onboarding "
               "funnel is not counted toward the funnel limit; the request budget is not counted (it is a run-time rate)")
FIX = {
    "dynamic-event-name": "send a name from a declared table of string literals, and put the changing part in the "
                          "value or a custom field",
    "per-frame-log": "log on the moment the thing happens, or from a loop that waits seconds; a per-frame call spends "
                     "the whole request budget",
    "unlogged-currency-change": "change the currency through its one grant or spend function and log the source or "
                                "sink event there, after the change landed",
    "logged-before-save": "send the event after the save answered success; a failed or retried save otherwise leaves "
                          "an event for a change that never landed, or logs it twice",
    "funnel-step-missing": "number the steps 1..n with no gap, each logged once where the player reaches it",
    "funnel-step-order": "log the steps in the order a player meets them, lowest first",
    "over-limit": "declare fewer names; put the detail in the value or one of the custom fields",
    "deprecated-member": "use the Log... member that replaced it (rbx_describe_class AnalyticsService)",
}
CANNOT_SEE = (
    "static: a name carried by a function's parameter is followed one call out, and through a required module by the "
    "module's name only; a name held in an upvalue, a table field or another module's variable is listed under "
    "unresolved, not judged",
    "a member reached by a computed key (service[name](...)) or through a wrapper library whose code is not in the "
    "place is not seen, and neither is a currency changed by SetAttribute, a ValueBase set with =, or a DataStore call",
    "nothing runs: whether a branch is reached, how often a function is called, the live request rate and what Roblox "
    "accepted are not proven; a per-frame log behind a time check is still flagged. The template's self-test on its "
    "recorder (rbx_play_probe self_test) and the experience's own analytics pages are that proof",
)
_OPEN, _CLOSE = ("(", "[", "{"), (")", "]", "}")


def _is(sc, i, text):
    t = sc.t(i)
    return t is not None and t.kind == "op" and t.text == text


def _split(sc, a, b):
    """[(start, end)] of the top-level comma-separated parts of tokens [a, b)."""
    if a >= b:
        return []
    out, d, s = [], 0, a
    for k in range(a, b):
        t = sc.toks[k]
        if t.kind != "op":
            continue
        if t.text in _OPEN:
            d += 1
        elif t.text in _CLOSE:
            d -= 1
        elif d == 0 and t.text in (",", ";"):
            out.append((s, k))
            s = k + 1
    if s < b:
        out.append((s, b))
    return out


def _interp(t):
    return t.kind == "str" and getattr(t, "q", None) == "`" and "{" in t.text


class _Facts:
    """One script's functions, calls and names, on save_audit's tokens and blocks."""

    def __init__(self, sc):
        self.sc = sc
        self.name = sc.path.split(".")[-1]
        self.server = sc.class_name != "LocalScript"
        self.fns = [nd for nd in sc.nodes if nd.kind == "function" and nd.end is not None]
        self.fn_at = {nd.start: nd for nd in self.fns}
        self.names, self.params = {}, {}
        for nd in self.fns:
            self.names[nd.start] = self._name(nd)
            self.params[nd.start] = self._params(nd)
        self.inputs = self._inputs()
        self.aliases = self._aliases()
        self.methods = {nd.start: set() for nd in self.fns}   # function -> the Log members it reaches
        self.top = set()                                      # the Log members the chunk itself reaches
        self.sites = {}                                       # call token -> the Log members that call reaches
        self.passes = {}                                      # function -> [(parameter index, member, parameter name)]

    def _name(self, nd):
        sc = self.sc
        if nd.name:
            return nd.name
        if _is(sc, nd.start - 1, "="):
            _s, chain = sc.chain_back(nd.start - 2)
            return chain
        return None

    def _params(self, nd):
        sc = self.sc
        j = nd.start + 1
        while j < nd.end and not _is(sc, j, "("):
            j += 1
        out = []
        for a, _b in _split(sc, j + 1, sc.match_paren(j)):
            t = sc.t(a)
            out.append(t.text if t is not None and t.kind == "name" else None)
        return out

    def _fn_arg(self, a, b):
        """The function nodes an argument [a, b) names: an inline function, or a name defined in this script."""
        sc = self.sc
        if sc.is_(a, "function", "kw") and a in self.fn_at:
            return [self.fn_at[a]]
        if b - a == 1 and sc.t(a).kind == "name":
            return [nd for nd in self.fns if self.names[nd.start] == sc.t(a).text]
        return []

    def _inputs(self):
        """{function start: parameter names a client controls} for remote and chat handlers."""
        sc, out = self.sc, {}
        for k, t in enumerate(sc.toks):
            if t.kind != "name":
                continue
            fns, first = [], 1
            if t.text in INPUT_SIGNALS and _is(sc, k + 1, ":") and sc.t(k + 2) is not None \
                    and sc.t(k + 2).text in ("Connect", "ConnectParallel", "Once") and _is(sc, k + 3, "("):
                args = _split(sc, k + 4, sc.match_paren(k + 3))
                fns, first = (self._fn_arg(*args[0]) if args else []), INPUT_SIGNALS[t.text]
            elif t.text == "OnServerInvoke" and _is(sc, k + 1, "="):
                fns = self._fn_arg(k + 2, k + 3)
            for nd in fns:
                out.setdefault(nd.start, set()).update(p for p in self.params[nd.start][first:] if p)
        return out

    def _aliases(self):
        """{local name: module name} for local X = require(...): the last string or name in the require is the module."""
        sc, out = self.sc, {}
        for k, t in enumerate(sc.toks):
            if t.kind == "name" and t.text == "require" and _is(sc, k + 1, "(") and _is(sc, k - 1, "=") \
                    and sc.t(k - 2) is not None and sc.t(k - 2).kind == "name":
                inner = sc.toks[k + 2:sc.match_paren(k + 1)]
                strs = [x.text for x in inner if x.kind == "str"]
                nms = [x.text for x in inner if x.kind == "name"]
                mod = strs[-1] if strs else (nms[-1] if nms else None)
                if mod:
                    out[sc.t(k - 2).text] = mod
        return out

    def enclosing(self, k):
        """The functions holding token k, innermost first."""
        return sorted((nd for nd in self.fns if nd.start < k <= nd.end), key=lambda nd: -nd.start)

    def is_call(self, k):
        sc, t = self.sc, self.sc.toks[k]
        return t.kind == "name" and _is(sc, k + 1, "(") and not sc.is_definition(k) and not sc.is_(k - 1, "function", "kw")

    def args(self, k):
        """The arguments of the call whose name is token k; a dot call on a method drops the receiver elsewhere."""
        return _split(self.sc, k + 2, self.sc.match_paren(k + 1))


def _tables(facts):
    """{table name: {values, keys}} for every table whose constructor holds only string literals, by name over all
    scripts: T = { A = "a", "b" }."""
    out = {}
    for f in facts:
        sc = f.sc
        for k, t in enumerate(sc.toks):
            if not (t.kind == "op" and t.text == "{" and _is(sc, k - 1, "=") and sc.t(k - 2) is not None
                    and sc.t(k - 2).kind == "name"):
                continue
            entries, good, n = {}, True, 0
            for a, b in _split(sc, k + 1, sc.match_paren(k)):
                key = None
                if _is(sc, a, "[") and _is(sc, sc.match_paren(a) + 1, "="):
                    c = sc.match_paren(a)
                    key = sc.t(a + 1).text if c == a + 2 and sc.t(a + 1).kind in ("str", "num") else None
                    a = c + 2
                elif sc.t(a).kind == "name" and _is(sc, a + 1, "="):
                    key, a = sc.t(a).text, a + 2
                else:
                    n += 1
                    key = str(n)
                if b - a == 1 and sc.t(a).kind == "str" and not _interp(sc.t(a)):
                    entries[key if key is not None else "?%d" % a] = sc.t(a).text
                else:
                    good = False
                    break
            if good and entries:
                d = out.setdefault(sc.t(k - 2).text, {"values": [], "keys": {}})
                d["values"].extend(v for v in entries.values() if v not in d["values"])
                d["keys"].update(entries)
    return out


def _expr_end(sc, a):
    """Where the expression starting at token a ends: a closing bracket or comma it does not own, a statement keyword,
    or two operands in a row."""
    d, k, n = 0, a, len(sc.toks)
    while k < n:
        t = sc.toks[k]
        if t.kind == "op" and t.text in _OPEN:
            d += 1
        elif t.kind == "op" and t.text in _CLOSE:
            if d == 0:
                return k
            d -= 1
        elif t.kind == "kw" and t.text == "function":
            nd = next((x for x in sc.nodes if x.kind == "function" and x.start == k), None)
            if nd is None or nd.end is None:
                return n
            k = nd.end
        elif d == 0:
            if t.kind == "op" and t.text in (",", ";"):
                return k
            if t.kind == "kw" and t.text not in ("and", "or", "not", "true", "false", "nil"):
                return k
            if k > a and t.kind in ("name", "num"):
                p = sc.toks[k - 1]
                if p.kind in ("name", "num", "str") or (p.kind == "op" and p.text in _CLOSE) or \
                        (p.kind == "kw" and p.text in ("true", "false", "nil", "end")):
                    return k
        k += 1
    return n


def _assignment(sc, name, before):
    """The expression last assigned to a plain name before token `before`: (start, end, token of the name) or None."""
    for j in range(before - 1, -1, -1):
        t = sc.toks[j]
        if t.kind != "name" or t.text != name or (j > 0 and sc.toks[j - 1].kind == "op" and sc.toks[j - 1].text in (".", ":")):
            continue
        # the names to its left and right, up to the =
        left, i = 0, j
        while i - 2 >= 0 and _is(sc, i - 1, ",") and sc.toks[i - 2].kind == "name":
            i -= 2
            left += 1
        if i > 0 and sc.toks[i - 1].kind == "op" and sc.toks[i - 1].text in ("{", "(", ",", ".", ":", "["):
            continue
        e = j + 1
        while _is(sc, e, ",") and sc.t(e + 1) is not None and sc.t(e + 1).kind == "name":
            e += 2
        if not _is(sc, e, "="):
            continue
        a = e + 1
        for _ in range(left):
            a = _expr_end(sc, a)
            if not _is(sc, a, ","):
                a = None
                break
            a += 1
        if a is None:
            return None
        return a, _expr_end(sc, a), j
    return None


class _Audit:
    def __init__(self, sources, currencies=None):
        self.facts, self.unreadable = [], []
        for s in sources:
            path = s.get("path")
            path = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            try:
                self.facts.append(_Facts(SA.Script(path, s.get("source") or "", s.get("className"))))
            except Exception as exc:  # noqa: BLE001 - a script this cannot read is named, never dropped
                self.unreadable.append({"script": path, "error": "%s: %s" % (type(exc).__name__, exc)})
        self.tables = _tables(self.facts)
        self.by_name = {}
        for f in self.facts:
            self.by_name.setdefault(f.name, []).append(f)
        self.findings, self.unresolved, self.seen = [], [], set()
        self.names = {"eventName": {}, "funnelName": {}, "currencyType": {}}   # value -> where first seen
        self.steps = []        # (funnel, step or None, facts, token)
        self.given = list(currencies or [])
        self.logs = []         # (facts, token, member)

    # -- findings
    def find(self, f, kind, k, why, lines=None):
        line = f.sc.toks[k].line
        if (f.sc.path, kind, line) in self.seen:
            return
        self.seen.add((f.sc.path, kind, line))
        self.findings.append({"script": f.sc.path, "kind": kind, "line": line, "lines": sorted(set(lines or [line])),
                              "text": f.sc.line_text(line), "why": why, "fix": FIX[kind]})

    # -- what an expression is
    def classify(self, f, a, b, depth=0):
        """{kind: literal | declared | enum | built | input | param | unknown, values, why, fn, index}"""
        sc = f.sc
        a, b = sc.strip(a, b)
        if a >= b:
            return {"kind": "unknown"}
        toks = sc.toks[a:b]
        if b - a == 1 and toks[0].kind == "str":
            if _interp(toks[0]):
                return {"kind": "built", "why": "an interpolated string"}
            return {"kind": "literal", "values": [toks[0].text]}
        if toks[0].kind == "name" and toks[0].text == "Enum":
            return {"kind": "enum"}
        if b - a == 1 and toks[0].kind == "name":
            return self._variable(f, a, depth)
        # T[...] and T.Field on a table of string literals
        if _is(sc, b - 1, "]"):
            o = next((i for i in range(a, b) if _is(sc, i, "[") and sc.match_paren(i) == b - 1), None)
            if o is not None and o > a and sc.t(o - 1).kind == "name" and sc.t(o - 1).text in self.tables:
                tb = self.tables[sc.t(o - 1).text]
                key = sc.t(o + 1).text if b - 1 == o + 2 and sc.t(o + 1).kind in ("str", "num") else None
                return {"kind": "declared", "values": [tb["keys"][key]] if key in tb["keys"] else list(tb["values"]),
                        "table": sc.t(o - 1).text}
        elif b - a >= 3 and all((x.kind == "name") if i % 2 == 0 else (x.kind == "op" and x.text == ".")
                                for i, x in enumerate(toks)) and toks[-3].text in self.tables:
            tb = self.tables[toks[-3].text]
            if toks[-1].text in tb["keys"]:
                return {"kind": "declared", "values": [tb["keys"][toks[-1].text]], "table": toks[-3].text}
        inputs = set()
        for nd in f.enclosing(a):
            inputs |= f.inputs.get(nd.start, set())
        for i, x in enumerate(toks):
            prev = toks[i - 1] if i else None
            dotted = prev is not None and prev.kind == "op" and prev.text in (".", ":")
            if x.kind == "name" and x.text in inputs and not dotted:
                return {"kind": "input", "why": "what a client sent as %s" % x.text}
        for i, x in enumerate(toks):
            prev = toks[i - 1] if i else None
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if x.kind == "op" and x.text == "..":
                return {"kind": "built", "why": "a concatenation"}
            if _interp(x):
                return {"kind": "built", "why": "an interpolated string"}
            if x.kind == "name" and x.text in BUILDERS and nxt is not None and nxt.kind == "op" and nxt.text == "(":
                return {"kind": "built", "why": "made by %s()" % x.text}
            if x.kind == "name" and x.text in DATA_FIELDS and prev is not None and prev.kind == "op" and prev.text == ".":
                return {"kind": "built", "why": "it reads .%s" % x.text}
        return {"kind": "unknown"}

    def _variable(self, f, a, depth):
        sc, name = f.sc, f.sc.toks[a].text
        owner = next((nd for nd in f.enclosing(a) if name in f.params[nd.start]), None)
        asg = _assignment(sc, name, a) if depth < 4 else None
        if asg is not None and (owner is None or asg[2] > owner.start):
            return self.classify(f, asg[0], asg[1], depth + 1)
        if owner is not None:
            if name in f.inputs.get(owner.start, ()):
                return {"kind": "input", "why": "what a client sent as %s" % name}
            return {"kind": "param", "fn": owner.start, "index": f.params[owner.start].index(name), "name": name}
        return {"kind": "unknown"}

    # -- pass 1: the Log calls
    def scan_logs(self):
        for f in self.facts:
            sc = f.sc
            names_service = any(t.text == "AnalyticsService" for t in sc.toks if t.kind in ("name", "str"))
            for k, t in enumerate(sc.toks):
                if t.kind != "name" or not f.is_call(k) or not (_is(sc, k - 1, ":") or _is(sc, k - 1, ".")):
                    continue
                if t.text in DEPRECATED and names_service:
                    self.find(f, "deprecated-member", k, "AnalyticsService:%s is tagged Deprecated in the catalog" % t.text)
                if t.text not in LOG_PARAMS:
                    continue
                self.logs.append((f, k, t.text))
                f.sites[k] = {t.text}
                enc = f.enclosing(k)
                for nd in enc:
                    f.methods[nd.start].add(t.text)
                if not enc:
                    f.top.add(t.text)
                args = f.args(k)
                if _is(sc, k - 1, "."):
                    args = args[1:]          # service.LogX(service, player, ...)
                named = dict(zip(LOG_PARAMS[t.text], args))
                self._names(f, k, t.text, named)
                self._funnel(f, k, t.text, named)
                self._fields(f, k, named.get("customFields"))

    def _names(self, f, k, member, named):
        for p in NAME_PARAMS:
            if p not in named:
                continue
            c = self.classify(f, *named[p])
            self._judge(f, k, member, p, c, named[p])

    def _judge(self, f, k, member, p, c, span, via=None):
        sc = f.sc
        if c["kind"] in ("built", "input"):
            self.find(f, "dynamic-event-name", k, "%s of %s%s is %s (%s): every distinct value is a new name, so it has no "
                      "bound" % (p, member, " (through %s)" % via if via else "", c["why"], sc.text(*span)[:80]))
        elif c["kind"] in ("literal", "declared"):
            if p in self.names:
                for v in c["values"]:
                    self.names[p].setdefault(v, {"script": sc.path, "line": sc.toks[k].line})
        elif c["kind"] == "param" and via is None:
            f.passes.setdefault(c["fn"], []).append((c["index"], member, p))
        elif c["kind"] != "enum":
            self.unresolved.append({"script": sc.path, "line": sc.toks[k].line, "member": member, "parameter": p,
                                    "expression": sc.text(*span)[:80]})

    def _funnel(self, f, k, member, named):
        if member not in FUNNELS or "step" not in named:
            return
        sc = f.sc
        funnel = ONBOARDING
        if member == "LogFunnelStepEvent":
            c = self.classify(f, *named["funnelName"]) if "funnelName" in named else {"kind": "unknown"}
            funnel = c["values"][0] if c["kind"] in ("literal", "declared") and len(c.get("values") or []) == 1 else None
        a, b = sc.strip(*named["step"])
        step = None
        if b - a == 1 and sc.t(a).kind == "num":
            try:
                step = int(float(sc.t(a).text))
            except ValueError:
                step = None
        self.steps.append((funnel, step, f, k))

    def _fields(self, f, k, span):
        if span is None:
            return
        sc = f.sc
        a, b = sc.strip(*span)
        if b - a == 1 and sc.t(a).kind == "name":
            asg = _assignment(sc, sc.t(a).text, a)
            if asg is None:
                return
            a, b = sc.strip(asg[0], asg[1])
        if _is(sc, a, "{") and sc.match_paren(a) == b - 1:
            n = len(_split(sc, a + 1, b - 1))
            if n > LIMITS["customFields"]:
                self.find(f, "over-limit", k, "%d custom fields in one event; the assumed limit is %d (CustomField01 to "
                          "CustomField03)" % (n, LIMITS["customFields"]), [sc.toks[k].line, sc.toks[a].line])

    # -- pass 2: who reaches a Log call, through this script's functions and required modules
    def targets(self, f, k):
        sc = f.sc
        _s, chain = sc.chain_back(k)
        if chain is None:
            return []
        norm = chain.replace(":", ".")
        segs = norm.split(".")
        out = [(f, nd) for nd in f.fns if (f.names[nd.start] or "").replace(":", ".") == norm]
        if len(segs) >= 2 and segs[0] == "self":
            out += [(f, nd) for nd in f.fns if "." in (f.names[nd.start] or "").replace(":", ".")
                    and (f.names[nd.start] or "").replace(":", ".").endswith("." + segs[-1])]
        if len(segs) == 2 and segs[0] in f.aliases:
            for g in self.by_name.get(f.aliases[segs[0]], []):
                out += [(g, nd) for nd in g.fns if "." in (g.names[nd.start] or "").replace(":", ".")
                        and (g.names[nd.start] or "").replace(":", ".").endswith("." + segs[1])]
        return out

    def reach(self):
        calls = []
        for f in self.facts:
            for k, t in enumerate(f.sc.toks):
                if t.kind == "name" and t.text not in LOG_PARAMS and f.is_call(k):
                    tg = self.targets(f, k)
                    if tg:
                        calls.append((f, k, tg))
        for _ in range(30):
            changed = False
            for f, k, tg in calls:
                m = set()
                for g, nd in tg:
                    m |= g.methods[nd.start]
                if not m or m <= f.sites.get(k, set()):
                    continue
                changed = True
                f.sites.setdefault(k, set()).update(m)
                enc = f.enclosing(k)
                for nd in enc:
                    f.methods[nd.start] |= m
                if not enc:
                    f.top |= m
            if not changed:
                break
        # a name a function takes as a parameter: judge what each caller passes
        for f, k, tg in calls:
            sc = f.sc
            colon = _is(sc, k - 1, ":")
            for g, nd in tg:
                if (":" in (g.names[nd.start] or "")) != colon:
                    continue
                for index, member, p in g.passes.get(nd.start, []):
                    args = f.args(k)
                    if index >= len(args):
                        continue
                    c = self.classify(f, *args[index])
                    self._judge(f, k, member, p, c, args[index], via=g.names[nd.start])

    # -- pass 3: the checks over reached calls
    def frames(self, f):
        """[(start, end, what)] token ranges that run every frame."""
        sc, out = f.sc, []
        for k, t in enumerate(sc.toks):
            if t.kind != "name" or not _is(sc, k + 1, "("):
                continue
            fns, what = [], None
            if t.text in ("Connect", "ConnectParallel") and _is(sc, k - 1, ":") and sc.t(k - 2) is not None \
                    and sc.t(k - 2).kind == "name" and sc.t(k - 2).text in FRAME_SIGNALS:
                args = f.args(k)
                fns, what = (f._fn_arg(*args[0]) if args else []), "RunService.%s" % sc.t(k - 2).text
            elif t.text == "BindToRenderStep" and (_is(sc, k - 1, ":") or _is(sc, k - 1, ".")):
                args = f.args(k)
                fns, what = (f._fn_arg(*args[-1]) if args else []), "BindToRenderStep"
            out += [(nd.start, nd.end, what) for nd in fns]
        for nd in sc.nodes:
            if nd.end is None or not (nd.kind == "repeat" or (nd.kind == "loop" and sc.is_(nd.start, "while", "kw"))):
                continue
            inner = [x for x in f.fns if nd.start < x.start < nd.end]
            frame = slow = False
            for k in range(nd.start, nd.end + 1):
                t = sc.toks[k]
                if t.kind != "name" or not _is(sc, k + 1, "(") or sc.in_functions(k, inner):
                    continue
                if t.text == "Wait" and _is(sc, k - 1, ":") and sc.t(k - 2) is not None and sc.t(k - 2).text in FRAME_SIGNALS:
                    frame = True
                elif t.text == "wait" and not (_is(sc, k - 1, ":") or (_is(sc, k - 1, ".") and not sc.is_(k - 2, "task"))):
                    args = f.args(k)
                    secs = None
                    if len(args) == 1 and args[0][1] - args[0][0] == 1 and sc.t(args[0][0]).kind == "num":
                        try:
                            secs = float(sc.t(args[0][0]).text)
                        except ValueError:
                            secs = None
                    if not args or (secs is not None and secs <= FRAME_SECONDS):
                        frame = True
                    else:
                        slow = True
            if frame and not slow:
                out.append((nd.start, nd.end, "a loop that waits one frame"))
        return out

    def check_frames(self):
        for f in self.facts:
            regions = self.frames(f)
            for k in sorted(f.sites):
                hit = next((r for r in regions if r[0] < k <= r[1]), None)
                if hit:
                    self.find(f, "per-frame-log", k, "this reaches %s inside %s: one request a frame for every player" % (
                        ", ".join(sorted(f.sites[k])), hit[2]), [f.sc.toks[hit[0]].line, f.sc.toks[k].line])

    def check_saves(self):
        for f in self.facts:
            sc = f.sc
            for k in sorted(f.sites):
                if ECONOMY not in f.sites[k]:
                    continue
                enc = f.enclosing(k)
                fn = next((nd for nd in enc if f.names[nd.start]), enc[-1] if enc else None)
                if fn is None:
                    continue
                for j in range(k + 1, fn.end):
                    t = sc.toks[j]
                    if t.kind != "name" or not _is(sc, j + 1, "(") or sc.is_definition(j):
                        continue
                    if (t.text in SAVE_WRITES and sc.ds_call_at(j)) or SAVE_NAME.match(t.text):
                        self.find(f, "logged-before-save", k, "the economy event is sent on line %d and %s is called "
                                  "after it on line %d: the event reports a change the save has not confirmed" % (
                                      sc.toks[k].line, t.text, t.line), [sc.toks[k].line, t.line])
                        break

    def currencies(self):
        return sorted(set(self.given) | set(self.names["currencyType"]))

    def check_currency(self):
        cur = set(self.currencies())
        if not cur:
            return False
        for f in self.facts:
            if not f.server:
                continue
            sc, done = f.sc, set()
            for k, t in enumerate(sc.toks):
                if t.kind != "op" or t.text not in ("+=", "-=", "="):
                    continue
                a = k
                while a > 0 and (sc.toks[a - 1].kind in ("name", "str", "num")
                                 or (sc.toks[a - 1].kind == "op" and sc.toks[a - 1].text in (".", "[", "]"))):
                    if sc.toks[a - 1].kind == "name" and a - 1 > 0 and sc.toks[a - 2].kind in ("name", "num", "str"):
                        a -= 1
                        break
                    a -= 1
                target = sc.toks[a:k]
                hit = next((x.text for x in target if x.kind in ("name", "str") and x.text in cur), None)
                if hit is None or len(target) < 2 or (a > 0 and sc.is_(a - 1, "local", "kw")):
                    continue
                if t.text == "=":
                    rhs = sc.toks[k + 1:k + 1 + len(target)]
                    after = sc.t(k + 1 + len(target))
                    if [x.text for x in rhs] != [x.text for x in target] or after is None or after.text not in ("+", "-"):
                        continue
                if sc.studio_only(k):
                    continue
                enc = f.enclosing(k)
                if any(ECONOMY in f.methods[nd.start] for nd in enc) or (not enc and ECONOMY in f.top):
                    continue
                key = (enc[0].start if enc else -1, hit)
                if key in done:
                    continue
                done.add(key)
                where = "function %s" % (f.names[enc[0].start] or "(anonymous)") if enc else "the script's own body"
                self.find(f, "unlogged-currency-change", k, "%s changes %s and reaches no LogEconomyEvent: this source or "
                          "sink is missing from the economy charts" % (where, hit))
        return True

    def check_funnels(self):
        by = {}
        for funnel, step, f, k in self.steps:
            by.setdefault(funnel, []).append((step, f, k))
        for funnel, recs in by.items():
            label = "the onboarding funnel" if funnel == ONBOARDING else "funnel %r" % funnel
            lit = [r for r in recs if r[0] is not None]
            if funnel is not None and lit and len(lit) == len(recs):
                have = {r[0] for r in lit}
                missing = [i for i in range(1, max(have) + 1) if i not in have]
                if missing:
                    _s, f, k = min(lit, key=lambda r: (r[0], r[1].sc.path, r[2]))
                    self.find(f, "funnel-step-missing", k, "%s logs steps %s and never %s" % (
                        label, sorted(have), missing), sorted({r[1].sc.toks[r[2]].line for r in lit if r[1] is f}))
            if funnel is not None:
                last = {}
                for step, f, k in sorted(lit, key=lambda r: (r[1].sc.path, r[2])):
                    enc = f.enclosing(k)
                    key = (f.sc.path, enc[0].start if enc else -1)
                    if key in last and step < last[key][0]:
                        self.find(f, "funnel-step-order", k, "%s: step %d is logged after step %d in the same function" % (
                            label, step, last[key][0]), [last[key][1], f.sc.toks[k].line])
                    last[key] = (step, f.sc.toks[k].line)
            top = max(lit, key=lambda r: r[0]) if lit else None
            if top is not None and top[0] > LIMITS["funnelSteps"]:
                self.find(top[1], "over-limit", top[2], "%s has a step %d; the assumed limit is %d steps a funnel" % (
                    label, top[0], LIMITS["funnelSteps"]))

    def check_limits(self):
        for p, key, what in (("eventName", "customEventNames", "custom event names"), ("funnelName", "funnels", "funnels"),
                             ("currencyType", "currencyTypes", "currency types")):
            n = len(self.names[p])
            if n > LIMITS[key]:
                where = next((x for x in self.logs if LOG_PARAMS[x[2]].count(p)), None)
                if where is not None:
                    self.find(where[0], "over-limit", where[1], "%d distinct %s are declared across the scripts; the "
                              "assumed limit is %d" % (n, what, LIMITS[key]))

    def counts(self):
        steps = {}
        for funnel, step, _f, _k in self.steps:
            if funnel is not None and step is not None:
                steps[funnel] = max(steps.get(funnel, 0), step)
        return {"logCalls": len(self.logs), "customEventNames": len(self.names["eventName"]),
                "funnels": len(self.names["funnelName"]), "currencyTypes": len(self.names["currencyType"]),
                "highestFunnelStep": max(steps.values()) if steps else 0}


def check(sources, currencies=None):
    """sources: [{path (list or dotted str), source, className?}] -> {scanned, analyticsScripts, findings, counts,
    currencies, currencyChecked, unresolved, unreadable}. Each finding: {script, kind, line, lines, text, why, fix}."""
    au = _Audit(sources, currencies)
    au.scan_logs()
    au.reach()
    au.check_frames()
    au.check_saves()
    checked = au.check_currency()
    au.check_funnels()
    au.check_limits()
    au.findings.sort(key=lambda x: (x["script"], x["line"], x["kind"]))
    seen, unresolved = set(), []
    for u in au.unresolved:
        key = (u["script"], u["line"], u["parameter"])
        if key not in seen:
            seen.add(key)
            unresolved.append(u)
    return {"scanned": len(au.facts), "analyticsScripts": sorted({f.sc.path for f, _k, _m in au.logs}),
            "findings": au.findings, "counts": au.counts(), "currencies": au.currencies(), "currencyChecked": checked,
            "unresolved": unresolved, "unreadable": au.unreadable,
            "unbalanced": [f.sc.path for f in au.facts if not f.sc.balanced]}


def answer(sources, origin, truncated=False, currencies=None):
    """The tool's answer over [{path, source, className?}] read from origin ("studio", "rbxlx", "rbxl", "folder")."""
    if not sources:
        return {"ok": False, "ran": True, "origin": origin, "scanned": 0,
                "error": "no scripts were read from the %s: nothing was checked, which is not clean" % origin}
    r = check(sources, currencies)
    found = r["findings"]
    out = {"ok": not found, "ran": True, "origin": origin, "scanned": r["scanned"], "truncated": bool(truncated),
           "analyticsScripts": len(r["analyticsScripts"]), "analyticsScriptPaths": r["analyticsScripts"][:100],
           "count": len(found), "clean": not found, "findings": found, "counts": r["counts"],
           "limits": dict(LIMITS), "limitsNote": LIMITS_NOTE, "currencies": r["currencies"],
           "unresolved": r["unresolved"][:100], "cannotSee": list(CANNOT_SEE)}
    notes = []
    if not r["analyticsScripts"]:
        notes.append("no script calls an AnalyticsService Log member: there are no events here to check")
    if not r["currencyChecked"]:
        out["notChecked"] = ["unlogged-currency-change: no currency is declared (no LogEconomyEvent names one, and "
                             "currencies was not given), so no currency change was looked for"]
    if notes:
        out["note"] = "; ".join(notes)
    if r["unreadable"]:
        out["unreadable"] = r["unreadable"]
    if r["unbalanced"]:
        out["unbalanced"] = r["unbalanced"]
        out["unbalancedNote"] = ("these scripts' blocks do not balance (an end too many or too few): their findings "
                                 "are read on a best-effort block structure; rbx_script_analyze names the syntax error")
    if found:
        kinds = {}
        for x in found:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        out["byKind"] = kinds
        out["error"] = "%d analytics finding(s): %s" % (
            len(found), "; ".join("%s %s line %s" % (x["kind"], x["script"], x["line"]) for x in found[:8]))
    return out
