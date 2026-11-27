"""A static check of player-written text: does it reach other players or a save without Roblox's text filter?

Roblox makes the developer responsible for filtering any text shown to players that the developer did not write, and an
experience found without filtering is taken down until it is added. Pet names, signs, titles and typed answers are where
it is missed: a client sends a string over a remote, the server sets it on a label or passes it on, and nobody filtered
it. This reads source text only; it never calls TextService.

Members, from the 0.748 catalog: TextService:FilterStringAsync(stringToFilter, fromUserId, textContext) answers a
TextFilterResult, whose GetNonChatStringForBroadcastAsync() is the text for everyone, GetNonChatStringForUserAsync(toUserId)
the text for one user, and GetChatForUserAsync(toUserId) chat for one user. FilterStringAsync is a server call.

What it follows, in server scripts (a Script, or a ModuleScript a server requires; a LocalScript is the client):
  sources   the arguments after the player of a function handed to <remote>.OnServerEvent:Connect, set as
            <remote>.OnServerInvoke, or handed to <player>.Chatted:Connect (its message)
  taint     inside that function only: a local or assignment whose right side names a tainted value is tainted, unless
            the right side calls FilterStringAsync or one of the result's methods, which makes it filtered
  sinks     text-show    a tainted value assigned to .Text, .PlaceholderText, .Value (a StringValue) or .Name, or set
                         with SetAttribute: replicated to every player who can see the instance
            send-others  a tainted value passed to :FireAllClients or :FireClient
            save         a tainted value passed to a DataStore SetAsync, UpdateAsync or IncrementAsync
Findings, by script, line and why:
  unfiltered-text       a source reaches a sink without the filter: the remote, the handler's line, the sink's line, the
                        author (the handler's player) and the result method that fits the audience
  client-only-filter    FilterStringAsync named in a LocalScript: the filter is a server call, and a client's word that
                        it filtered is not one
  per-keystroke-filter  FilterStringAsync inside a function connected to a TextBox's GetPropertyChangedSignal("Text"),
                        Changed or a per-frame signal: filter once, when the text is submitted
A value passed to another function (or stored in a table and read elsewhere) is not followed: each such path is listed
under unresolved with its line, never taken as filtered. A place whose scripts have no such handler and name no TextBox
fails: nothing was traced, which is not clean.
"""
from mif_roblox import save_audit as SA

FILTER_CALLS = ("FilterStringAsync", "GetNonChatStringForBroadcastAsync", "GetNonChatStringForUserAsync",
                "GetChatForUserAsync")
TEXT_PROPS = ("Text", "PlaceholderText", "Value", "Name")
SEND = ("FireAllClients", "FireClient")
SAVE = ("SetAsync", "UpdateAsync", "IncrementAsync")
PER_KEYSTROKE = ("GetPropertyChangedSignal", "Changed", "Heartbeat", "RenderStepped", "Stepped")
METHOD_FOR = {"text-show": "GetNonChatStringForBroadcastAsync() - a label or value every player can see",
              "send-others": "GetNonChatStringForBroadcastAsync() for FireAllClients, or GetNonChatStringForUserAsync("
                             "toUserId) for FireClient to one player",
              "save": "filter it when it is shown again, with the method for that audience: a saved string is shown "
                      "later to players who never typed it"}
FIX = ("on the server: local result = TextService:FilterStringAsync(text, player.UserId) in a pcall, then show or send "
       "only what the result's method for the audience answers (an error answers nothing, never the raw text)")
CANNOT_SEE = (
    "static, inside one handler: a value passed into another function, kept in a table or a module's state, or read "
    "back from a save later is listed under unresolved, not followed",
    "a remote's arguments are found where its handler is connected in the same script; a handler connected through a "
    "wrapper library (a networking module) is not seen",
    "TextBox text a client keeps to itself is not a finding: only text that reaches the server, other players or a "
    "save is",
)


def _params(sc, fn):
    """The parameter names of the function node fn, in order."""
    k = fn.start + 1
    while k < len(sc.toks) and not sc.is_(k, "(", "op"):
        k += 1
    if k >= len(sc.toks):
        return []
    end = sc.match_paren(k)
    return [t.text for t in sc.toks[k + 1:end] if t.kind == "name"]


def _body(sc, fn):
    k = fn.start + 1
    while k < len(sc.toks) and not sc.is_(k, "(", "op"):
        k += 1
    return (sc.match_paren(k) + 1 if k < len(sc.toks) else fn.start + 1), (fn.end if fn.end is not None else len(sc.toks))


def _handlers(sc):
    """[(kind, remote chain, function node, line)] of the server-side text sources in this script."""
    out = []
    for k, tk in enumerate(sc.toks):
        if tk.kind != "name" or not sc.is_(k - 1, ".", "op"):
            continue
        if tk.text in ("OnServerEvent", "Chatted"):
            s, chain = sc.chain_back(k)
            name = chain[:-len(tk.text) - 1] or sc.line_text(tk.line)[:80]
            for fn in SA._handler_nodes(sc, k):
                out.append(("chat" if tk.text == "Chatted" else "remote", name, fn, tk.line))
        elif tk.text == "OnServerInvoke" and sc.is_(k + 1, "=", "op"):
            _s, chain = sc.chain_back(k)
            a = k + 2
            fns = ([nd for nd in sc.nodes if nd.kind == "function" and nd.start == a] if sc.is_(a, "function", "kw")
                   else SA._functions_named(sc, sc.chain_fwd(a)[1] or ""))
            for fn in fns:
                out.append(("remote", chain[:-len(tk.text) - 1] or sc.line_text(tk.line)[:80], fn, tk.line))
    return out


def _names_in(sc, a, b):
    return {sc.toks[i].text for i in range(a, b) if sc.toks[i].kind == "name"}


def _calls_filter(sc, a, b):
    return any(sc.toks[i].kind == "name" and sc.toks[i].text in FILTER_CALLS for i in range(a, b))


def _stmt_end(sc, k, end):
    """The end of the expression starting at token k: the line's end, or past a bracket opened on it."""
    line, i, depth = sc.toks[k].line, k, 0
    while i < end:
        t = sc.toks[i]
        if t.kind == "op" and t.text in ("(", "[", "{"):
            depth += 1
        elif t.kind == "op" and t.text in (")", "]", "}"):
            depth -= 1
            if depth < 0:
                return i
        elif depth == 0 and t.line != line:
            return i
        i += 1
    return end


def _trace(sc, fn, sources):
    """(sinks [(kind, token, what)], unresolved [(token, what)]) of tainted values reaching sinks in fn's body."""
    a, b = _body(sc, fn)
    tainted, sinks, unresolved = set(sources), [], []
    for k in range(a, b):
        t = sc.toks[k]
        # assignments: local x = expr / x = expr / x.Text = expr
        if t.kind == "op" and t.text == "=":
            r_end = _stmt_end(sc, k + 1, b)
            rhs = _names_in(sc, k + 1, r_end) & tainted
            s, lhs = sc.chain_back(k - 1)
            if lhs is None:
                continue
            last = lhs.replace(":", ".").rsplit(".", 1)[-1]
            if rhs and not _calls_filter(sc, k + 1, r_end):
                if "." in lhs and last in TEXT_PROPS:
                    sinks.append(("text-show", k, "%s = %s" % (lhs, sc.text(k + 1, r_end))))
                elif "." in lhs:
                    unresolved.append((k, "kept in %s, not followed" % lhs))
                else:
                    names = [lhs]
                    j = s - 1
                    while sc.is_(j, ",", "op") and sc.t(j - 1) is not None and sc.t(j - 1).kind == "name":
                        names.append(sc.t(j - 1).text)
                        j -= 2
                    tainted.update(names)
            elif "." not in lhs and lhs in tainted and not rhs:
                tainted.discard(lhs)         # overwritten with something the client did not send
            continue
        # calls: x:Method(args) / f(args)
        if t.kind == "name" and sc.is_(k + 1, "(", "op") and not sc.is_definition(k):
            close = sc.match_paren(k + 1)
            hit = _names_in(sc, k + 2, close) & tainted
            if not hit or t.text in FILTER_CALLS:
                continue
            method = sc.is_(k - 1, ":", "op")
            if method and t.text in SEND:
                sinks.append(("send-others", k, "%s(%s)" % (t.text, sc.text(k + 2, close))))
            elif method and t.text in SAVE:
                sinks.append(("save", k, "%s(%s)" % (t.text, sc.text(k + 2, close))))
            elif method and t.text == "SetAttribute":
                sinks.append(("text-show", k, "SetAttribute(%s)" % sc.text(k + 2, close)))
            elif t.text not in ("print", "warn", "tostring", "type", "typeof", "pcall", "assert", "error", "len",
                                "sub", "lower", "upper", "match", "find", "gsub", "format"):
                _s, chain = sc.chain_back(k)
                unresolved.append((k, "passed to %s(...), not followed" % (chain or t.text)))
    return sinks, unresolved


def _connected_to(sc, fn):
    """The signal names whose :Connect( hands this function node its callback."""
    out = []
    for k, tk in enumerate(sc.toks):
        if tk.kind == "name" and tk.text in ("Connect", "Once") and sc.is_(k - 1, ":", "op") and sc.is_(k + 1, "(", "op"):
            if fn in SA._handler_nodes(sc, k - 2):
                _s, chain = sc.chain_back(k - 2)
                if chain:
                    out.append(chain)
                elif sc.is_(k - 2, ")", "op"):
                    # x:GetPropertyChangedSignal("Text"):Connect(
                    j = k - 2
                    depth = 0
                    while j > 0:
                        if sc.is_(j, ")", "op"):
                            depth += 1
                        elif sc.is_(j, "(", "op"):
                            depth -= 1
                            if depth == 0:
                                break
                        j -= 1
                    _s2, chain2 = sc.chain_back(j - 1)
                    if chain2:
                        out.append(chain2)
    return out


def check(sources):
    findings, unresolved, unreadable, traced, textboxes = [], [], [], [], 0
    for src in sources:
        path = ".".join(src["path"]) if isinstance(src.get("path"), list) else str(src.get("path"))
        try:
            sc = SA.Script(path, src.get("source"), src.get("className"))
        except Exception as exc:  # noqa: BLE001 - a source that does not tokenize is named, never dropped
            unreadable.append({"script": path, "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        textboxes += sum(1 for t in sc.toks if t.kind in ("name", "string") and t.text == "TextBox")
        client = src.get("className") == "LocalScript"
        for k, tk in enumerate(sc.toks):
            if tk.kind == "name" and tk.text == "FilterStringAsync":
                if client:
                    findings.append({"script": path, "kind": "client-only-filter", "line": tk.line,
                                     "text": sc.line_text(tk.line),
                                     "why": "FilterStringAsync is a server call: a client cannot filter for others, "
                                            "and a server must not trust a client that says it did", "fix": FIX})
                for fn in [nd for nd in sc.nodes if nd.kind == "function" and nd.start < k < (nd.end or 0)]:
                    sig = [c for c in _connected_to(sc, fn) if c.rsplit(".", 1)[-1].split(":")[-1] in PER_KEYSTROKE
                           or c.endswith("GetPropertyChangedSignal")]
                    if sig:
                        findings.append({"script": path, "kind": "per-keystroke-filter", "line": tk.line,
                                         "text": sc.line_text(tk.line), "signal": sig[0],
                                         "why": "the filter runs on every change of the text (each keystroke), which "
                                                "spends TextService's budget and filters half-typed text",
                                         "fix": "filter once, on the server, when the text is submitted (FocusLost "
                                                "with enterPressed, or a submit button's remote)"})
                        break
        if client:
            continue
        for kind, remote, fn, line in _handlers(sc):
            ps = _params(sc, fn)
            author, args = (ps[0] if ps else None), ps[1:] if kind == "remote" else ps[:1]
            if kind == "chat":
                author = None
            traced.append({"script": path, "remote": remote, "kind": kind, "line": line, "arguments": args})
            sinks, unres = _trace(sc, fn, args)
            for sink, k, what in sinks:
                findings.append({"script": path, "kind": "unfiltered-text", "remote": remote, "handlerLine": line,
                                 "line": sc.toks[k].line, "sink": sink, "what": what[:200],
                                 "text": sc.line_text(sc.toks[k].line),
                                 "author": "%s.UserId" % author if author else "the chatting player's UserId",
                                 "method": METHOD_FOR[sink],
                                 "why": "a string the client sent over %s reaches %s with no TextService filter on the "
                                        "server" % (remote, {"text-show": "a property other players see",
                                                             "send-others": "other clients",
                                                             "save": "a save"}[sink]),
                                 "fix": FIX})
            for k, what in unres:
                unresolved.append({"script": path, "remote": remote, "line": sc.toks[k].line, "what": what})
    return {"findings": findings, "unresolved": unresolved, "unreadable": unreadable, "traced": traced,
            "textBoxMentions": textboxes, "scanned": len(sources)}


def answer(sources, origin, truncated=False):
    if not sources:
        return {"ok": False, "ran": True, "origin": origin, "scanned": 0,
                "error": "no scripts were read from the %s: nothing was checked, which is not clean" % origin}
    r = check(sources)
    out = {"ran": True, "origin": origin, "truncated": bool(truncated), "scanned": r["scanned"],
           "traced": r["traced"][:200], "unresolved": r["unresolved"][:200], "findings": r["findings"],
           "count": len(r["findings"]), "cannotSee": list(CANNOT_SEE), "members": list(FILTER_CALLS)}
    if r["unreadable"]:
        out["unreadable"] = r["unreadable"]
    if not r["traced"] and not r["textBoxMentions"]:
        out.update(ok=False, error="no remote handler, Chatted handler or TextBox was found in %d scripts: no player "
                                   "text was traced, which is not clean" % r["scanned"])
        return out
    out["ok"] = not r["findings"]
    if r["findings"]:
        out["error"] = "%d finding(s): %s" % (len(r["findings"]), "; ".join(
            "%s line %s (%s%s)" % (f["script"], f["line"], f["kind"], ", " + f["remote"] if f.get("remote") else "")
            for f in r["findings"][:8]))
    return out

