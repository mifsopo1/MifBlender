"""The roadmap's grounding: what a plan's steps need, derived - never typed - from the source, the shipped data and the
facts the server reads from a live editor. docs/30_ROADMAP.md is the design.

PURE. Nothing here talks to an editor, to Blender or to the network: server.py's mif_roadmap gathers the live facts
and hands them in, and tools/test_mif_roadmap.py drives every status with no editor at all.

WHAT IT DERIVES FROM THE SOURCE (scan_source), per endpoint handler H_<name> in Source/MifBridge/Private/*.cpp:
  guards     every MIF_ENGINE_AT_LEAST / MIF_ENGINE_BEFORE in the handler, WITH its direction (the rule that
             tools/ue_capability_query.py imports from here - one copy of it);
  refusals   a SHORT preprocessor arm (an #if/#else branch of a few lines) that calls Fail(...), with the
             conditions it sits under - "on an engine before 5.5 this answers <message>", "without plugin X it
             answers <message>". A long arm with a Fail inside it is a branch with an input check, not a refusal
             of the version, and is not counted;
  plugins    every MIF_WITH_<X> the handler sits under or branches on; MifBridge.Build.cs names the plugin each
             one stands for (AddPluginModules("MIF_WITH_X", "Plugin", ...)) and which ones every build compiles off
             (PublicDefinitions.Add("MIF_WITH_X=0") with no AddPluginModules);
  cooked     a handler that checks for cooked targets (IsCookedOrContainerPackage, RefuseIfCookedGraph,
             CookedGraphError, MifNoteCookedContext): it may read, refuse or note differently on cooked content;
  crash      a Fail(...) whose message names a crash - the bridge's own words for a case it refuses because the
             editor would go down.
LIMIT, said rather than hidden: a check made in a shared helper rather than in the handler body is not seen.

THE CHECK ALLOWLIST. A catalog step may carry a read-only `check` (is this step already done in the project?). It may
call only an endpoint on CHECK_ENDPOINTS - not anything in the readOnly bucket, which holds save_package and
run_console because that bucket is about the undo stack, not about mutation (docs/15 section 1).
"""
import difflib
import glob
import io
import json
import os
import re

# ---- the guard rule, shared with tools/ue_capability_query.py ---------------------------------------------------------
HANDLER_RE = re.compile(r"^\s*(?:static\s+)?void\s+(H_[a-z0-9_]+)\s*\(")
BOUNDARY_RE = re.compile(r"MIF_ENGINE_(AT_LEAST|BEFORE)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def vkey(v):
    """5.10 sorts after 5.9, which a string compare gets wrong."""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def applies_at(direction, boundary, engine):
    """Does this guard put `engine` on the SHORT side of the branch?

    AT_LEAST(5,7) guards code that exists from 5.7 - so 5.3 to 5.6 do NOT get it.
    BEFORE(5,7)   guards code that exists up to 5.6 - so 5.7 and later do NOT get it."""
    e, b = vkey(engine), vkey(boundary)
    return e < b if direction == "AT_LEAST" else e >= b


def holds_at(direction, boundary, engine):
    """Is the guard's condition itself TRUE at `engine` - the code under `#if MIF_ENGINE_<direction>(...)` compiled?"""
    return not applies_at(direction, boundary, engine)


# ---- the read-only checks a catalog step may run ----------------------------------------------------------------------
# Endpoints whose handlers read and never write, each checked by name against MifBridgeSafety.cpp's unsafe list and the
# readOnly bucket's trap (save_package, run_console). Adding one means reading its handler first.
CHECK_ENDPOINTS = frozenset({
    "list_level_actors", "find_assets", "list_datatables", "list_blueprints", "landscape_info", "list_foliage_instances",
    "nav_status", "list_state_trees", "list_project_plugins", "project_asset_distribution", "list_sublevels",
    "list_niagara_emitters", "describe_package", "list_dirty_packages",
})

# The calls that answer a preview to commit when the project's change cap is on (docs/15 section 6; bridge manual
# section 1). A step that uses one says so, so a preview answer is expected rather than read as a failure.
PREVIEW_ENDPOINTS = frozenset({"spawn_many", "write_datatable_rows", "delete_datatable_rows", "snap_actors_to_ground",
                               "batch"})

COOKED_MARKERS = ("IsCookedOrContainerPackage", "RefuseIfCookedGraph", "CookedGraphError", "MifNoteCookedContext")
REFUSAL_ARM_MAX = 15          # non-blank lines: an arm longer than this is a code branch, not a refusal
_TEXT_RE = re.compile(r'TEXT\("((?:[^"\\]|\\.)*)"\)')
_ATOM_ENGINE = re.compile(r"(!\s*)?MIF_ENGINE_(AT_LEAST|BEFORE)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_ATOM_WITH = re.compile(r"(!\s*)?(?:defined\s*\(\s*)?(MIF_WITH_[A-Z0-9_]+)\)?")
_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def plugin_root(start):
    """The folder holding MifBridge.uplugin, walking up from `start` (a checkout's tools/mcp-server is two folders down,
    a store package's Content/Python/MifBridge/mcp-server four), or None for a server with no plugin beside it."""
    cur = os.path.abspath(start)
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "MifBridge.uplugin")):
            return cur
        up = os.path.dirname(cur)
        if up == cur:
            break
        cur = up
    return None


def _parse_expr(expr):
    """(atoms, complete) for a preprocessor expression. atoms: [("engine", (dir, "5.7"), positive) |
    ("with", "MIF_WITH_X", positive)]. complete: the expression is exactly a conjunction of those atoms, so its
    negation (an #else arm) can be attributed when it holds ONE atom. An || anywhere makes it opaque."""
    expr = re.sub(r"//.*$", "", expr or "").strip()
    if "||" in expr:
        return None, False           # opaque: no line under it is attributed to any condition
    atoms = []
    rest = expr
    for m in _ATOM_ENGINE.finditer(expr):
        atoms.append(("engine", (m.group(2), "%s.%s" % (m.group(3), m.group(4))), not m.group(1)))
    rest = _ATOM_ENGINE.sub(" ", rest)
    for m in _ATOM_WITH.finditer(rest):
        atoms.append(("with", m.group(2), not m.group(1)))
    rest = _ATOM_WITH.sub(" ", rest)
    leftover = re.sub(r"[()&\s]", "", rest)
    return atoms, (leftover == "" and bool(atoms))


def _negate(atoms, complete):
    """The atoms an #else arm sits under: only a single complete atom negates into one we can name."""
    if complete and atoms and len(atoms) == 1:
        kind, key, pos = atoms[0]
        return [(kind, key, not pos)]
    return None          # opaque: the arm's condition cannot be attributed


def _fail_message(lines, i):
    """The message a Fail( on line i passes: its TEXT("...") pieces, joined, up to the parenthesis that closes the call
    (at most 8 lines) - never the next statement's text."""
    text = "\n".join(lines[i:i + 8])
    start = text.find("Fail(")
    depth, in_str, esc, end = 0, False, False, len(text)
    for k in range(start + 4, len(text)):
        c = text[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = k
                break
    parts = _TEXT_RE.findall(text[start:end + 1])
    msg = "".join(parts).replace('\\"', '"').replace("\\n", " ")
    return re.sub(r"\s+", " ", msg).strip()[:400]


def build_plugin_map(root):
    """({macro: plugin}, {macro forced off in every build: Build.cs line}) from MifBridge.Build.cs."""
    path = os.path.join(root, "Source", "MifBridge", "MifBridge.Build.cs")
    try:
        text = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}, {}
    plugins = dict(re.findall(r'AddPluginModules\(\s*"(MIF_WITH_[A-Z0-9_]+)"\s*,\s*"([A-Za-z0-9_]+)"', text))
    off = {}
    for n, line in enumerate(text.split("\n"), 1):
        m = re.search(r'PublicDefinitions\.Add\(\s*"(MIF_WITH_[A-Z0-9_]+)=0"\s*\)', line)
        if not m or m.group(1) in plugins:
            continue
        # ALWAYS OFF only when nothing else in the file can turn it on: an `X=1` in another branch, or an
        # `"X=" + (b ? "1" : "0")`, is a build-time choice (TrajectoryTools is one), not a macro every build compiles off
        if '"%s=1"' % m.group(1) in text or '"%s=" +' % m.group(1) in text:
            continue
        off[m.group(1)] = "MifBridge.Build.cs:%d" % n
    return plugins, off


def scan_source(root):
    """{endpoint: facts} for every handler in <root>/Source/MifBridge/Private, or None when there is no Source.

    facts = {"site": file:line, "guards": [(dir, boundary, site)], "refusals": [{conds, message, site}],
             "plugins": {macro: "whole" | "partial"}, "cooked": [site], "crash": [{message, site}]}"""
    src = os.path.join(root, "Source", "MifBridge", "Private") if root else None
    if not src or not os.path.isdir(src):
        return None
    out = {}
    for path in sorted(glob.glob(os.path.join(src, "*.cpp"))):
        try:
            lines = io.open(path, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        base = os.path.basename(path)
        _scan_file(base, lines, out)
    return out


def _scan_file(base, lines, out):
    stack = []           # frames: {"arms": [(atoms, complete)], "cur": atoms-or-None, "start": line, "body": [..]}
    current, indent, end_brace = None, None, None
    handler_under, handler_body = None, []
    for i, raw in enumerate(lines):
        line = raw.rstrip("\r")
        h = HANDLER_RE.match(line)
        if h:
            current = h.group(1)[2:]
            indent = line[:len(line) - len(line.lstrip())]
            end_brace = indent + "}"
            rec = out.setdefault(current, {"site": "%s:%d" % (base, i + 1), "guards": [], "refusals": [],
                                           "plugins": {}, "cooked": [], "crash": []})
            # the conditions the handler is DEFINED under: a whole-handler plugin or engine dependence. A definition
            # whose whole body is a short Fail (the !MIF_WITH_X arm's twin) is a refusal under those conditions.
            under = []
            for fr in stack:
                for kind, key, pos in (fr["cur"] or []):
                    under.append((kind, key, pos))
                    if kind == "with":
                        rec["plugins"][key] = "whole"
            handler_under, handler_body = (under if under and all(fr["cur"] is not None for fr in stack) else None), []
            continue
        d = _DIRECTIVE.match(line)
        if d:
            kw, rest = d.group(1), d.group(2)
            if kw in ("if", "ifdef", "ifndef"):
                atoms, complete = _parse_expr(rest if kw == "if" else ("!" if kw == "ifndef" else "") + rest.strip())
                stack.append({"arms": [(atoms, complete)], "cur": atoms, "start": i, "body": [], "partial": not complete})
            elif kw in ("elif", "else") and stack:
                fr = stack[-1]
                _close_arm(base, fr, current, out, lines)
                negs = []
                for atoms, complete in fr["arms"]:
                    n = _negate(atoms, complete)
                    if n is None:
                        negs = None
                        break
                    negs += n
                if kw == "elif":
                    atoms, complete = _parse_expr(rest)
                    fr["arms"].append((atoms, complete))
                    fr["cur"] = (negs + atoms) if (negs is not None and atoms is not None) else None
                else:
                    fr["cur"] = negs
                fr["start"], fr["body"] = i, []
            elif kw == "endif" and stack:
                _close_arm(base, stack[-1], current, out, lines)
                stack.pop()
            if current and kw in ("if", "elif") :
                rec = out[current]
                for m in BOUNDARY_RE.finditer(line):
                    rec["guards"].append((m.group(1), "%s.%s" % (m.group(2), m.group(3)), "%s:%d" % (base, i + 1)))
                for m in re.finditer(r"MIF_WITH_[A-Z0-9_]+", line):
                    rec["plugins"].setdefault(m.group(0), "partial")
            continue
        if current is None:
            continue
        if line == end_brace:
            if handler_under:
                body = [j for j in handler_body if lines[j].strip() and not lines[j].strip().startswith("//")]
                fails = [j for j in body if "Fail(" in lines[j]]
                if fails and len(body) <= REFUSAL_ARM_MAX:
                    out[current]["refusals"].append({"conds": list(handler_under), "scope": "endpoint",
                                                     "message": _fail_message(lines, fails[0]),
                                                     "site": "%s:%d" % (base, fails[0] + 1)})
            current, handler_under = None, None
            continue
        rec = out[current]
        handler_body.append(i)
        for fr in stack:
            fr["body"].append(i)
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        for m in BOUNDARY_RE.finditer(line):        # a guard used outside a directive (a constexpr, a macro arg)
            rec["guards"].append((m.group(1), "%s.%s" % (m.group(2), m.group(3)), "%s:%d" % (base, i + 1)))
        if any(k in line for k in COOKED_MARKERS):
            rec["cooked"].append("%s:%d" % (base, i + 1))
        if "Fail(" in line:
            msg = _fail_message(lines, i)
            if "crash" in msg.lower():
                rec["crash"].append({"message": msg, "site": "%s:%d" % (base, i + 1)})


def _close_arm(base, fr, current, out, lines):
    """At the end of an arm: if it is short and calls Fail(...), it is a REFUSAL under the arm's conditions."""
    if current is None or fr["cur"] is None:
        return
    body = [j for j in fr["body"] if lines[j].strip() and not lines[j].strip().startswith("//")]
    if not body or len(body) > REFUSAL_ARM_MAX:
        return
    fails = [j for j in body if "Fail(" in lines[j]]
    if not fails:
        return
    # The innermost arm's conditions only: an enclosing #if is not carried (a limit docs/30 states).
    msg = _fail_message(lines, fails[0])
    # WHOLE ENDPOINT OR SOME CALLS. A refusal that names its own endpoint ("create_cloth_asset needs UE 5.6") refuses
    # every call; any other short refusal arm is about an option or an input ("dependencies:false needs UE 5.4") and
    # refuses only the calls that ask for it - quoted, never promoted to "this step cannot run".
    scope = "endpoint" if msg.lower().startswith(current.lower() + " ") else "some calls"
    out[current]["refusals"].append({"conds": list(fr["cur"]), "scope": scope, "message": msg,
                                     "site": "%s:%d" % (base, fails[0] + 1)})


# ---- the catalog -------------------------------------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "with", "from", "into", "by", "at", "my", "i",
         "want", "make", "do", "how", "some", "this", "that", "it", "is", "be", "me", "can", "new", "add", "create",
         "help", "plan", "steps", "step", "roadmap", "need", "would", "like", "using", "use", "via", "through"}


def _stems(text):
    out = set()
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.add(w)
        for suf in ("ies", "es", "s", "ing", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                out.add(w[:-len(suf)] + ("y" if suf == "ies" else ""))
    return out


def load_catalog(path):
    """(goals dict, error or None)."""
    try:
        with io.open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return {}, "the roadmap catalog at %s is unreadable (%s)" % (path, exc)
    goals = doc.get("goals") if isinstance(doc, dict) else None
    if not isinstance(goals, dict) or not goals:
        return {}, "the roadmap catalog at %s holds no goals" % path
    return goals, None


def catalog_problems(goals, registered, check_params=None):
    """Every way the catalog can name something that is not there - for tools/test_mif_roadmap.py."""
    out = []
    for gid, g in goals.items():
        for key in ("title", "words", "steps"):
            if not g.get(key):
                out.append("%s: no %s" % (gid, key))
        seen = set()
        for k, s in enumerate(g.get("steps") or []):
            where = "%s step %d (%s)" % (gid, k + 1, s.get("id"))
            if not s.get("id") or s["id"] in seen:
                out.append("%s: missing or repeated id" % where)
            seen.add(s.get("id"))
            if not s.get("say"):
                out.append("%s: no say" % where)
            tools = s.get("tools") or []
            if not tools and not s.get("manual"):
                out.append("%s: names no tool and is not manual - a step must be grounded in a call or say why not"
                           % where)
            for t in tools:
                if t not in registered:
                    out.append("%s: %s is not a registered tool" % (where, t))
            man = s.get("manual")
            if man is not None and not (isinstance(man, dict) and man.get("why") and man.get("instead")):
                out.append("%s: manual needs both why and instead" % where)
            chk = s.get("check")
            if chk is not None:
                if chk.get("endpoint") not in CHECK_ENDPOINTS:
                    out.append("%s: check endpoint %r is not on CHECK_ENDPOINTS" % (where, chk.get("endpoint")))
                if not chk.get("count"):
                    out.append("%s: check names no count field" % where)
                if check_params is not None and chk.get("endpoint") in check_params:
                    allowed = check_params[chk["endpoint"]]
                    for p in (chk.get("params") or {}):
                        if allowed is not None and p not in allowed:
                            out.append("%s: check param %r is not accepted by %s" % (where, p, chk["endpoint"]))
    return out


def match_goal(goal, goals):
    """[(score, goal id)] best first. A goal word in the catalog's `words` counts 2, in its title 1."""
    want = _stems(goal)
    if not want:
        return []
    scored = []
    for gid, g in goals.items():
        words = set()
        for w in g.get("words") or []:
            words |= _stems(w)
        title = _stems(g.get("title")) | _stems(gid.replace("_", " "))
        score = 2 * len(want & words) + len((want & title) - words)
        if score:
            scored.append((score, gid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def goal_matched(scored, goal):
    """The matched goal id or None. One shared word is not a match unless the goal is a word or two."""
    if not scored:
        return None
    top = scored[0][0]
    if top >= 4 or (top >= 2 and len(_stems(goal)) <= 2):
        if len(scored) > 1 and scored[1][0] == top:
            return None          # a tie is a question for the assistant, not a coin toss
        return scored[0][1]
    return None


# ---- grounding ---------------------------------------------------------------------------------------------------------
def _is_version_refusal(conds):
    """A refusal OF AN ENGINE VERSION sits on the side where the guarded code is ABSENT - the #else of AT_LEAST, the
    #if of BEFORE - with any plugin atom present. A Fail on the side where the code EXISTS is an input check inside the
    newer branch ("the landscape has no edit layer at index 3"), not a statement about the version."""
    eng = [(k, p) for kind, k, p in conds if kind == "engine"]
    if not eng:
        return False
    if any(kind == "with" and not p for kind, _k, p in conds):
        return False
    return all((d == "AT_LEAST" and not p) or (d == "BEFORE" and p) for (d, _b), p in eng)


def _conds_text(conds):
    return " and ".join("%s%s" % ("" if p else "!", ("MIF_ENGINE_%s(%s)" % (k[0], k[1].replace(".", ", ")))
                                  if kind == "engine" else k) for kind, k, p in conds)


def _engine_conds_hold(conds, engine):
    """(all engine atoms hold at engine, there was at least one)."""
    eng = [(k, p) for kind, k, p in conds if kind == "engine"]
    if not eng:
        return False, False
    return all(holds_at(d, b, engine) == p for (d, b), p in eng), True


def ground_tool(tool, facts):
    """One tool's derived facts and problems. `facts` is the dict mif_roadmap assembles (docs/30 section 3)."""
    reg = facts.get("registered") or set()
    out = {"tool": tool}
    problems = []
    if tool not in reg:
        near = difflib.get_close_matches(tool, sorted(reg), n=3, cutoff=0.75)
        near += [t for t in sorted(reg) if tool and len(tool) > 3 and tool in t and t not in near][:3 - len(near)]
        out["registered"] = False
        if near:
            out["didYouMean"] = near
        problems.append(("blocked", "no MifBridge tool is named %s" % tool))
        out["problems"] = [{"level": lv, "why": w} for lv, w in problems]
        return out
    out["registered"] = True
    needs = (facts.get("needs") or {}).get(tool) or {}
    eps = sorted(needs.get("endpoints") or ())
    ops = sorted(needs.get("ops") or ())
    if eps:
        out["endpoints"] = eps
    if ops:
        out["blenderOps"] = ops
    pack = (facts.get("pack") or {}).get(tool)
    if pack:
        out["pack"] = pack
    rules = (facts.get("rules") or {}).get(tool)
    if rules:
        out["facts"] = rules
    engine = facts.get("engine")
    src = facts.get("source")
    plugin_map = facts.get("pluginMap") or {}
    forced_off = facts.get("pluginsOff") or {}
    live = facts.get("live") or {}
    plugins_live = live.get("plugins")         # {plugin name lower: enabled} or None
    differs, refuses, plug_rows, cooked_sites, crash = [], [], [], [], []
    if src is not None:
        for ep in eps:
            rec = src.get(ep)
            if not rec:
                continue
            for d, b, site in rec["guards"]:
                row = {"endpoint": ep, "guard": "MIF_ENGINE_%s(%s)" % (d, b.replace(".", ", ")), "site": site}
                if engine:
                    row["thisEngineGetsGuardedCode"] = holds_at(d, b, engine)
                if not engine or applies_at(d, b, engine):
                    differs.append(row)
            for r in rec["refusals"]:
                if not _is_version_refusal(r["conds"]):
                    continue
                hold, _has = _engine_conds_hold(r["conds"], engine) if engine else (None, True)
                entry = {"endpoint": ep, "refuses": r["message"], "scope": r["scope"], "site": r["site"],
                         "under": _conds_text(r["conds"])}
                if engine is None:
                    refuses.append(dict(entry, appliesHere=None))
                elif hold:
                    refuses.append(dict(entry, appliesHere=True))
                    if r["scope"] == "endpoint":
                        problems.append(("blocked", "%s refuses every call on UE %s: \"%s\" (%s)"
                                         % (ep, engine, r["message"][:220], r["site"])))
                    else:
                        problems.append(("caution", "on UE %s %s refuses some calls - read whether yours is one: "
                                         "\"%s\" (%s)" % (engine, ep, r["message"][:220], r["site"])))
            for macro, how in rec["plugins"].items():
                plugin = plugin_map.get(macro)
                row = {"endpoint": ep, "macro": macro, "scope": "whole" if how == "whole" else "partial"}
                if plugin:
                    row["plugin"] = plugin
                off_refusal = [r for r in rec["refusals"]
                               if any(kind == "with" and k == macro and not p for kind, k, p in r["conds"])]
                if off_refusal:
                    row["refusesWithout"] = off_refusal[0]["message"]
                    if off_refusal[0]["scope"] == "endpoint":
                        row["scope"] = "whole"
                whole = row["scope"] == "whole"
                if macro in forced_off:
                    row["builtOff"] = forced_off[macro]
                    problems.append(("blocked" if whole else "caution",
                                     "%s: every MifBridge build compiles %s off (%s), so %s answers a named refusal%s"
                                     % (ep, macro, forced_off[macro], "it" if whole else "the part that needs it",
                                        (": \"%s\"" % row["refusesWithout"][:200]) if row.get("refusesWithout") else "")))
                elif plugin and plugins_live is not None:
                    on = plugins_live.get(plugin.lower())
                    row["enabledHere"] = on
                    if not on:
                        problems.append(("blocked" if whole else "caution",
                                         "%s needs the %s plugin (%s) %s, and it is %s in this editor - "
                                         "set_plugin_enabled turns it on once the user agrees, then the editor restarts"
                                         % (ep, plugin, macro, "for every call" if whole else "for part of what it does",
                                            "not enabled" if on is False else "not installed")))
                plug_rows.append(row)
            if rec["cooked"]:
                cooked_sites.append({"endpoint": ep, "sites": rec["cooked"][:3]})
            crash += [dict(c, endpoint=ep) for c in rec["crash"]]
    if differs:
        # INFORMATION, NOT A CAUTION. Most guards are the same behaviour through a renamed engine API
        # (GetSectionBase / GetSectionBaseOffset); a guard that changes the ANSWER on this engine is a refusal arm, and
        # those are judged above. Promoting every branch to a caution would make cautions noise on older engines.
        out["engineBranches"] = {"count": len(differs), "first": differs[:3]}
    if refuses:
        out["engineRefusals"] = refuses[:6]
    if plug_rows:
        out["plugins"] = plug_rows
    if cooked_sites:
        out["cookedHandling"] = cooked_sites
        if (live.get("cookedCount") or 0) > 0:
            problems.append(("caution", "checks for cooked targets, and this project holds cooked content - on a "
                             "cooked asset it may refuse or read less; work on an editable copy or new asset"))
    if crash:
        out["crashGuards"] = crash[:4]
    # ---- live: registry, write gate, undo, generative ----------------------------------------------------------------
    unavailable = (live.get("unavailable") or {}).get(tool)
    if unavailable:
        problems.append(("blocked", unavailable))
    ue_eps = live.get("endpoints")
    if ue_eps is not None:
        missing = [e for e in eps if e not in ue_eps]
        if missing:
            # NOT "older" alone: an external provider's endpoints (kr_*) are absent when its plugin is not installed,
            # however new the bridge is - measured on a probe with no reconstructor plugin, 2026-10-19
            problems.append(("blocked", "the running editor registers no endpoint %s - its MifBridge is older than "
                             "this server, or the plugin that provides it is not installed there" % ", ".join(missing)))
        gated = [e for e in eps if (ue_eps.get(e) or {}).get("writeGated")]
        buckets = sorted({(ue_eps.get(e) or {}).get("bucket") for e in eps if e in ue_eps} - {None})
        if buckets:
            out["undo"] = ("Ctrl-Z reverts it" if buckets == ["transacted"] else
                           "not a transaction - Ctrl-Z does not revert it" if "transacted" not in buckets else
                           "partly transacted: %s" % ", ".join(buckets))
        if gated:
            out["writeGated"] = gated
            mode = live.get("writeMode")
            if mode and mode != "full":
                problems.append(("blocked", "%s is write-gated and the bridge is in %s mode - the USER raises the "
                                 "write mode (MIF_BRIDGE_WRITE_MODE=full); never route around it"
                                 % (", ".join(gated), mode)))
    if any(e in PREVIEW_ENDPOINTS for e in eps) and (live.get("changeCap") or {}).get("on"):
        out["preview"] = ("the change cap is %s: a call changing more than that answers a preview to commit"
                          % (live.get("changeCap") or {}).get("cap"))
    if tool.startswith("bl_gen_") and live.get("generative") not in (None, "generative"):
        problems.append(("blocked", "a generative tool, and this project is %s - only the user changes that "
                         "(mif_project_mode)" % live.get("generative")))
    # ---- dev-only evidence and ledger --------------------------------------------------------------------------------
    ledger = facts.get("ledger")
    if ledger is not None:
        rows = []
        for n in [tool] + eps + ops:
            rows += ledger.get(n, [])
        if rows:
            seen, kept = set(), []
            for rid, eng in rows:
                if rid not in seen:
                    seen.add(rid)
                    kept.append((rid, eng))
            here = [rid for rid, eng in kept if engine and _engine_cell_has(eng, engine)]
            out["ledger"] = {"rows": [rid for rid, _ in kept][:6], "count": len(kept)}
            if here:
                out["ledger"]["onThisEngine"] = here[:6]
    ev = facts.get("evidence")
    if ev is not None and engine and eps:
        verdicts = {}
        for e in eps:
            v = ev(e, engine)
            if v:
                verdicts[e] = v
        if verdicts:
            # the state per endpoint; the suites only where they are the news (a failure, or nothing ran here)
            out["evidence"] = {e: (v["state"] if v.get("state") == "passes" else v) for e, v in verdicts.items()}
            fails = [e for e, v in verdicts.items() if v.get("state") == "fails"]
            if fails:
                problems.append(("caution", "a suite FAILS on UE %s for %s" % (engine, ", ".join(fails))))
            mixed = [e for e, v in verdicts.items() if v.get("state") == "mixed"]
            if mixed:
                problems.append(("note", _MIXED_NOTE % (", ".join(mixed), "UE %s" % engine)))
    if ops and facts.get("blender"):
        out.update(ground_blender_ops(ops, facts, problems))
    out["problems"] = [{"level": lv, "why": w} for lv, w in problems]
    return out


# A note, not a caution: it leaves the step's status alone. See ue_capability_query.evidence_row.
_MIXED_NOTE = ("suites calling %s pass on %s and at least one fails: a suite's verdict is the whole suite's, so the failing "
               "row is often another tool's")


def _engine_cell_has(cell, engine):
    """A ledger engine cell - one version, a range '5.3-5.8', a list, or a dash - covers `engine`?"""
    cell = (cell or "").replace("–", "-").replace(" ", "")
    if not cell or cell in ("-", "—"):
        return False
    for part in re.split(r"[,/;]", cell):
        m = re.match(r"^(\d+\.\d+)(?:\.\d+)?(?:-(\d+\.\d+)(?:\.\d+)?|\+)?$", part)
        if not m:
            continue
        lo = vkey(m.group(1))
        hi = vkey(m.group(2)) if m.group(2) else ((99, 99) if part.endswith("+") else lo)
        if lo <= vkey(engine) <= hi:
            return True
    return False


_ORDER = {"blocked": 0, "unknown": 1, "caution": 2, "ready": 3}


def ground_step(step, facts):
    """A step's status and its tools' facts. Status is one of done, ready, caution, blocked, manual, unknown."""
    out = {"id": step.get("id"), "say": step.get("say")}
    tools = [ground_tool(t, facts) for t in (step.get("tools") or [])]
    if tools:
        out["tools"] = tools
    check = (facts.get("checks") or {}).get(step.get("id"))
    if step.get("manual"):
        # DONE BY HAND IS THE ONE WAY A MANUAL STEP ENDS: nothing can check it, and without this nextStep stops on it for good.
        if check is not None and check.get("byHand"):
            out.update(status="done", check=check)
            return out
        out["status"] = "manual"
        out["cannot"] = dict(step["manual"])
        return out
    if step.get("note"):
        out["note"] = step["note"]
    live = facts.get("live")
    if check is not None:
        out["check"] = check
        if check.get("done") is True:
            out["status"] = "done"
            return out
    probs = [p for t in tools for p in t.get("problems") or []]
    if any(p["level"] == "blocked" for p in probs):
        status = "blocked"
    elif check is not None and check.get("done") is None:
        status = "unknown"
    elif any(p["level"] == "caution" for p in probs):
        status = "caution"
    elif live is None:
        status = "unknown"           # nothing stops it in the source - and nothing live was read to say it runs here
    else:
        status = "ready"
    out["status"] = status
    if probs:
        out["why"] = [p["why"] for p in probs][:8]
    return out


def ground_plan(steps, facts):
    """(grounded steps, summary). An empty plan or one whose every tool is unknown is the caller's failure to report."""
    grounded = [ground_step(s, facts) for s in steps]
    counts = {}
    for g in grounded:
        counts[g["status"]] = counts.get(g["status"], 0) + 1
    nxt = next((g["id"] for g in grounded if g["status"] not in ("done",)), None)
    return grounded, {"counts": counts, "nextStep": nxt}


def normalize_steps(steps):
    """A caller's plan as step dicts: bare tool names and {say, tools} both accepted. (steps, error)."""
    if not isinstance(steps, list) or not steps:
        return None, "steps must be a non-empty list of {say, tools:[...]} or of tool names"
    out = []
    for k, s in enumerate(steps, 1):
        if isinstance(s, str):
            out.append({"id": "step%d" % k, "say": s, "tools": [s.strip()]})
        elif isinstance(s, dict):
            tools = s.get("tools")
            if isinstance(tools, str):
                tools = [tools]
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                return None, "step %d: tools must be a list of tool names" % k
            if not tools and not s.get("manual"):
                return None, "step %d names no tool - a grounded step is made of calls" % k
            out.append({"id": str(s.get("id") or "step%d" % k), "say": str(s.get("say") or ""),
                        "tools": [t.strip() for t in tools], **({"manual": s["manual"]} if s.get("manual") else {})})
        else:
            return None, "step %d is neither a tool name nor an object" % k
    return out, None


# WHAT A PROJECT DOES REPEATEDLY, from the call journals (Unreal's Saved/MifBridge/journal.jsonl, Blender's per-user file). Both record the call's
# name and never its arguments' meaning here: only ep, ok and the process it ran in are read, so nothing a person typed reaches a reply.
_NOT_A_STEP = re.compile(r"^(list_|get_|describe_|find_|read_|self_audit$|pie_status$|project_paths$|scene_info$|status$|call_log$|object_info$|"
                         r"list$|mif_|inspect_|sample_|check_|validate$|look$)")


def journal_sessions(lines):
    """The calls that ended ok in a journal's lines, in order, one list per process that made them. A torn or foreign line is skipped."""
    out = {}
    for line in lines:
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict) and rec.get("ev") == "end" and rec.get("ok") is True and isinstance(rec.get("ep"), str):
            out.setdefault(rec.get("pid"), []).append(rec["ep"])
    return list(out.values())


def learn_workflows(sessions, min_count=3, min_len=2, max_len=6, top=5):
    """The longest runs of writes that recur: (rows, writes read). A row is {tools, count, sessions}. A run counts once per place it starts and
    never overlaps itself, a call repeated in a row counts as one, and a run inside a longer one that recurs as often is left out."""
    runs = []
    writes = 0
    for sess in sessions:
        run = []
        for ep in sess:
            if _NOT_A_STEP.match(ep):
                continue
            writes += 1
            if not run or run[-1] != ep:
                run.append(ep)
        runs.append(run)
    seen = {}
    for idx, run in enumerate(runs):
        for n in range(min_len, max_len + 1):
            at = 0
            while at + n <= len(run):
                key = tuple(run[at:at + n])
                if len(set(key)) < 2:
                    at += 1
                    continue
                row = seen.setdefault(key, {"count": 0, "sessions": set(), "end": {}})
                if at >= row["end"].get(idx, 0):    # not overlapping its own last match in this run
                    row["count"] += 1
                    row["sessions"].add(idx)
                    row["end"][idx] = at + n
                at += 1
    rows = [(k, v) for k, v in seen.items() if v["count"] >= min_count]
    rows.sort(key=lambda kv: (-len(kv[0]) * kv[1]["count"], -len(kv[0]), kv[0]))
    picked = []
    for key, v in rows:
        joined = "\0" + "\0".join(key) + "\0"
        if any(joined in "\0" + "\0".join(p["tools"]) + "\0" and v["count"] <= p["count"] for p in picked):
            continue
        picked.append({"tools": list(key), "count": v["count"], "sessions": len(v["sessions"])})
        if len(picked) >= top:
            break
    return picked, writes


def read_count(result, path):
    """The number a check reads: a list's length or a number at a dotted path, or None."""
    cur = result
    for part in (path or "").split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, list):
        return len(cur)
    if isinstance(cur, bool):
        return int(cur)
    if isinstance(cur, (int, float)):
        return cur
    return None


def ledger_index(rows_iter, names):
    """{name: [(row id, engine cell)]} for the rows that name one of `names` - rows from split_ledger_by_pack.rows()."""
    names = set(names)
    word = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
    out = {}
    for rid, _section, body in rows_iter:
        if rid.endswith("-superseded"):
            continue
        hit = set(word.findall(body)) & names
        if not hit:
            continue
        cells = [c.strip() for c in body.strip().strip("|").split("|")] if body.startswith("| ") else []
        eng = cells[-1] if len(cells) >= 3 else ""
        for n in hit:
            out.setdefault(n, []).append((rid, eng))
    return out


# ---- the Blender half: version gates per op (1.3 wave 3) -------------------------------------------------------------
# WHAT IS MEASURED, and why it is three sources and not one. The add-on mostly finds a version difference AT RUN TIME
# (an enum's valid items, a hasattr, a refusal that names bpy.app.version_string) rather than in a static comparison, so
# a source scan alone sees little. So an op's Blender facts are:
#   gates       every `raise MifOpError(...)` in the op's own function that sits under an `if` comparing bpy.app.version
#               (judged against the Blender version: appliesHere), or whose message names the running build's version
#               (a run-time discovery: appliesHere None - "it refuses on some builds, and says so by name");
#   drift       tools/blender_version_matrix_values.json: the (op, result field) pairs the version matrix accepted as
#               differing between installed builds while the op SUCCEEDED on every one - the op works everywhere and
#               these fields read differently by version. It records no versions, so it is information, never a gate;
#   evidence    the per-Blender-version suite verdicts (the verification board, or the evidence snapshot beside server.py).
# LIMIT, said rather than hidden: a refusal made in a helper the op calls is not seen, as on the C++ side.
_BL_VERSION_TEXT = re.compile(r"version_string|bpy\.app\.version|Blender \d+\.\d")


def _ast_mentions_version(node):
    import ast
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in ("version", "version_string"):
            v = n.value
            if isinstance(v, ast.Attribute) and v.attr == "app":
                return True
    return False


def _first_text(call):
    """The first literal text in a MifOpError(...) call - its message's opening words."""
    import ast
    for n in ast.walk(call):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.strip():
            return n.value.strip()[:220]
    return ""


def _version_constants(tree):
    """{NAME: its value} for the module's constant tuples of whole numbers: a condition reading `_GP_EXPORT_CRASHES[0]`
    names a version range kept in one place, and a judge that cannot read it cautions on every build."""
    import ast
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                value = ast.literal_eval(n.value)
            except (ValueError, SyntaxError):
                continue
            flat = [x for v in value for x in (v if isinstance(v, tuple) else (v,))] if isinstance(value, tuple) else []
            if flat and all(type(x) is int for x in flat):
                out[n.targets[0].id] = value
    return out


def _with_constants(text, constants):
    """A condition's source with NAME[i] and NAME replaced by the module constant's value, so the snapshot's copy and
    the judge read it with no add-on source beside them."""
    for name, value in constants.items():
        if isinstance(value[0], tuple):
            text = re.sub(r"\b%s\[(\d+)\]" % re.escape(name),
                          lambda m, v=value: repr(v[int(m.group(1))]) if int(m.group(1)) < len(v) else m.group(0), text)
        text = re.sub(r"(?<![\w.])%s\b(?!\[)" % re.escape(name), repr(value), text)
    return text


def scan_blender_addon(addon_dir):
    """{op: [{"under": [(condition source, negated)], "message", "site"}]} for every version-shaped refusal in an op's
    own function. None when there is no add-on folder (a store package ships the add-on separately)."""
    import ast
    if not addon_dir or not os.path.isdir(addon_dir):
        return None
    out = {}
    for path in sorted(glob.glob(os.path.join(addon_dir, "ops_*.py"))):
        try:
            src = io.open(path, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError):
            continue
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        constants = _version_constants(tree)
        table = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPS" for t in n.targets) \
                    and isinstance(n.value, ast.Dict):
                for k, v in zip(n.value.keys, n.value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, ast.Name):
                        table[k.value] = v.id
        base = os.path.basename(path)
        for op, fname in table.items():
            fn = funcs.get(fname)
            if fn is None:
                continue
            rows = []

            def visit(node, conds):
                for field, value in ast.iter_fields(node):
                    kids = value if isinstance(value, list) else [value]
                    for kid in kids:
                        if not isinstance(kid, ast.AST) or isinstance(kid, (ast.FunctionDef, ast.Lambda)):
                            continue
                        here = conds
                        if isinstance(node, ast.If) and field in ("body", "orelse") and _ast_mentions_version(node.test):
                            here = conds + [(_with_constants(ast.get_source_segment(src, node.test) or "", constants),
                                             field == "orelse")]
                        if isinstance(kid, ast.Raise) and isinstance(kid.exc, ast.Call) and \
                                getattr(kid.exc.func, "id", getattr(kid.exc.func, "attr", "")) == "MifOpError":
                            text = ast.get_source_segment(src, kid.exc) or ""
                            if here or _BL_VERSION_TEXT.search(text):
                                rows.append({"under": [list(c) for c in here], "message": _first_text(kid.exc),
                                             "site": "%s:%d" % (base, kid.lineno)})
                        visit(kid, here)
            visit(fn, [])
            if rows:
                out[op] = rows
    return out


def _bl_version_tuple(version):
    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version or ""))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def eval_bl_condition(text, version):
    """True / False for a condition over bpy.app.version at `version` ("4.2" or "4.2.3"), or None when its answer
    depends on something else (a hasattr, the call's own arguments) - never a guess. Three-valued: `x and version <
    (5, 0)` is False on 5.0 whatever x is, and None (it depends on the call) on 4.2."""
    import ast
    import operator
    v = _bl_version_tuple(version)
    if v is None:
        return None
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    ops = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.Eq: operator.eq, ast.NotEq: operator.ne}

    def ver_of(n):
        if isinstance(n, ast.Attribute) and n.attr == "version" and isinstance(n.value, ast.Attribute) \
                and n.value.attr == "app":
            return v
        if isinstance(n, ast.Subscript) and ver_of(n.value) is not None and isinstance(n.slice, ast.Slice) \
                and n.slice.lower is None and isinstance(n.slice.upper, ast.Constant):
            return v[:n.slice.upper.value]
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "tuple" and len(n.args) == 1:
            return ver_of(n.args[0])
        return None

    def lit_of(n):
        if isinstance(n, ast.Tuple) and all(isinstance(e, ast.Constant) and type(e.value) is int for e in n.elts):
            return tuple(e.value for e in n.elts)
        return None

    def ev(n):
        if isinstance(n, ast.BoolOp):
            vals = [ev(x) for x in n.values]
            if isinstance(n.op, ast.And):
                return False if False in vals else (None if None in vals else True)
            return True if True in vals else (None if None in vals else False)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            x = ev(n.operand)
            return None if x is None else not x
        if isinstance(n, ast.Compare) and all(type(o) in ops for o in n.ops):
            # a chain reads link by link, as Python does: `(4, 3, 0) <= bpy.app.version < (5, 0, 0)`
            terms = [n.left] + list(n.comparators)
            got = []
            for op, a, b in zip(n.ops, terms, terms[1:]):
                left, right = ver_of(a), lit_of(b)
                flip = left is None
                if flip:
                    left, right = ver_of(b), lit_of(a)
                if left is None or right is None:
                    got.append(None)
                    continue
                left = left[:len(right)] if len(left) > len(right) else left
                got.append(ops[type(op)](right, left) if flip else ops[type(op)](left, right))
            return False if False in got else (None if None in got else True)
        return None
    return ev(node)


def gate_applies(row, version):
    """Whether one gate row's conditions all hold at `version`: True, False, or None (a run-time discovery, or a
    condition this cannot read)."""
    if not row.get("under"):
        return None
    got = []
    for text, negated in row["under"]:
        r = eval_bl_condition(text, version)
        got.append(r if r is None or not negated else not r)
    return False if False in got else (None if None in got else True)


def load_value_drift(path):
    """{op: [fields]} from blender_version_matrix_values.json, or None when the file is absent or unreadable."""
    try:
        with io.open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return None
    out = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, list) and len(row) == 2 and all(isinstance(x, str) for x in row):
            out.setdefault(row[0], []).append(row[1])
    return out


def ground_blender_ops(ops, facts, problems):
    """The Blender-version facts of a tool's ops, into a dict for the tool's row; problems appended in place."""
    bl = facts.get("blender") or {}
    version = bl.get("version")
    out = {}
    gates = bl.get("gates")
    if gates is not None:
        rows = []
        for op in ops:
            for g in gates.get(op) or []:
                applies = gate_applies(g, version) if version else None
                row = {"op": op, "refuses": g["message"], "site": g["site"],
                       "under": " and ".join(("not (%s)" % t) if neg else t for t, neg in g["under"]) or
                                "a check made at run time - the refusal names the running build",
                       "appliesHere": applies}
                if applies is False:
                    continue                     # judged against this build: it does not apply here
                rows.append(row)
                # a static version arm that this build does not rule out: the refusal can meet a call HERE (None:
                # whether it does depends on the call - a name's length, an action). A run-time discovery is listed
                # only: it names the build when it refuses, and the evidence below says how the suites fared.
                if version and g["under"]:
                    problems.append(("caution", "on Blender %s %s refuses some calls - read whether yours is one: "
                                     "\"%s\" (%s)" % (version, op, g["message"][:200], g["site"])))
        if rows:
            out["blenderVersionGates"] = rows[:6]
    drift = bl.get("drift")
    if drift:
        moved = {op: drift[op][:6] for op in ops if op in drift}
        if moved:
            out["blenderVersionDiffers"] = {"fields": moved,
                                            "note": "the version matrix ran these ops on every installed Blender: they "
                                                    "succeed on each, and these result fields read differently by "
                                                    "version"}
    ev = bl.get("evidence")
    if ev is not None and version:
        verdicts = {}
        for op in ops:
            got = ev(op, version)
            if got:
                verdicts[op] = got
        if verdicts:
            out["blenderEvidence"] = {op: (v["state"] if v.get("state") == "passes" else v)
                                      for op, v in verdicts.items()}
            fails = [op for op, v in verdicts.items() if v.get("state") == "fails"]
            if fails:
                problems.append(("caution", "a suite FAILS on Blender %s for %s" % (version, ", ".join(fails))))
            mixed = [op for op, v in verdicts.items() if v.get("state") == "mixed"]
            if mixed:
                problems.append(("note", _MIXED_NOTE % (", ".join(mixed), "Blender %s" % version)))
    return out


# ---- the evidence snapshot: per-engine suite verdicts carried beside server.py (1.3 wave 3) --------------------------
SNAPSHOT_NAME = "evidence_snapshot.json"
SNAPSHOT_CODES = {"p": "passes", "f": "fails", "m": "mixed", "s": "skips", "n": "neverhere"}


def snapshot_evidence(snap, backend):
    """evidence(name, version) over a snapshot's rows for one backend ('ue' or 'blender'), or None. The same answer
    shape as the board's: {"state"} - suites are not carried, only the verdict."""
    engines = ((snap or {}).get("engines") or {}).get(backend)
    rows = (snap or {}).get(backend)
    if not isinstance(engines, list) or not isinstance(rows, dict):
        return None
    index = {str(e): i for i, e in enumerate(engines)}

    def evidence(name, version):
        cells = rows.get(name)
        m = re.match(r"^(\d+\.\d+)", str(version or ""))
        if not isinstance(cells, str) or not m or m.group(1) not in index:
            return None
        code = cells[index[m.group(1)]] if index[m.group(1)] < len(cells) else "-"
        state = SNAPSHOT_CODES.get(code)
        return {"state": state, "source": "snapshot"} if state else None
    return evidence
