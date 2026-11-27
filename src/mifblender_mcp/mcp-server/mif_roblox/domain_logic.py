"""Roblox game-logic checks: what a game's saves, remotes, odds and config hold, measured in a live Studio.

Lessons in the internal notes (saves, receipts and duplication; server-authoritative actions; monetization; balance and testing)
turned into tools:
- rbx_play_probe's fault_store, census and self_test (play_with_logic below): a DataStore stand-in that fails, stalls or
  holds a foreign lock on cue, a census of every player's items before and after, and the scaffold templates' self-tests.
- rbx_remote_fuzz: every remote fired from a client with forged arguments, the server's state compared around each.
- rbx_module_values and rbx_odds_check: a ModuleScript's returned table, and a roll function sampled against the odds
  the game shows, both through the plugin's module_require op (edit mode, a fresh clone, nothing runs a server).
- rbx_script_scaffold's kinds save, action, prompt_once, economy and analytics (templates()).
- rbx_save_audit and rbx_analytics_audit: static checks of a game's scripts (save_audit.py, analytics_audit.py),
  from the open Studio or, with no Studio, from a saved place or a folder.
- rbx_place_manifest: one canonical hash of a saved place file and the diff of two (place_manifest.py), read with
  no Studio.

Everything a play test needs is written under Workspace.MifScratch (allowed in scratch mode) and removed after, even
when the run fails. The Luau is fixed text in logic_luau.py; a caller's settings reach it only as attribute JSON.
NEVER RAISE.
"""
import functools
import json
import math
import os
import re
from typing import Any

from mif_roblox import logic_luau as LU
from mif_roblox import values as V

__all__ = []

SCRATCH = "Workspace.MifScratch"
SCRATCH_PLACE = "MifScratch.rbxlx"    # the plugin's own constant: rbx_launch's scratch place, all of it scratch
STAND_IN = "DataStoreStandIn"
RUNNER = "MifLogicRunner"
FUZZ_FOLDER = "MifRemoteFuzz"
MARK = "MIFLOGIC"
FUZZ_MARK = "MIFFUZZ"
TEST_MARK = "MIFSELFTEST"
EVENT_OPS = ("UpdateAsync", "SetAsync", "GetAsync", "RemoveAsync", "IncrementAsync", "*")
EVENT_DOS = ("fail", "stall", "lock", "unlock")
EVENT_KEYS = {"op", "store", "key", "nth", "count", "after", "before", "do", "seconds", "then", "field", "session",
              "message"}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")


def _T():
    from mif_roblox import tools as T
    return T


def _tool(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            out = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - never raise
            out = {"ok": False, "error": "%s failed inside the bridge: %s: %s" % (fn.__name__, type(exc).__name__, exc)}
        _T()._post_verdict(fn.__name__, out)
        return out
    __all__.append(fn.__name__)
    return wrapper


def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


# ---------------------------------------------------------------- reading what the scripts printed
_LINE = re.compile(r"^(MIF[A-Z]+) (\S+) (\d+)/(\d+) (.*)$", re.S)


def _messages(r):
    """Every Output line of a probe result, server first, then each client's, in order; the recovered-from-log forms too."""
    out = []

    def take(seq):
        for m in seq or []:
            s = m.get("message") if isinstance(m, dict) else m
            if isinstance(s, str):
                out.append(s)
    take(r.get("logs"))
    for c in r.get("clients") or []:
        if isinstance(c, dict):
            take(c.get("logs") or c.get("output"))
    return out


_ESCAPED = re.compile(rb"~([0-9a-f]{2})")


def unescape(text):
    """The JSON text of logic_luau.EMIT_FN's ~XX form: every ~ and two hex digits back to its byte, then UTF-8."""
    return _ESCAPED.sub(lambda m: bytes([int(m.group(1), 16)]), text.encode("latin-1")).decode("utf-8")


def marked(r, mark):
    """{tag: [decoded value, ...]} from lines 'MARK tag i/n payload', each value's parts joined in order. A part that
    starts with ~ is logic_luau.EMIT_FN's escaped form (that ~ is a flag, not data): the parts are joined and then
    unescaped, so a cut inside a ~XX or a UTF-8 character is whole again. A raw part (an older script's) never starts
    with ~ and is joined as it came."""
    found, open_ = {}, {}
    for line in _messages(r):
        m = _LINE.match(line.strip("\r\n"))
        if not m or m.group(1) != mark:
            continue
        tag, i, n, part = m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
        if i == 1:
            open_[tag] = []
        if tag not in open_:
            continue
        open_[tag].append(part)
        if i == n:
            parts = open_.pop(tag)
            escaped = parts[0].startswith("~")
            text = "".join(p[1:] if escaped and p.startswith("~") else p for p in parts)
            try:
                found.setdefault(tag, []).append(json.loads(unescape(text) if escaped else text))
            except (ValueError, UnicodeError) as exc:
                found.setdefault(tag, []).append({"parseError": str(exc), "text": text[:300]})
    return found


# ---------------------------------------------------------------- staging scripts under Workspace.MifScratch
def _ensure_scratch():
    T = _T()
    if T._rbx("resolve", target=V.split_path(SCRATCH)).get("ok"):
        return None
    mk = T.rbx_create("Folder", "Workspace", name="MifScratch")
    return None if mk.get("ok") else "no %s and it could not be made: %s" % (SCRATCH, mk.get("error"))


def _stage(items):
    """items: [(class, path, props, source or None, attributes)]. Made in order; (paths made, error)."""
    T = _T()
    made = []
    for cls, path, props, source, attrs in items:
        parent, name = path.rsplit(".", 1)
        if T._rbx("resolve", target=V.split_path(path)).get("ok"):
            return made, "%s is already there: an earlier run left it, or it is yours; nothing was run" % path
        mk = T.rbx_create(cls, parent, name=name, props=props or None)
        if not mk.get("ok"):
            return made, "%s could not be made: %s" % (path, mk.get("error"))
        made.append(path)
        if source is not None:
            w = T.rbx_script_write(path, source)
            if not w.get("ok"):
                return made, "%s's source was not written: %s" % (path, w.get("error"))
        if attrs:
            a = T.rbx_attributes(path, set=attrs)
            if not a.get("ok"):
                return made, "%s's settings were not written: %s" % (path, a.get("error"))
    return made, None


def _unstage(made):
    T = _T()
    gone = []
    for path in reversed(made):
        d = T.rbx_destroy(path)
        if d.get("ok"):
            gone.append(path)
    out = {"wrote": made, "removed": sorted(gone) == sorted(made)}
    if not out["removed"]:
        out["left"] = [p for p in made if p not in gone]
    return out


def _left_problem(staged):
    """The problem line for staged scripts that were not removed, or None. A probe script left in the place would be
    saved with it, so a run that could not take its scripts out is never ok."""
    if not isinstance(staged, dict) or staged.get("removed") is not False:
        return None
    return ("the staged play-test scripts were not removed: %s is still in the place; remove it with rbx_destroy "
            "before saving" % ", ".join(staged.get("left") or staged.get("wrote") or ["something"]))


# ---------------------------------------------------------------- the census
def _items(v):
    """(kind, {item: count}) for one owner's census value: a list is items (uids, each once), a dict of numbers is counts,
    any other dict is uids as keys."""
    if isinstance(v, list):
        c = {}
        for x in v:
            c[str(x)] = c.get(str(x), 0) + 1
        return "uids", c
    if isinstance(v, dict):
        if "error" in v and len(v) == 1:
            return "error", {}
        if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v.values()):
            return "counts", {str(k): x for k, x in v.items()}
        return "uids", {str(k): 1 for k in v}
    return "unknown", {}


def census_report(snaps):
    """snaps: the runner's census values in order (before, after). Each owner's items, what changed, and every uid held
    more than once (by two owners, or twice by one) in the last snapshot."""
    out = {"snapshots": [], "duplicates": [], "errors": []}
    last = None
    for s in snaps:
        if s.get("error"):
            out["errors"].append("%s: %s" % (s.get("at"), s["error"]))
        owners = {}
        for owner, v in (s.get("owners") or {}).items():
            kind, counts = _items(v)
            if kind == "error":
                out["errors"].append("%s %s: %s" % (s.get("at"), owner, v.get("error")))
            owners[owner] = {"kind": kind, "items": counts}
        out["snapshots"].append({"at": s.get("at"), "t": s.get("t"), "owners": owners})
        last = owners
    if len(out["snapshots"]) >= 2:
        a, b = out["snapshots"][0]["owners"], out["snapshots"][-1]["owners"]
        changes = {}
        for owner in sorted(set(a) | set(b)):
            ia, ib = (a.get(owner) or {}).get("items") or {}, (b.get(owner) or {}).get("items") or {}
            d = {k: ib.get(k, 0) - ia.get(k, 0) for k in set(ia) | set(ib) if ib.get(k, 0) != ia.get(k, 0)}
            if d:
                changes[owner] = d
        out["changes"] = changes
    if last:
        held = {}
        for owner, o in last.items():
            if o["kind"] != "uids":
                continue
            for item, n in o["items"].items():
                held.setdefault(item, []).extend([owner] * n)
        out["duplicates"] = [{"item": k, "heldBy": v} for k, v in sorted(held.items()) if len(v) > 1]
    return out


# ---------------------------------------------------------------- rbx_play_probe: fault_store, census, self_test
def _check_events(events):
    if not isinstance(events, list) or len(events) > 50:
        return "fault_store is a list of up to 50 events"
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            return "fault_store[%d] is an object" % i
        extra = set(ev) - EVENT_KEYS
        if extra:
            return "fault_store[%d] has unknown fields %s; the fields are %s" % (i, sorted(extra), sorted(EVENT_KEYS))
        if ev.get("do") not in EVENT_DOS:
            return "fault_store[%d].do is one of %s" % (i, ", ".join(EVENT_DOS))
        if ev.get("op", "*") not in EVENT_OPS:
            return "fault_store[%d].op is one of %s" % (i, ", ".join(EVENT_OPS))
        for k in ("store", "key", "field", "session", "message"):
            if k in ev and not isinstance(ev[k], str):
                return "fault_store[%d].%s is a string" % (i, k)
        for k, lo in (("nth", 1), ("count", 0)):
            if k in ev and (not isinstance(ev[k], int) or isinstance(ev[k], bool) or ev[k] < lo):
                return "fault_store[%d].%s is a whole number, at least %d" % (i, k, lo)
        for k in ("after", "before", "seconds"):
            if k in ev and (not isinstance(ev[k], (int, float)) or isinstance(ev[k], bool) or not 0 <= ev[k] <= 600):
                return "fault_store[%d].%s is 0-600 seconds" % (i, k)
        if ev.get("then", "fail") != "fail":
            return "fault_store[%d].then is \"fail\" (after a stall) or absent" % i
    return None


def _check_census(census):
    if not isinstance(census, dict) or set(census) - {"module", "function", "per"}:
        return None, "census is {module: a ModuleScript path, function: its name (default census), per: player or all}"
    parts = V.split_path(census.get("module"))
    if not parts:
        return None, "census.module is the path of a ModuleScript"
    fn = census.get("function", "census")
    if not isinstance(fn, str) or not _NAME.match(fn):
        return None, "census.function is a function name in that module"
    per = census.get("per", "player")
    if per not in ("player", "all"):
        return None, "census.per is player (fn(player) per player) or all (fn() once, {owner: items})"
    return {"module": parts, "fn": fn, "per": per}, None


def play_with_logic(probe_kwargs, fault_store=None, seed=None, census=None, self_test=None):
    """rbx_play_probe with the stand-in, the census and the self-tests staged under Workspace.MifScratch for one run."""
    T = _T()
    implicit = fault_store is None and self_test is not None   # the stand-in comes with a self-test; a template that
    if implicit:                                                # keeps no data (economy, action) never asks it
        fault_store = []
    if fault_store is not None:
        err = _check_events(fault_store)
        if err:
            return _no(err)
    if seed is not None and (not isinstance(seed, dict) or not all(isinstance(v, dict) for v in seed.values())):
        return _no("seed is {store name: {key: value}}: the stand-in's records at the start")
    cen = None
    if census is not None:
        cen, err = _check_census(census)
        if err:
            return _no(err)
    tests = []
    if self_test is not None:
        tests = [self_test] if isinstance(self_test, str) else self_test
        if not isinstance(tests, list) or not tests or not all(isinstance(t, str) and _NAME.match(t) for t in tests):
            return _no("self_test is a template name (or a list of them), as given to rbx_script_scaffold")
    run_mode = probe_kwargs.get("run_mode") is True
    if run_mode and cen and cen["per"] == "player":
        # Run mode starts no player: a census per player would wait for one that never joins and count nobody.
        return _no("run_mode starts no player, so a census per player has nobody to count: use census.per all, or a "
                   "multiplayer test (run_mode false, players 1-8)")
    seconds, settle = probe_kwargs.get("seconds", 2), probe_kwargs.get("settle", 0)
    if tests and seconds < 20:
        return _no("self-tests need seconds of 20 or more (the save template's take about 15)")
    err = _ensure_scratch()
    if err:
        return _no(err)
    before = settle + 2
    after = max(before + 1, settle + seconds - 2)
    items = []
    if fault_store is not None:
        items.append(("ModuleScript", "%s.%s" % (SCRATCH, STAND_IN), None, LU.STAND_IN,
                      {"Schedule": json.dumps(fault_store), "Seed": json.dumps(seed or {})}))
    items.append(("Script", "%s.%s" % (SCRATCH, RUNNER), {"RunContext": "Server"}, LU.RUNNER,
                  {"Mark": MARK, "Census": json.dumps(cen) if cen else "", "SelfTests": json.dumps(tests),
                   "Players": 0 if run_mode else (probe_kwargs.get("players") or 1), "Before": before, "After": after}))
    made, err = _stage(items)
    r = None
    try:
        if not err:
            r = T.rbx_play_probe(**probe_kwargs)
    finally:
        staged = _unstage(made)
    if err:
        left = _left_problem(staged)
        return _no(err + ("; " + left if left else ""), staged=staged)
    out = dict(r, staged=staged)
    got = marked(r, MARK)
    problems = []
    left = _left_problem(staged)
    if left:
        problems.append(left)
    if not r.get("ok"):
        problems.append("the play probe failed: %s" % r.get("error"))
    if "done" not in got:
        problems.append("the runner printed nothing back (its script did not run, or the session ended first)")
    if cen:
        rep = census_report(got.get("census") or [])
        out["census"] = rep
        if len(rep["snapshots"]) < 2:
            problems.append("the census took %d of 2 snapshots" % len(rep["snapshots"]))
        if rep["errors"]:
            problems.append("census errors: %s" % "; ".join(rep["errors"][:5]))
        if rep["snapshots"] and not any(s["owners"] for s in rep["snapshots"]):
            problems.append("the census counted nobody: an empty census is not a clean one")
        if rep["duplicates"]:
            problems.append("items held twice: %s" % ", ".join("%s by %s" % (d["item"], d["heldBy"]) for d in rep["duplicates"][:10]))
    if fault_store is not None:
        si = (got.get("standIn") or [None])[-1]
        out["standIn"] = si
        if not isinstance(si, dict) or si.get("error"):
            problems.append("the stand-in gave no report: %s" % ((si or {}).get("error") if isinstance(si, dict) else "none"))
        elif not sum((si.get("calls") or {}).values()) and implicit:
            out["note"] = "the place never asked the stand-in for a store (fine for a template that keeps no data)"
        elif not sum((si.get("calls") or {}).values()):
            problems.append("the place never asked the stand-in for a store: route its DataStoreService calls through the "
                            "seam (dataStores() in rbx_script_scaffold's save template)")
        elif fault_store and not any(e.get("acted") for e in si.get("events") or []):
            out["warning"] = "no scheduled event acted: check op, store, key, nth and the time window"
        # the internal notes L141: live UpdateAsync refuses a transform that yields; the stand-in failed each one the same way
        ny = si.get("yieldedTransforms") if isinstance(si, dict) else None
        if isinstance(ny, int) and not isinstance(ny, bool) and ny > 0:
            problems.append("%d UpdateAsync transform%s yielded (a DataStore, HTTP or task.wait call inside the callback): "
                            "live refuses each one with \"Callbacks cannot yield\", and the stand-in failed them the same "
                            "way. Do the yielding read before UpdateAsync" % (ny, "" if ny == 1 else "s"))
    if tests:
        res = {}
        printed = marked(r, TEST_MARK)
        for t in tests:
            # each template of that name prints under <name>.<kind>: save, action, prompt_once, economy
            mine = {tag: v[-1] for tag, v in printed.items() if tag.split(".")[0] == t and v}
            res[t] = mine or None
            if not mine:
                problems.append("self-test %s printed no result (is its SelfTest script in the place, and seconds long enough?)" % t)
            for tag, v in sorted(mine.items()):
                if v.get("passed") != v.get("total") or not v.get("total"):
                    bad = [c.get("name") for c in v.get("cases") or [] if not c.get("pass")]
                    problems.append("self-test %s: %s of %s passed; failed: %s" % (tag, v.get("passed"), v.get("total"), bad))
        out["selfTests"] = res
    if problems:
        out.update(ok=False, error="; ".join(problems))
    else:
        out["ok"] = True
        out.pop("error", None)
    return out


# ---------------------------------------------------------------- rbx_remote_fuzz
@_tool
def rbx_remote_fuzz(roots: list = None, players: int = 1, seconds: float = 45, census: dict = None, watch: list = None,
                    ignore: list = None, wait: float = 0.1, controls: int = 3, race: dict = None) -> dict:
    "Roblox Studio: fire every RemoteEvent, UnreliableRemoteEvent and RemoteFunction under roots (default ReplicatedStorage) from a client in a play test with forged arguments - nothing, nil, NaN, +inf, -inf, 2^53, -1, -1e9, 0.5, an empty table, a table, a nested table, a string, a 1,000-character string, true, a huge Vector3, an instance, the player, another player (players 2) and NaN after a string - each twice in a row, and compare the server's state before and after each: every player's leaderstats values and attributes, the objects in watch (Value, attributes, child count) and census {module, function} called per player. Changes seen with no call fired (controls rounds) are drift and are not blamed; ignore names keys to skip. ok false when a forged call changed state, when no remote was found, or when the run ended before every case. With race {remote, args (JSON values), balance (leaderstats.Coins, or @Coins for an attribute, on the firing player), price}, BEFORE the fuzz (so no forged call has spent the balance or is still being paid) the client fires that one remote twice in the same frame with those valid arguments, and the server reads the balance before and 1 s after: race answers before, after, couldPay (how many of the two the balance before could pay) and wentThrough ((before - after) / price); ok false when more went through than the balance could pay, when the balance could not be read, when the balance before pays neither call (nothing could show a race), or when more than two went through (other spends landed in the window: not judged). The fuzzer's two scripts are written under Workspace.MifScratch (scratch mode is enough) and removed after (ok false, naming what is left, when they could not be; both do nothing outside Studio); the test runs as a multiplayer test, so players have test UserIds (-1, -2) and the game's real saves are not its players'."
    roots = ["ReplicatedStorage"] if roots is None else roots
    if not isinstance(roots, list) or not roots or len(roots) > 20:
        return _no("roots is a list of up to 20 paths (default ReplicatedStorage)")
    root_parts = [V.split_path(p) for p in roots]
    if not all(root_parts):
        return _no("each root is a dot path from game")
    if not isinstance(players, int) or isinstance(players, bool) or not 1 <= players <= 4:
        return _no("players is 1-4 (2 or more adds the another-player case)")
    if not isinstance(seconds, (int, float)) or not 20 <= seconds <= 60:
        return _no("seconds is 20-60: the client fires for seconds minus 14 from when it starts, and the server reports 3 s before the end")
    if not isinstance(wait, (int, float)) or not 0 <= wait <= 2:
        return _no("wait (seconds the server waits before each snapshot) is 0-2")
    if not isinstance(controls, int) or isinstance(controls, bool) or not 0 <= controls <= 10:
        return _no("controls is 0-10")
    cen = None
    if census is not None:
        cen, err = _check_census(census)
        if err:
            return _no(err)
        if cen["per"] != "player":
            return _no("the fuzz census is per player: census.per player")
    watch_parts = [V.split_path(p) for p in (watch or [])]
    if watch is not None and (not isinstance(watch, list) or not all(watch_parts)):
        return _no("watch is a list of paths")
    if ignore is not None and not (isinstance(ignore, list) and all(isinstance(k, str) for k in ignore)):
        return _no("ignore is a list of keys as the reply names them (Player1.leaderstats.Coins, Player1@Attr)")
    race_cfg, err = _check_race(race)
    if err:
        return _no(err)
    T = _T()
    err = _ensure_scratch()
    if err:
        return _no(err)
    cfg = {"roots": root_parts, "census": cen, "watch": watch_parts, "ignore": ignore or [], "wait": wait,
           "controls": controls, "budget": seconds - 14, "deadline": seconds - 3}
    if race_cfg:
        cfg["race"] = race_cfg
    folder = "%s.%s" % (SCRATCH, FUZZ_FOLDER)
    made, err = _stage([("Folder", folder, None, None, {"Config": json.dumps(cfg), "Mark": FUZZ_MARK}),
                        ("Script", folder + ".FuzzServer", {"RunContext": "Server"}, LU.FUZZ_SERVER, None),
                        ("Script", folder + ".FuzzClient", {"RunContext": "Client"}, LU.FUZZ_CLIENT, None)])
    r = None
    try:
        if not err:
            r = T.rbx_play_probe("log", seconds=seconds, players=players, timeout=seconds + 120)
    finally:
        staged = _unstage(made[:1])           # the folder takes its scripts with it
    left = _left_problem(staged)
    if err:
        return _no(err + ("; " + left if left else ""), staged=staged)
    res = (marked(r, FUZZ_MARK).get("result") or [None])[-1]
    out = {"staged": staged, "probe": {k: r.get(k) for k in ("ok", "error", "refusedRule", "endedEarly", "recovered") if k in r}}
    if not isinstance(res, dict):
        return dict(out, ok=False, error="the fuzzer printed no result (probe %s: %s)%s" % (
            "ok" if r.get("ok") else "failed", r.get("error") or "no fuzz output; did the client script run?",
            "; " + left if left else ""))
    skipped = res.get("skipped") or []
    if isinstance(skipped, dict):
        skipped = list(skipped.values())
    needs_players = [s for s in skipped if s.endswith("(needs players 2)")]
    cut = [s for s in skipped if s not in needs_players]
    findings = res.get("findings") or []
    if isinstance(findings, dict):
        findings = list(findings.values())
    remotes = res.get("remotes") or []
    if isinstance(remotes, dict):
        remotes = list(remotes.values())
    out.update(remotes=remotes, remotesFired=res.get("batches"), argumentCases=res.get("cases"), findings=findings,
               drift=res.get("drift") or [], notRun=cut, skipped=needs_players, players=res.get("players"),
               census=res.get("census"), client=res.get("client") or [])
    problems = [left] if left else []
    if res.get("incomplete"):
        problems.append("the run did not finish: %s" % res["incomplete"])
    if not remotes:
        problems.append("no remote was found under %s: nothing was fuzzed" % ", ".join(roots))
    if findings:
        problems.append("%d forged call(s) changed state: %s" % (len(findings), "; ".join(
            "%s(%s) changed %s" % (f.get("remote"), f.get("arg"), ", ".join(sorted(f.get("changed") or {}))[:200])
            for f in findings[:8])))
    if cut:
        problems.append("%d case(s) did not run in the time: raise seconds or narrow roots" % len(cut))
    if cen and not res.get("census"):
        problems.append("the census module or function was not found in the play server")
    if race_cfg:
        rr, why = _race_verdict(res.get("race"), race_cfg)
        out["race"] = rr
        if why:
            problems.append(why)
    if problems:
        out.update(ok=False, error="; ".join(problems))
    else:
        out["ok"] = True
    return out


def _check_race(race):
    """(cfg, None) for rbx_remote_fuzz's race, (None, None) when not asked, or (None, the refusal)."""
    if race is None:
        return None, None
    if not isinstance(race, dict) or set(race) - {"remote", "args", "balance", "price"}:
        return None, "race is {remote, args, balance, price}"
    parts = V.split_path(race.get("remote")) if isinstance(race.get("remote"), str) else None
    if not parts:
        return None, "race.remote is a dot path from game to a RemoteEvent or RemoteFunction"
    args = race.get("args", [])
    if not isinstance(args, list) or len(args) > 20:
        return None, "race.args is a list of up to 20 JSON values, the remote's valid arguments"
    bal = race.get("balance")
    if not isinstance(bal, str) or not re.fullmatch(r"@[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", bal):
        return None, "race.balance is the firing player's number: \"leaderstats.Coins\" or \"@Coins\" (an attribute)"
    price = race.get("price")
    if not isinstance(price, (int, float)) or isinstance(price, bool) or not price > 0:
        return None, "race.price is what one call costs, a number above 0"
    return {"remote": parts, "args": args, "balance": bal, "price": price}, None


def _race_verdict(got, cfg):
    """(the race answer, None or why it fails): more of the two calls went through than the balance before could pay."""
    if not isinstance(got, dict):
        return {"ran": False}, "the race did not run (no race result from the play server)"
    before, after, price = got.get("before"), got.get("after"), cfg["price"]
    out = {"remote": got.get("remote"), "class": got.get("class"), "fired": 2, "sameFrame": True, "before": before,
           "after": after, "price": price, "answers": got.get("answers")}
    if got.get("error"):
        return dict(out, error=got["error"]), "the race could not fire: %s" % got["error"]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before, after)):
        return out, "the race's balance %s did not read as a number before and after" % cfg["balance"]
    could = max(0, min(2, int(before // price)))
    went = (before - after) / price
    out.update(couldPay=could, wentThrough=round(went, 3))
    if could == 0:
        # a balance that pays neither call cannot show a race: nothing was ever allowed through (found live 2026-11-26)
        return out, ("the balance before (%g) pays neither call at price %g: the race proves nothing - give the firing "
                     "player at least one price" % (before, price))
    if went > 2 + 1e-9:
        return out, ("%g went through from two calls: other spends landed in the 1 s window (another script, or a "
                     "handler still paying an earlier call), so the race was not judged" % went)
    if went > could + 1e-9:
        return out, ("%s went through %g times from two calls in one frame, against a balance (%g) that pays %d: "
                     "the second call was not refused" % (out["remote"], went, before, could))
    return out, None


# ---------------------------------------------------------------- module_require: rbx_module_values, rbx_odds_check
def _module_require(target, call=None, timeout=30):
    T = _T()
    parts, err = T._target(target)
    if err:
        return None, err
    # require() runs the module's top level, and every function call runs its body, as the plugin: with the plugin's
    # identity, able to write anywhere in the place, Studio's settings or the live game's data. So it is code, not a read
    # (found 2026-11-22: read and scratch mode let any ModuleScript run, two allowed calls past every rule of the gate).
    # It runs in full mode, or in scratch mode on MifBridge's own scratch place, where all of the place is scratch.
    ping = T._rbx("ping", nonce="module_require", timeout=15)
    if not ping.get("ok"):
        return None, ping
    mode = ping.get("mode")
    if not (mode == "full" or (mode == "scratch" and ping.get("placeName") == SCRATCH_PLACE)):
        return None, _no("requiring %s runs its code (its top level, and any function called) with the plugin's identity, "
                         "which can write anywhere: it runs only in full mode, or in scratch mode on MifBridge's own "
                         "scratch place, and the panel is in %s mode%s: nothing was run"
                         % (V.join_path(parts), mode, "" if mode == "read" else " on %s" % ping.get("placeName")),
                         refusedRule="read-mode" if mode == "read" else "full-mode", mode=mode,
                         unlock="Raise the mode to full in the MifBridge panel.")
    args = {"path": parts}
    if call:
        args["call"] = call
    r = T._rbx("module_require", timeout=timeout + 10, callTimeout=timeout, **args)
    if not r.get("ok") and "module_require" in str(r.get("error") or "") and "unknown" in str(r.get("error") or "").lower():
        r = dict(r, error="%s: the module_require op is in plugin 0.17.5 and later; install it and restart Studio" % r.get("error"))
    return parts, r


@_tool
def rbx_module_values(target: Any, timeout: float = 10) -> dict:
    "Roblox Studio: what a ModuleScript returns, as JSON, read in edit mode: the plugin requires a fresh clone (so require's cache is never stale) beside the original, outside the undo history, and removes it. Functions and instances are named, never called ({$function: key}, {$instance: path}); NaN and infinities are named too. Refused module-error with the message when the module errors, module-yields when it yields (it is not waited on). No server runs, but require runs the module's top level as code with the plugin's identity, so it runs only in full mode, or in scratch mode on MifBridge's own scratch place (read mode refuses it read-mode, scratch mode elsewhere full-mode). For a balance sim or store art that must read the game's real config instead of a copy. Read only."
    if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
        return _no("timeout is 1-60 seconds")
    parts, r = _module_require(target, timeout=timeout)
    if parts is None:
        return r
    if not r.get("ok"):
        return r
    out = {k: v for k, v in r.items() if k not in ("value",)}
    out.update(ok=True, path=V.join_path(parts), value=r.get("value"))
    if "value" not in r:
        out.update(ok=False, error="the plugin answered no value")
    return out


def _audit_sources(root, file, folder, max_scripts):
    """The scripts a static audit reads: ((sources, origin, truncated, {file | folder | root: ...}), None), or
    (None, refusal). From the connected Studio (script_scan under root), a saved place (file: .rbxlx or binary .rbxl)
    or a folder of .lua/.luau files - the last two need no Studio."""
    from mif_roblox import save_audit as SA
    if not isinstance(max_scripts, int) or isinstance(max_scripts, bool) or not 1 <= max_scripts <= 20000:
        return None, _no("max_scripts is 1-20000")
    if file is not None and folder is not None:
        return None, _no("give file or folder, not both")
    if file is not None:
        if not isinstance(file, str) or not os.path.isfile(file):
            return None, _no("file is an existing place file (.rbxlx or .rbxl): %r" % (file,))
        ext = os.path.splitext(file)[1].lower()
        if ext not in (".rbxlx", ".rbxl"):
            return None, _no("file is a .rbxlx or .rbxl place (got %s)" % (ext or "no extension"))
        try:
            sources, origin = SA.sources_from_file(file)
        except Exception as exc:  # noqa: BLE001 - an unreadable place is named with its reason
            return None, _no("%s could not be read as a place: %s: %s" % (os.path.basename(file), type(exc).__name__, exc),
                             ran=True)
        return (sources[:max_scripts], origin, len(sources) > max_scripts, {"file": file}), None
    if folder is not None:
        if not isinstance(folder, str) or not os.path.isdir(folder):
            return None, _no("folder is an existing folder of .lua/.luau files: %r" % (folder,))
        sources = SA.sources_from_folder(folder)
        return (sources[:max_scripts], "folder", len(sources) > max_scripts, {"folder": folder}), None
    T = _T()
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return None, T._target(root, "root")[1]
    r = T._rbx("script_scan", root=parts)
    if not r.get("ok"):
        return None, r
    sources = [x for x in r.get("sources") or [] if isinstance(x, dict) and isinstance(x.get("source"), str)]
    return (sources[:max_scripts], "studio", bool(r.get("truncated")) or len(sources) > max_scripts,
            {"root": V.join_path(parts) or "game"}), None


@_tool
def rbx_save_audit(root: Any = "game", file: str = None, folder: str = None, max_scripts: int = 5000,
                   checks: list = None) -> dict:
    "Roblox: a static check of a game's save layer and its game-pass grants; checks is a list of save and passes (both by default). PASSES: a permanent grant (SetAttribute(..., true), `= true` on a target naming a pass, a perk or saved data, a BoolValue set true, table.insert, or a call named grant, give, award, unlock, apply, activate, enable, reward or perk) reachable from a MarketplaceService.PromptGamePassPurchaseFinished handler with no MarketplaceService:UserOwnsGamePassAsync read on its path (kind pass-granted-unchecked: wasPurchased is the prompt closing, not Roblox's record), and a PlayerAdded path that grants inside a condition or loop reading a pass from saved data with no ownership read (kind pass-restored-from-save: a refunded pass stays granted); a handler is followed into the functions this script defines and calls by name. SAVE: the flaw that ran servers on memory: a table-backed or stand-in store chosen where RunService:IsStudio() can be false, and a failed GetDataStore, GetAsync, UpdateAsync (any DataStore call in a pcall or xpcall) whose failure path sets a memory flag, puts a table or a stand-in where the store was, returns one from the store's getter, or is read as a switch that reads or writes an in-memory table instead of refusing. Each finding has its script path, kind (stand-in-outside-studio, memory-fallback, pass-granted-unchecked, pass-restored-from-save), lines, the line's text, why, and the fix (for a save: retry, then refuse saves, loads and receipts and say so; for a pass: grant only when UserOwnsGamePassAsync answers true, and ask again on every join). Reads the scripts under root in the connected Studio (script_scan), or with file a saved place (.rbxlx, or binary .rbxl), or with folder every .lua/.luau file under it (Rojo names give the class) - file and folder need no Studio. ok false when anything is found; a run that read no scripts fails. A stand-in FACTORY (a function that builds a stand-in in a local and returns it, or returns another factory's call) is judged where it is called, outside test scripts; its own return is not a choice, a call passed straight to another call is judged in the callee, and a Studio flag a caller can override (if opts.studio ~= nil then opts.studio else RunService:IsStudio()) reads as IsStudio(). The save template (rbx_script_scaffold kind save) passes: its stand-in is reached only under IsStudio() from Workspace.MifScratch. Static, so it says what it cannot see (cannotSee): values passed through arguments or other modules are followed by name only, run-time stand-ins are not known, and nothing runs - rbx_play_probe fault_store is the play-test proof. Read only."
    from mif_roblox import save_audit as SA
    checks = list(SA.CHECKS) if checks is None else checks
    if not (isinstance(checks, list) and checks and all(c in SA.CHECKS for c in checks)):
        return _no("checks is a non-empty list of %s" % " and ".join(SA.CHECKS))
    got, refusal = _audit_sources(root, file, folder, max_scripts)
    if got is None:
        return refusal
    sources, origin, truncated, where = got
    out = SA.answer(sources, origin, truncated=truncated, checks=checks)
    out.update(where)
    return out


@_tool
def rbx_analytics_audit(root: Any = "game", file: str = None, folder: str = None, currencies: list = None,
                        max_scripts: int = 5000) -> dict:
    "Roblox: a static check of a game's analytics events (AnalyticsService's Log members, by the 0.748 catalog's parameter order) for the misuse that spends an experience's limits or reports changes that never happened. Findings, each with its script path, kind, lines, the line's text, why and the fix: dynamic-event-name (an event, funnel, step, currency, journey or progression name that is a concatenation, an interpolated string, string.format or tostring, an instance's or player's .Name, .DisplayName, .UserId or .Text, or an argument a client sent over OnServerEvent, OnServerInvoke or Chatted - each distinct value is a new name), per-frame-log (a log call, or a call to a function that reaches one, inside a function connected to RunService's Heartbeat, Stepped, RenderStepped, PreRender, PreAnimation, PreSimulation or PostSimulation, bound with BindToRenderStep, or in a while or repeat loop whose only wait is one frame), unlogged-currency-change (a declared currency's field or index changed with +=, -= or x.C = x.C + n in a server script by a function that reaches no LogEconomyEvent; a currency is declared by being a LogEconomyEvent's currencyType or by currencies, matched by exact name; Studio-only code is skipped), logged-before-save (an economy event sent before a DataStore UpdateAsync, SetAsync or IncrementAsync, or a call named save..., later in the same function), funnel-step-missing and funnel-step-order (literal step numbers with a gap from 1, or a lower step logged after a higher one in one function), over-limit, and deprecated-member (a Fire... member the catalog tags Deprecated). A name is declared when it is a string literal, an Enum item or a lookup in a table of string literals; a name that is a function's parameter is followed one call out, also through a required module. THE LIMITS ARE ASSUMED, not measured: 10 funnels, 100 steps a funnel, 100 custom event names, 3 custom fields, 10 currency types and 120 + 20 x players requests a minute are Roblox's documented numbers as read on 2026-11-22 (limits, limitsNote); the request rate is a run-time figure and is not counted. counts has what was counted (logCalls, customEventNames, funnels, currencyTypes, highestFunnelStep), unresolved the names it could not classify, notChecked what did not run (no declared currency means no currency change was looked for). Reads the scripts under root in the connected Studio (script_scan), or with file a saved place (.rbxlx, or binary .rbxl), or with folder every .lua/.luau file under it (Rojo names give the class) - file and folder need no Studio. ok false when anything is found; a run that read no scripts fails. The analytics template (rbx_script_scaffold kind analytics) passes. Static, so it says what it cannot see (cannotSee), and nothing runs: the bridge never calls a Log member. Read only."
    from mif_roblox import analytics_audit as AA
    if currencies is not None and (not isinstance(currencies, list) or len(currencies) > 50 or not all(
            isinstance(c, str) and c and len(c) <= 100 for c in currencies)):
        return _no("currencies is a list of up to 50 currency names (each a non-empty string), or omitted")
    got, refusal = _audit_sources(root, file, folder, max_scripts)
    if got is None:
        return refusal
    sources, origin, truncated, where = got
    out = AA.answer(sources, origin, truncated=truncated, currencies=currencies)
    out.update(where)
    return out


@_tool
def rbx_place_manifest(file: str, against: str = None, services: list = None, out: str = None, must_match: bool = False,
                       limit: int = 200) -> dict:
    "Roblox: one canonical sha256 of a saved place, and with against the diff of two - did a rebuild make the same place, and what did a change touch, before a publish. Offline: file is a saved place (.rbxlx, or binary .rbxl), no Studio. Every instance is a row - path, className, each property the file stores as {t: type, v: value}, attributes, tags, a script's sha256, bytes and lines - hashed, and the rows are hashed in path order into sha256; services gives each top-level instance its own count and sha256. Left out, and named in leftOut with the reason: the referent ids a file gives its instances (a Ref property is kept as the path it points at), UniqueId, HistoryId, ScriptGuid, the bridge's own ServerStorage.MifBridgePort marker and the order a file lists children in - two saves of one place differ in those and hash equal. No default is filtered (defaults says why): a file stores every serialized property. A string or binary value over 160 bytes is its sha256 and length; attributes are decoded when they hold only strings, booleans and numbers, else given as a sha256 (attributesAs). services names the top-level instances to keep (['Workspace', 'ServerScriptService']); a name the file does not hold is refused. out writes the whole manifest as JSON (read back before the answer); the answer itself carries the first limit rows as path, className and sha256. against is the earlier place file or a manifest written by out: the answer's diff has same, counts, and added, removed and changed rows, each changed row naming its properties with was and now (Attributes, Tags, Source and ClassName among them); siblings of one path are matched equal rows first, and an added or removed subtree is one row with its descendants counted. A renamed instance reads as one removed and one added. must_match true fails the call (ok false) when the two differ - the gate for a build that should not have changed. A manifest carries its format and a diff across .rbxlx and .rbxl is refused: the two store values differently. A binary property this reader does not decode is never guessed: it is left out, named in undecoded, and complete is false. An empty or unreadable file, a place with no instance and an edited manifest fail. Read only apart from out."
    from mif_roblox import place_manifest as PM
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5000:
        return _no("limit is 1-5000")
    if not isinstance(must_match, bool):
        return _no("must_match is true or false")
    if must_match and against is None:
        return _no("must_match needs against: the earlier place file or manifest to match")
    if services is not None and (not isinstance(services, list) or not services
                                 or not all(isinstance(s, str) and s for s in services)):
        return _no("services is a list of top-level instance names, such as ['Workspace', 'ServerScriptService']")
    if not isinstance(file, str) or not os.path.isfile(file):
        return _no("file is an existing place file (.rbxlx or .rbxl): %r" % (file,))
    ext = os.path.splitext(file)[1].lower()
    if ext not in (".rbxlx", ".rbxl"):
        return _no("file is a .rbxlx or .rbxl place (got %s)" % (ext or "no extension"))
    if against is not None:
        if not isinstance(against, str) or not os.path.isfile(against):
            return _no("against is an existing place file or a manifest written by out: %r" % (against,))
        aext = os.path.splitext(against)[1].lower()
        if aext not in (".rbxlx", ".rbxl", ".json"):
            return _no("against is a .rbxlx or .rbxl place, or a .json manifest (got %s)" % (aext or "no extension"))
    if out is not None:
        if not isinstance(out, str) or not out.lower().endswith(".json"):
            return _no("out is a .json file to write the manifest to")
        if not os.path.isdir(os.path.dirname(os.path.abspath(out))):
            return _no("out's folder does not exist: %s" % os.path.dirname(os.path.abspath(out)))
        if os.path.exists(out) and against is not None and os.path.samefile(out, against):
            return _no("out would overwrite against: write the new manifest to another file")

    def load(path):
        with open(path, "rb") as fh:
            blob = fh.read()
        if path.lower().endswith(".json"):
            return PM.load_manifest(blob.decode("utf-8", "replace"))
        fmt, rows, undecoded = PM.read(blob, os.path.splitext(path)[1].lower())
        return fmt, rows, undecoded, None

    def unread(path, exc):
        return _no("%s could not be read as a place: %s: %s" % (os.path.basename(path), type(exc).__name__, exc), ran=True)

    try:
        fmt, rows, undecoded, _s = load(file)
    except Exception as exc:  # noqa: BLE001 - an unreadable place is named with its reason
        return unread(file, exc)
    if services is not None:
        held = sorted(PM.per_service(rows))
        rows, missing = PM.only_services(rows, services)
        if missing:
            return _no("%s holds no top-level %s; it holds %s" % (os.path.basename(file), ", ".join(missing),
                                                                  ", ".join(held)), ran=True)
    if not rows:
        return _no("%s gave no rows: nothing was read, so nothing was hashed" % os.path.basename(file), ran=True)
    sha = PM.place_sha(rows)

    def short(r):
        return {"path": V.join_path(r["path"]), "className": r["className"], "sha256": r["sha256"]}
    ans = {"ok": True, "file": file, "format": fmt, "sha256": sha, "instances": len(rows),
           "classes": len({r["className"] for r in rows}), "scripts": sum(1 for r in rows if r.get("script")),
           "services": PM.per_service(rows), "leftOut": PM.LEFT_OUT, "defaults": PM.DEFAULTS_NOTE,
           "undecoded": undecoded, "complete": not undecoded, "rows": [short(r) for r in rows[:limit]],
           "truncated": len(rows) > limit}
    if undecoded:
        ans["note"] = ("%d binary propert%s could not be decoded and %s left out of the hash (undecoded): a change there "
                       "is not seen. Save the place as .rbxlx for a complete manifest"
                       % (len(undecoded), "y" if len(undecoded) == 1 else "ies", "is" if len(undecoded) == 1 else "are"))
    if out is not None:
        doc = PM.manifest(fmt, rows, undecoded, source=os.path.basename(file), services=services)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        try:
            with open(out, encoding="utf-8") as fh:
                back = PM.load_manifest(fh.read())
        except Exception as exc:  # noqa: BLE001 - a manifest that does not read back is not a manifest
            return dict(ans, ok=False, error="the manifest written to %s does not read back: %s" % (out, exc))
        if PM.place_sha(back[1]) != sha:
            return dict(ans, ok=False, error="the manifest written to %s reads back another sha256" % out)
        ans.update(out=out, outBytes=os.path.getsize(out))
    if against is None:
        return ans
    try:
        afmt, arows, aund, aserv = load(against)
    except Exception as exc:  # noqa: BLE001 - an unreadable place or manifest is named with its reason
        return unread(against, exc)
    if aserv is not None and sorted(aserv) != sorted(services or []):
        return dict(ans, ok=False, refusedRule="services-mismatch",
                    error="%s was written for services %s: pass the same services to compare the same part of the place"
                          % (os.path.basename(against), aserv))
    if services is not None and aserv is None:
        arows, missing = PM.only_services(arows, services)
        if missing or not arows:
            return dict(ans, ok=False, error="%s holds no top-level %s" % (os.path.basename(against), ", ".join(missing)))
    if afmt != fmt:
        return dict(ans, ok=False, refusedRule="format-mismatch",
                    error="%s is %s and %s is %s: the two formats store values differently, so save both the same way"
                          % (os.path.basename(file), fmt, os.path.basename(against), afmt))
    d = PM.diff(arows, rows)
    asha = PM.place_sha(arows)
    counts = {k: len(d[k]) for k in ("added", "removed", "changed")}
    same = asha == sha

    def named(r):
        return dict(r, path=V.join_path(r["path"]))
    ans["diff"] = {"against": against, "againstSha256": asha, "same": same, "counts": counts,
                   "added": [named(r) for r in d["added"][:limit]], "removed": [named(r) for r in d["removed"][:limit]],
                   "changed": [named(r) for r in d["changed"][:limit]], "truncated": any(n > limit for n in counts.values())}
    if aund != undecoded:
        ans["diff"]["note"] = "the two files leave different binary properties undecoded: see undecoded on each"
    if same != (not any(counts.values())):
        return dict(ans, ok=False, error="the hashes say %s and the diff lists %d row(s): the two must agree, so neither "
                                         "is to be trusted here" % ("same" if same else "different", sum(counts.values())))
    if must_match and not same:
        ans.update(ok=False, refusedRule="must-match",
                   error="the place differs from %s: %d added, %d removed, %d changed (diff)"
                         % (os.path.basename(against), counts["added"], counts["removed"], counts["changed"]))
    return ans


@_tool
def rbx_text_filter_audit(root: Any = "game", file: str = None, folder: str = None, max_scripts: int = 5000) -> dict:
    "Roblox: a static check of player-written text: does a string a client sent reach other players or a save without Roblox's text filter (TextService:FilterStringAsync and its TextFilterResult's GetNonChatStringForBroadcastAsync, GetNonChatStringForUserAsync and GetChatForUserAsync)? The developer must filter any displayed text they did not write. In server scripts it follows the arguments after the player of each function connected to a remote's OnServerEvent, set as OnServerInvoke, or connected to Chatted, inside that function: a value assigned from one is followed, one that went through the filter is not. Findings, each with its script, line, why and fix: unfiltered-text (the remote, the handler's line, the sink - text-show: set on .Text, .PlaceholderText, .Value, .Name or with SetAttribute; send-others: FireAllClients or FireClient; save: SetAsync, UpdateAsync or IncrementAsync - the author, the handler's player.UserId, and the result method that fits the audience), client-only-filter (FilterStringAsync named in a LocalScript) and per-keystroke-filter (FilterStringAsync in a function connected to GetPropertyChangedSignal, Changed or a per-frame signal). A value passed into another function or kept in a table is listed under unresolved with its line, never taken as filtered. Reads the scripts under root in the connected Studio (script_scan), or with file a saved place (.rbxlx, or binary .rbxl), or with folder every .lua/.luau file under it (Rojo names give the class) - file and folder need no Studio. ok false when anything is found; a run that read no scripts, or whose scripts hold no such handler and name no TextBox, fails. Static, so it says what it cannot see (cannotSee); nothing runs and TextService is never called. Read only."
    from mif_roblox import text_filter_audit as TFA
    got, refusal = _audit_sources(root, file, folder, max_scripts)
    if got is None:
        return refusal
    sources, origin, truncated, where = got
    out = TFA.answer(sources, origin, truncated=truncated)
    out.update(where)
    return out


def _gammq(a, x):
    """The regularized upper incomplete gamma Q(a, x) (Numerical Recipes 6.2): chi-square's p-value is Q(k/2, x/2)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi_square(counts, odds):
    """(statistic, degrees of freedom, p-value, rows) for observed counts against odds {outcome: weight}."""
    total_w = float(sum(odds.values()))
    n = sum(counts.values())
    rows, stat, k = [], 0.0, 0
    for outcome, w in odds.items():
        p = w / total_w
        e = n * p
        o = counts.get(outcome, 0)
        rows.append({"outcome": outcome, "expectedShare": p, "observedShare": o / n if n else None, "observed": o,
                     "expected": round(e, 2)})
        if e > 0:
            stat += (o - e) ** 2 / e
            k += 1
    df = max(k - 1, 1)
    return stat, df, _gammq(df / 2.0, stat / 2.0), rows


_PERCENT = re.compile(r"(\d+(?:\.(\d+))?)\s*%")


def _odds_table(odds, where):
    if isinstance(odds, dict) and odds and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
                                               for v in odds.values()) and sum(odds.values()) > 0:
        return {str(k): float(v) for k, v in odds.items()}, None
    return None, "%s is {outcome: weight or percent}, numbers 0 or more, not all 0" % where


def _dig(value, key):
    cur = value
    for k in key.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and k.isdigit() and 0 < int(k) <= len(cur):
            cur = cur[int(k) - 1]
        else:
            return None
    return cur


@_tool
def rbx_odds_check(target: Any, function: str, levels: list, n: int = 20000, alpha: float = 0.001, config: Any = None,
                   labels: dict = None) -> dict:
    "Roblox Studio: sample a game's roll function against the odds it shows. target is the ModuleScript holding function (a dotted key into what it returns, 'roll' or 'Eggs.roll'); each of levels is {args: [the call's arguments, e.g. a luck level], odds: {outcome: weight or percent} or a dotted key into config's returned table}. The plugin requires a fresh clone of target in edit mode (no server, no Run mode: the module and what it requires load on their own, or the call fails with the module's error) and calls the function n times per level; the counts of each first return (a table's Name or Id) are compared with the odds by a chi-square test, failing below alpha. An outcome the odds do not list fails. labels {outcome: path of a TextLabel} reads the odds text the UI shows (its first 'nn.n%') and compares it with the first level's odds to the label's own decimals. Read only of the place, but requiring the module and calling its function run its code with the plugin's identity, so it runs only in full mode, or in scratch mode on MifBridge's own scratch place."
    if not isinstance(function, str) or not re.match(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$", function):
        return _no("function is a dotted key into the module's table: roll, or Eggs.roll")
    if not isinstance(levels, list) or not 1 <= len(levels) <= 10:
        return _no("levels is 1-10 of {args: [...], odds: {outcome: weight} or a config key}")
    if not isinstance(n, int) or isinstance(n, bool) or not 100 <= n <= 200000:
        return _no("n (rolls per level) is 100-200000")
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 0.5:
        return _no("alpha is above 0 and below 0.5")
    if labels is not None and not (isinstance(labels, dict) and labels and all(V.split_path(p) for p in labels.values())):
        return _no("labels is {outcome: path of the label showing its odds}")
    cfg_value = None
    for i, lv in enumerate(levels):
        if not isinstance(lv, dict) or set(lv) - {"args", "odds"} or "odds" not in lv:
            return _no("levels[%d] is {args: [...], odds: ...}" % i)
        if not isinstance(lv.get("args", []), list):
            return _no("levels[%d].args is a list" % i)
        if isinstance(lv["odds"], str) and config is None:
            return _no("levels[%d].odds names a config key, so config (the config ModuleScript's path) is needed" % i)
    if config is not None and any(isinstance(lv["odds"], str) for lv in levels):
        c = rbx_module_values(config)
        if not c.get("ok"):
            return dict(c, error="config could not be read: %s" % c.get("error"))
        cfg_value = c.get("value")
    tables = []
    for i, lv in enumerate(levels):
        odds = _dig(cfg_value, lv["odds"]) if isinstance(lv["odds"], str) else lv["odds"]
        t, err = _odds_table(odds, "levels[%d].odds%s" % (i, " (%s in config)" % lv["odds"] if isinstance(lv["odds"], str) else ""))
        if err:
            return _no(err, found=odds if isinstance(lv["odds"], str) else None)
        tables.append(t)
    T = _T()
    out_levels, problems = [], []
    for i, lv in enumerate(levels):
        cat, err = T._catalog()
        if err:
            return err
        args = []
        for a in lv.get("args", []):
            w, why = V.encode("Variant", a, cat.enums)
            if why:
                return _no("levels[%d].args: %s" % (i, why))
            args.append(w)
        parts, r = _module_require(target, call={"fn": function, "args": args, "n": n}, timeout=60)
        if parts is None:
            return r
        if not r.get("ok"):
            return dict(r, level=i)
        r = r.get("call") if isinstance(r.get("call"), dict) else r     # plugin 0.17.5 answers the calls under "call"
        counts = {str(k): int(v) for k, v in (r.get("counts") or {}).items()}
        calls = sum(counts.values())
        if not calls:
            return _no("the roll function returned nothing in %s calls (errors %s: %s)" % (
                r.get("calls"), r.get("errors"), r.get("firstError")), level=i)
        stat, df, p, rows = chi_square(counts, tables[i])
        unknown = sorted(set(counts) - set(tables[i]))
        small = [row["outcome"] for row in rows if 0 < row["expected"] < 5]
        lvl = {"args": lv.get("args", []), "odds": tables[i], "calls": calls, "errors": r.get("errors"),
               "firstError": r.get("firstError"), "chiSquare": round(stat, 3), "df": df, "p": p, "rows": rows,
               "unlisted": unknown, "pass": p >= alpha and not unknown and not r.get("errors"),
               "stoppedAt": r.get("stoppedAt")}
        if small:
            lvl["note"] = "expected counts under 5 for %s: raise n for a fair test" % small
        out_levels.append(lvl)
        if unknown:
            problems.append("level %d rolled %s, which its odds do not list" % (i, unknown))
        if p < alpha:
            problems.append("level %d: the counts differ from the odds (chi-square %.2f, df %d, p %.2g < %g)" % (i, stat, df, p, alpha))
        if r.get("errors"):
            problems.append("level %d: %s calls errored: %s" % (i, r.get("errors"), r.get("firstError")))
    out = {"target": V.join_path(V.split_path(target)), "function": function, "n": n, "alpha": alpha, "levels": out_levels}
    if labels:
        shown = []
        base = tables[0]
        total = sum(base.values())
        for outcome, path in sorted(labels.items()):
            g = T.rbx_get_props(path, ["Text"])
            text = (g.get("values") or {}).get("Text") if g.get("ok") else None
            m = _PERCENT.search(text or "")
            want = 100.0 * base.get(str(outcome), 0) / total if total else None
            row = {"outcome": outcome, "label": path, "text": text}
            if not g.get("ok"):
                row["error"] = g.get("error")
                problems.append("label %s could not be read: %s" % (path, g.get("error")))
            elif not m:
                row["error"] = "no percent in the text"
                problems.append("label %s shows no percent: %r" % (path, text))
            else:
                decimals = len(m.group(2) or "")
                row.update(shown=float(m.group(1)), odds=round(want, decimals + 2), decimals=decimals,
                           match=abs(float(m.group(1)) - want) <= 0.5 * 10 ** -decimals + 1e-9)
                if not row["match"]:
                    problems.append("label %s shows %s%%, the odds are %.4g%%" % (path, m.group(1), want))
            shown.append(row)
        out["labels"] = shown
    if problems:
        out.update(ok=False, error="; ".join(problems))
    else:
        out["ok"] = True
    return out


# ---------------------------------------------------------------- rbx_script_scaffold's kinds
TEMPLATE_KINDS = ("save", "action", "prompt_once", "economy", "scenery", "warps", "subscription", "analytics",
                  "filtered_text")
PROMPT_KINDS = ("favorite", "notifications")


def templates(kind, name, prompt="favorite"):
    """[(class, parent, name, props, source)] for one template kind."""
    SSS, RS, SPS = "ServerScriptService", "ReplicatedStorage", "StarterPlayer.StarterPlayerScripts"
    srv = {"RunContext": "Server"}
    if kind == "save":
        return [("RemoteFunction", RS, name + "AdRemote", None, None),
                ("ModuleScript", SSS, name + "Store", None, LU.render(LU.SAVE_MODULE, name)),
                ("Script", SSS, name + "Server", srv, LU.render(LU.SAVE_SERVER, name)),
                ("LocalScript", SPS, name + "AdClient", None, LU.render(LU.SAVE_AD_CLIENT, name)),
                ("Script", SSS, name + "SaveSelfTest", srv, LU.render(LU.SAVE_SELFTEST, name))]
    if kind == "action":
        return [("RemoteEvent", RS, name + "Remote", None, None),
                ("ModuleScript", SSS, name + "Action", None, LU.render(LU.ACTION_MODULE, name)),
                ("Script", SSS, name + "ActionServer", srv, LU.render(LU.ACTION_SERVER, name)),
                ("LocalScript", SPS, name + "ActionClient", None, LU.render(LU.ACTION_CLIENT, name)),
                ("Script", SSS, name + "ActionSelfTest", srv, LU.render(LU.ACTION_SELFTEST, name))]
    if kind == "prompt_once":
        return [("RemoteEvent", RS, name + "PromptRemote", None, None),
                ("ModuleScript", SSS, name + "Prompt", None, LU.render(LU.PROMPT_MODULE, name, kind=prompt)),
                ("Script", SSS, name + "PromptServer", srv, LU.render(LU.PROMPT_SERVER, name)),
                ("LocalScript", SPS, name + "PromptClient", None, LU.render(LU.PROMPT_CLIENT, name)),
                ("Script", SSS, name + "PromptSelfTest", srv, LU.render(LU.PROMPT_SELFTEST, name))]
    if kind == "economy":
        return [("ModuleScript", SSS, name + "Clock", None, LU.render(LU.ECONOMY_CLOCK, name)),
                ("ModuleScript", SSS, name + "Economy", None, LU.render(LU.ECONOMY_MODULE, name)),
                ("Script", SSS, name + "EconomyServer", srv, LU.render(LU.ECONOMY_SERVER, name)),
                ("Script", SSS, name + "EconomySelfTest", srv, LU.render(LU.ECONOMY_SELFTEST, name))]
    if kind == "scenery":
        return [("RemoteEvent", RS, name + "SceneryReport", None, None),
                ("ModuleScript", RS, name + "Textures", None, LU.render(LU.SCENERY_TEXTURES, name)),
                ("ModuleScript", RS, name + "Scenery", None, LU.render(LU.SCENERY_MODULE, name)),
                ("Script", SSS, name + "SceneryServer", srv, LU.render(LU.SCENERY_SERVER, name)),
                ("LocalScript", SPS, name + "SceneryClient", None, LU.render(LU.SCENERY_CLIENT, name)),
                ("Script", SSS, name + "ScenerySelfTest", srv, LU.render(LU.SCENERY_SELFTEST, name))]
    if kind == "warps":
        return [("RemoteFunction", RS, name + "WarpRemote", None, None),
                ("ModuleScript", SSS, name + "Warps", None, LU.render(LU.WARP_MODULE, name)),
                ("Script", SSS, name + "WarpServer", srv, LU.render(LU.WARP_SERVER, name)),
                ("LocalScript", SPS, name + "WarpClient", None, LU.render(LU.WARP_CLIENT, name)),
                ("Script", SSS, name + "WarpSelfTest", srv, LU.render(LU.WARP_SELFTEST, name))]
    if kind == "subscription":
        return [("RemoteFunction", RS, name + "SubscriptionRemote", None, None),
                ("ModuleScript", SSS, name + "Subscription", None, LU.render(LU.SUB_MODULE, name)),
                ("Script", SSS, name + "SubscriptionServer", srv, LU.render(LU.SUB_SERVER, name)),
                ("LocalScript", SPS, name + "SubscriptionClient", None, LU.render(LU.SUB_CLIENT, name)),
                ("Script", SSS, name + "SubscriptionSelfTest", srv, LU.render(LU.SUB_SELFTEST, name))]
    if kind == "analytics":
        return [("ModuleScript", SSS, name + "Analytics", None, LU.render(LU.ANALYTICS_MODULE, name)),
                ("Script", SSS, name + "AnalyticsServer", srv, LU.render(LU.ANALYTICS_SERVER, name)),
                ("Script", SSS, name + "AnalyticsSelfTest", srv, LU.render(LU.ANALYTICS_SELFTEST, name))]
    if kind == "filtered_text":
        return [("RemoteEvent", RS, name + "TextRemote", None, None),
                ("ModuleScript", SSS, name + "Text", None, LU.render(LU.FILTER_MODULE, name)),
                ("Script", SSS, name + "TextServer", srv, LU.render(LU.FILTER_SERVER, name)),
                ("LocalScript", SPS, name + "TextClient", None, LU.render(LU.FILTER_CLIENT, name)),
                ("Script", SSS, name + "TextSelfTest", srv, LU.render(LU.FILTER_SELFTEST, name))]
    return None


def scaffold_clash(T, ops):
    """The refusal for a scaffold whose objects are already there, else None (found 2026-11-26: a second scaffold of one
    name made a second copy of every object and then failed on the duplicate name, leaving both copies behind). Every
    rbx_create in ops is resolved before the batch runs."""
    there = []
    for op in ops:
        if op.get("tool") != "rbx_create":
            continue
        a = op.get("args") or {}
        path = "%s.%s" % (a.get("parent"), a.get("name"))
        if T._rbx("resolve", target=V.split_path(path)).get("ok"):
            there.append(path)
    if not there:
        return None
    return {"ok": False, "ran": False, "refusedRule": "exists", "exists": there,
            "error": "this scaffold's objects are already in the place (%s): nothing was written. Use another name, or "
                     "remove them first with rbx_destroy" % ", ".join(there[:6])}


def scaffold(kind, name, prompt="favorite"):
    """Write one template kind with rbx_batch (one undo step), read back by path and sha256."""
    T = _T()
    if kind not in TEMPLATE_KINDS:
        return _no("kind is pair (the default), %s" % ", ".join(TEMPLATE_KINDS))
    if prompt not in PROMPT_KINDS:
        return _no("prompt is %s" % " or ".join(PROMPT_KINDS))
    objs = templates(kind, name, prompt)
    ops = []
    for cls, parent, nm, props, _src in objs:
        a = {"class_name": cls, "parent": parent, "name": nm}
        if props:
            a["props"] = props
        ops.append({"tool": "rbx_create", "args": a})
    for _cls, parent, nm, _props, src in objs:
        if src is not None:
            ops.append({"tool": "rbx_script_write", "args": {"target": parent + "." + nm, "source": src}})
    clash = scaffold_clash(T, ops)
    if clash:
        return clash
    r = T.rbx_batch(ops)
    written = {parent + "." + nm: T._sha(src) for _c, parent, nm, _p, src in objs if src is not None}
    seconds = {"save": 30, "scenery": 30, "warps": 30}.get(kind, 25)
    return dict(r, kind=kind, objects=[p + "." + n for _c, p, n, _q, _s in objs], sha256=written,
                selfTest=[parent + "." + nm for _c, parent, nm, _p, _s in objs if nm.endswith("SelfTest")][0],
                prove="rbx_play_probe('log', seconds=%d, players=1, self_test='%s')" % (seconds, name))
