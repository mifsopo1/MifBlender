"""Release machinery on the MCP side (batch 15): the coverage number per domain.

    rbx_coverage   counted members a typed rbx_* call has reached AND read back on this machine, per the internal notes domain, over
                   the catalog's denominator (4,758 in 561 classes on 0.748)

THE DEFINITION: a member counts when it is counted (reachable, not Deprecated,
Hidden or NotScriptable) and a typed tool reached it with a read-back. Each way is recorded as its `how`:
    read            a property rbx_get_props read and decoded (a value that came back as its text only, {"$t", "str"},
                    counts nothing)
    write           a property rbx_set_props wrote, read back equal, whose value CHANGED (Studio held something else before)
    write-same      the same, but Studio already held the value: read back, yet no proof the write does anything
    call            a function rbx_call judged a read (by its name, or named in tools.PURE_CALLS after a live run)
    call-verified   a write rbx_call proved through its paired getter, or through a tools.PROOFS row (a getter of another
                    name, a property read before and after, or an event watched from before the call)
    event           an event rbx_watch saw fire at least once (a read or stop of a watch this process started)
    composite-<how> a member another rbx_* tool read back in the same call and listed in its answer's `proved`, checked
                    against the plugin ops that call made (tools._proved; how is read, write, call or event)
    play-read, play-call, play-event
                    a property read, a read call answered, or an event seen by rbx_play_probe kind members in a running
                    game, taken only from the run's own nonce-matched reply (members.trusted). A Player's members count only
                    this way (decision 9): a Player made in Edit is no game's player
    play-call-verified
                    a member a members-plan cause called in the running game (the cause's own, play_members.CAUSE_PROVES,
                    or a PLAY_CALLS row), proven by an event that fired in the cause's window and not just before it, or by
                    a check read before and after that changed
An unverified write, a refusal, a failure and a watch that saw nothing count nothing. "Written and read back" (WRITTEN)
is reported beside the one number: a property counts on a read alone, so the second number says how much was written. The record is
kept per Studio build in the machine's coverage folder (below). Every live suite run and every agent session on this machine adds
to it, so the number says how much of the surface has been shown to work here, and since when. It is not a claim about
code that has not run.

WHERE IT LIVES (2026-11-21): ONE record per Studio build for the whole machine, <LOCALAPPDATA>/MifBridge/Roblox/coverage/
<version>.json, whatever MIF_ROBLOX_HOME says - the play lock's pattern (launch.play_lock_path). Before, it lived in each
home, so every lane and suite with its own home kept a private record and the machine's number missed their proofs.
MIF_ROBLOX_COVERAGE moves the folder, for a suite.

WHO MAY WRITE IT: only a call a running Roblox Studio answered (live_studio: the daemon's connected plugin, whose process -
read from its connection - is a RobloxStudio image), recorded under the build that Studio reported. An offline fake answers
from a Python process and records nothing.

HOW IT IS WRITTEN: every writer (one per MCP server process; several lanes at once) holds the record's file lock
(launch.RecordLock on <version>.lock), reads it, merges, writes a per-process temp file and replaces atomically. A record
that cannot be read is never taken for "no record": the write waits for it, and a record whose content is not a record is
kept aside as <version>.corrupt-<time>.json before a new one starts. The old code did take a failed read for no record:
three processes writing at once read PermissionError on 14% of reads while another replaced the file (RB2100), and each
such read started a fresh record - the 2026-11-21 08:53 restart.

A WRITER ON OLD CODE (RB2129): an MCP server started before a fix keeps the code it loaded. The record carries a format
number and the last writer's stamp (FORMAT, writer); every writer of this format also keeps the last good copy beside it
(<version>.good.json). A record that holds fewer members than that copy - reset, or replaced by a writer that started
over - is restored from it under the lock by the next writer or reader, and the event is logged to
<version>.restore.log with the stamp of the record found (none: a writer older than format 2). Each process that writes
leaves its stamp in writers/<pid>.json; rbx_status names every running MCP server whose code is older (older_writers).

DECLINED (spec "a declined list with reasons"): roblox_declined.json beside the catalog lists, one row per member, the
counted members no plugin can prove and why - each rule with its the internal notes or doc section. They stay in the 4,758 (the
catalog's is_counted is unchanged); rbx_coverage answers the number over both denominators, and a declined member that
shows up reached is a problem the offline gate fails on (declined_problems).

tools.py calls note() from its _reports wrapper, one line, after every reported tool returns. note() never raises and never
changes an answer.
"""
import json
import os
import re
import threading
import time

__all__ = ["rbx_coverage"]
_LOCK = threading.Lock()
# What THIS process's calls proved (spec "Fix the coverage floors: count what this run proved, not the cumulative
# record"): the record only grows, so a suite whose floor reads it passes after its first green run however much it
# regresses. note() adds every member it records here too, {"Owner.Member": {"how": [...]}}, and a live suite judges
# its floor on run_proved(); the record check stays as a second assertion.
RUN_PROVED = {}
READ_RETRY_SECONDS = 5.0      # a record held open by another process (a replace in flight, a scanner) is read again
FORMAT = 2                    # 1 (no field): P1's machine-wide record; 2: writer stamp, last good copy, the hows below
HOWS = ("read", "write", "write-same", "call", "call-verified", "event",
        "composite-read", "composite-write", "composite-call", "composite-event", "play-read", "play-call", "play-event",
        "play-call-verified")
WRITTEN = ("write", "call-verified", "composite-write", "play-call-verified")      # "written and read back"
COMPOSITE_HOWS = ("read", "write", "call", "event")
DECLINED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roblox_declined.json")


def _T():
    from mif_roblox import tools
    return tools


def coverage_dir():
    """The machine's coverage folder: not the home, so every lane's and suite's live proofs land in one record."""
    return os.environ.get("MIF_ROBLOX_COVERAGE") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MifBridge", "Roblox", "coverage")


def _path(version):
    safe = "".join(c for c in str(version or "unknown") if c.isalnum() or c in ".-_")
    return os.path.join(coverage_dir(), "%s.json" % safe)


def _read(path, wait=READ_RETRY_SECONDS):
    """(doc, state): state "ok", "missing" (no file), "busy" (could not be opened within wait) or "bad" (opened, but not
    a record). Only "missing" may start a new record."""
    deadline = time.monotonic() + wait
    while True:
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            return None, "missing"
        except OSError:
            if time.monotonic() >= deadline:
                return None, "busy"
            time.sleep(0.02)
            continue
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("members"), dict):
            return doc, "ok"
        if time.monotonic() >= deadline:
            return None, "bad"
        time.sleep(0.05)


def _load(path):
    doc, _state = _read(path, wait=1.0)
    return doc


def _merge_into(doc, members):
    """Merge {key: row} into doc: the earliest first, the latest last, every way it was proven. True when a key was new."""
    added = False
    for k, row in members.items():
        if not isinstance(row, dict):
            continue
        have = doc["members"].get(k)
        if have is None:
            doc["members"][k] = {"first": row.get("first"), "how": sorted(set(row.get("how") or [])),
                                 "last": row.get("last") or row.get("first")}
            added = True
            continue
        firsts = [t for t in (have.get("first"), row.get("first")) if isinstance(t, str)]
        lasts = [t for t in (have.get("last"), row.get("last")) if isinstance(t, str)]
        have["first"] = min(firsts) if firsts else None
        have["last"] = max(lasts) if lasts else None
        have["how"] = sorted(set(have.get("how") or []) | set(row.get("how") or []))
    return added


def _replace(path, doc, wait=READ_RETRY_SECONDS):
    """Write doc beside path under this process's own temp name and replace path with it, retrying while a reader holds
    the old file open (Windows refuses the replace then). Raises when it never could."""
    tmp = "%s.%d.tmp" % (path, os.getpid())
    from mif_roblox import durable
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=0, sort_keys=True)
        durable.sync(fh)                     # before the rename: a power loss once emptied the record and its copy
    deadline = time.monotonic() + wait
    while True:
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if time.monotonic() >= deadline:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(0.02)


def _good_path(path):
    return path[:-5] + ".good.json"


def _prev_path(path):
    """The good copy before the last one: a second chance when the good copy itself cannot be read (2026-11-22)."""
    return path[:-5] + ".good.prev.json"


def _stamp():
    """Who wrote the record: this process, its code's format, and when."""
    return {"pid": os.getpid(), "format": FORMAT, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "code": os.path.abspath(__file__)}


def _restore(path, doc):
    """Put back every member the last good copy holds and doc lost (a reset, or a writer that started over), and log it.
    Returns how many were put back. Called under the record's lock."""
    good, state = _read(_good_path(path), wait=1.0)
    if state != "ok":
        good, state = _read(_prev_path(path), wait=1.0)
    if state != "ok":
        return 0
    lost = {k: v for k, v in good["members"].items() if k not in doc["members"]}
    if not lost:
        return 0
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "restored": len(lost), "found": len(doc["members"]),
             "good": len(good["members"]), "foundSince": doc.get("since"), "goodSince": good.get("since"),
             "foundWriter": doc.get("writer") or "no writer stamp: a writer older than format %d" % FORMAT,
             "restoredBy": os.getpid()}
    _merge_into(doc, lost)
    starts = [t for t in (doc.get("since"), good.get("since")) if isinstance(t, str)]
    doc["since"] = min(starts) if starts else doc.get("since")
    try:
        with open(path[:-5] + ".restore.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass
    return len(lost)


def record(version, members, since=None):
    """Merge {"Owner.Member": {"first", "last", "how"}} into the machine's record for this build, under its file lock.
    Returns (added, path). Never starts over a record it could not read: raises OSError instead (busy), and keeps a record
    that is not one aside as <version>.corrupt-<time>.json before starting a new one. A record smaller than its last good
    copy is restored from it first (a writer on old code reset it, RB2129); the record and the good copy are then written
    with this writer's stamp."""
    from mif_roblox import launch as L
    path = _path(version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _LOCK, L.RecordLock(path[:-5] + ".lock", wait=30):
        doc, state = _read(path)
        prior = json.loads(json.dumps(doc)) if state == "ok" and doc else None   # the record as it was: next good copy
        if state == "busy":
            raise OSError("the coverage record %s could not be opened for %ss" % (path, READ_RETRY_SECONDS))
        if state == "bad":
            os.replace(path, "%s.corrupt-%s.json" % (path[:-5], time.strftime("%Y%m%dT%H%M%S")))
        if doc is None:
            doc = {"studio": version, "since": since or time.strftime("%Y-%m-%dT%H:%M:%S"), "members": {}}
        restored = _restore(path, doc)
        starts = [r.get("first") for r in members.values() if isinstance(r, dict) and isinstance(r.get("first"), str)]
        starts += [t for t in (doc.get("since"), since) if isinstance(t, str)]
        doc["since"] = min(starts) if starts else doc.get("since")
        added = _merge_into(doc, members)
        doc["format"] = FORMAT
        doc["writer"] = _stamp()
        # The good copy is one generation behind, written from the record as it was before this write, never from this
        # write's own doc, so one bad moment cannot empty both (2026-11-22); the copy before it is kept as well.
        if _read(_good_path(path), wait=1.0)[1] == "ok":
            try:
                os.replace(_good_path(path), _prev_path(path))
            except OSError:
                pass
        _replace(_good_path(path), prior if prior is not None else doc)
        _replace(path, doc)
    _register()
    return added or bool(restored), path


def _current(version):
    """The record for reading: restored first when it holds fewer members than its last good copy. Never raises."""
    path = _path(version)
    doc = _load(path)
    good = _load(_good_path(path))
    if good and (doc is None or set(good["members"]) - set(doc["members"])):
        try:
            record(version, {})
            doc = _load(path)
        except Exception:  # noqa: BLE001 - a reader that cannot repair still reports what it read
            pass
    return doc


_REGISTERED = []
_OLDER_CACHE = []                 # [monotonic, named]: rbx_status asks often, the process table is read every 30 s


def writers_dir():
    return os.path.join(coverage_dir(), "writers")


def _register():
    """Leave this process's stamp in writers/<pid>.json once: older_writers reads it to tell this code from older code."""
    if _REGISTERED:
        return
    try:
        import psutil
        os.makedirs(writers_dir(), exist_ok=True)
        st = dict(_stamp(), started=psutil.Process(os.getpid()).create_time())
        with open(os.path.join(writers_dir(), "%d.json" % os.getpid()), "w", encoding="utf-8") as fh:
            json.dump(st, fh)
        _REGISTERED.append(st)
    except Exception:  # noqa: BLE001 - the stamp is a report, never a reason a call fails
        return


def _code_format(module_file):
    """The FORMAT a domain_release.py on disk declares: 1 for code from before the field, None when there is no file."""
    try:
        with open(module_file, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    import re
    m = re.search(r"^FORMAT = (\d+)", text, re.M)
    return int(m.group(1)) if m else 1


def _python_pids():
    """The pids of running Python processes. On Windows tasklist's own image filter (0.3 s with 750 processes running,
    measured 2026-11-21) - asking psutil each process's name took 2.5 s, too slow for rbx_status."""
    import subprocess
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python*", "/FO", "CSV", "/NH"], capture_output=True,
                                 text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            pids = []
            for line in out.splitlines():
                cells = [c.strip('"') for c in line.split('","')]
                if len(cells) > 1 and cells[1].isdigit():
                    pids.append(int(cells[1]))
            return pids
        except (OSError, subprocess.SubprocessError):
            pass
    import psutil
    return [p.pid for p in psutil.process_iter(["name"]) if (p.info.get("name") or "").lower().startswith("python")]


def _mcp_servers():
    """Running MifBridge MCP servers that carry the Roblox code: [{pid, started, server, module}]. Never raises."""
    out = []
    try:
        import psutil
        for pid in _python_pids():
            try:
                if pid == os.getpid():
                    continue
                p = psutil.Process(pid)
                server = next((a for a in p.cmdline() or [] if a.replace("\\", "/").endswith("mcp-server/server.py")), None)
                if not server:
                    continue
                server = os.path.abspath(server if os.path.isabs(server) else os.path.join(p.cwd(), server))
                module = os.path.join(os.path.dirname(server), "mif_roblox", "domain_release.py")
                if os.path.isfile(module):
                    out.append({"pid": pid, "started": p.create_time(), "server": server, "module": module})
            except Exception:  # noqa: BLE001 - a process that ended or hides its command line is skipped
                continue
    except Exception:  # noqa: BLE001
        return out
    return out


def older_writers(servers=None):
    """Running MCP servers whose coverage code is older than this one's FORMAT, each with why: its code on disk declares
    an older format (or none), or it started before its code on disk changed and has left no stamp since - it still
    runs what it loaded. A server with a stamp of this format, or whose code on disk is this format and older than the
    process, is not named. Never raises."""
    if servers is None and _OLDER_CACHE and time.monotonic() - _OLDER_CACHE[0] < 30:
        return list(_OLDER_CACHE[1])
    named = []
    for s in (_mcp_servers() if servers is None else servers):
        try:
            stamp = None
            try:
                with open(os.path.join(writers_dir(), "%d.json" % s["pid"]), encoding="utf-8") as fh:
                    stamp = json.load(fh)
            except (OSError, ValueError):
                stamp = None
            if (isinstance(stamp, dict) and (stamp.get("format") or 0) >= FORMAT
                    and abs((stamp.get("started") or 0) - (s.get("started") or 0)) < 2):
                continue
            disk = _code_format(s["module"])
            if disk is None:
                continue
            if disk < FORMAT:
                why = "its code on disk is format %d (this is %d): it can reset the record" % (disk, FORMAT)
            elif os.path.getmtime(s["module"]) > (s.get("started") or 0):
                why = ("it started before its code on disk changed and has written no stamp since: it may still run "
                       "format-%d-or-older code" % (FORMAT - 1))
            else:
                continue
            named.append({"pid": s["pid"], "server": s["server"], "started": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(s.get("started") or 0)), "why": why})
        except Exception:  # noqa: BLE001
            continue
    if servers is None:
        _OLDER_CACHE[:] = [time.monotonic(), named]
    return named


def live_studio():
    """The Roblox Studio that answers on this MCP server's daemon: {"version", "pid", "port"}, or None when no running
    Roblox Studio is connected (no daemon, no plugin, or a plugin in another process - an offline suite's fake). Only
    then does a call's proof count. Never raises."""
    try:
        T = _T()
        _code, st, _b = T._http("GET", "/status", timeout=2.0)
        plugin = (st or {}).get("plugin") or {}
        version = plugin.get("studioVersion")
        if not plugin.get("connected") or not isinstance(version, str) or not version:
            return None
        import psutil
        pid = plugin.get("studioPid")
        if not isinstance(pid, int):
            # off Windows the daemon cannot read its peer's pid: find the process connected to the daemon's port
            port = st.get("port")
            pid = next((c.pid for c in psutil.net_connections(kind="tcp")
                        if c.raddr and c.raddr.port == port and c.pid), None)
        if not isinstance(pid, int):
            return None
        if not psutil.Process(pid).name().lower().startswith("robloxstudio"):
            return None
        return {"version": version, "pid": pid, "port": st.get("port")}
    except Exception:  # noqa: BLE001 - no proof of a real Studio means nothing is recorded
        return None


PLAY_HOWS = ("play-read", "play-call", "play-event", "play-call-verified")
PLAY_ONLY = {"Player"}      # classes whose members count only from a play test (coverage plan decision 9)


def proved_problems(out, ops, cat):
    """What is wrong with a composite answer's proved list, as strings: an entry that is not {member, how, op}, a member
    that is not in the catalog, a how that is not one of COMPOSITE_HOWS, or an op this call did not make and get an ok
    answer to (ops: [(op, ok)], the call's own slice of tools._ops). Empty when every entry stands."""
    bad = []
    made = {op for op, ok in (ops or []) if ok}
    for e in out.get("proved") or [] if isinstance(out, dict) else []:
        if not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in ("member", "how", "op")):
            bad.append("not {member, how, op}: %r" % (e,))
            continue
        cls, _dot, name = e["member"].partition(".")
        if cat.find_member(cls, name)[1] is None:
            bad.append("%s is not a member in the catalog" % e["member"])
        elif e["how"] not in COMPOSITE_HOWS:
            bad.append("%s: how %r is not one of %s" % (e["member"], e["how"], ", ".join(COMPOSITE_HOWS)))
        elif e["op"] not in made:
            bad.append("%s: the call made no %s op that Studio answered" % (e["member"], e["op"]))
    return bad


def keys_from(tool, args, kwargs, out, cat, ops=None):
    """The counted members this answer proves, as {"Owner.Member": how}. Empty for anything not proven. ops is the
    call's own [(op, ok)]: a composite's proved entry counts only when its op is among them."""
    if not isinstance(out, dict):
        return {}
    found = {}

    def add(cls, name, how):
        owner, row = cat.find_member(cls, name)
        if row is not None and owner:
            from mif_roblox import catalog as C
            # coverage plan decision 9: a Player's members are proven only on a
            # player who joined a play test, never on a Player made in Edit - no game ever has one
            if C.is_counted(row) and (owner not in PLAY_ONLY or how.startswith("play-")):
                key = "%s.%s" % (owner, name)
                if found.get(key) not in WRITTEN:        # one answer, one how per member: the stronger proof wins
                    found[key] = how
    if tool == "rbx_play_probe":
        # kind members (spec "members proven inside a play test count"): each step the run proved, only from a result
        # tools._members_answer trusted - this run's nonce, the reply itself, not a log it was recovered from. A run whose
        # other steps failed still proves the ones that answered.
        m = out.get("members") if isinstance(out.get("members"), dict) else {}
        if m.get("trusted") is True:
            for row in out.get("proved") or []:
                key, how = (row or {}).get("member"), (row or {}).get("how")
                if isinstance(key, str) and "." in key and how in PLAY_HOWS:
                    add(*key.split(".", 1), how)
        # the run's own composite rows (tools._session_proved: the session's start and EndTest, a test run's result),
        # checked like any composite's against the ops this call made; a members row is never read as one
        comp = [e for e in out.get("provedSession") or [] if isinstance(e, dict)]
        if out.get("ok") and comp and not proved_problems({"proved": comp}, ops, cat):
            for e in comp:
                cls, _dot, name = e["member"].partition(".")
                add(cls, name, "composite-" + e["how"])
        return found
    if not out.get("ok"):
        return {}
    if tool == "rbx_get_props":
        props = kwargs.get("props", args[1] if len(args) > 1 else None)
        cls = out.get("className")
        values = out.get("values") if isinstance(out.get("values"), dict) else {}
        undecoded = set(out.get("undecoded") or [])
        if cls and isinstance(props, list):
            for p in props:
                if isinstance(p, str) and p in values and p not in undecoded:
                    add(cls, p, "read")
    elif tool == "rbx_set_props" and out.get("verified") is True:
        cls = out.get("className")
        changed = out.get("changed") if isinstance(out.get("changed"), dict) else {}
        for p in (out.get("after") or {}):
            add(cls, p, "write" if changed.get(p) is True else "write-same")
    elif tool == "rbx_call" and (out.get("readOnly") or out.get("verified") is True):
        member = out.get("member")
        if isinstance(member, str) and ":" in member:
            cls, name = member.split(":", 1)
            add(cls, name, "call" if out.get("readOnly") else "call-verified")
    elif tool == "rbx_watch" and isinstance(out.get("watched"), str) and (out.get("count") or len(out.get("events") or [])):
        cls, _dot, name = out["watched"].partition(".")
        _o, row = cat.find_member(cls, name)
        if row is not None and row[0] == "E":
            add(cls, name, "event")
    if out.get("proved") and not proved_problems(out, ops, cat):
        for e in out["proved"]:
            cls, _dot, name = e["member"].partition(".")
            add(cls, name, "composite-" + e["how"])
    return found


def note(tool, args, kwargs, out, answered=None, ops=None):
    """Record what this answer proved. Never raises, never changes out. True when it added a member not recorded before.
    answered False (tools._reports: no daemon replied during the call - a suite's stand-in for _rbx) records nothing.
    ops: the plugin ops the call made, [(op, ok)], which a composite's proved list is checked against."""
    try:
        if answered is False or not isinstance(out, dict):
            return False
        if tool not in ("rbx_get_props", "rbx_set_props", "rbx_call", "rbx_watch") and not (
                out.get("proved") or out.get("provedSession")):
            return False
        T = _T()
        cat, err = T._catalog()
        if err:
            return
        found = keys_from(tool, args, kwargs, out, cat, ops=ops)
        if not found:
            return False
        studio = live_studio()
        if not studio:
            return False                     # no running Roblox Studio answered: a fake or a dead plugin proves nothing
        _run_add(found)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        added, _path_ = record(studio["version"], {k: {"first": now, "last": now, "how": [how]}
                                                   for k, how in found.items()})
        return added
    except Exception:  # noqa: BLE001 - coverage is a record, never a reason a tool fails
        return False


def _run_add(found):
    with _LOCK:
        for k, how in found.items():
            row = RUN_PROVED.setdefault(k, {"how": []})
            if how not in row["how"]:
                row["how"].append(how)


def run_proved(cat, domains=None):
    """This process's proofs by domain: {domain: {"reached", "writtenAndReadBack", "members"}} for the domains named
    (every domain with a proof when None), counted as rbx_coverage counts the record (revoked members left out)."""
    from mif_roblox import domains as RC
    with _LOCK:
        doc = {"members": {k: {"how": list(r["how"])} for k, r in RUN_PROVED.items()}}
    members = counting(doc, cat)
    out = {d: {"reached": 0, "writtenAndReadBack": 0, "members": []} for d in (domains or ())}
    for key, row in members.items():
        d = RC.domain(cat, key.split(".", 1)[0])
        if domains is not None and d not in domains:
            continue
        r = out.setdefault(d, {"reached": 0, "writtenAndReadBack": 0, "members": []})
        r["reached"] += 1
        r["members"].append(key)
        if set(row.get("how") or []) & set(WRITTEN):
            r["writtenAndReadBack"] += 1
    for r in out.values():
        r["members"].sort()
    return out


def panel_summary(cat=None):
    """What the plugin's panel shows in its CoverageRow (spec "coverage per domain"): {studio, since, reached,
    denominator, domains: [[name, reached, counted], ...] every domain, most reached first}. The daemon sends it in its
    hello answer, and the MCP side with a verdict whenever a call adds a member. Never raises: None when unknown."""
    try:
        if cat is None:
            from mif_roblox import catalog as C
            cat = C.load()
        from mif_roblox import domains as RC
        ds = RC.domains(cat)
        doc = _current(cat.studio.get("version")) or {"members": {}, "since": None}
        members = counting(doc, cat)
        by = {}
        for key in members:
            d = RC.domain(cat, key.split(".", 1)[0])
            by[d] = by.get(d, 0) + 1
        rows = sorted(([n, by.get(n, 0), d["members"]] for n, d in ds.items()), key=lambda r: (-r[1], r[0]))
        return {"studio": cat.studio.get("version"), "since": doc.get("since"), "reached": len(members),
                "denominator": cat.doc.get("counts", {}).get("reachable"), "domains": rows}
    except Exception:  # noqa: BLE001 - the panel is a view
        return None


# ---------------------------------------------------------------- the built-in Studio MCP server, named at a wall
# Roblox's own MCP server does what the plugin's
# security wall stops: input simulation and running Luau. When the bridge refuses one of those members, the answer
# names the built-in tool that does it, so the person picks. The bridge never starts, enables or calls that server.
WALLS = (
    (r"^VirtualInputManager\.Send(Key|TextInputCharacter)", "user_keyboard_input"),
    (r"^VirtualInputManager\.Send(Mouse|Scroll|Touch)", "user_mouse_input"),
    (r"^VirtualUser\.(SetKey|TypeKey)", "user_keyboard_input"),
    (r"^VirtualUser\.(Button|Click|MoveMouse)", "user_mouse_input"),
    (r"^LogService\.ExecuteScript$", "execute_luau"),
)


# the internal notes decision 9 (Mifsopo, 2026-11-20: yes): setup offers the built-in server beside MifBridge. The offer is text the person
# pastes into their own client, never a write to their config. The bridge calls that server in one place only:
# rbx_place_save with use_builtin_mcp true (builtin_save.py: list_roblox_studios, inspect_instance, save_place; 2026-11-22).
# What is read from disk (0.748, RB1414, RB1422): the Assistant plugin's copy of the "Enable Studio as MCP server" switch
# (Assistant-ExternalMCPEnabled), else AssistantSettings/<user id>.json "mcp-server"; StudioMCP/tools-cache.json lists
# the tools the server last offered. The StudioMCP/auth.token beside it is the person's and is never opened.
ROBLOX_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Roblox")
BUILTIN_NAME = "Roblox_Studio"
BUILTIN_ARGS = ["/c", "%LOCALAPPDATA%\\Roblox\\mcp.bat"]
BUILTIN_ENABLE = "Studio: Assistant > Manage MCP Servers > Enable Studio as MCP server"


def _read_bools(pattern, pick):
    import glob
    seen = []
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f, encoding="utf-8") as fh:
                v = pick(json.load(fh))
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        if isinstance(v, bool):
            seen.append(v)
    return seen


def _builtin_enabled():
    """True / False / "unknown". The switch itself is a cloud user setting; Studio's Assistant plugin keeps a copy in its
    plugin setting Assistant-ExternalMCPEnabled (<user id>/InstalledPlugins/0/settings.json), which is read first:
    measured 0.748 (RB1422), AssistantSettings/<user id>.json read false while Studios connected to the server."""
    seen = _read_bools(os.path.join(ROBLOX_DIR, "*", "InstalledPlugins", "0", "settings.json"),
                       lambda d: d.get("Assistant-ExternalMCPEnabled"))
    if not seen:
        seen = _read_bools(os.path.join(ROBLOX_DIR, "AssistantSettings", "*.json"),
                           lambda d: (d.get("mcp-server") or {}).get("enabled"))
    return any(seen) if seen else "unknown"


def _assistant_settings():
    """One row per AssistantSettings/<user id>.json that parses: whether it holds the "mcp-server" key at all, that
    key's enabled value, and "mcp-prompt-sharing" enabled. Measured 0.748, 2026-11-22 (RB2889): after a power-loss
    reboot the file held no mcp-server key and Studio showed the switch off, while the plugin copy _builtin_enabled
    reads first can still say on; turning the switch back on also turns "Data access" (prompt sharing) on."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(ROBLOX_DIR, "AssistantSettings", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        srv, share = d.get("mcp-server"), d.get("mcp-prompt-sharing")
        rows.append({"file": os.path.basename(f), "hasMcpServer": "mcp-server" in d,
                     "mcpServerEnabled": srv.get("enabled") if isinstance(srv, dict) else None,
                     "promptSharing": share.get("enabled") if isinstance(share, dict) else None})
    return rows


def switch_on_after(start):
    """The newest write time of an AssistantSettings file that holds mcp-server enabled true, when that write is later
    than start (a process start time, epoch seconds); else None. A Studio started before the switch was turned on did
    not join Roblox's built-in server until it was relaunched (2026-11-22, RB3064). The file's mtime is its last
    write, which any Assistant setting can make, so this is a hint for a refusal, never a refusal of its own."""
    import glob
    best = None
    try:
        for f in glob.glob(os.path.join(ROBLOX_DIR, "AssistantSettings", "*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
                on = isinstance(d, dict) and isinstance(d.get("mcp-server"), dict) and d["mcp-server"].get("enabled") is True
                t = os.path.getmtime(f)
            except (OSError, ValueError):
                continue
            if on and (best is None or t > best):
                best = t
    except Exception:  # noqa: BLE001 - a hint must never break the refusal it is added to
        return None
    return best if best is not None and isinstance(start, (int, float)) and best > start else None


def builtin_warnings(rows=None):
    """Warnings about the built-in server's switch as Studio's settings file shows it. Never raises."""
    try:
        rows = _assistant_settings() if rows is None else rows
    except Exception:  # noqa: BLE001 - a warning must never break the call that asks for it
        return []
    out = []
    for r in rows:
        if not r.get("hasMcpServer"):
            out.append({"code": "mcp-server-key-missing", "file": r.get("file"),
                        "warning": "Studio's AssistantSettings/%s holds no \"mcp-server\" key: a reboot dropped the "
                                   "'Enable Studio as MCP server' switch on this machine (RB2889), so no Studio joins "
                                   "Roblox's built-in MCP server and rbx_place_save cannot save" % r.get("file"),
                        "fix": "the person: %s" % BUILTIN_ENABLE})
        if r.get("promptSharing") is True:
            out.append({"code": "prompt-sharing-on", "file": r.get("file"),
                        "warning": "Studio's \"Data access\" setting is on in AssistantSettings/%s: it gives Roblox "
                                   "access to prompt information, and turning the MCP switch on turns it on by "
                                   "default (RB2889)" % r.get("file"),
                        "fix": "the person, if they do not want that: turn Data access off on the same page (%s)"
                               % BUILTIN_ENABLE.split(" > Enable")[0]})
    return out


def _builtin_tools():
    """(names, date) from the tool list the server last cached, or (None, None)."""
    try:
        with open(os.path.join(ROBLOX_DIR, "StudioMCP", "tools-cache.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        names = sorted(t["name"] for t in doc.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("name"), str))
        stamp = doc.get("date")
        when = time.strftime("%Y-%m-%d", time.localtime(stamp)) if isinstance(stamp, (int, float)) else None
        return (names or None), when
    except (OSError, ValueError, AttributeError, OverflowError):
        return None, None


def builtin_config(client):
    """The text a person pastes to add the built-in server to one client, beside MifBridge's own entry."""
    entry = {"command": "cmd.exe", "args": list(BUILTIN_ARGS)}
    if client == "claude-code":
        return "claude mcp add %s -- cmd.exe /c %%LOCALAPPDATA%%\\Roblox\\mcp.bat" % BUILTIN_NAME
    if client == "codex":
        return ('[mcp_servers.%s]\ncommand = "cmd.exe"\nargs = ["/c", "%%LOCALAPPDATA%%\\\\Roblox\\\\mcp.bat"]'
                % BUILTIN_NAME)
    return json.dumps({"mcpServers": {BUILTIN_NAME: entry}}, indent=2)


CLIENTS = ("claude-code", "cursor", "claude-desktop", "codex")


def builtin_mcp():
    """What this machine shows of Roblox's built-in server and how to add it beside MifBridge: its launcher (installed
    with Studio), whether the person's "Enable Studio as MCP server" switch is on, the tools it last offered, and the
    entry for each client. Never raises."""
    names, when = _builtin_tools()
    launcher = os.path.join(ROBLOX_DIR, "mcp.bat")
    return {"launcher": launcher if os.path.isfile(launcher) else None, "enabled": _builtin_enabled(),
            "enable": BUILTIN_ENABLE, "toolsSeen": names, "toolsSeenOn": when,
            "client": {"command": "cmd.exe", "args": list(BUILTIN_ARGS), "name": BUILTIN_NAME},
            "configs": {c: builtin_config(c) for c in CLIENTS}, "warnings": builtin_warnings(),
            "note":"Optional, beside MifBridge: Roblox's own server (execute_luau, input, save_place, start_stop_play). "
                    "The person turns it on and adds it; MifBridge calls it only for rbx_place_save with use_builtin_mcp "
                    "true (save_place on its own Studio)"}


def wall_hint(tool, out):
    """Add builtInMcp to a security refusal of a member the built-in server covers. Never raises; only adds a key."""
    try:
        if tool != "rbx_call" or not isinstance(out, dict) or out.get("refusedRule") != "security":
            return
        import re
        m = re.match(r"^(\w+\.\w+) is ", str(out.get("error") or ""))
        if not m:
            return
        for rx, name in WALLS:
            if re.search(rx, m.group(1)):
                out["builtInMcp"] = dict(builtin_mcp(), tool=name,
                                         note="Roblox's own Studio MCP server has %s for this; MifBridge does not call it. "
                                              "The person enables it and adds it to their client" % name)
                return
    except Exception:  # noqa: BLE001
        return


def summary(cat):
    """{reached, denominator, since} for rbx_status: the file read, no classification. Never raises."""
    try:
        doc = _current(cat.studio.get("version")) or {"members": {}}
        members = counting(doc, cat)
        out = {"reached": len(members), "denominator": cat.doc.get("counts", {}).get("reachable"),
               "since": doc.get("since"), "perDomain": "rbx_coverage",
               "writtenAndReadBack": sum(1 for r in members.values() if set(r.get("how") or []) & set(WRITTEN))}
        out["olderWriters"] = older_writers()
        return out
    except Exception:  # noqa: BLE001
        return None


def rbx_coverage(domain: str = None, list_members: bool = False) -> dict:
    "Roblox Studio: how much of Studio's API MifBridge has shown working on this machine - per API domain, the counted members a typed rbx_* call reached and read back (a property read and decoded, a write read back equal - how write when the value changed, write-same when Studio already held it - a read-only call, a write its paired getter proved, an event rbx_watch saw fire, or a member another tool read back and listed as proved), over the catalog's denominator (4,758 on 0.748). Beside it: writtenAndReadBack (members written and read back changed), byHow, declined (members no plugin can prove, each with its reason in roblox_declined.json) and percentOfProvable over the denominator less those; revoked names members whose proof was judged too weak after it landed (roblox_revoked.json), left out of every number until a later proof lands. Recorded per Studio build in ONE record for the machine (LOCALAPPDATA/MifBridge/Roblox/coverage, whatever MIF_ROBLOX_HOME says; MIF_ROBLOX_COVERAGE moves it for a suite) by every reported call a running Roblox Studio answered - every lane, live suite and agent session alike, never an offline fake - since the date given; concurrent writers merge under a file lock and never reset it. With domain, that domain only; list_members names the members. Read only; Studio need not run."
    T = _T()
    cat, err = T._catalog()
    if err:
        return err
    from mif_roblox import domains as RC
    ds = RC.domains(cat)
    if domain is not None and domain not in ds:
        return {"ok": False, "ran": False, "error": "domain is one of %s" % ", ".join(sorted(ds))}
    path = _path(cat.studio.get("version"))
    doc = _current(cat.studio.get("version")) or {"members": {}, "since": None}
    members = counting(doc, cat)
    revoked = sorted(set(doc["members"]) - set(members))
    declined = declined_members(cat)
    by, dec_by = {}, {}
    for key in members:
        cls = key.split(".", 1)[0]
        by.setdefault(RC.domain(cat, cls), []).append(key)
    for key in declined:
        d = RC.domain(cat, key.split(".", 1)[0])
        dec_by[d] = dec_by.get(d, 0) + 1
    written = {k for k, r in members.items() if set(r.get("how") or []) & set(WRITTEN)}
    hows = {}
    for r in members.values():
        for h in r.get("how") or []:
            hows[h] = hows.get(h, 0) + 1
    rows = {}
    for name, d in sorted(ds.items()):
        if domain is not None and name != domain:
            continue
        got = sorted(by.get(name, []))
        provable = d["members"] - dec_by.get(name, 0)
        row = {"reached": len(got), "counted": d["members"],
               "percent": round(100.0 * len(got) / d["members"], 1) if d["members"] else None,   # a domain of 0 members
               "writtenAndReadBack": sum(1 for k in got if k in written), "declined": dec_by.get(name, 0),
               "percentOfProvable": round(100.0 * sum(1 for k in got if k not in declined) / provable, 1)
               if provable else None}
        if list_members:
            row["members"] = got
        rows[name] = row
    total = sum(len(v) for v in by.values())
    den = cat.doc.get("counts", {}).get("reachable")
    reached_declined = sorted(k for k in members if k in declined)
    provable = den - len(declined) if den else None
    out = {"ok": True, "studio": cat.studio.get("version"), "since": doc.get("since"), "file": path,
           "reached": total, "denominator": den, "percent": round(100.0 * total / den, 2) if den else None,
           "writtenAndReadBack": len(written), "declined": len(declined), "provable": provable,
           "percentOfProvable": round(100.0 * (total - len(reached_declined)) / provable, 2) if provable else None,
           "reachedDeclined": reached_declined, "revoked": revoked, "byHow": dict(sorted(hows.items())),
           "format": doc.get("format", 1), "writer": doc.get("writer"),
           "domains": rows,
           "definition": "counted members (reachable, not Deprecated/Hidden/NotScriptable) a typed rbx_* call reached "
                         "with a read-back on this machine for this Studio build; writtenAndReadBack: of those, the ones "
                         "written and read back changed (how %s); percentOfProvable: reached over the counted members "
                         "less the declined ones (roblox_declined.json)" % ", ".join(WRITTEN)}
    if reached_declined:
        out["warning"] = ("%d declined members show up reached: their decline rows are wrong, or the record counts "
                          "something it should not (declined_problems)" % len(reached_declined))
    if not total:
        out["note"] = "nothing recorded yet for this build: run the live suites (tools/roblox_live.py)"
    return out


def load_declined(path=None):
    """roblox_declined.json: {"studio", "rules": {rule: {why, cite}}, "members": {"Owner.Name": rule}}, or None."""
    try:
        with open(path or DECLINED_PATH, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) and isinstance(doc.get("members"), dict) else None
    except (OSError, ValueError):
        return None


def declined_members(cat=None, path=None):
    """{"Owner.Name": rule} for the catalog's build; empty when the file is for another build or absent."""
    doc = load_declined(path)
    if not doc:
        return {}
    if cat is not None and doc.get("studio") != cat.studio.get("version"):
        return {}
    return dict(doc["members"])


IMPORTER_CLASSES = ("BaseImportData", "ImportSession")
DEBUGGER_CLASSES = ("DebuggerManager", "ScriptDebugger", "DebuggerBreakpoint")
REFUSED_ARG_TYPES = ("Secret", "SharedTable", "AdReward", "ClipEvaluator", "CollectionHandle", "DockWidgetPluginGuiInfo")
# Generative beyond GenerationService and the Generate calls (COVERAGE_PLAN 7.2 #2, 2026-11-22): speech synthesis, the
# procedural model classes, and the calls that build or rig an avatar by a model
GENERATIVE_CLASSES = ("GenerationService", "AudioTextToSpeech", "ProceduralModel", "GeneratedFolder")
GENERATIVE_MEMBERS = ("AvatarCreationService.AutoSetupAvatarAsync", "AvatarCreationService.LoadAvatar2DPreviewAsync",
                      "TextGenerator.GenerateTextAsync")
# A measured rule's citation names where the measurement is: a the internal notes, a docs section, the plan, a suite, the catalog
CITATION = re.compile(r"\bRB\d+\b|docs/\d+|COVERAGE_PLAN|plan \d|\bcatalog\b|tools/test_roblox|values\.REFUSED_TYPES")


def decline_rule_holds(cat, member, rule, rule_doc=None, evidence=None):
    """Does the catalog still show what the rule says of this member? The rules are the ones the decline rows cite; a
    member is declined only by a row, never by a rule alone (COVERAGE_PLAN 2.4). A measured rule (kind "measured": a
    refusal, a silence, a safety rule - nothing the catalog shows) holds while the member is counted, its own evidence
    row says what was measured, and the rule's citation names a the internal notes, a doc section or a suite."""
    from mif_roblox import catalog as C
    cls, _dot, name = member.partition(".")
    owner, row = cat.find_member(cls, name)
    if row is None or owner != cls or not C.is_counted(row):
        return False
    if isinstance(rule_doc, dict) and rule_doc.get("kind") == "measured":
        return bool(isinstance(evidence, str) and evidence.strip() and CITATION.search(str(rule_doc.get("cite") or "")))
    types = [str(row[1] or "").rstrip("?")] + [str(p[1]).rstrip("?") for p in row[5]]
    if rule == "callback":
        return row[0] == "C"
    if rule == "opens-dialog":
        return row[0] == "F" and bool(C.OPENS_DIALOG.match(name))
    if rule == "needs-function":
        return row[0] in "FE" and any(p[1] == "Function" for p in row[5])
    if rule == "refused-type":
        return (row[0] == "P" and types[0] in ("QFont", "QDir")) or (row[0] == "F" and bool(set(types) & set(REFUSED_ARG_TYPES)))
    if rule == "importer":
        return any(cat.is_a(cls, b) for b in IMPORTER_CLASSES)
    if rule == "debugger":
        return cls in DEBUGGER_CLASSES
    if rule == "generative":
        return (cls in GENERATIVE_CLASSES or member in GENERATIVE_MEMBERS
                or (cls == "AvatarCreationService" and "Generat" in name))
    return False


def declined_problems(cat, record_doc=None, path=None):
    """Why the declined list cannot be trusted, as strings: a file for another build, a rule with no reason or citation,
    a row whose rule the catalog no longer shows, and - given a record - a declined member that shows up reached (its
    row is wrong, or the record counted what no plugin can prove). Empty when it stands."""
    doc = load_declined(path)
    if doc is None:
        return ["no declined list at %s" % (path or DECLINED_PATH)]
    bad = []
    if doc.get("studio") != cat.studio.get("version"):
        bad.append("the declined list is for Studio %s, the catalog %s" % (doc.get("studio"), cat.studio.get("version")))
    rules = doc.get("rules") or {}
    for rule, r in rules.items():
        if not (isinstance(r, dict) and r.get("why") and r.get("cite")):
            bad.append("rule %s has no reason or no citation" % rule)
    evidence = doc.get("evidence") if isinstance(doc.get("evidence"), dict) else {}
    for member, rule in sorted(doc["members"].items()):
        if rule not in rules:
            bad.append("%s: rule %r is not one of the file's rules" % (member, rule))
        elif not decline_rule_holds(cat, member, rule, rules.get(rule), evidence.get(member)):
            measured = isinstance(rules.get(rule), dict) and rules[rule].get("kind") == "measured"
            bad.append("%s: %s" % (member, "a measured rule (%s) with no evidence row for it, or a citation naming no "
                                            "ledger row or doc" % rule if measured else
                                   "the catalog no longer shows %s for it" % rule))
    if record_doc:
        for k in sorted(set(record_doc.get("members") or {}) & set(doc["members"])):
            bad.append("%s is declined (%s) but the record shows it reached (%s)"
                       % (k, doc["members"][k], ", ".join(record_doc["members"][k].get("how") or [])))
    return bad


REVOKED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roblox_revoked.json")
_STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")


def load_revoked(path=None):
    """roblox_revoked.json (COVERAGE_PLAN 7.2 #11): {"studio", "members": {"Owner.Name": {at, reason, the internal notes}}} - proofs
    judged too weak after they landed. The record is never edited by hand and a restore would put a deleted member back,
    so a revoked member stays in the file and stops COUNTING: rbx_coverage, rbx_status and the panel leave it out until
    a proof later than `at` lands (the record's `last`). None when absent or not a list."""
    try:
        with open(path or REVOKED_PATH, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) and isinstance(doc.get("members"), dict) else None
    except (OSError, ValueError):
        return None


def counting(doc, cat=None, path=None):
    """The record's members that count: every one, less each revoked member with no proof after its revoke time."""
    members = (doc or {}).get("members") or {}
    rev = load_revoked(path)
    if not rev or (cat is not None and rev.get("studio") != cat.studio.get("version")):
        return dict(members)
    out = {}
    for k, row in members.items():
        r = rev["members"].get(k)
        at = r.get("at") if isinstance(r, dict) else None
        last = row.get("last") if isinstance(row, dict) else None
        if isinstance(at, str) and not (isinstance(last, str) and last > at):
            continue                                  # revoked, and nothing proved it again since
        out[k] = row
    return out


def revoked_problems(cat, path=None):
    """Why the revoke list cannot be trusted: another build, a member not in the catalog, an entry with no reason, no
    the internal notes (RB...) or no clock-read time. Empty when it stands."""
    doc = load_revoked(path)
    if doc is None:
        return ["no revoke list at %s" % (path or REVOKED_PATH)]
    bad = []
    if doc.get("studio") != cat.studio.get("version"):
        bad.append("the revoke list is for Studio %s, the catalog %s" % (doc.get("studio"), cat.studio.get("version")))
    for member, r in sorted(doc["members"].items()):
        cls, _dot, name = member.partition(".")
        if cat.find_member(cls, name)[1] is None:
            bad.append("%s is not a member in the catalog" % member)
        if not (isinstance(r, dict) and r.get("reason") and re.search(r"\bRB\d+\b", str(r.get("ledger") or ""))
                and _STAMP.match(str(r.get("at") or ""))):
            bad.append("%s: an entry needs reason, ledger (an RB row) and at (YYYY-MM-DDTHH:MM:SS)" % member)
    return bad


def merge_file(src):
    """Merge another coverage record (a lane home's, from before the record was machine-wide) into the machine's, keeping
    its dates. Returns {"file", "studio", "members", "added"}; refuses a file that is not a record or names no build."""
    doc, state = _read(src, wait=1.0)
    if state != "ok":
        return {"ok": False, "file": src, "error": "not a coverage record (%s)" % state}
    version = doc.get("studio")
    if not isinstance(version, str) or not version:
        return {"ok": False, "file": src, "error": "the record names no Studio build"}
    members = {k: v for k, v in doc["members"].items() if isinstance(v, dict) and isinstance(v.get("first"), str)}
    before = len((_load(_path(version)) or {"members": {}})["members"])
    _added, path = record(version, members, since=doc.get("since"))
    after = len((_load(path) or {"members": {}})["members"])
    return {"ok": True, "file": src, "into": path, "studio": version, "members": len(members), "added": after - before}


if __name__ == "__main__":
    # python -m mif_roblox.domain_release --merge <record.json>...: fold older per-home records into the machine's
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "--merge":
        print("usage: python -m mif_roblox.domain_release --merge <record.json>...")
        sys.exit(2)
    results = [merge_file(p) for p in sys.argv[2:]]
    for r in results:
        print(json.dumps(r))
    sys.exit(0 if results and all(r.get("ok") for r in results) else 1)
