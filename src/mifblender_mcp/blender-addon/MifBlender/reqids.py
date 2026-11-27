"""Request ids: a call repeated with the same requestId answers its first reply and never runs twice.

WHY, FOUND 2026-11-22. A caller whose read timed out cannot tell a call that never ran from one whose reply was late:
Blender runs every op on its main thread, so a long bake, a modal operator or a background Blender serving another
client holds the reply while the work completes. The MCP server gives Unreal calls a request id the plugin keeps with
its result (MifBridgeJournal.cpp, "REQUEST IDS"); Blender calls carried none, so a Blender write retried after a timeout
ran twice - a second bevel, a second import. This is the Blender twin of that design, with its field names.

THE FRAME. A request may carry `requestId` beside endpoint and token (an add-on older than this ignores it). The socket
thread checks it AFTER the token and BEFORE the request reaches the main thread (begin):
  - an id never seen: recorded as running, and the request runs;
  - an id whose call has finished: its first reply is answered again with replayed:true, requestId and
    firstRunSecondsAgo - from the socket thread, without the main thread, so a replay works while Blender is busy;
  - an id whose call is still running: requestRunning:true, retryable:true, nothing started again;
  - an id already used for a different call (another endpoint or other params): requestIdConflict:true, nothing ran.
The reply is stored as it is framed for the wire (end), so the replay is byte for byte what the first call answered -
and it is stored before the send, which is the send that fails when the caller has gone.

WHAT IS FORGOTTEN, so a repeat runs: a request refused because its caller had stopped waiting (callerGaveUp - nothing
ran), and a window Blender's job cancelled before it started (the waiter gave up first). Both ran nothing, so replaying
their refusal would refuse a call that can now run - the reason the Unreal side forgets a modal-dialog refusal.

THE BOUNDS, and why each. An id is a retry handle, not a history, and the memory is the user's Blender's:
  KEEP_SECONDS      600 s after a call ends - the MCP server's _RID_KEEP_SECONDS, the window in which it reuses an
                    id for the identical call made again; past it the MCP server sends a fresh id anyway
  MAX_RECORDS       1024 - the MCP server sends an id on every call, so a busy session makes hundreds a minute
  MAX_REPLY_BYTES   1 MiB - a reply over it is not kept (the record keeps ok and error, and a repeat answers alreadyRan
                    instead of running again). Writes answer in a few KB; the replies that reach megabytes are reads
                    (read_mesh_data, list_objects of a big scene), which are safe to send again, and one 64 MiB read
                    must not push every write's record out
  MAX_TOTAL_BYTES   16 MiB of kept replies in all, the oldest dropped first (their records stay: ok and error)
The same numbers as the Unreal plugin's (MifRequestKeepSeconds, GRequestMaxRecords, GRequestMaxResponseChars,
GRequestMaxTotalChars), in bytes rather than characters. A running record is never dropped: its call is still going.

request_status reads a record without repeating anything (status). It is answered from the socket thread too, so it
reaches a Blender whose main thread is still inside the call it asks about.

NO bpy HERE: the socket threads call it, and only the socket threads and the main thread's job end touch it, under one
lock.
"""
import hashlib
import json
import re
import threading
import time

KEEP_SECONDS = 600.0
MAX_RECORDS = 1024
MAX_REPLY_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024

# Not recorded: reading a record must not make one.
UNRECORDED = frozenset({"request_status"})

_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_LOCK = threading.Lock()
_RECORDS = {}           # id -> record dict
_KEPT = [0]             # bytes of kept replies


def valid(rid):
    """None when rid is a request id, else the refusal's text. The Unreal header's form: 1-64 of letters, digits - _ . :"""
    if isinstance(rid, str) and _ID.match(rid):
        return None
    return ("requestId %r is not a request id: 1-64 characters of letters, digits and - _ . : - nothing ran. Send a "
            "fresh random id per call, and the same id only to repeat that call." % (str(rid)[:80],))


def digest(endpoint, params):
    """What makes two calls the same call: the endpoint and the params, key order aside."""
    try:
        text = json.dumps({"e": endpoint, "p": params}, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = repr((endpoint, params))
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _prune(now):
    for rid in [r for r, rec in _RECORDS.items() if rec["done"] and now - rec["endedAt"] > KEEP_SECONDS]:
        _KEPT[0] -= len(_RECORDS.pop(rid).get("reply") or b"")
    done = sorted((r for r, rec in _RECORDS.items() if rec["done"]), key=lambda r: _RECORDS[r]["startedAt"])
    while len(_RECORDS) > MAX_RECORDS and done:
        _KEPT[0] -= len(_RECORDS.pop(done.pop(0)).get("reply") or b"")
    if _KEPT[0] > MAX_TOTAL_BYTES:
        for rid in done:
            rec = _RECORDS[rid]
            if rec.get("reply"):
                _KEPT[0] -= len(rec["reply"])
                rec["reply"] = None
                if _KEPT[0] <= MAX_TOTAL_BYTES:
                    break


def begin(rid, endpoint, params, agent=None):
    """Before the request runs. None = run it (the id is now recorded as running); a dict = the whole answer."""
    now = time.monotonic()
    key = digest(endpoint, params)
    with _LOCK:
        _prune(now)
        rec = _RECORDS.get(rid)
        if rec is None:
            _RECORDS[rid] = {"endpoint": endpoint, "agent": agent, "digest": key, "startedAt": now,
                             "startedUtc": time.time(), "done": False}
            return None
        if rec["endpoint"] != endpoint or rec["digest"] != key:
            return {"ok": False, "requestIdConflict": True, "requestId": rid, "firstEndpoint": rec["endpoint"],
                    "error": "request id '%s' was already used %.0f s ago for a different call (%s%s) - nothing ran. A "
                             "request id names ONE call: send a fresh one for a new call, and repeat an id only with "
                             "the identical endpoint and params."
                             % (rid, now - rec["startedAt"], rec["endpoint"],
                                ", with other params" if rec["endpoint"] == endpoint else "")}
        if not rec["done"]:
            return {"ok": False, "requestRunning": True, "retryable": True, "requestId": rid,
                    "error": "the call with request id '%s' (%s) is still running, %.1f s in - it was NOT started "
                             "again. Its reply is kept: repeat this call or ask request_status {requestId} once it "
                             "finishes." % (rid, endpoint, now - rec["startedAt"])}
        out = None
        if rec.get("reply"):
            try:
                out = json.loads(rec["reply"].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                out = None
        if not isinstance(out, dict):
            out = {"ok": False, "alreadyRan": True, "firstOk": rec["ok"],
                   "error": "the call with request id '%s' (%s) already ran %.0f s ago and answered ok:%s%s - it was "
                            "NOT run again. Its reply (%d bytes) was not kept, so read the target to see what it did."
                            % (rid, endpoint, now - rec["endedAt"], "true" if rec["ok"] else "false",
                               " (%s)" % rec["error"][:200] if rec.get("error") else "", rec.get("replyBytes") or 0)}
        out.update(replayed=True, requestId=rid, firstRunSecondsAgo=round(now - rec["endedAt"]))
        return out


def end(rid, body, response):
    """After the request ran: `body` is its reply as framed for the wire (bytes, without the length header), `response`
    the dict it came from. A refusal that ran nothing is forgotten instead (see the module docstring)."""
    if not rid:
        return
    if isinstance(response, dict) and response.get("callerGaveUp") is True:
        forget(rid)
        return
    now = time.monotonic()
    with _LOCK:
        rec = _RECORDS.get(rid)
        if rec is None or rec["done"]:
            return
        rec.update(done=True, endedAt=now, endedUtc=time.time(),
                   ok=isinstance(response, dict) and response.get("ok") is True,
                   error=str(response.get("error"))[:300] if isinstance(response, dict) and response.get("error")
                   else None,
                   replyBytes=len(body) if body is not None else 0)
        if body is not None and len(body) <= MAX_REPLY_BYTES:
            rec["reply"] = bytes(body)
            _KEPT[0] += len(body)
        else:
            rec["reply"] = None
        _prune(now)


def end_unsent(rid, response):
    """end() for a reply nobody will frame - a window Blender's job that finished after its waiter gave up."""
    if not rid:
        return
    try:
        body = json.dumps(response).encode("utf-8")
    except (TypeError, ValueError):
        body = None
    end(rid, body, response)


def forget(rid):
    with _LOCK:
        rec = _RECORDS.pop(rid, None)
        if rec is not None:
            _KEPT[0] -= len(rec.get("reply") or b"")


def running(rid):
    with _LOCK:
        rec = _RECORDS.get(rid)
        return rec is not None and not rec["done"]


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000)


def status(rid, include=False):
    """What happened to a call by its (valid) requestId - state running, done or unknown. Changes nothing."""
    out = {"ok": True, "endpoint": "request_status", "requestId": rid, "keepSeconds": KEEP_SECONDS,
           "bounds": {"records": MAX_RECORDS, "replyBytes": MAX_REPLY_BYTES, "totalBytes": MAX_TOTAL_BYTES}}
    now = time.monotonic()
    with _LOCK:
        _prune(now)
        rec = _RECORDS.get(rid)
        if rec is None:
            out.update(state="unknown",
                       note="this Blender has no record of that id: the call never reached it, ran nothing (its caller "
                            "had stopped waiting, or it was cancelled before it started), finished more than %.0f s "
                            "ago, or Blender has restarted since (records are in memory; the call journal's start "
                            "records are on disk - bl_call_log disk=true). Read the target the call would have changed "
                            "to settle it." % KEEP_SECONDS)
            return out
        out.update(requestEndpoint=rec["endpoint"], startedAt=_iso(rec["startedUtc"]))
        if rec.get("agent"):
            out["agent"] = rec["agent"]
        if not rec["done"]:
            out.update(state="running", seconds=round(now - rec["startedAt"], 1))
            return out
        out.update(state="done", endedAt=_iso(rec["endedUtc"]), seconds=round(rec["endedAt"] - rec["startedAt"], 1),
                   resultOk=rec["ok"], responseKept=rec.get("reply") is not None,
                   responseBytes=rec.get("replyBytes") or 0)
        if rec.get("error"):
            out["resultError"] = rec["error"]
        reply = rec.get("reply")
    if include and reply is not None:
        try:
            out["result"] = json.loads(reply.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            pass
    return out


def answer_status(params):
    """request_status as the socket thread answers it (server._handle_client), so it is read while the main thread is
    still inside the call it asks about. ops_scene.op_request_status is the op-table entry. Changes nothing."""
    from .ops_common import reject_unknown, take_bool
    reject_unknown(params, {"requestId", "includeResult"}, "request_status",
                   {"id": "spell it requestId", "rid": "spell it requestId"})
    return status_for(params.get("requestId"), take_bool(params, "includeResult", default=False))


def status_for(rid, include=False):
    """request_status's answer for a requestId as sent: refused when missing or malformed, else status()."""
    from .ops_common import MifOpError
    if isinstance(rid, str):
        rid = rid.strip()
    if not rid:
        raise MifOpError("request_status needs requestId - the requestId the call carried (a timed-out reply from the "
                         "MCP server names it)")
    bad = valid(rid)
    if bad:
        raise MifOpError(bad)
    return status(rid, include)
