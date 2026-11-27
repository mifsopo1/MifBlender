"""Roblox Open Cloud from the MCP side: the Assets API (POST /assets/v1/assets, then GET /assets/v1/operations/{id}).

The person's own API key is read from a file only (MIF_ROBLOX_OPENCLOUD_KEY_FILE, default
%LOCALAPPDATA%/MifBridge/roblox_opencloud.key). It is never returned, printed, logged or put in an error: every text that
leaves this module passes through redact() first, and an answer says only whether a key was found and how long it is.

Shapes are Roblox's documented ones (create.roblox.com/docs/cloud/guides/usage-assets): a multipart form with a `request`
JSON part ({assetType, displayName, description, creationContext: {creator: {userId}}}) and a `fileContent` part with
the file's content type; the answer names an operation `operations/{id}`, polled until `done`, whose `response` carries
assetId and moderationResult.moderationState.

NEVER RAISE: every failure is {"ok": False, ...}. An upload is permanent and moderated, so a caller must confirm it.
"""
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = "https://apis.roblox.com"
RECHECKS = 3                  # re-reads of an operation that ended with no response, RECHECK_EVERY seconds apart
RECHECK_EVERY = 5.0
MAX_BYTES = 20 * 1024 * 1024  # Roblox: up to 20 MB an asset a call ...
VIDEO_MAX_BYTES = 3750 * 1024 * 1024  # ... except Video: up to 3.75 GB, 5 minutes, 4096x2160 (usage-assets page)
# Extension -> (content type, the assetType it uploads as by default, every assetType Roblox's usage page takes it as).
# The page (create.roblox.com/docs/cloud/guides/usage-assets, read 2026-11-20): Animation .rbxm/.rbxmx; Audio .mp3 .ogg
# .wav .flac; Decal and Image .png .jpeg .bmp .tga (under 8000 x 8000); Model .fbx .gltf .glb .rbxm .rbxmx; Video .mp4 .mov.
CONTENT = {
    ".fbx": ("model/fbx", "Model", ("Model",)),
    ".gltf": ("model/gltf+json", "Model", ("Model",)),
    ".glb": ("model/gltf-binary", "Model", ("Model",)),
    ".rbxm": ("model/x-rbxm", "Model", ("Model", "Animation")),
    ".rbxmx": ("model/x-rbxm", "Model", ("Model", "Animation")),
    ".png": ("image/png", "Decal", ("Decal", "Image")),
    ".jpg": ("image/jpeg", "Decal", ("Decal", "Image")),
    ".jpeg": ("image/jpeg", "Decal", ("Decal", "Image")),
    ".bmp": ("image/bmp", "Decal", ("Decal", "Image")),
    ".tga": ("image/tga", "Decal", ("Decal", "Image")),
    ".mp3": ("audio/mpeg", "Audio", ("Audio",)),
    ".ogg": ("audio/ogg", "Audio", ("Audio",)),
    ".wav": ("audio/wav", "Audio", ("Audio",)),
    ".flac": ("audio/flac", "Audio", ("Audio",)),
    ".mp4": ("video/mp4", "Video", ("Video",)),
    ".mov": ("video/mov", "Video", ("Video",)),
}
# Roblox's numeric asset type ids (Enum.AssetType values), used by the inventory read that settles an unknown outcome.
ASSET_TYPE_IDS = {"Image": 1, "Audio": 3, "Mesh": 4, "Decal": 13, "Model": 10, "Animation": 24, "Plugin": 38, "Video": 62}


# A suite replaces this with a fake: (method, url, headers, body, timeout) -> (status, headers dict, body bytes).
def _urlopen(method, url, headers, body, timeout):
    if isinstance(body, FileBody):  # a large video: streamed from disk, a fresh reader per try
        headers = dict(headers, **{"Content-Length": str(len(body))})
        body = body.reader()
    req =urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read() or b""


REQUEST = _urlopen


def key_file():
    return os.environ.get("MIF_ROBLOX_OPENCLOUD_KEY_FILE") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MifBridge", "roblox_opencloud.key")


def read_key():
    """(key, None) or (None, a refusal). The refusal names the file, never any of its content. Never raises.

    The file may be UTF-8, UTF-8 with a BOM (Notepad, Set-Content -Encoding utf8) or UTF-16 with a BOM (Windows
    PowerShell 5.1's > and Out-File). Found 2026-11-22: a BOM read as utf-8 stayed in the key, the header encode then
    raised with the key in its text, and a UTF-16 file raised UnicodeDecodeError whose text carried the file's bytes. So
    the key is decoded from the bytes by its BOM, and a key that is not printable ASCII is refused without its content."""
    path = key_file()
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return None, {"ok": False, "ran": False, "refusedRule": "no-key", "keyFile": path,
                      "error": "no Open Cloud key file at %s (%s): the person saves their own key there" % (path, exc.strerror)}
    try:
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            key = raw.decode("utf-16").strip()
        else:
            key = raw.decode("utf-8-sig").strip()
    except (UnicodeError, ValueError):
        key = "\x00"     # undecodable: refused below as not printable ASCII, with none of its bytes
    del raw
    if not key or any(c.isspace() for c in key):
        return None, {"ok": False, "ran": False, "refusedRule": "no-key", "keyFile": path,
                      "error": "the key file at %s is empty or holds more than one line" % path}
    if any(not 0x21 <= ord(c) <= 0x7e for c in key):
        return None, {"ok": False, "ran": False, "refusedRule": "no-key", "keyFile": path,
                      "error": "the key file at %s does not read as printable ASCII (an API key is): save it as UTF-8 with "
                               "the key alone on one line" % path}
    return key, None


def exc_text(exc, key=None):
    """An exception as safe text for an answer: its type name and str(), with the key (read here when not given)
    redacted. Never repr(): a repr escapes characters, and an escaped key no longer matches a redaction."""
    if key is None:
        key, _e = read_key()
    try:
        text = "%s: %s" % (type(exc).__name__, exc)
    except Exception:  # noqa: BLE001 - a str() that raises is named by its type alone
        text = type(exc).__name__
    return redact(text, key)[:600]


# ---- which experiences a write may reach. A key's scopes usually cover every universe the person owns ("*", measured
# 2026-11-21 on two keys), so the key cannot keep a write off a live game: this list does. A write (a data store write,
# a publish, a setting change, a cloud Luau task) names its universe and place, and each must be on the person's list in
# roblox_write_targets.json, or named by the caller in that call's allow_ids. Nothing listed means every write needs it.
def write_targets_file():
    return os.environ.get("MIF_ROBLOX_WRITE_TARGETS_FILE") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MifBridge", "roblox_write_targets.json")


def write_targets():
    """{"universes": set, "places": set} from the file ({"universes": [ids], "places": [ids]}); empty when missing or
    unreadable (closed, never open)."""
    out = {"universes": set(), "places": set()}
    try:
        with open(write_targets_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return out
    for k in out:
        for v in (data.get(k) if isinstance(data, dict) else None) or []:
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                out[k].add(v)
    return out


def write_target_check(universe_id=None, place_id=None, allow_ids=None):
    """None when every id the write reaches (a universe, a place) is on the write-target list or in allow_ids, else the
    refusal (refusedRule write-target) naming each id that is not. An id that could not be read is refused too."""
    if allow_ids is not None and not (isinstance(allow_ids, list) and all(
            isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in allow_ids)):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "allow_ids is a list of universe and place ids"}
    listed, named = write_targets(), set(allow_ids or [])
    bad = []
    for kind, v in (("universe", universe_id), ("place", place_id)):
        if v is False:
            continue
        if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            bad.append("the %s this write reaches did not read" % kind)
        elif v not in listed[kind + "s"] and v not in named:
            bad.append("%s %d" % (kind, v))
    if not bad:
        return None
    return {"ok": False, "ran": False, "refusedRule": "write-target", "sent": False, "notAllowed": bad,
            "targetsFile": write_targets_file(),
            "error": "this write would reach %s, which is not on the write-target list (%s) and not named in allow_ids: "
                     "nothing was sent. The key's scopes cover every experience, so this list is what keeps a write off a "
                     "live game" % (", ".join(bad), write_targets_file())}


# "Save failed. Server is busy ... PlaceOngoingTeamCreateSession": an Open Cloud publish (Assets API) and SavePlaceAsync
# answered it while a Studio had the place open from Roblox (RB1967), and SavePlaceAsync from a local copy answered it 31 s
# and 56 s after a Studio that held the place's id was stopped, then saved 5.6 and 6.7 minutes after one (RB2002): the
# hold outlives the Studio by minutes. The place had Team Create on throughout, which does not block a save by itself.
SESSION_HELD = "PlaceOngoingTeamCreateSession"
SESSION_STEP = ("close every Studio that has this place open from Roblox (its session holds the place), wait about five "
                "minutes - the hold outlives a stopped Studio (RB2002) - then publish again")


def session_held(answer):
    """True when an answer (an operation's error, an exception text) names the place's edit session."""
    return SESSION_HELD in (answer if isinstance(answer, str) else json.dumps(answer, default=str))


def redact(value, key):
    """Any string inside value with the key replaced. Applied to everything this module returns."""
    if not key:
        return value
    if isinstance(value, str):
        # the key as written and as a repr or an escape writes it (read_key admits printable ASCII only, where these are
        # the same; kept so a key read some other way still cannot pass escaped)
        for form in {key, repr(key)[1:-1], key.encode("unicode_escape").decode("ascii")}:
            value = value.replace(form, "<redacted key>")
        return value
    if isinstance(value, dict):
        return {redact(k, key): redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    return value


def _json(body):
    try:
        return json.loads(body.decode("utf-8") or "null")
    except (ValueError, UnicodeDecodeError):
        return {"raw": body[:500].decode("utf-8", "replace")}


def outcome_unknown(status):
    """True for an answer that does not say whether the server acted: no answer at all (0: a timeout, a reset) or a 5xx
    (a gateway can answer 502/504 after the service took the request)."""
    return status == 0 or status >= 500


RETRY_AFTER_MAX = 60.0     # seconds one Retry-After may make a call wait


def _call(method, path, key, body=None, content_type=None, timeout=60, tries=5, base=BASE, idempotent=None):
    """One API call with the key header, backing off on 429 and 5xx. (status, json, seconds waited for limits).

    A call that is not idempotent (by default every POST: an asset create, a data store increment, a Luau task create) is
    retried only on 429, where nothing was done. On no answer or a 5xx its outcome is unknown, and sending it again could
    make it land twice (found 2026-11-22), so it is answered as it came and the caller settles it by a read
    (outcome_unknown). idempotent True marks a POST that only reads (key introspection)."""
    if idempotent is None:
        idempotent = method.upper() != "POST"
    headers = {"x-api-key": key, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    waited = 0.0
    status, out = 0, None
    for attempt in range(tries):
        try:
            status, h, raw = REQUEST(method, base + path, headers, body, timeout)
        except (OSError, http.client.HTTPException) as exc:
            # a dropped or garbled answer (RemoteDisconnected, IncompleteRead, BadStatusLine) is no answer: status 0,
            # outcome unknown, never a raise out of the tool (found 2026-11-22)
            status, h, raw = 0, {}, ("%s: %s" % (type(exc).__name__, exc)).encode("utf-8")
        # Redact at the source, before any text is cut short: a truncated message would keep a part of the key that a
        # whole-key redaction later no longer matches (the offline suite caught exactly that).
        raw = raw.replace(key.encode("utf-8"), b"<redacted key>")
        out = _json(raw)
        if status == 429 or (idempotent and outcome_unknown(status)):
            retry = h.get("Retry-After") or h.get("retry-after")
            delay = float(retry) if retry and str(retry).replace(".", "", 1).isdigit() else min(30.0, 2.0 ** attempt)
            # a server's Retry-After is honored up to RETRY_AFTER_MAX: an hour's answer must not hang the tool an hour
            delay = min(delay, RETRY_AFTER_MAX)
            if attempt + 1 < tries:
                time.sleep(delay)
                waited += delay
                continue
        break
    return status, out, waited


class FileBody:
    """A multipart body whose file part stays on disk (a video may be 3.75 GB): head bytes, the file, tail bytes."""

    def __init__(self, head, path, tail):
        self.head, self.path, self.tail = head, path, tail

    def __len__(self):
        return len(self.head) + os.path.getsize(self.path) + len(self.tail)

    def reader(self):
        import io
        body = self

        class _Reader(io.RawIOBase):
            def __init__(self):
                super().__init__()
                self.parts = [io.BytesIO(body.head), open(body.path, "rb"), io.BytesIO(body.tail)]

            def readable(self):
                return True

            def readinto(self, b):
                while self.parts:
                    n = self.parts[0].readinto(b)
                    if n:
                        return n
                    self.parts.pop(0).close()
                return 0
        return io.BufferedReader(_Reader(), 1 << 20)


def multipart(request_json, file_name, content_type, data):
    """(body, content type). data is the file's bytes, or its path to stream it (the body is then a FileBody)."""
    boundary = "MifBridge" + uuid.uuid4().hex
    parts = [
        ("--%s\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n" % boundary).encode(),
        json.dumps(request_json).encode("utf-8"),
        ("\r\n--%s\r\nContent-Disposition: form-data; name=\"fileContent\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
         % (boundary, os.path.basename(file_name).replace('"', ""), content_type)).encode(),
        data,
        ("\r\n--%s--\r\n" % boundary).encode(),
    ]
    ctype = "multipart/form-data; boundary=%s" % boundary
    if isinstance(data, str):
        return FileBody(b"".join(parts[:3]), data, parts[4]), ctype
    return b"".join(parts), ctype


def upload(file, display_name, user_id, description="", asset_type=None, poll_timeout=300, poll_every=2.0):
    """Create one asset and poll its operation to the end. Never raises; never returns the key."""
    key, err = read_key()
    if err:
        return err
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext not in CONTENT:
            return {"ok": False, "ran": False, "refusedRule": "type",
                    "error": "%s is not a format the Assets API lists; it takes %s" % (ext or "no extension", ", ".join(sorted(CONTENT)))}
        content_type, default_type, allowed = CONTENT[ext]
        asset_type = asset_type or default_type
        if asset_type not in allowed:
            return {"ok": False, "ran": False, "refusedRule": "type",
                    "error": "a %s file uploads as %s, not %s (Roblox's usage page)" % (ext, " or ".join(allowed), asset_type)}
        size = os.path.getsize(file)
        cap = VIDEO_MAX_BYTES if asset_type == "Video" else MAX_BYTES
        if size > cap:
            return {"ok": False, "ran": False, "refusedRule": "size", "bytes": size,
                    "error": "%d bytes is over the Assets API's %s a call for %s" % (
                        size, "3.75 GB" if asset_type == "Video" else "20 MB", asset_type)}
        if size > MAX_BYTES:
            data = file  # streamed from disk, never held whole
        else:
            with open(file, "rb") as f:
                data = f.read()
        req = {"assetType": asset_type, "displayName": display_name, "description": description or "",
               "creationContext": {"creator": {"userId": str(user_id)}}}
        body, ctype = multipart(req, file, content_type, data)
        del data
        t0 = time.time()
        status, created, waited = _call("POST", "/assets/v1/assets", key, body, ctype, timeout=120)
        out = {"request": req, "file": file, "bytes": size, "contentType": content_type, "createStatus": status,
               "rateLimitWaitSeconds": waited}
        if outcome_unknown(status):
            # Sent once, never again (_call): Roblox may have taken it. Settle it from the creator's inventory; anything
            # but a listed asset is unknown, so the upload log's made "unknown" holds a retry back until it is checked.
            found = settle_by_listing(user_id, asset_type, display_name, t0)
            out["listing"] = found
            if found.get("made") == "yes":
                out.update(ok=True, ran=True, made="yes", madeOnListing=True, assetId=str(found["assetId"]),
                           assetType=asset_type, moderationState=None,
                           note="the create answered %s; the asset was found by name in the creator's inventory, created "
                                "after this upload began (moderation unread: GET /assets/v1/assets/{id})" % status)
                return redact(out, key)
            out.update(ok=False, ran=None, outcomeUnknown=True, made="unknown", created=created,
                       error="POST /assets/v1/assets answered %s (no answer, or a server error): Roblox may have made the "
                             "asset. It was sent once and not again; the creator's inventory %s. Check Creator Hub before "
                             "uploading this file again (retry_unknown true then)"
                             % (status or "nothing", "did not list it" if found.get("made") == "no"
                                else "could not be read (%s)" % found.get("error")))
            return redact(out, key)
        if status != 200 or not isinstance(created, dict) or not created.get("path"):
            price = PRICE.search(json.dumps(created)) if isinstance(created, (dict, list)) else None
            if price:
                # Measured 2026-11-20: a Video upload answers 400 INVALID_ARGUMENT "Actual price (2000 Robux) is larger
                # than expected price (0 Robux)." Nothing is made or charged. The bridge never offers a price.
                out.update(ok=False, ran=True, created=created, refusedRule="costs-robux", made="no",
                           priceRobux=int(price.group(1)),
                           error="Roblox charges %s Robux for this %s upload and refused it at the expected price of 0: "
                                 "nothing was made or charged. MifBridge never pays; the person uploads it in Creator Hub "
                                 "if they want to pay" % (price.group(1), asset_type))
                return redact(out, key)
            out.update(ok=False, ran=True, created=created,
                       error="POST /assets/v1/assets answered %s: %s" % (status, json.dumps(created)[:600]))
            return redact(out, key)
        op_path = created["path"]
        out["operation"] = op_path
        op = created
        polls = 0
        while not op.get("done") and time.time() - t0 < poll_timeout:
            time.sleep(poll_every)
            polls += 1
            st, op, w2 = _call("GET", "/assets/v1/" + op_path, key)
            out["rateLimitWaitSeconds"] += w2
            if st != 200 or not isinstance(op, dict):
                out.update(ok=False, ran=True, outcomeUnknown=True, polls=polls, lastPoll={"status": st, "body": op},
                           error="polling %s answered %s; the asset may still be made - poll it again" % (op_path, st))
                return redact(out, key)
        out.update(seconds=round(time.time() - t0, 2), polls=polls, done=bool(op.get("done")))
        if not op.get("done"):
            out.update(ok=False, ran=True, outcomeUnknown=True,
                       error="the operation was not done after %ss; poll %s again" % (poll_timeout, op_path))
            return redact(out, key)
        resp = op.get("response") or {}
        if not resp:
            # Seen once (spec "an operation that ends without a response is looked up"): done, no response, "Unknown
            # Error" - and a retry by name made a second asset, so the first may have made one too. Read the operation
            # again before calling it anything; if it still has no response, the outcome is unknown, never "failed".
            for _ in range(RECHECKS):
                time.sleep(RECHECK_EVERY)
                st, again, w3 = _call("GET", "/assets/v1/" + op_path, key)
                out["rateLimitWaitSeconds"] += w3
                out["rechecks"] = out.get("rechecks", 0) + 1
                if st == 200 and isinstance(again, dict) and (again.get("response") or {}).get("assetId"):
                    op, resp = again, again["response"]
                    out["madeOnRecheck"] = True
                    break
        if not resp:
            found = settle_by_listing(user_id, asset_type, display_name, t0)
            out["listing"] = found
            if found.get("made") == "yes":
                out.update(ok=True, ran=True, made="yes", madeOnListing=True, assetId=str(found["assetId"]),
                           assetType=asset_type, moderationState=None,
                           note="the operation ended with no response; the asset was found by name in the creator's "
                                "inventory, created after this upload began (moderation unread: rbx_asset_upload "
                                "moderation_wait, or GET /assets/v1/assets/{id})")
                return redact(out, key)
            if found.get("made") == "no":
                out.update(ok=False, ran=True, made="no", operationResult=op,
                           error="the operation finished without a response (%s) and nothing named %r was in the "
                                 "creator's %s inventory %ss later: not made. A retry is allowed"
                                 % (json.dumps(op.get("error") or op)[:300], display_name, asset_type,
                                    found.get("afterSeconds")))
                return redact(out, key)
            out.update(ok=False, ran=True, outcomeUnknown=True, made="unknown", operationResult=op,
                       error="the operation finished without a response (%s) and %d re-reads of %s found no asset: it may "
                             "or may not have made one. A retry of this file and name is refused until the person checks "
                             "Creator Hub (retry_unknown true then)" % (json.dumps(op.get("error") or op)[:300],
                                                                        out.get("rechecks", 0), op_path))
            return redact(out, key)
        mod = (resp.get("moderationResult") or {}).get("moderationState")
        out.update(ok=bool(resp.get("assetId")), ran=True, assetId=resp.get("assetId"), revisionId=resp.get("revisionId"),
                   assetType=resp.get("assetType"), moderationState=mod, response=resp)
        return redact(out, key)
    except Exception as exc:  # noqa: BLE001 - never raise, and never let an exception text carry the key out
        return redact({"ok": False, "ran": None, "outcomeUnknown": True,
                       "error": "upload failed: %s" % exc_text(exc, key)}, key)


PRICE = __import__("re").compile(r"Actual price \((\d+) Robux\)")
INVENTORY = "https://inventory.roblox.com"


def recent_creations(user_id, asset_type, limit=25):
    """The creator's newest items of one asset type, newest first: inventory.roblox.com GET
    /v2/users/{userId}/inventory/{assetTypeId}?limit=&sortOrder=Desc. A public read: the API key is NEVER sent there
    (measured 2026-11-20: Open Cloud's own lists - itemconfiguration /v1/creations/get-assets and
    /asset-delivery-api - answer 403 "Scope not authorized." to a key with asset read and write). Rows: assetId,
    assetName, created (ISO). It answers only while the person's inventory is visible to others (Roblox privacy
    setting); otherwise ok false with the status."""
    tid = ASSET_TYPE_IDS.get(asset_type)
    if tid is None or asset_type in ("Image", "Mesh"):
        return {"ok": False, "error": "%s assets are not listed in an inventory" % asset_type}
    if limit not in (10, 25, 50, 100):
        limit = 25
    try:
        st, _h, raw = REQUEST("GET", "%s/v2/users/%d/inventory/%d?limit=%d&sortOrder=Desc"
                              % (INVENTORY, int(user_id), tid, limit), {"Accept": "application/json"}, None, 30)
    except Exception as exc:  # noqa: BLE001 - never raise: it settles an upload whose outcome is already unknown
        return {"ok": False, "status": 0, "error": "the inventory read failed: %s" % type(exc).__name__}
    body = _json(raw)
    if st != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return {"ok": False, "status": st, "error": "the inventory read answered %s: %s" % (st, json.dumps(body)[:300])}
    rows = [{"assetId": r.get("assetId"), "assetName": r.get("assetName"), "created": r.get("created")}
            for r in body["data"] if isinstance(r, dict)]
    return {"ok": True, "assetType": asset_type, "rows": rows}


def _iso_seconds(text):
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


LIST_RECHECKS = 2       # inventory reads before "not made", LIST_EVERY seconds apart
LIST_EVERY = 10.0
CLOCK_SLACK = 120.0     # seconds an asset's created time may read before this machine's clock at the upload's start


def settle_by_listing(user_id, asset_type, display_name, started, known_ids=None):
    """Was an asset made by an upload whose operation gave no answer? -> {made yes, assetId} when the creator's inventory
    lists one named display_name created after the upload began (less CLOCK_SLACK), {made no, afterSeconds} when
    LIST_RECHECKS reads LIST_EVERY seconds apart list none, {made unknown, error} when the list cannot be read.
    Measured 2026-11-20: an upload is in the list at once (its created time 0.2 s after the operation's
    revisionCreateTime), newest first."""
    out = {"made": "unknown", "reads": 0}
    # an asset of the same name already logged as made by an earlier upload is not this one's (found 2026-11-22: a
    # same-named asset made inside the clock slack was taken for the upload whose answer was lost)
    known = {str(i) for i in (known_ids if known_ids is not None else
                              [r.get("assetId") for r in _rows() if r.get("made") == "yes" and r.get("assetId")])}
    for i in range(LIST_RECHECKS):
        if i:
            time.sleep(LIST_EVERY)
        got = recent_creations(user_id, asset_type)
        out["reads"] += 1
        if not got.get("ok"):
            out.update(made="unknown", error=got.get("error"))
            return out
        for r in got["rows"]:
            c = _iso_seconds(r.get("created"))
            if r.get("assetName") == display_name and c is not None and c >= started - CLOCK_SLACK \
                    and str(r.get("assetId")) not in known:
                out.update(made="yes", assetId=r.get("assetId"), created=r.get("created"))
                return out
    out.update(made="no", afterSeconds=round(time.time() - started, 1))
    return out


def asset_state(asset_id):
    """One asset's state and moderation: GET /assets/v1/assets/{id}?readMask=... (scope asset:read). Never the key."""
    key, err = read_key()
    if err:
        return err
    st, out, _w = _call("GET", "/assets/v1/assets/%d?readMask=assetId,assetType,displayName,state,moderationResult,"
                               "revisionId,revisionCreateTime" % int(asset_id), key, tries=3)
    if st != 200 or not isinstance(out, dict):
        return redact({"ok": False, "status": st, "error": "GET /assets/v1/assets/%s answered %s: %s"
                       % (asset_id, st, json.dumps(out)[:300])}, key)
    return redact({"ok": True, "assetId": out.get("assetId"), "assetType": out.get("assetType"), "state": out.get("state"),
                   "moderationState": (out.get("moderationResult") or {}).get("moderationState"),
                   "revisionId": out.get("revisionId"), "revisionCreateTime": out.get("revisionCreateTime")}, key)


def uploads_log():
    from mif_roblox import daemon as D
    return os.path.join(D.home(), "uploads.jsonl")


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_log():
    """(rows, bad line numbers, error or None) of uploads.jsonl, line by line. A line that is not a JSON object (a write
    torn by a crash, a hand edit) is counted and skipped; it never empties the log, which until 2026-11-23 switched off
    every duplicate guard at once. A missing log is empty and whole; an unreadable one is the error."""
    rows, bad = [], []
    try:
        with open(uploads_log(), "rb") as fh:
            for n, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    row = None
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    bad.append(n)
    except FileNotFoundError:
        return [], [], None
    except OSError as exc:
        return [], [], "%s could not be read (%s)" % (uploads_log(), type(exc).__name__)
    return rows, bad, None


def _rows():
    return read_log()[0]


def log_damage():
    """None when uploads.jsonl reads whole, else the refusal a guarded write answers: the duplicate guards read that log,
    so a damaged one could let the same asset be made twice."""
    _rows_, bad, err = read_log()
    if not bad and not err:
        return None
    return {"ok": False, "ran": False, "refusedRule": "log-damaged", "logDamaged": True, "badLines": bad[:20],
            "uploadsLog": uploads_log(),
            "error": err or ("uploads.jsonl has %d line(s) that are not a record (line %s): the duplicate guards read that "
                             "log, so nothing was sent" % (len(bad), ", ".join(str(n) for n in bad[:20]))),
            "steps": ["open %s, fix or delete the named line(s) (each record is one JSON object per line), then call again"
                      % uploads_log()]}


def prior_unknown(digest, display_name):
    """The last logged upload of this file and name, when its outcome is unknown (made "unknown") and no later upload of
    the same pair settled it; else None."""
    last = None
    for r in _rows():
        if r.get("sha256") == digest and r.get("displayName") == display_name:
            last = r
    return last if last and last.get("made") == "unknown" else None


def default_type(file):
    """The asset type a file uploads as when none is given (CONTENT), or None."""
    row = CONTENT.get(os.path.splitext(str(file))[1].lower())
    return row[1] if row else None


def prior_made(digest, mesh_hash, asset_type, creator=None):
    """The newest logged upload that made an asset of this type from the same file sha256 or the same mesh data, for
    the same creator, else None. A row written before rows carried assetType matches on its sha256 only. A row of
    another creator never matches; one written before rows carried the creator still matches (sending a duplicate
    makes a permanent asset), and the caller says the creator was not recorded (creatorUnknown)."""
    hit = None
    for r in _rows():
        if r.get("made") != "yes" or not r.get("assetId") or r.get("route"):
            continue
        if creator is not None and r.get("creator") is not None and str(r.get("creator")) != str(creator):
            continue
        if r.get("assetType") not in (None, asset_type):
            continue
        if r.get("sha256") == digest or (mesh_hash and r.get("meshHash") == mesh_hash and r.get("assetType") == asset_type):
            hit = r
    return hit


def log_upload(digest, display_name, out, mesh_hash=None, asset_type=None, route=None, creator=None):
    """One line per upload that reached Roblox: never the key (out has been redacted), never the file's bytes. route
    "studio" marks a save from Studio (rbx_save_to_roblox, rbx_plugin_upload: CreateAssetAsync, not Open Cloud)."""
    if not isinstance(out, dict) or out.get("ran") is False:
        return
    made = "yes" if out.get("assetId") else ("unknown" if out.get("made") == "unknown" or out.get("outcomeUnknown")
                                            else "no")
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sha256": digest, "displayName": display_name, "made": made,
           "assetId": out.get("assetId"), "operation": out.get("operation"), "assetType": asset_type,
           "meshHash": mesh_hash}
    if creator is not None:
        row["creator"] = creator
    if route:
        row["route"] = route
    try:
        os.makedirs(os.path.dirname(uploads_log()), exist_ok=True)
        with open(uploads_log(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def introspect():
    """What the key may do: POST /api-keys/v1/introspect with the key in the JSON body (never a URL), a read. -> {ok,
    name, authorizedUserId, enabled, expired, expirationTimeUtc, scopes [{name, operations, resources}]}. The key is
    never in the answer."""
    key, err = read_key()
    if err:
        return err
    body = json.dumps({"apiKey": key}).encode("utf-8")
    st, out, _w = _call("POST", "/api-keys/v1/introspect", key, body, "application/json", timeout=30, tries=2,
                        idempotent=True)
    if st != 200 or not isinstance(out, dict):
        return redact({"ok": False, "ran": True, "status": st, "body": out,
                       "error": "POST /api-keys/v1/introspect answered %s: %s" % (st, json.dumps(out)[:400])}, key)
    scopes = []
    for s in out.get("scopes") or []:
        if not isinstance(s, dict):
            continue
        res = {k: v for k, v in s.items() if k not in ("name", "operations")}
        scopes.append({"name": s.get("name"), "operations": list(s.get("operations") or []), "resources": res})
    return redact({"ok": True, "name": out.get("name"), "authorizedUserId": out.get("authorizedUserId"),
                   "enabled": out.get("enabled"), "expired": out.get("expired"),
                   "expirationTimeUtc": out.get("expirationTimeUtc"), "scopes": scopes}, key)


def scope_summary():
    """The key's scopes as "name:operation" strings (from introspect), for a 401/403 answer to name what the key holds.
    Never the key, never the resource ids."""
    info = introspect()
    if not info.get("ok"):
        return {"error": info.get("error")}
    return sorted("%s:%s" % (s["name"], op) for s in info["scopes"] for op in s["operations"])


def has_scope(info, name, operation):
    """(True/False, the resources it covers) for one scope and operation in an introspect() answer."""
    for s in (info or {}).get("scopes") or []:
        if s.get("name") == name and operation in (s.get("operations") or []):
            return True, s.get("resources")
    return False, None


# ---- the names of an experience's secrets, never their content (Roblox's Secrets Store API, BETA: create.roblox.com/
# docs/cloud/reference/domains/apis, read 2026-11-22; not yet measured against a live experience). GET /cloud/v2/
# universes/{u}/secrets under universe.secret:read, limit 1-500 and cursor, answers {secrets [{id, secret, key_id,
# domain, create_time, update_time}], nextPageCursor}; an experience holds at most 500. Only each id is kept: the
# other fields are dropped here, so no caller can pass them on.
SECRETS_SCOPE = "universe.secret:read"
SECRETS_STEP = ("Creator Hub > Open Cloud > API Keys > this key > Access Permissions: add the secrets store with read "
                "(%s) for this experience, then Save." % SECRETS_SCOPE)
SECRET_PAGES = 4              # pages of 500 read before the list is called unread (an experience holds at most 500)


def secret_names(universe_id):
    """The names of the secrets an experience holds, with the person's key. -> {ok, names [sorted], status} or {ok False,
    status, error, refusedRule?, scope?, steps?}. A read; no secret's content, key id or domain is kept or answered."""
    key, err = read_key()
    if err:
        return err
    names, cursor = set(), None
    for _ in range(SECRET_PAGES):
        path = "/cloud/v2/universes/%d/secrets?limit=500" % int(universe_id)
        if cursor:
            path += "&cursor=%s" % urllib.parse.quote(str(cursor), safe="")
        st, body, _w = _call("GET", path, key, timeout=30, tries=3)
        rows = body.get("secrets") if isinstance(body, dict) else None
        if st != 200:
            out = {"ok": False, "ran": True, "status": st,
                   "error": "GET /cloud/v2/universes/%d/secrets answered %s: %s" % (universe_id, st, json.dumps(body)[:300])}
            if st in (401, 403):
                out.update(refusedRule="key-scope", scope=SECRETS_SCOPE, steps=[SECRETS_STEP],
                           error="the key may not list this experience's secrets (it needs %s): the list answered %s"
                                 % (SECRETS_SCOPE, st))
            return redact(out, key)
        if not isinstance(rows, list):  # the body is not echoed: a list in another shape could hold a secret's content
            return {"ok": False, "ran": True, "status": st,
                    "error": "GET /cloud/v2/universes/%d/secrets answered 200 with no secrets list" % universe_id}
        names.update(r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str))
        cursor = body.get("nextPageCursor")
        if not cursor:
            return redact({"ok": True, "ran": True, "status": 200, "names": sorted(names)}, key)
    return {"ok": False, "ran": True, "status": 200,
            "error": "more than %d pages of secrets: the list could not be read to its end" % SECRET_PAGES}


def operation(op_path):
    """Read one operation again (after an outcomeUnknown)."""
    key, err = read_key()
    if err:
        return err
    st, op, _w = _call("GET", "/assets/v1/" + op_path.lstrip("/"), key)
    return redact({"ok": st == 200, "status": st, "operation": op}, key)
