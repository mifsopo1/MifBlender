"""artifact_inventory: a read-only report over the artifact registry and any folder the caller names. Design: docs/35_ARTIFACTS.md.

IT CHANGES NOTHING. No file, no registry line, no preference, no undo step, and no reply to any report asks for a move: the
archive plan is text to review, and no op applies it. Deleting is out of scope here; delete a file by hand, or through a tool
that shows what goes first and waits for a yes.
"""
from __future__ import annotations

import hashlib
import os
import re

import bpy

from . import artifacts
from .ops_common import MifOpError, reject_unknown, take, take_bool

_KEYS = ("roots", "status", "limit", "offset", "hash", "archiveRoot", "maxFiles")
_CLASSES = ("referenced", "protected", "unknown", "candidate", "gone")
_HASH_BUDGET = 4 * 1024 ** 3
_NOT_CHECKED = [
    "unopened .blend files: a file one of them links is not seen as used",
    "other running Blender sessions and their unsaved work",
    "Unreal: nothing here reads an asset's import source or an Unreal project's files",
    "paths in scripts, drivers and render output settings",
    "files the record cannot know: animation frames, and anything made outside the participating ops",
]


def _session_references():
    """{canonical path: [{kind, name, users}]}, [(compiled pattern, row)], the linked libraries: what the OPEN session uses,
    read through manage_external_data's own rows so there is one reading of it. A UDIM set or an image sequence names many
    files through one path, so those are patterns."""
    from .ops_datablock import _EXT_KINDS, _external_rows
    exact, patterns, libraries = {}, [], set()
    for row in _external_rows(list(_EXT_KINDS)):
        ref = {"kind": row["kind"], "name": row["name"], "users": row["users"]}
        ab = row["absolutePath"]
        if "<UDIM>" in ab:
            patterns.append((re.compile(re.escape(artifacts.canonical(ab)).replace(re.escape(os.path.normcase("<UDIM>")),
                                                                                    r"\d{4}")), ref))
            continue
        block = getattr(getattr(bpy.data, _EXT_KINDS[row["kind"]], {}).get(row["name"]) or object(), "source", None)
        if row["kind"] == "IMAGE" and block == "SEQUENCE":
            patterns.append((re.compile(re.escape(artifacts.canonical(os.path.dirname(ab)) + os.sep) + r".*"), ref))
        key = artifacts.canonical(ab)
        exact.setdefault(key, []).append(ref)
        if row["kind"] == "LIBRARY":
            libraries.add(key)
    return exact, patterns, libraries


def _walk(roots, max_files, skip):
    """Every file under the named folders, links not followed and a git checkout not entered. (files, protected dirs, links
    not followed, unreadable, truncated)."""
    files, protected, links, unreadable = [], [], [], []

    def on_error(exc):
        unreadable.append(str(getattr(exc, "filename", exc)))

    for root in roots:
        for here, dirs, names in os.walk(root, followlinks=False, onerror=on_error):
            if ".git" in dirs or ".git" in names:
                protected.append(here)
                dirs[:] = []
                continue
            kept = []
            for d in dirs:
                full = os.path.join(here, d)
                is_junction = getattr(os.path, "isjunction", lambda _p: False)(full)
                if os.path.islink(full) or is_junction or _reparse(full):
                    links.append(full)
                else:
                    kept.append(d)
            dirs[:] = kept
            for name in names:
                full = os.path.join(here, name)
                if os.path.islink(full) or _reparse(full):
                    links.append(full)
                    continue
                if artifacts.canonical(full) in skip:
                    continue
                if len(files) >= max_files:
                    return files, protected, links, unreadable, True
                files.append(full)
    return files, protected, links, unreadable, False


def _reparse(path):
    """A Windows reparse point (a junction before Python 3.12 has os.path.isjunction): never crossed."""
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def _inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:                                                  # another drive
        return False


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_plan(rows, archive_root):
    plan, total, cross_drive = [], 0, 0
    drive = os.path.splitdrive(archive_root)[0].lower()
    for row in rows:
        if row["class"] != "candidate":
            continue
        d, rest = os.path.splitdrive(row["path"])
        dest = os.path.join(archive_root, d.replace(":", "") or "root", rest.lstrip("\\/"))
        same = d.lower() == drive
        total += row["bytes"]
        cross_drive += 0 if same else row["bytes"]
        plan.append({"source": row["path"], "proposedDestination": dest, "bytes": row["bytes"],
                     "knownReferences": [], "retention": row.get("purpose"),
                     "destinationExists": os.path.exists(dest),
                     "unresolved": ["the things coverage.notChecked lists: a consumer there would break"]})
    return {"applied": False, "entries": plan[:500], "entriesTotal": len(plan), "footprintBytes": total,
            "diskBytesFreedByMoving": cross_drive,
            "note": "A plan to review. Nothing was moved and no op applies it. Moving within one drive frees no disk space, "
                    "which is why footprintBytes and diskBytesFreedByMoving differ."}


def op_artifact_inventory(params):
    """What the registry knows and what a named folder holds, each file classed referenced, protected, unknown, candidate or
    gone with why. Reads only: nothing is written, moved or deleted, and no reply is a verdict that a file is unused.

    params:
      roots (list)        folders to walk too: a file there with no record is unknown. Default: the registry's files only
      status (list)       rows of these classes only; the summary still counts all
      limit, offset       a page of rows (default limit 100, at most 1000), largest first
      hash (bool)         SHA-256 of the files listed as referenced or candidate, up to 4 GB: duplicates[] names files with
                          identical bytes. Identical bytes alone do not make either removable
      archiveRoot (str)   also answer archivePlan: where each candidate would go. A plan to review, never applied
      maxFiles (int)      stop walking roots after this many files (default 20000); coverage.truncated says it stopped
    """
    reject_unknown(params, _KEYS, "artifact_inventory")
    tail = "Nothing was read."
    roots = params.get("roots")
    if roots is not None and (not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r for r in roots)):
        raise MifOpError("roots is a non-empty list of folders. %s" % tail)
    resolved = []
    for r in roots or []:
        ab = os.path.abspath(bpy.path.abspath(r))
        if not os.path.isdir(ab):
            raise MifOpError("roots names %r, which is not a folder. %s" % (r, tail))
        resolved.append(ab)
    wanted = params.get("status")
    if wanted is not None and (not isinstance(wanted, list) or not wanted or any(w not in _CLASSES for w in wanted)):
        raise MifOpError("status is a list of %s. %s" % (", ".join(_CLASSES), tail))
    limit = take(params, "limit", default=100, kind=int)
    offset = take(params, "offset", default=0, kind=int)
    max_files = take(params, "maxFiles", default=20000, kind=int)
    if not 1 <= limit <= 1000 or offset < 0 or max_files < 1:
        raise MifOpError("limit is 1 to 1000, offset 0 or more, maxFiles 1 or more. %s" % tail)
    want_hash = take_bool(params, "hash", default=False)
    archive = take(params, "archiveRoot", default=None, kind=str)
    archive_root = os.path.abspath(bpy.path.abspath(archive)) if archive else None

    path = artifacts.registry_path()
    records, bad = artifacts.read(path)
    exact, patterns, libraries = _session_references()
    blend = bpy.data.filepath
    open_blend = artifacts.canonical(blend) if blend else None
    dirty = bool(bpy.data.is_dirty)

    canonical_roots = [artifacts.canonical(r) for r in resolved]
    in_scope = {p: r for p, r in records.items() if not resolved or any(_inside(p, c) for c in canonical_roots)}
    walked, git_dirs, links, unreadable, truncated = _walk(resolved, max_files, skip=set(in_scope))

    def protected_reason(key):
        if key == open_blend:
            return ("the file open in this Blender, with unsaved changes" if dirty else "the file open in this Blender")
        if key in libraries:
            return "a library the open file links"
        return None

    rows = []

    def add(shown, key, record):
        refs = list(exact.get(key) or []) + [ref for pat, ref in patterns if pat.fullmatch(key)]
        try:
            st = os.stat(shown)
            state = "same"
            size = st.st_size
            if record is not None and (st.st_size != record.get("bytes") or st.st_mtime_ns != record.get("mtimeNs")):
                state = "changed"
        except OSError:
            st, size, state = None, (record or {}).get("bytes") or 0, "gone"
        source = (record or {}).get("blend")
        klass, why = artifacts.classify(record, refs, protected_reason(key), state, bool(source) and os.path.isfile(source))
        row = {"path": shown, "class": klass, "why": why, "bytes": size}
        if record is not None:
            row.update(producer=record.get("op"), purpose=record.get("purpose"), recordedAt=record.get("at"),
                       source=source)
        if refs:
            row["references"] = refs[:10]
        rows.append(row)

    for key, record in in_scope.items():
        add(record["path"], key, record)
    for full in walked:
        add(full, artifacts.canonical(full), None)
    for d in git_dirs:
        rows.append({"path": d, "class": "protected", "why": ["a git checkout or worktree: not entered, and never scratch"],
                     "bytes": 0, "directory": True})
    if open_blend and not any(artifacts.canonical(r["path"]) == open_blend for r in rows):
        add(blend, open_blend, records.get(open_blend))
    rows.sort(key=lambda r: (-r["bytes"], r["path"]))

    summary = {k: {"files": 0, "bytes": 0} for k in _CLASSES}
    for row in rows:
        summary[row["class"]]["files"] += 1
        summary[row["class"]]["bytes"] += row["bytes"]
    listed = [r for r in rows if wanted is None or r["class"] in wanted]
    out = {"summary": summary, "rowsTotal": len(listed), "rows": listed[offset:offset + limit]}
    if offset + limit < len(listed):
        out["nextOffset"] = offset + limit
    st = artifacts.stats()
    out["registry"] = {"recording": artifacts.enabled(), "path": path, "exists": os.path.isfile(path), "records": len(records),
                       "unreadableLines": bad, "writesFailedThisSession": st["failed"]}
    if st["lastError"]:
        out["registry"]["lastError"] = st["lastError"]
    out["session"] = {"file": blend or None, "unsavedChanges": dirty}
    out["coverage"] = {"filesExamined": len(rows), "rootsWalked": resolved, "truncated": truncated,
                       "linksNotFollowed": links[:50], "linksNotFollowedTotal": len(links),
                       "unreadable": unreadable[:50], "unreadableTotal": len(unreadable), "notChecked": list(_NOT_CHECKED),
                       "note": "No consumer found is not proof a file is unused. This answer never says a project is clean."}
    if not records:
        out["emptyRegistry"] = ("no record, which says nothing about this project's files"
                                + ("" if artifacts.enabled() else ": recording is off (the Record artifacts preference)")
                                + ". To look at older files, name their folder in roots: they read as unknown.")
    if want_hash:
        by_hash, skipped, spent = {}, [], 0
        for row in rows:
            if row["class"] not in ("referenced", "candidate") or row.get("directory") or not row["bytes"]:
                continue
            if spent + row["bytes"] > _HASH_BUDGET:
                skipped.append(row["path"])
                continue
            try:
                digest = _sha256(row["path"])
            except OSError:
                skipped.append(row["path"])
                continue
            spent += row["bytes"]
            by_hash.setdefault(digest, []).append(row)
        out["duplicates"] = [{"sha256": h, "bytes": g[0]["bytes"], "files": [{"path": r["path"], "class": r["class"]} for r in g]}
                             for h, g in by_hash.items() if len(g) > 1][:200]
        out["hashSkipped"] = {"total": len(skipped), "first": skipped[:20]}
        out["duplicatesNote"] = "Identical bytes alone do not make either file removable: a referenced copy stays referenced."
    if archive_root:
        out["archivePlan"] = _archive_plan(rows, archive_root)
    return out


OPS = {"artifact_inventory": op_artifact_inventory}
