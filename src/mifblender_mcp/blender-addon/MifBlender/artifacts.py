"""The artifact registry: which files this add-on's ops wrote, kept so a later session can tell a tool's output from a file
nobody can account for. Design: docs/35_ARTIFACTS.md.

OPT-IN. Off by default (the "Record artifacts" preference, or MIF_BLENDER_ARTIFACTS=1 / 0). Off, nothing here touches the
disk. On, one JSON object per line goes to artifacts.jsonl beside the call journal: per user, not beside a .blend.

WHERE THE RECORD COMES FROM. tools/audit_output_paths.py holds every op that writes a caller's path to check_output_path, so
that one call names the path and this file adds no list of writers. check_output_path runs BEFORE the write, so a path only
becomes a record when the op has ended ok and a file is there, written since the note. set_render_settings checks a path it
only stores; that path has no new file behind it and is not recorded.

NOT RECORDED, and said in every report: a file an op's helper writes without check_output_path, animation frames (the path
checked is a prefix, not a file), and anything the person or another program made.

A RECORD NEVER FAILS AN OP. The file an op was asked for matters more than the note about it; a failed write counts in
stats() and the report says so.

NO bpy AT MODULE SCOPE, so tools/test_blender_artifacts.py can load this file on its own for the compaction rows.
"""
import io
import json
import os
import time

SCHEMA = 1
MAX_RECORDS = 5000                      # compaction keeps the newest per path, down to this
_COMPACT_AT = 2 * MAX_RECORDS           # lines in the file before a compaction is due

# What an op's output is FOR. "intermediate" is what the same source can make again, the only purpose a report may call a
# candidate. "source" is the person's work. An op not named here is "unclassified", which reads as unknown: a guess here
# would be the very inference the registry exists to replace.
_INTERMEDIATE = frozenset((
    "export_mesh", "export_scene", "export_actions", "export_camera", "export_groom", "export_vertex_positions",
    "mixdown_audio", "extract_audio", "render_still", "render_viewport", "compose_atlas", "preview_garment",
    "bake_texture", "bake_texture_set", "bake_vertex_animation", "bake_hidden_face_map", "build_lod_chain"))
_SOURCE = frozenset(("save_file", "save_image", "set_asset_library", "create_udim_image", "material_from_maps"))

_PENDING = []
_STATS = {"written": 0, "failed": 0, "lastError": None, "lines": None}


def purpose_of(op):
    return "intermediate" if op in _INTERMEDIATE else "source" if op in _SOURCE else "unclassified"


def _preference():
    try:
        import bpy
        addon = bpy.context.preferences.addons.get(__package__)
        return bool(getattr(addon.preferences, "record_artifacts", False)) if addon else False
    except Exception:                                                   # noqa: BLE001 - no Blender, no preference
        return False


def enabled():
    forced = (os.environ.get("MIF_BLENDER_ARTIFACTS") or "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return _preference()


def registry_path():
    override = (os.environ.get("MIF_BLENDER_ARTIFACTS_FILE") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    from . import journal
    return os.path.join(os.path.dirname(journal.journal_path()), "artifacts.jsonl")


def canonical(path):
    """The one spelling a path is compared by: absolute, links resolved, case folded where the filesystem folds it."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def note_output(path):
    """check_output_path names a file an op is about to write. Nothing is recorded until the op has ended."""
    try:
        if enabled():
            _PENDING.append((path, time.time()))
    except Exception:                                                   # noqa: BLE001 - see the header
        pass


def _blend_now():
    try:
        import bpy
        return bpy.data.filepath or None
    except Exception:                                                   # noqa: BLE001
        return None


def flush(op, ok):
    """The op has ended: each path it named that holds a file written since the note becomes a record. A refusal, and a
    path whose file is older than the note, leave nothing."""
    if not _PENDING:
        return 0
    pending, _PENDING[:] = list(_PENDING), []
    if not ok:
        return 0
    try:
        return _record(op, pending)
    except Exception as exc:                                            # noqa: BLE001 - see the header
        _STATS["failed"] += len(pending)
        _STATS["lastError"] = str(exc)[:200]
        return 0


def _record(op, pending):
    rows = []
    for path, noted in pending:
        try:
            st = os.stat(path)
        except OSError:
            continue
        # mtime can round down by up to two seconds on some filesystems
        if st.st_mtime < noted - 2.0 or not os.path.isfile(path):
            continue
        rows.append({"v": SCHEMA, "path": canonical(path), "op": op, "purpose": purpose_of(op),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bytes": st.st_size, "mtimeNs": st.st_mtime_ns,
                     "blend": _blend_now()})
    if rows:
        _append(rows)
    return len(rows)


def _append(rows):
    path = registry_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "a", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        _STATS["written"] += len(rows)
        if _STATS["lines"] is None:
            with io.open(path, "rb") as fh:
                _STATS["lines"] = sum(1 for _ in fh)
        else:
            _STATS["lines"] += len(rows)
        if _STATS["lines"] > _COMPACT_AT:
            _compact(path)
    except OSError as exc:
        _STATS["failed"] += len(rows)
        _STATS["lastError"] = str(exc)[:200]


def _compact(path):
    """The newest record per path, the newest MAX_RECORDS paths, replaced in one step. Only a writer compacts: a report
    changes nothing."""
    records, _bad = read(path)
    keep = sorted(records.values(), key=lambda r: str(r.get("at")))[-MAX_RECORDS:]
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for row in keep:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    _STATS["lines"] = len(keep)


def read(path=None):
    """({canonical path: its newest record}, lines that did not read). A missing file is no records, and no bad lines."""
    path = path or registry_path()
    records, bad = {}, 0
    try:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                        raise ValueError
                    records[row["path"]] = row
                except ValueError:
                    bad += 1
    except OSError:
        pass
    return records, bad


def stats():
    return dict(_STATS)


def classify(row, references, protected_why, file_state, source_exists):
    """(class, [why]) for one file, the only place the rule lives. `row` is its record or None. `references` are the known
    consumers; `protected_why` a reason it is out of reach, or None; `file_state` is "same", "changed" or "gone" against the
    record; `source_exists` whether the record's .blend is on disk.

    Out of reach comes first: the open file, a linked library. Then a known consumer, so a byte-identical copy with its own
    consumer is never offered. No record is unknown, so
    age, extension or a missing entry can never make a file disposable. Nothing here says a file is unused: a candidate is
    "recorded as re-makable, and no consumer was found in what was read".
    """
    if protected_why:
        return "protected", [protected_why]
    if references:
        return "referenced", ["%s %s uses it" % (r.get("kind"), r.get("name")) for r in references[:5]]
    if row is None:
        return "unknown", ["no record: no participating op is on file as having written this"]
    if file_state == "gone":
        return "gone", ["recorded as written by %s, and no file is there now" % row.get("op")]
    if file_state == "changed":
        return "unknown", ["its size or time differs from the record: something else wrote it since %s" % row.get("op")]
    purpose = row.get("purpose")
    if purpose == "source":
        return "protected", ["%s wrote it as a source, which is the person's work" % row.get("op")]
    if purpose != "intermediate":
        return "unknown", ["%s's output has no recorded purpose" % row.get("op")]
    if not row.get("blend"):
        return "unknown", ["%s ran in an unsaved file, so nothing on disk can make this again" % row.get("op")]
    if not source_exists:
        return "unknown", ["its source %s is gone, so nothing can make this again" % row.get("blend")]
    return "candidate", ["%s wrote it from %s; no consumer found in what was read" % (row.get("op"), row.get("blend"))]
