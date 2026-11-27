"""Where a tool may write on disk: the project folder, the tool's scratch folder, and folders the person lists.

WHY (Mifsopo, 2026-11-22, on a code audit's S13 "writes that take any path": "Confine writes" - "Project, scratch and
listed folders only; anything else is refused with a message naming the setting"). Tools that write a file took any
path a caller named - an export, a render, a capture, an .ini, a REAPER render - so a caller steered by text it read
could write anywhere the user can. Reads stay free.

THE RULE, the same on every backend (this file for the MCP server; a byte copy, MifBlender/write_roots.py, in the
Blender add-on; MifBridgeSafety.cpp's RefuseFileOutsideProject in Unreal; Gate.cs in the Unity package):
  a write is allowed when its target, with symlinks and `..` resolved, is the project folder or under it, the
  backend's scratch folder or under it, or a listed folder or under it. Anything else is refused before it is written,
  with refusedRule "write-root", the target, the roots it was checked against, and how to list a folder.

THE SETTING: one per-user file, WRITE_ROOTS_FILE_ENV (MIF_WRITE_ROOTS_FILE) else ~/.mifbridge/write_roots.json, beside
the token files:  {"folders": ["D:/Renders", "F:/Work"]}. Read on every check, so an edit applies to the next call with
no restart. A file that is missing lists nothing; one that cannot be parsed lists nothing and the refusal says why.
"""
import json
import os

WRITE_ROOTS_FILE_ENV = "MIF_WRITE_ROOTS_FILE"
RULE = "write-root"


def roots_file():
    raw = (os.environ.get(WRITE_ROOTS_FILE_ENV) or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.join(os.path.expanduser("~"), ".mifbridge", "write_roots.json")


def listed_folders():
    """(folders, problem). The person's listed folders, absolute; problem is a sentence when the file is unreadable."""
    path = roots_file()
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return [], None
    except (OSError, ValueError) as exc:
        return [], "%s could not be read (%s), so it lists no folder" % (path, exc)
    raw = data.get("folders") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], "%s has no \"folders\" list, so it lists no folder" % path
    return [os.path.abspath(os.path.expanduser(str(f))) for f in raw if isinstance(f, str) and f.strip()], None


def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(p))))


def _inside(target, root):
    t, r = _real(target), _real(root)
    return t == r or t.startswith(r.rstrip("\\/") + os.sep)


def check(target, project=None, scratch=()):
    """None when `target` may be written; else the refusal dict (ok False, nothing written).

    project: the project folder, or None when this backend has none. scratch: the backend's scratch folder(s)."""
    named = [("project", project)] if project else []
    named += [("scratch", s) for s in scratch if s]
    folders, problem = listed_folders()
    named += [("listed", f) for f in folders]
    for _kind, root in named:
        try:
            if _inside(target, root):
                return None
        except (OSError, ValueError):
            continue
    path = roots_file()
    example = json.dumps({"folders": folders + [os.path.dirname(os.path.abspath(target)).replace("\\", "/")]})
    return {
        "ok": False, "refusedRule": RULE, "target": os.path.abspath(target),
        "writeRoots": [{"kind": k, "path": os.path.abspath(r)} for k, r in named], "setting": path,
        "error": ("refused: %s is outside the folders tools may write to - %s. To allow it, list its folder in %s "
                  "(%s), for example %s; the next call reads it. Nothing was written.%s"
                  % (os.path.abspath(target),
                     "; ".join("%s %s" % (k, os.path.abspath(r)) for k, r in named) or "none is set",
                     path, WRITE_ROOTS_FILE_ENV + " moves that file", example,
                     (" " + problem + ".") if problem else "")),
    }
