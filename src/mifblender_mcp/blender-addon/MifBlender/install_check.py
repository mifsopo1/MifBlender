"""Is this MifBlender install whole, and is it the only one Blender can see? Plain Python - no bpy - so the logic is
tested offline (tools/test_blender_install_check.py) and ping can call it on every request.

WHY, 2026-09-16: on Blender 5.2.1 an upgrade left 43 `MifBlender.pre-install-*.bak`
folders beside the add-on inside scripts/addons - Blender lists every folder holding an __init__.py as another add-on -
and installing the new zip over the enabled add-on produced a HYBRID: the new __init__.py, the old ops_file.py and
server.py, and no ops_undo.py at all. The release zip was whole (26 files); the tree it left was not, and the Add-ons
list still showed one ticked MifBridge entry. Nothing a caller could read said either thing.

WHAT IT CHECKS
  duplicates   sibling folders beside this package whose name starts with "MifBlender" and that hold an __init__.py -
               each is a second add-on to Blender, whatever its suffix
  manifest     build_zip.py writes MANIFEST.json into the release zip: every packaged file and its sha256. Against it:
                 missing     listed and not on disk
                 changed     on disk with other bytes
                 unexpected  a .py on disk the release does not have - an old module left behind
               A source checkout or a hand copy has no manifest, and says so rather than passing a check it cannot make.
"""
import hashlib
import json
import os

MANIFEST_NAME = "MANIFEST.json"
_SKIP_DIRS = {"__pycache__"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def package_files(pkg_dir):
    """Every file under the package, as forward-slash paths relative to it, skipping __pycache__."""
    out = []
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            out.append(os.path.relpath(os.path.join(root, name), pkg_dir).replace(os.sep, "/"))
    return out


def duplicates(pkg_dir):
    """Sibling add-on folders Blender would load as another MifBlender, or None if the parent cannot be listed."""
    parent, own = os.path.dirname(os.path.abspath(pkg_dir)), os.path.basename(os.path.abspath(pkg_dir))
    try:
        names = os.listdir(parent)
    except OSError:
        return None
    return sorted(n for n in names
                  if n != own and n.lower().startswith("mifblender")
                  and os.path.isfile(os.path.join(parent, n, "__init__.py")))


def check(pkg_dir):
    pkg_dir = os.path.abspath(pkg_dir)
    dups = duplicates(pkg_dir)
    out = {"path": pkg_dir, "duplicates": dups, "manifestPresent": False,
           "missing": None, "changed": None, "unexpected": None}
    manifest_path = os.path.join(pkg_dir, MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                listed = dict(json.load(fh).get("files") or {})
        except (OSError, ValueError, AttributeError) as exc:
            out["manifestError"] = str(exc)[:200]
            listed = None
        if listed is not None:
            out["manifestPresent"] = True
            on_disk = set(package_files(pkg_dir))
            out["missing"] = sorted(f for f in listed if f not in on_disk)
            changed = []
            for f, want in sorted(listed.items()):
                if f in on_disk:
                    try:
                        if sha256_of(os.path.join(pkg_dir, f)) != want:
                            changed.append(f)
                    except OSError:
                        changed.append(f)
            out["changed"] = changed
            out["unexpected"] = sorted(f for f in on_disk if f.endswith(".py") and f not in listed)
    problems = bool(dups) or bool(out["missing"] or out["changed"] or out["unexpected"]) or "manifestError" in out
    out["ok"] = not problems
    out["note"] = _note(out)
    return out


def _note(r):
    parts = []
    if r["duplicates"]:
        parts.append("%d other MifBlender folder(s) sit beside this one in %s, and Blender loads each as another add-on: "
                     "%s. Close Blender, move them OUT of that addons folder (not just renamed inside it), restart."
                     % (len(r["duplicates"]), os.path.dirname(r["path"]), ", ".join(r["duplicates"][:10])))
    if r.get("manifestError"):
        parts.append("MANIFEST.json could not be read (%s), so the files were not checked." % r["manifestError"])
    bad = [(k, r[k]) for k in ("missing", "changed", "unexpected") if r[k]]
    if bad:
        parts.append("this install does not match the release it came from - %s. An install over a running add-on can "
                     "leave old files behind: close Blender, replace the whole MifBlender folder from the zip, restart."
                     % "; ".join("%s %s" % (k, ", ".join(v[:10])) for k, v in bad))
    if not r["manifestPresent"] and not r.get("manifestError"):
        parts.append("no MANIFEST.json - a source checkout or a hand copy - so the files are not checked against a "
                     "release.")
    return " ".join(parts) or None


def manifest_for(pkg_dir, files):
    """{"package", "files": {relpath: sha256}} for the given package-relative files - what build_zip.py writes."""
    return {"package": os.path.basename(os.path.abspath(pkg_dir)),
            "files": {f: sha256_of(os.path.join(pkg_dir, f)) for f in sorted(files)}}
