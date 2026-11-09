"""MifBlender ops: datablock and file hygiene - remap and dedupe, batch rename, external files, object relations, asset
libraries, collection exporters and keying sets (1.3 gap audit, 2026-10-19).

THE GAP. An FBX or glTF import, or appending one asset twice, leaves M_Metal, M_Metal.001, M_Metal.002 - three slots and
three Unreal materials on the way out. A .blend handed to another machine opens with pink textures. Engine naming
conventions (SM_, M_, T_, A_) were applied one object at a time. Nothing made a linked duplicate single-user, made an
instance real, browsed an asset library or kept a collection's export settings in the file. Each op here is one of
those jobs with a read-back of what it changed.
"""
from __future__ import annotations

import hashlib
import os
import re

import bpy

from .ops_common import (MifOpError, get_object, limit_rows, reject_unknown, select_only, selection_restore,
                         selection_snapshot, take, take_bool, take_int)

# KIND -> the bpy.data collection it lives in. Bones are renamed through their armature (batch_rename armature).
_KINDS = {
    "OBJECT": "objects", "MESH": "meshes", "MATERIAL": "materials", "IMAGE": "images", "ACTION": "actions",
    "COLLECTION": "collections", "NODE_GROUP": "node_groups", "CAMERA": "cameras", "LIGHT": "lights",
    "ARMATURE": "armatures", "CURVE": "curves", "TEXTURE": "textures", "WORLD": "worlds", "SCENE": "scenes",
    "TEXT": "texts", "SOUND": "sounds", "FONT": "fonts", "LATTICE": "lattices", "PARTICLE": "particles",
    "BRUSH": "brushes", "PALETTE": "palettes", "MOVIECLIP": "movieclips", "MASK": "masks", "SPEAKER": "speakers",
    "VOLUME": "volumes", "POINTCLOUD": "pointclouds", "HAIR_CURVES": "hair_curves", "CACHE_FILE": "cache_files",
}
_NAME_MAX = 255 if bpy.app.version >= (5, 0, 0) else 63      # bytes; Blender's MAX_ID_NAME - 2 per build


def _kind(params, tail, default=None, allowed=None):
    raw = take(params, "kind", "type", default=default, kind=str)
    if raw is None:
        raise MifOpError("kind is required: one of %s. %s" % (", ".join(sorted(allowed or _KINDS)), tail))
    k = str(raw).upper()
    pool = allowed or _KINDS
    if k not in pool or not hasattr(bpy.data, _KINDS.get(k, "")):
        have = [x for x in sorted(pool) if hasattr(bpy.data, _KINDS.get(x, ""))]
        raise MifOpError("kind %r is not one of %s on Blender %s. %s"
                         % (raw, ", ".join(have), bpy.app.version_string, tail))
    return k, getattr(bpy.data, _KINDS[k])


def _named(coll, name, kind, tail):
    if not isinstance(name, str) or not name:
        raise MifOpError("a %s name is required. %s" % (kind.lower(), tail))
    block = coll.get(name)
    if block is None:
        have = sorted(b.name for b in coll)[:25]
        raise MifOpError("no %s named '%s'. Present: %s%s. %s" % (kind.lower(), name, ", ".join(have) or "(none)",
                                                                  " ..." if len(coll) > 25 else "", tail))
    return block


# --------------------------------------------------------------------------------------------- remap_datablock

_REMAP_KEYS = ("kind", "type", "mode", "source", "sources", "target", "remove", "dryRun", "names")
_REMAP_NOTES = {
    "from": "name the datablock being replaced with source (or sources for several)",
    "to": "name the datablock that takes over its users with target",
}
_DUP_SUFFIX = re.compile(r"^(.*)\.(\d{3,})$")


def _sock_value(sock):
    v = getattr(sock, "default_value", None)
    if v is None:
        return None
    try:
        return tuple(round(float(x), 5) for x in v)
    except TypeError:
        try:
            return round(float(v), 5)
        except (TypeError, ValueError):
            return str(v)


def _tree_fingerprint(tree, seen=None):
    """What a node tree computes: each node's type, its settings that are not names, its unlinked input values, and
    the links - named by node TYPE and order, so a copy whose nodes are called 'Principled BSDF.001' still matches."""
    if tree is None:
        return None
    seen = set() if seen is None else seen
    order = {n.name: i for i, n in enumerate(sorted(tree.nodes, key=lambda n: (n.bl_idname, n.location[0],
                                                                              n.location[1])))}
    rows = []
    for n in sorted(tree.nodes, key=lambda n: order[n.name]):
        props = []
        for p in n.bl_rna.properties:
            if p.identifier in ("name", "label", "location", "width", "height", "select", "dimensions", "location_absolute",
                                "show_options", "show_preview", "show_texture", "hide", "mute", "parent",
                                "width_hidden", "use_custom_color", "color", "warning_propagation", "color_tag"):
                continue
            if p.type in ("POINTER", "COLLECTION"):
                if p.identifier == "image":
                    img = getattr(n, "image", None)
                    props.append(("image", img.name if img is not None else None))
                elif p.identifier == "node_tree":
                    sub = getattr(n, "node_tree", None)
                    props.append(("tree", sub.name if sub is not None else None))
                continue
            try:
                held = getattr(n, p.identifier)
            except AttributeError:
                continue
            if isinstance(held, float):
                held = round(held, 5)
            elif not isinstance(held, (int, str, bool)):
                try:
                    held = tuple(round(float(x), 5) for x in held)
                except TypeError:
                    held = str(held)
            props.append((p.identifier, held))
        ins = [(s.identifier, _sock_value(s)) for s in n.inputs if not s.is_linked]
        rows.append((n.bl_idname, tuple(props), tuple(ins)))
    links = sorted((order[k.from_node.name], k.from_socket.identifier, order[k.to_node.name], k.to_socket.identifier)
                   for k in tree.links)
    return hashlib.sha1(repr((rows, links)).encode("utf-8")).hexdigest()


def _fingerprint(kind, block):
    """A content hash of a datablock, or None when this kind is not compared."""
    if kind == "MATERIAL":
        extra = (tuple(round(c, 5) for c in block.diffuse_color), round(block.metallic, 5), round(block.roughness, 5),
                 getattr(block, "blend_method", None), bool(block.use_nodes))
        return hashlib.sha1(repr((extra, _tree_fingerprint(block.node_tree if block.use_nodes else None)))
                            .encode("utf-8")).hexdigest()
    if kind == "NODE_GROUP":
        return _tree_fingerprint(block)
    if kind == "IMAGE":
        if block.size[0] == 0 or block.size[1] == 0:
            return None
        import numpy as np                                 # noqa: PLC0415 - Blender bundles it
        a = np.empty(block.size[0] * block.size[1] * block.channels, dtype=np.float32)
        try:
            block.pixels.foreach_get(a)
        except (RuntimeError, ValueError):
            return None
        return hashlib.sha1(repr((tuple(block.size), block.channels)).encode() + a.tobytes()).hexdigest()
    if kind == "MESH":
        import numpy as np                                 # noqa: PLC0415
        co = np.empty(len(block.vertices) * 3, dtype=np.float32)
        block.vertices.foreach_get("co", co)
        loops = np.empty(len(block.loops), dtype=np.int32)
        block.loops.foreach_get("vertex_index", loops)
        return hashlib.sha1(np.round(co, 5).tobytes() + loops.tobytes()
                            + repr([m.name if m else None for m in block.materials]).encode()).hexdigest()
    if kind == "ACTION":
        rows = []
        for fc in getattr(block, "fcurves", []):
            pts = [0.0] * (len(fc.keyframe_points) * 2)
            fc.keyframe_points.foreach_get("co", pts)
            rows.append((fc.data_path, fc.array_index, tuple(round(p, 5) for p in pts)))
        return hashlib.sha1(repr(sorted(rows)).encode()).hexdigest()
    if kind == "WORLD":
        return hashlib.sha1(repr((tuple(round(c, 5) for c in block.color),
                                  _tree_fingerprint(block.node_tree if block.use_nodes else None))).encode()).hexdigest()
    return None


_DEDUPE_KINDS = ("MATERIAL", "NODE_GROUP", "IMAGE", "MESH", "ACTION", "WORLD")


def op_remap_datablock(params):
    """Replace every use of one datablock with another (ID.user_remap), or DEDUPE `.001` copies onto their base.

    params:
      kind (str)            MATERIAL | MESH | IMAGE | NODE_GROUP | ACTION | WORLD | ... (bpy.data kinds); required
      mode (str)            REMAP (default) | DEDUPE
      REMAP:
        source / sources    the datablock(s) whose users move
        target (str)        the datablock that takes them over (same kind)
      DEDUPE:
        names (list)        optional: only these base names (default every base with a numbered copy)
                            A copy `Name.001` is remapped onto `Name` only when their CONTENT matches (materials and
                            node groups: node types, settings, unlinked values and links; images: pixels; meshes:
                            vertices, loops, material slots; actions: keys; worlds: color and tree). A copy that differs
                            is listed in kept with the reason, never merged.
      remove (bool)         remove the replaced datablocks once they have no users (default false)
      dryRun (bool)         report what would move; nothing changes
    Reply per remap: the users the source had before, the target's users before and after.
    """
    reject_unknown(params, _REMAP_KEYS, "remap_datablock", notes=_REMAP_NOTES)
    tail = "NOTHING was changed."
    kind, coll = _kind(params, tail)
    mode = str(take(params, "mode", default="REMAP", kind=str)).upper()
    dry = take_bool(params, "dryRun", default=False)
    remove = take_bool(params, "remove", default=False)
    pairs, kept = [], []
    if mode == "REMAP":
        if params.get("names") is not None:
            raise MifOpError("names filters DEDUPE; REMAP takes source(s) and target. %s" % tail)
        raw = take(params, "sources", "source", required=True)
        names = [raw] if isinstance(raw, str) else raw
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("source is a %s name or a list of them. %s" % (kind.lower(), tail))
        target = _named(coll, take(params, "target", required=True, kind=str), kind, tail)
        for n in dict.fromkeys(names):
            src = _named(coll, n, kind, tail)
            if src is target:
                raise MifOpError("'%s' is both a source and the target. %s" % (n, tail))
            if src.library is not None or target.library is not None:
                raise MifOpError("'%s' or '%s' is linked from a library - remap local datablocks (library_override "
                                 "LOCALIZE makes one local). %s" % (src.name, target.name, tail))
            pairs.append((src, target, "named"))
    elif mode == "DEDUPE":
        if kind not in _DEDUPE_KINDS:
            raise MifOpError("DEDUPE compares content, which is measured for %s; %s copies can be remapped by name with "
                             "REMAP. %s" % (", ".join(_DEDUPE_KINDS), kind, tail))
        for k in ("source", "sources", "target"):
            if params.get(k) is not None:
                raise MifOpError("'%s' is REMAP's; DEDUPE finds the copies itself. %s" % (k, tail))
        only = params.get("names")
        if only is not None and (not isinstance(only, list) or not all(isinstance(n, str) for n in only)):
            raise MifOpError("names is a list of base names. %s" % tail)
        fps = {}
        for block in sorted(coll, key=lambda b: b.name):
            m = _DUP_SUFFIX.match(block.name)
            if not m or block.library is not None:
                continue
            base = coll.get(m.group(1))
            if base is None or base.library is not None or (only is not None and base.name not in only):
                continue
            if kind == "IMAGE" and (block.source == "VIEWER" or base.source == "VIEWER"):
                continue
            fb = fps.get(base.name) or fps.setdefault(base.name, _fingerprint(kind, base))
            fc = _fingerprint(kind, block)
            if fb is None or fc is None:
                kept.append({"name": block.name, "base": base.name, "why": "no content to compare (no pixels?)"})
            elif fb != fc:
                kept.append({"name": block.name, "base": base.name, "why": "content differs from the base"})
            else:
                pairs.append((block, base, "content matches"))
    else:
        raise MifOpError("mode must be REMAP or DEDUPE, got %r. %s" % (params.get("mode"), tail))

    rows = []
    for src, dst, why in pairs:
        row = {"source": src.name, "target": dst.name, "reason": why, "sourceUsers": src.users,
               "targetUsersBefore": dst.users}
        if not dry:
            src.user_remap(dst)
            row["targetUsersAfter"] = dst.users
            row["sourceUsersAfter"] = src.users
        rows.append(row)
    removed = []
    if remove and not dry:
        for src, _dst, _why in pairs:
            # a fake user is a user nothing else holds; it stays only when the caller kept it on purpose elsewhere
            if src.users <= (1 if src.use_fake_user else 0):
                removed.append(src.name)
                coll.remove(src)
    out = {"kind": kind, "mode": mode, "dryRun": dry, "remaps": rows, "remapped": len(rows), "removed": removed,
           "kept": kept}
    if not rows:
        out["note"] = ("no %s copy matched its base - nothing to merge" % kind.lower() if mode == "DEDUPE"
                       else "nothing to remap")
    elif dry:
        out["note"] = "dryRun: nothing was changed - the rows are what a call without dryRun does"
    elif not remove:
        out["note"] = "the replaced datablocks are kept with no users; remove:true (or purge_orphans) deletes them"
    return out


# --------------------------------------------------------------------------------------------- batch_rename

_RENAME_KEYS = ("kind", "type", "armature", "names", "match", "find", "replace", "regex", "caseSensitive", "prefix",
                "suffix", "removePrefix", "removeSuffix", "case", "skipIfPrefixed", "dryRun", "limit")
_RENAME_NOTES = {
    "pattern": "find (plain text, or a regular expression with regex:true) and replace",
    "allowCollision": "a rename onto a name in use is refused - Blender would silently suffix one of the two .001",
}
_CASES = ("UPPER", "LOWER", "TITLE", "NONE")


def _new_name(name, params, rx):
    n = name
    find = params.get("find")
    if find:
        n = rx.sub(params.get("replace") or "", n)
    for key, fn in (("removePrefix", lambda s, p: s[len(p):] if s.startswith(p) else s),
                    ("removeSuffix", lambda s, p: s[:-len(p)] if p and s.endswith(p) else s)):
        p = params.get(key)
        if p:
            n = fn(n, p)
    case = str(params.get("case") or "NONE").upper()
    if case == "UPPER":
        n = n.upper()
    elif case == "LOWER":
        n = n.lower()
    elif case == "TITLE":
        n = n.title()
    pre, suf = params.get("prefix") or "", params.get("suffix") or ""
    if pre and not (params.get("skipIfPrefixed", True) and n.startswith(pre)):
        n = pre + n
    if suf and not n.endswith(suf):
        n = n + suf
    return n


def op_batch_rename(params):
    """Rename many datablocks of one kind at once - find/replace (plain or regex), prefix, suffix, strip, case.

    WHY. Engine naming conventions (SM_, SK_, M_, T_, A_) before a hand-off, `Cube.003` names after an import, 40 actions
    from a pack: rename_object does one object and rename_bones takes a map. This is wm.batch_rename's job without its
    UI context, and it REFUSES a rename onto a name in use - Blender would silently append .001 to one of the two.

    params:
      kind (str)              OBJECT | MESH | MATERIAL | IMAGE | ACTION | COLLECTION | NODE_GROUP | ... | BONE
      armature (str)          with kind BONE: the armature object whose bones are renamed (vertex groups, constraints'
                              subtargets and F-curve paths follow, as Blender's own rename does)
      names (list)            only these; default every datablock of the kind (linked ones are never renamed)
      match (str)             only names this regular expression matches (re.search)
      find, replace (str)     replace every occurrence; regex true takes find as a regular expression (\\1 groups)
      caseSensitive (bool)    default true
      removePrefix, removeSuffix (str)   stripped when present
      case (str)              UPPER | LOWER | TITLE | NONE (default)
      prefix, suffix (str)    added; skipIfPrefixed (default true) leaves a name that already has the prefix
      dryRun (bool)           report the plan; nothing is renamed
      limit (int)             rows listed in the reply (the rename covers all)
    Steps run in that order: find/replace, strip, case, prefix/suffix. Unchanged names are skipped.
    """
    reject_unknown(params, _RENAME_KEYS, "batch_rename", notes=_RENAME_NOTES)
    tail = "NOTHING was renamed."
    raw_kind = str(take(params, "kind", "type", required=True, kind=str)).upper()
    arm = None
    if raw_kind == "BONE":
        arm = get_object(take(params, "armature", required=True, kind=str))
        if arm.type != "ARMATURE":
            raise MifOpError("'%s' is a %s, not an armature. %s" % (arm.name, arm.type, tail))
        if arm.mode == "EDIT":
            raise MifOpError("'%s' is in edit mode - leave it first. %s" % (arm.name, tail))
        pool = list(arm.data.bones)
        kind = "BONE"
    else:
        if params.get("armature") is not None:
            raise MifOpError("armature goes with kind BONE. %s" % tail)
        kind, coll = _kind(params, tail)
        pool = [b for b in coll if b.library is None]
    case = str(params.get("case") or "NONE").upper()
    if case not in _CASES:
        raise MifOpError("case must be one of %s, got %r. %s" % (", ".join(_CASES), params.get("case"), tail))
    for key in ("find", "replace", "prefix", "suffix", "removePrefix", "removeSuffix", "match"):
        if params.get(key) is not None and not isinstance(params.get(key), str):
            raise MifOpError("'%s' must be a string. %s" % (key, tail))
    if params.get("replace") is not None and not params.get("find"):
        raise MifOpError("replace needs find. %s" % tail)
    if not any(params.get(k) for k in ("find", "prefix", "suffix", "removePrefix", "removeSuffix")) and case == "NONE":
        raise MifOpError("nothing to do: give find/replace, prefix, suffix, removePrefix, removeSuffix or case. %s" % tail)
    flags = 0 if take_bool(params, "caseSensitive", default=True) else re.IGNORECASE
    rx = None
    if params.get("find"):
        try:
            rx = re.compile(params["find"] if take_bool(params, "regex", default=False) else re.escape(params["find"]),
                            flags)
        except re.error as exc:
            raise MifOpError("find is not a valid regular expression: %s. %s" % (exc, tail))
    mrx = None
    if params.get("match"):
        try:
            mrx = re.compile(params["match"], flags)
        except re.error as exc:
            raise MifOpError("match is not a valid regular expression: %s. %s" % (exc, tail))
    only = params.get("names")
    if only is not None:
        if not isinstance(only, list) or not all(isinstance(n, str) for n in only):
            raise MifOpError("names is a list of %s names. %s" % (kind.lower(), tail))
        have = {b.name for b in pool}
        absent = [n for n in only if n not in have]
        if absent:
            raise MifOpError("no %s named %s. %s" % (kind.lower(), ", ".join(absent[:20]), tail))
        pool = [b for b in pool if b.name in set(only)]
    if mrx is not None:
        pool = [b for b in pool if mrx.search(b.name)]
    p2 = dict(params)
    p2["skipIfPrefixed"] = take_bool(params, "skipIfPrefixed", default=True)
    plan = []
    for b in pool:
        new = _new_name(b.name, p2, rx)
        if new != b.name:
            plan.append((b, new))
    renaming = {b.name for b, _ in plan}
    taken = {b.name for b in (arm.data.bones if arm is not None else coll)} - renaming
    clashes, seen = [], {}
    for b, new in plan:
        if not new.strip():
            clashes.append({"name": b.name, "newName": new, "why": "the new name is empty"})
        elif len(new.encode("utf-8")) > _NAME_MAX:
            clashes.append({"name": b.name, "newName": new, "why": "longer than %d bytes, Blender's name limit on %s"
                            % (_NAME_MAX, bpy.app.version_string)})
        elif new in taken:
            clashes.append({"name": b.name, "newName": new, "why": "a %s not being renamed is called that"
                            % kind.lower()})
        elif new in seen:
            clashes.append({"name": b.name, "newName": new, "why": "'%s' would be renamed to it too" % seen[new]})
        seen.setdefault(new, b.name)
    if clashes:
        raise MifOpError("%d rename(s) would collide - Blender would silently append .001 to one name of each pair: %s. "
                         "%s" % (len(clashes), "; ".join("%s -> %s (%s)" % (c["name"], c["newName"], c["why"])
                                                         for c in clashes[:12]), tail))
    dry = take_bool(params, "dryRun", default=False)
    rows = [{"name": b.name, "newName": new} for b, new in plan]
    if not dry and plan:
        # TWO PHASES, so a chain (A -> B while B -> C) never lands on a name still held: everything moves to a unique
        # parking name first, then to its own.
        parked = []
        for i, (b, new) in enumerate(plan):
            b.name = "_MifRename%d_%s" % (i, hashlib.sha1(b.name.encode("utf-8")).hexdigest()[:8])
            parked.append((b, new))
        for b, new in parked:
            b.name = new
        wrong = [(r["newName"], b.name) for r, (b, _n) in zip(rows, parked) if b.name != r["newName"]]
        if wrong:
            raise MifOpError("renamed, and %d name(s) did not land as planned: %s - check with a listing op."
                             % (len(wrong), wrong[:10]))
    listed, extra = limit_rows(rows, params, "renames", excludes=False)
    out = {"kind": kind, "armature": arm.name if arm is not None else None, "considered": len(pool),
           "renamed": 0 if dry else len(plan), "planned": len(plan), "dryRun": dry, "renames": listed}
    out.update(extra)
    if dry:
        out["note"] = "dryRun: nothing was renamed"
    return out


# --------------------------------------------------------------------------------------------- manage_external_data

_EXT_KEYS = ("action", "directory", "method", "kinds", "missingOnly", "limit")
_EXT_ACTIONS = ("REPORT", "PACK_ALL", "UNPACK", "MAKE_RELATIVE", "MAKE_ABSOLUTE", "FIND_MISSING")
_UNPACK_METHODS = ("USE_LOCAL", "WRITE_LOCAL", "USE_ORIGINAL", "WRITE_ORIGINAL")
_EXT_KINDS = {"IMAGE": "images", "SOUND": "sounds", "FONT": "fonts", "MOVIECLIP": "movieclips",
              "CACHE_FILE": "cache_files", "LIBRARY": "libraries", "VOLUME": "volumes"}


def _external_rows(kinds):
    rows = []
    for kind in kinds:
        for b in getattr(bpy.data, _EXT_KINDS[kind], []):
            path = getattr(b, "filepath", "")
            if kind == "IMAGE" and b.source in ("GENERATED", "VIEWER"):
                continue
            if kind == "FONT" and path == "<builtin>":
                continue
            if not path:
                continue
            packed = bool(getattr(b, "packed_file", None)) or bool(getattr(b, "packed_files", None))
            ab = os.path.abspath(bpy.path.abspath(path, library=getattr(b, "library", None)))
            if kind == "IMAGE" and b.source == "TILED":
                exists = any(os.path.isfile(ab.replace("<UDIM>", str(t.number))) for t in b.tiles)
            elif kind == "IMAGE" and b.source == "SEQUENCE":
                exists = os.path.isdir(os.path.dirname(ab))
            else:
                exists = os.path.exists(ab)
            rows.append({"kind": kind, "name": b.name, "filepath": path, "absolutePath": ab,
                         "relative": path.startswith("//"), "packed": packed, "exists": exists,
                         "users": b.users})
    return rows


def _summary(rows):
    return {"files": len(rows), "packed": sum(r["packed"] for r in rows),
            "missing": sum(1 for r in rows if not r["exists"] and not r["packed"]),
            "absolute": sum(1 for r in rows if not r["relative"]), "relative": sum(r["relative"] for r in rows)}


def op_manage_external_data(params):
    """File > External Data as one op: REPORT every external path, PACK_ALL, UNPACK, MAKE_RELATIVE, MAKE_ABSOLUTE,
    FIND_MISSING - with the counts before and after.

    WHY. A .blend handed to another machine or a render farm opens with pink textures, and nothing said so: save_file had
    no pack or relative option and only list_images reported `packed`, for images alone.

    params:
      action (str)          REPORT (default) | PACK_ALL | UNPACK | MAKE_RELATIVE | MAKE_ABSOLUTE | FIND_MISSING
      method (str)          UNPACK: USE_LOCAL | WRITE_LOCAL (default: //textures/ beside the .blend, existing files
                            kept) | USE_ORIGINAL | WRITE_ORIGINAL
      directory (str)       FIND_MISSING: the folder tree searched by file name
      kinds (list)          REPORT rows: IMAGE, SOUND, FONT, MOVIECLIP, CACHE_FILE, LIBRARY, VOLUME (default all)
      missingOnly (bool)    REPORT only the files that are neither on disk nor packed
      limit (int)           rows listed
    MAKE_RELATIVE, UNPACK and the local methods need a saved .blend (a relative path is relative to it).
    """
    reject_unknown(params, _EXT_KEYS, "manage_external_data")
    tail = "NOTHING was changed."
    action = str(take(params, "action", default="REPORT", kind=str)).upper()
    if action not in _EXT_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(_EXT_ACTIONS), params.get("action"), tail))
    kinds = params.get("kinds")
    if kinds is None:
        kinds = list(_EXT_KINDS)
    elif not isinstance(kinds, list) or any(str(k).upper() not in _EXT_KINDS for k in kinds):
        raise MifOpError("kinds is a list of %s. %s" % (", ".join(_EXT_KINDS), tail))
    kinds = [str(k).upper() for k in kinds]
    blend = bpy.data.filepath
    before = _external_rows(list(_EXT_KINDS))
    detail = {}
    if action in ("MAKE_RELATIVE", "UNPACK") and not blend:
        method = str(params.get("method") or "WRITE_LOCAL").upper()
        if action == "MAKE_RELATIVE" or method in ("USE_LOCAL", "WRITE_LOCAL"):
            raise MifOpError("this file has never been saved, so there is no folder a relative or local path could be "
                             "relative to - save_file first. %s" % tail)
    for key, owner in (("method", "UNPACK"), ("directory", "FIND_MISSING")):
        if params.get(key) is not None and action != owner:
            raise MifOpError("%s is %s's. %s" % (key, owner, tail))
    try:
        if action == "PACK_ALL":
            # NATIVE, per file: Blender's file.pack_all stops at the first missing file and packs nothing after it
            # ("Unable to pack file, source path ... not found", measured on 5.0.1) - one lost texture would keep every
            # present one unpacked. Each present file is packed; each missing one is listed.
            not_packed = []
            for r in before:
                if r["packed"] or r["kind"] not in ("IMAGE", "SOUND", "FONT", "VOLUME"):
                    continue
                block = getattr(bpy.data, _EXT_KINDS[r["kind"]]).get(r["name"])
                if block is None or not hasattr(block, "pack"):
                    continue
                if not r["exists"]:
                    not_packed.append({"kind": r["kind"], "name": r["name"], "why": "the file is missing"})
                    continue
                try:
                    block.pack()
                except RuntimeError as exc:
                    not_packed.append({"kind": r["kind"], "name": r["name"], "why": str(exc).strip()[:200]})
            detail["notPacked"] = not_packed
        elif action == "UNPACK":
            method = str(take(params, "method", default="WRITE_LOCAL", kind=str)).upper()
            if method not in _UNPACK_METHODS:
                raise MifOpError("method must be one of %s, got %r. %s" % (", ".join(_UNPACK_METHODS), method, tail))
            detail["method"] = method
            bpy.ops.file.unpack_all(method=method)
        elif action == "MAKE_RELATIVE":
            bpy.ops.file.make_paths_relative()
        elif action == "MAKE_ABSOLUTE":
            bpy.ops.file.make_paths_absolute()
        elif action == "FIND_MISSING":
            folder = take(params, "directory", required=True, kind=str)
            folder = os.path.abspath(bpy.path.abspath(folder))
            if not os.path.isdir(folder):
                raise MifOpError("directory '%s' is not a folder. %s" % (folder, tail))
            detail["directory"] = folder
            bpy.ops.file.find_missing_files(directory=folder)
    except RuntimeError as exc:
        raise MifOpError("Blender refused %s: %s" % (action, str(exc).strip()))
    after = _external_rows(list(_EXT_KINDS)) if action != "REPORT" else before
    rows = [r for r in after if r["kind"] in kinds]
    if take_bool(params, "missingOnly", default=False):
        rows = [r for r in rows if not r["exists"] and not r["packed"]]
    listed, extra = limit_rows(rows, params, "external files", excludes=False)
    out = {"action": action, "blendFile": blend or None, "before": _summary(before), "after": _summary(after),
           "files": listed}
    out.update(extra)
    out.update(detail)
    if action != "REPORT":
        old = {(r["kind"], r["name"]): r for r in before}
        out["changed"] = [{"kind": r["kind"], "name": r["name"], "filepathBefore": old[(r["kind"], r["name"])]["filepath"],
                           "filepath": r["filepath"], "packedBefore": old[(r["kind"], r["name"])]["packed"],
                           "packed": r["packed"]}
                          for r in after if (r["kind"], r["name"]) in old
                          and (old[(r["kind"], r["name"])]["filepath"], old[(r["kind"], r["name"])]["packed"])
                          != (r["filepath"], r["packed"])][:200]
    if out["after"]["missing"]:
        out["warning"] = ("%d external file(s) are neither on disk nor packed - the file opens with them missing here "
                          "too; FIND_MISSING searches a folder for them" % out["after"]["missing"])
    return out


# --------------------------------------------------------------------------------------------- object_relations

_REL_KEYS = ("action", "objects", "object", "source", "targets", "what", "keepHierarchy", "parent")
_REL_ACTIONS = ("MAKE_SINGLE_USER", "MAKE_INSTANCES_REAL", "LINK_DATA")
_SINGLE_WHAT = ("DATA", "MATERIALS", "ACTIONS")
_LINK_WHAT = ("DATA", "MATERIALS", "MODIFIERS", "COLLECTIONS")


def _objects_param(params, key_list, key_one, tail, required=True):
    raw = params.get(key_list, params.get(key_one))
    if raw is None:
        if required:
            raise MifOpError("%s is required (an object name or a list). %s" % (key_list, tail))
        return []
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("%s is an object name or a list of them. %s" % (key_list, tail))
    return [get_object(n) for n in dict.fromkeys(names)]


def op_object_relations(params):
    """Object > Relations: MAKE_SINGLE_USER, MAKE_INSTANCES_REAL, LINK_DATA - users before and after.

    params:
      action (str)          MAKE_SINGLE_USER | MAKE_INSTANCES_REAL | LINK_DATA
      objects / object      the objects acted on (MAKE_SINGLE_USER, MAKE_INSTANCES_REAL)
      what (list)           MAKE_SINGLE_USER: DATA (default), MATERIALS, ACTIONS; LINK_DATA: DATA (default),
                            MATERIALS, MODIFIERS, COLLECTIONS
      source, targets       LINK_DATA: the object whose data the targets take
      keepHierarchy (bool)  MAKE_INSTANCES_REAL: keep the instanced hierarchy's parenting (default true)
      parent (bool)         MAKE_INSTANCES_REAL: parent the new objects to the instancer (default false)
    """
    reject_unknown(params, _REL_KEYS, "object_relations")
    tail = "NOTHING was changed."
    action = str(take(params, "action", required=True, kind=str)).upper()
    if action not in _REL_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(_REL_ACTIONS), params.get("action"), tail))
    # NOT DROPPED SILENTLY: the other two actions never read them, so keepHierarchy on a LINK_DATA answered ok and did nothing.
    stray = [k for k in ("keepHierarchy", "parent") if params.get(k) is not None]
    if stray and action != "MAKE_INSTANCES_REAL":
        raise MifOpError("%s go with MAKE_INSTANCES_REAL, not %s. %s" % (" and ".join(stray), action, tail))
    what = params.get("what")
    if isinstance(what, str):
        what = [what]
    if action == "MAKE_SINGLE_USER":
        objs = _objects_param(params, "objects", "object", tail)
        what = [str(w).upper() for w in (what or ["DATA"])]
        bad = [w for w in what if w not in _SINGLE_WHAT]
        if bad:
            raise MifOpError("what for MAKE_SINGLE_USER is a list of %s, got %s. %s" % (", ".join(_SINGLE_WHAT), bad,
                                                                                       tail))
        rows = []
        for o in objs:
            row = {"object": o.name}
            if "DATA" in what and o.data is not None:
                row["dataBefore"], row["dataUsersBefore"] = o.data.name, o.data.users
                if o.data.users > 1:
                    o.data = o.data.copy()
                row["data"], row["dataUsers"] = o.data.name, o.data.users
            if "MATERIALS" in what:
                mats = []
                for slot in o.material_slots:
                    m = slot.material
                    if m is None:
                        continue
                    if m.users > 1:
                        new = m.copy()
                        if slot.link == "OBJECT":
                            slot.material = new
                        else:
                            o.data.materials[slot.slot_index] = new
                        mats.append({"before": m.name, "after": new.name})
                row["materials"] = mats
            if "ACTIONS" in what:
                ad = o.animation_data
                if ad is not None and ad.action is not None and ad.action.users > 1:
                    old = ad.action
                    ad.action = old.copy()
                    row["action"] = {"before": old.name, "after": ad.action.name}
            rows.append(row)
        return {"action": action, "what": what, "objects": rows}

    if action == "LINK_DATA":
        if params.get("objects") is not None or params.get("object") is not None:
            raise MifOpError("LINK_DATA takes source and targets. %s" % tail)
        src = get_object(take(params, "source", required=True, kind=str))
        targets = _objects_param(params, "targets", "targets", tail)
        if src in targets:
            raise MifOpError("'%s' is the source and a target. %s" % (src.name, tail))
        what = [str(w).upper() for w in (what or ["DATA"])]
        bad = [w for w in what if w not in _LINK_WHAT]
        if bad:
            raise MifOpError("what for LINK_DATA is a list of %s, got %s. %s" % (", ".join(_LINK_WHAT), bad, tail))
        if "DATA" in what:
            wrong = [t.name for t in targets if t.type != src.type]
            if wrong:
                raise MifOpError("DATA links one %s datablock, and %s are not %s objects. %s"
                                 % (src.type, ", ".join(wrong), src.type, tail))
        rows = []
        for t in targets:
            row = {"object": t.name}
            if "DATA" in what and src.data is not None:
                row["dataBefore"] = t.data.name if t.data is not None else None
                t.data = src.data
                row["data"] = t.data.name
            if "MATERIALS" in what:
                if t.data is None or not hasattr(t.data, "materials"):
                    raise MifOpError("'%s' (%s) has no material slots. %s" % (t.name, t.type, tail))
                if t.data is not src.data:
                    t.data.materials.clear()
                    for m in src.data.materials:
                        t.data.materials.append(m)
                for i, slot in enumerate(src.material_slots):
                    if slot.link == "OBJECT" and i < len(t.material_slots):
                        t.material_slots[i].link = "OBJECT"
                        t.material_slots[i].material = slot.material
                row["materials"] = [s.material.name if s.material else None for s in t.material_slots]
            if "MODIFIERS" in what:
                for m in list(t.modifiers):
                    t.modifiers.remove(m)
                for m in src.modifiers:
                    nm = t.modifiers.new(m.name, m.type)
                    for p in m.bl_rna.properties:
                        if p.is_readonly or p.identifier in ("name", "type", "rna_type"):
                            continue
                        try:
                            setattr(nm, p.identifier, getattr(m, p.identifier))
                        except (AttributeError, TypeError, ValueError):
                            pass
                    if m.type == "NODES":
                        # THE INPUTS MOVED IN 5.2. To 5.0 they are id-properties on the modifier; 5.2's NodesModifier
                        # has none ("bpy_struct.keys(): this type doesn't support IDProperties", a raw TypeError out of
                        # LINK_DATA MODIFIERS, found 2026-10-28 by the version matrix) and holds them in
                        # properties.inputs[identifier], a group of value / type / attribute_name - measured on 5.2.1.
                        src_in = getattr(getattr(m, "properties", None), "inputs", None)
                        dst_in = getattr(getattr(nm, "properties", None), "inputs", None)
                        if src_in is None or dst_in is None:
                            for key in m.keys():
                                try:
                                    nm[key] = m[key]
                                except (TypeError, KeyError):
                                    pass
                        else:
                            for key in src_in.keys():
                                for sub in src_in[key].keys():
                                    try:
                                        dst_in[key][sub] = src_in[key][sub]
                                    except (TypeError, KeyError):
                                        pass
                row["modifiers"] = [m.name for m in t.modifiers]
            if "COLLECTIONS" in what:
                linked = {c.as_pointer() for c in t.users_collection}
                for c in src.users_collection:
                    if c.as_pointer() not in linked:
                        c.objects.link(t)
                row["collections"] = [c.name for c in t.users_collection]
            rows.append(row)
        return {"action": action, "what": what, "source": src.name, "targets": rows,
                "dataUsers": src.data.users if src.data is not None else None}

    # MAKE_INSTANCES_REAL: the operator works on the selection, so the instancers are selected alone and the selection
    # put back after; what it made is what is new in bpy.data.objects.
    objs = _objects_param(params, "objects", "object", tail)
    if what is not None:
        raise MifOpError("what is MAKE_SINGLE_USER's and LINK_DATA's. %s" % tail)
    idle = [o.name for o in objs if o.instance_type == "NONE" and not any(m.type == "NODES" for m in o.modifiers)
            and not o.particle_systems]
    if idle:
        raise MifOpError("%s instance nothing (no collection instance, particle instancing or Geometry Nodes "
                         "instances). %s" % (", ".join(idle), tail))
    before = set(bpy.data.objects)
    snap = selection_snapshot()
    try:
        select_only(objs)
        try:
            bpy.ops.object.duplicates_make_real(use_base_parent=take_bool(params, "parent", default=False),
                                                use_hierarchy=take_bool(params, "keepHierarchy", default=True))
        except RuntimeError as exc:
            raise MifOpError("Blender refused Make Instances Real: %s. %s" % (str(exc).strip(), tail))
    finally:
        made = [o for o in bpy.data.objects if o not in before]
        selection_restore(snap)
    bpy.context.view_layer.update()
    rows = [{"name": o.name, "type": o.type, "data": o.data.name if o.data is not None else None,
             "parent": o.parent.name if o.parent else None} for o in made]
    out = {"action": action, "objects": [o.name for o in objs], "made": len(made), "newObjects": rows[:500]}
    if not made:
        out["warning"] = ("Blender reported success and no object was made - the instances may not be realizable "
                          "this way (a Geometry Nodes tree whose instances are realized inside it, or none drawn)")
    return out


# --------------------------------------------------------------------------------------------- list_library_assets

_LIB_KEYS = ("library", "folder", "file", "kinds", "catalog", "tag", "name", "limit", "offset", "catalogs")
_ASSET_KINDS = {"OBJECT": "objects", "MATERIAL": "materials", "COLLECTION": "collections", "NODE_GROUP": "node_groups",
                "ACTION": "actions", "WORLD": "worlds", "BRUSH": "brushes", "SCENE": "scenes"}


def _catalog_tree(folder):
    """{uuid: path} from blender_assets.cats.txt at the library's root."""
    from .ops_pose import _read_catalogs               # noqa: PLC0415 - the one catalog reader
    return {v.lower(): k for k, v in _read_catalogs(folder).items()} if folder else {}


def op_list_library_assets(params):
    """Browse an asset library without opening its files: the assets in each .blend with kind, catalog, tags,
    description, author and whether a preview is stored; filtered and paged; the catalog tree.

    HOW IT READS A FILE'S ASSETS. bpy.data.libraries.load(link=True, assets_only=True) links each asset into this
    session long enough to read its asset_data, and the links are removed again - the open file is not changed (a
    library this file already linked keeps what it had). Bring one in with append_from_file.

    params:
      library (str)        the name of an asset library in Preferences > File Paths
      folder (str)         or a folder searched for .blend files (subfolders too)
      file (str)           or one .blend
      kinds (list)         OBJECT, MATERIAL, COLLECTION, NODE_GROUP, ACTION, WORLD, BRUSH, SCENE (default all)
      catalog (str)        a catalog path; assets in it or below it
      tag (str)            assets carrying this tag (case-insensitive)
      name (str)           assets whose name contains this (case-insensitive)
      limit, offset (int)  paging (default 200 from 0)
      catalogs (bool)      include the catalog tree (default true)
    """
    reject_unknown(params, _LIB_KEYS, "list_library_assets")
    tail = "Nothing was read."
    given = [k for k in ("library", "folder", "file") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("give exactly one of library (a name in Preferences > File Paths), folder or file. %s" % tail)
    root = None
    if given[0] == "library":
        from .ops_pose import _library                     # noqa: PLC0415 - the one library resolver
        name = take(params, "library", kind=str)
        if str(name).upper() == "CURRENT_FILE":
            raise MifOpError("CURRENT_FILE's assets are this file's own - list them with the listing ops. %s" % tail)
        _lib, root = _library(name, "read")
        files = None
    elif given[0] == "folder":
        root = os.path.abspath(bpy.path.abspath(take(params, "folder", kind=str)))
        if not os.path.isdir(root):
            raise MifOpError("folder '%s' is not a folder. %s" % (root, tail))
        files = None
    else:
        f = os.path.abspath(bpy.path.abspath(take(params, "file", kind=str)))
        if not os.path.isfile(f) or not f.lower().endswith(".blend"):
            raise MifOpError("file '%s' is not a .blend on disk. %s" % (f, tail))
        files = [f]
        root = os.path.dirname(f)
    if files is None:
        files = []
        for dirpath, _dirs, names in os.walk(root):
            files += [os.path.join(dirpath, n) for n in sorted(names) if n.lower().endswith(".blend")]
    kinds = params.get("kinds")
    kinds = list(_ASSET_KINDS) if kinds is None else [str(k).upper() for k in (kinds if isinstance(kinds, list)
                                                                                   else [kinds])]
    bad = [k for k in kinds if k not in _ASSET_KINDS]
    if bad:
        raise MifOpError("kinds is a list of %s, got %s. %s" % (", ".join(_ASSET_KINDS), bad, tail))
    limit = take_int(params, "limit", default=200)
    offset = take_int(params, "offset", default=0)
    if limit < 1 or offset < 0:
        raise MifOpError("limit is 1 or more and offset 0 or more. %s" % tail)
    cats = _catalog_tree(root)
    want_cat = take(params, "catalog", default=None, kind=str)
    want_tag = (take(params, "tag", default=None, kind=str) or "").lower() or None
    want_name = (take(params, "name", default=None, kind=str) or "").lower() or None
    rows, unreadable = [], []
    current = os.path.normcase(os.path.abspath(bpy.data.filepath)) if bpy.data.filepath else None
    for path in files:
        if current and os.path.normcase(path) == current:
            unreadable.append({"file": path, "why": "the open file itself - its assets are local"})
            continue
        existing_lib = next((lb for lb in bpy.data.libraries
                             if os.path.normcase(os.path.abspath(bpy.path.abspath(lb.filepath))) == os.path.normcase(path)),
                            None)
        before = {k: set(getattr(bpy.data, a)) for k, a in _ASSET_KINDS.items()}
        try:
            with bpy.data.libraries.load(path, link=True, assets_only=True) as (src, dst):
                for k in kinds:
                    setattr(dst, _ASSET_KINDS[k], list(getattr(src, _ASSET_KINDS[k])))
        except (OSError, RuntimeError) as exc:
            unreadable.append({"file": path, "why": str(exc).strip()[:200]})
            continue
        new_ids = []
        try:
            for k in kinds:
                for idb in getattr(bpy.data, _ASSET_KINDS[k]):
                    if idb in before[k] or idb.library is None:
                        continue
                    new_ids.append(idb)
                    ad = idb.asset_data
                    if ad is None:
                        continue
                    cid = str(getattr(ad, "catalog_id", "") or "").lower()
                    cpath = cats.get(cid) if cid and cid != "00000000-0000-0000-0000-000000000000" else None
                    rows.append({"name": idb.name, "kind": k, "file": path,
                                 "relativeFile": os.path.relpath(path, root) if root else path,
                                 "catalogId": cid or None, "catalog": cpath,
                                 "tags": [t.name for t in ad.tags], "description": ad.description or None,
                                 "author": getattr(ad, "author", "") or None,
                                 "hasPreview": bool(getattr(idb, "preview", None) is not None
                                                    and any(getattr(idb.preview, "image_size", (0, 0))))})
        finally:
            lib = None
            for idb in new_ids:
                lib = lib or idb.library
            if new_ids:
                bpy.data.batch_remove(new_ids)
            if lib is not None and existing_lib is None and lib.name in bpy.data.libraries:
                try:
                    bpy.data.libraries.remove(lib)
                except (ReferenceError, RuntimeError):
                    pass
    if want_cat is not None:
        p = "/".join(x for x in want_cat.split("/") if x)
        rows = [r for r in rows if r["catalog"] and (r["catalog"] == p or r["catalog"].startswith(p + "/"))]
    if want_tag:
        rows = [r for r in rows if any(t.lower() == want_tag for t in r["tags"])]
    if want_name:
        rows = [r for r in rows if want_name in r["name"].lower()]
    rows.sort(key=lambda r: (r["catalog"] or "~", r["kind"], r["name"]))
    page = rows[offset:offset + limit]
    out = {"root": root, "filesScanned": len(files) - len(unreadable), "unreadable": unreadable, "total": len(rows),
           "offset": offset, "returned": len(page), "truncated": offset + len(page) < len(rows), "assets": page}
    if take_bool(params, "catalogs", default=True):
        out["catalogs"] = sorted(set(cats.values()))
    if not files:
        out["warning"] = "no .blend file under %s - nothing was listed" % root
    return out


# --------------------------------------------------------------------------------------------- set_collection_exporter

_CEX_KEYS = ("action", "collection", "format", "filepath", "settings", "index", "all")
_CEX_ACTIONS = ("LIST", "ADD", "EDIT", "REMOVE", "EXPORT")
_CEX_FORMATS = {"FBX": "IO_FH_fbx", "GLTF": "IO_FH_gltf2", "USD": "IO_FH_usd", "OBJ": "IO_FH_obj",
                "ALEMBIC": "IO_FH_alembic", "PLY": "IO_FH_ply", "STL": "IO_FH_stl"}


def _exporter_rows(coll):
    rows = []
    for i, ex in enumerate(coll.exporters):
        props = ex.export_properties
        fp = getattr(props, "filepath", "")
        ab = os.path.abspath(bpy.path.abspath(fp)) if fp else None
        row = {"index": i, "collection": coll.name, "handler": ex.name, "filepath": fp, "absolutePath": ab,
               "fileExists": bool(ab and os.path.isfile(ab))}
        if row["fileExists"]:
            st = os.stat(ab)
            row["bytes"], row["mtime"] = st.st_size, st.st_mtime
        rows.append(row)
    return rows


def _apply_settings(props, settings, tail):
    bad = []
    for k, v in (settings or {}).items():
        if k not in props.bl_rna.properties or props.bl_rna.properties[k].is_readonly:
            bad.append(k)
            continue
        try:
            setattr(props, k, v)
        except (TypeError, ValueError, AttributeError) as exc:
            raise MifOpError("settings.%s = %r was refused by the exporter: %s. %s" % (k, v, exc, tail))
    if bad:
        have = sorted(p.identifier for p in props.bl_rna.properties if not p.is_readonly and p.identifier != "rna_type")
        raise MifOpError("settings has key(s) this exporter does not take: %s. It takes: %s. %s"
                         % (", ".join(bad), ", ".join(have[:80]), tail))


def op_set_collection_exporter(params):
    """Collection exporters (Blender 4.2+): LIST, ADD, EDIT, REMOVE a collection's exporters, and EXPORT one collection's
    or every collection's now - the settings stay in the .blend, so the next session exports the same way.

    params:
      action (str)        LIST (default) | ADD | EDIT | REMOVE | EXPORT
      collection (str)    the collection (LIST without it lists every collection's exporters)
      format (str)        ADD: FBX | GLTF | USD | OBJ | ALEMBIC | PLY | STL (the build's file handlers)
      filepath (str)      ADD / EDIT: where it writes ('//' is relative to the .blend)
      settings (dict)     ADD / EDIT: the exporter's own properties by identifier (e.g. global_scale, use_selection)
      index (int)         EDIT / REMOVE / EXPORT: which exporter of the collection (default 0; EXPORT default all)
      all (bool)          EXPORT: every collection's exporters in the file
    EXPORT replies per exporter the file, its size and whether its mtime is newer than before the call.
    """
    reject_unknown(params, _CEX_KEYS, "set_collection_exporter")
    tail = "NOTHING was changed."
    if "exporters" not in bpy.types.Collection.bl_rna.properties:
        raise MifOpError("collection exporters arrived in Blender 4.2; this is %s - export_scene writes a collection "
                         "with the same settings on every call. %s" % (bpy.app.version_string, tail))
    action = str(take(params, "action", default="LIST", kind=str)).upper()
    if action not in _CEX_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(_CEX_ACTIONS), params.get("action"), tail))
    cname = take(params, "collection", default=None, kind=str)
    coll = None
    if cname is not None:
        coll = bpy.data.collections.get(cname)
        if coll is None:
            raise MifOpError("no collection named '%s'. Present: %s. %s"
                             % (cname, ", ".join(sorted(c.name for c in bpy.data.collections)[:25]) or "(none)", tail))
    if action == "LIST":
        colls = [coll] if coll is not None else list(bpy.data.collections)
        rows = [r for c in colls for r in _exporter_rows(c)]
        return {"action": action, "exporters": rows, "count": len(rows)}
    every = take_bool(params, "all", default=False)
    if action == "EXPORT" and every:
        if coll is not None:
            raise MifOpError("all exports every collection's exporters; collection names one - pass one. %s" % tail)
        colls = [c for c in bpy.data.collections if len(c.exporters)]
        targets = [(c, i) for c in colls for i in range(len(c.exporters))]
    else:
        if coll is None:
            raise MifOpError("collection is required for %s. %s" % (action, tail))
        targets = None
    # A STRING, read before ADD or EDIT writes anything (2026-10-28, found reading and measured by the version matrix):
    # str() took a mapping or a list and wrote "{'mif': ...}" as the exporter's path.
    fpath = take(params, "filepath", kind=str)
    settings = take(params, "settings", default=None, kind=dict)
    if action == "ADD":
        fmt = str(take(params, "format", required=True, kind=str)).upper()
        if fmt not in _CEX_FORMATS:
            raise MifOpError("format must be one of %s, got %r. %s" % (", ".join(_CEX_FORMATS), fmt, tail))
        # A file handler written in C (OBJ, USD, PLY, STL, Alembic; FBX from 5.0) is not a bpy.types attribute, so the
        # operator itself is asked: it names a handler it does not know (measured on 4.2.17 and 5.0.1, every one of
        # the seven names below is accepted on both).
        n0 = len(coll.exporters)
        try:
            with bpy.context.temp_override(collection=coll):
                bpy.ops.collection.exporter_add(name=_CEX_FORMATS[fmt])
        except RuntimeError as exc:
            raise MifOpError("Blender %s has no %s exporter (%s) - its add-on may be off. %s"
                             % (bpy.app.version_string, fmt, str(exc).strip(), tail))
        if len(coll.exporters) != n0 + 1:
            raise MifOpError("Blender did not add a %s exporter to '%s'. %s" % (fmt, coll.name, tail))
        ex = coll.exporters[n0]
        try:
            if fpath is not None:
                ex.export_properties.filepath = fpath
            _apply_settings(ex.export_properties, settings, tail)
        except MifOpError:
            with bpy.context.temp_override(collection=coll):
                bpy.ops.collection.exporter_remove(index=n0)
            raise
        return {"action": action, "collection": coll.name, "exporter": _exporter_rows(coll)[n0]}
    index = take_int(params, "index", default=None)
    if action in ("EDIT", "REMOVE") or (action == "EXPORT" and targets is None):
        n = len(coll.exporters)
        if n == 0:
            raise MifOpError("'%s' has no exporter - ADD one first. %s" % (coll.name, tail))
        if action == "EXPORT" and index is None:
            targets = [(coll, i) for i in range(n)]
        else:
            i = 0 if index is None else index
            if not 0 <= i < n:
                raise MifOpError("index %d is outside '%s''s %d exporter(s). %s" % (i, coll.name, n, tail))
            targets = [(coll, i)]
    if action == "EDIT":
        c, i = targets[0]
        props = c.exporters[i].export_properties
        # PUT BACK ON A REFUSAL: _apply_settings writes key by key, so a refused key after a good one left the good one
        # (and the path) written under "NOTHING was changed".
        keep = {}
        for k in list(settings or {}) + ["filepath"]:
            if k in props.bl_rna.properties and not props.bl_rna.properties[k].is_readonly:
                v = getattr(props, k)
                keep[k] = v if isinstance(v, (str, int, float, bool)) else tuple(v) if hasattr(v, "__len__") else v
        try:
            if fpath is not None:
                props.filepath = fpath
            _apply_settings(props, settings, tail)
        except MifOpError:
            for k, v in keep.items():
                try:
                    setattr(props, k, v)
                except (TypeError, ValueError, AttributeError):
                    pass
            raise
        return {"action": action, "collection": c.name, "exporter": _exporter_rows(c)[i]}
    if action == "REMOVE":
        c, i = targets[0]
        row = _exporter_rows(c)[i]
        with bpy.context.temp_override(collection=c):
            bpy.ops.collection.exporter_remove(index=i)
        return {"action": action, "collection": c.name, "removed": row, "remaining": len(c.exporters)}
    # EXPORT
    if not targets:
        raise MifOpError("no collection in this file has an exporter. %s" % tail)
    results = []
    for c, i in targets:
        before = _exporter_rows(c)[i]
        if not before["filepath"]:
            results.append(dict(before, exported=False, why="the exporter has no filepath - EDIT it first"))
            continue
        t0 = before.get("mtime")
        try:
            with bpy.context.temp_override(collection=c):
                bpy.ops.collection.exporter_export(index=i)
        except RuntimeError as exc:
            results.append(dict(before, exported=False, why=str(exc).strip()[:300]))
            continue
        after = _exporter_rows(c)[i]
        after["exported"] = bool(after.get("fileExists") and (t0 is None or after.get("mtime", 0) > t0))
        if not after["exported"]:
            after["why"] = "Blender reported success and the file is %s" % ("missing" if not after.get("fileExists")
                                                                           else "no newer than before")
        results.append(after)
    failed = [r for r in results if not r.get("exported")]
    out = {"action": action, "exports": results, "exported": len(results) - len(failed)}
    if failed:
        out["warning"] = "%d exporter(s) wrote nothing - see why on each row" % len(failed)
    return out


# --------------------------------------------------------------------------------------------- set_keying_set

_KS_KEYS = ("action", "name", "paths", "active", "frame")


def _ks_rows(scene):
    rows = []
    for i, ks in enumerate(scene.keying_sets):
        rows.append({"index": i, "name": ks.bl_label, "active": scene.keying_sets.active_index == i,
                     "paths": [{"id": p.id.name if p.id else None, "idType": p.id_type, "dataPath": p.data_path,
                                "index": p.array_index if not p.use_entire_array else -1, "group": p.group or None}
                               for p in ks.paths]})
    return rows


def op_set_keying_set(params):
    """A scene keying set: LIST, CREATE, ADD_PATHS, REMOVE_PATHS, KEY (insert keys on every path at a frame), DELETE.

    params:
      action (str)       LIST (default) | CREATE | ADD_PATHS | REMOVE_PATHS | KEY | DELETE
      name (str)         the keying set (CREATE makes it; the others find it)
      paths (list)       ADD_PATHS / REMOVE_PATHS: [{object, dataPath, index (-1 or omitted: the whole array), group}] -
                         dataPath on the object, e.g. 'location' or 'pose.bones["hand.L"].rotation_quaternion'
      active (bool)      CREATE / ADD_PATHS: make it the scene's active keying set (default true on CREATE)
      frame (float)      KEY: the frame keyed (default the current frame)
    KEY reports the F-curves that gained a key.
    """
    reject_unknown(params, _KS_KEYS, "set_keying_set")
    tail = "NOTHING was changed."
    scene = bpy.context.scene
    action = str(take(params, "action", default="LIST", kind=str)).upper()
    acts = ("LIST", "CREATE", "ADD_PATHS", "REMOVE_PATHS", "KEY", "DELETE")
    if action not in acts:
        raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(acts), params.get("action"), tail))
    if action == "LIST":
        return {"action": action, "scene": scene.name, "keyingSets": _ks_rows(scene)}
    name = take(params, "name", required=True, kind=str)
    ks = next((k for k in scene.keying_sets if k.bl_label == name), None)
    if action == "CREATE":
        if ks is not None:
            raise MifOpError("scene '%s' already has a keying set '%s'. %s" % (scene.name, name, tail))
        make_active = take_bool(params, "active", default=True)   # read BEFORE the keying set exists
        ks = scene.keying_sets.new(idname="MifKS_%s" % re.sub(r"\W", "_", name), name=name)
        if make_active:
            scene.keying_sets.active = ks
        if params.get("paths"):
            _ks_add(ks, params["paths"], tail)
        return {"action": action, "scene": scene.name, "keyingSet": _ks_rows(scene)[list(scene.keying_sets).index(ks)]}
    if ks is None:
        raise MifOpError("scene '%s' has no keying set '%s'. Present: %s. %s"
                         % (scene.name, name, ", ".join(k.bl_label for k in scene.keying_sets) or "(none)", tail))
    if action == "DELETE":
        # KeyingSets has new() and no remove(); the operator removes the ACTIVE one, so the named one is made active
        scene.keying_sets.active = ks
        with bpy.context.temp_override(scene=scene):
            bpy.ops.anim.keying_set_remove()
        if any(k.bl_label == name for k in scene.keying_sets):
            raise MifOpError("Blender did not remove keying set '%s'." % name)
        return {"action": action, "scene": scene.name, "deleted": name, "keyingSets": _ks_rows(scene)}
    if action == "ADD_PATHS":
        _ks_add(ks, take(params, "paths", required=True, kind=list), tail)
        if take_bool(params, "active", default=False):
            scene.keying_sets.active = ks
    elif action == "REMOVE_PATHS":
        want = take(params, "paths", required=True, kind=list)
        gone = 0
        for spec in want:
            obj, dp, idx = _ks_path_spec(spec, tail)
            for p in list(ks.paths):
                if p.id == obj and p.data_path == dp and (idx is None or p.array_index == idx):
                    ks.paths.remove(p)
                    gone += 1
        if not gone:
            raise MifOpError("no path of '%s' matched %r. %s" % (name, want, tail))
    elif action == "KEY":
        frame = params.get("frame")
        frame = float(scene.frame_current if frame is None else frame)
        made = []
        for p in ks.paths:
            if p.id is None:
                continue
            idx = -1 if p.use_entire_array else p.array_index
            try:
                ok = p.id.keyframe_insert(p.data_path, index=idx, frame=frame, group=p.group or "")
            except (TypeError, RuntimeError) as exc:
                raise MifOpError("keying '%s' %s failed: %s" % (p.id.name, p.data_path, exc))
            if ok:
                made.append({"id": p.id.name, "dataPath": p.data_path, "index": idx})
        return {"action": action, "keyingSet": name, "frame": frame, "keyed": made, "paths": len(ks.paths)}
    return {"action": action, "scene": scene.name, "keyingSet": _ks_rows(scene)[list(scene.keying_sets).index(ks)]}


def _ks_path_spec(spec, tail):
    if not isinstance(spec, dict) or not isinstance(spec.get("object"), str) or not isinstance(spec.get("dataPath"), str):
        raise MifOpError("each path is {object, dataPath, index, group}, got %r. %s" % (spec, tail))
    obj = get_object(spec["object"])
    idx = spec.get("index")
    if idx is not None and (isinstance(idx, bool) or not isinstance(idx, int)):
        raise MifOpError("index is a whole number (-1 for the whole array), got %r. %s" % (idx, tail))
    return obj, spec["dataPath"], (None if idx is None or idx < 0 else idx)


def _ks_add(ks, paths, tail):
    if not isinstance(paths, list) or not paths:
        raise MifOpError("paths is a non-empty list of {object, dataPath, index, group}. %s" % tail)
    plan = []
    for spec in paths:
        obj, dp, idx = _ks_path_spec(spec, tail)
        try:
            obj.path_resolve(dp)
        except ValueError:
            raise MifOpError("'%s' has no property at data path %r. %s" % (obj.name, dp, tail))
        plan.append((obj, dp, idx, spec.get("group")))
    for obj, dp, idx, grp in plan:
        kw = {"index": -1 if idx is None else idx}
        if grp:
            kw["group_method"] = "NAMED"
            kw["group_name"] = str(grp)
        ks.paths.add(obj, dp, **kw)


OPS = {
    "remap_datablock": op_remap_datablock,
    "batch_rename": op_batch_rename,
    "manage_external_data": op_manage_external_data,
    "object_relations": op_object_relations,
    "list_library_assets": op_list_library_assets,
    "set_collection_exporter": op_set_collection_exporter,
    "set_keying_set": op_set_keying_set,
}
