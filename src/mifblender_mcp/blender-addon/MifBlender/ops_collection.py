"""Collections - how a Blender scene is actually organised, and the hole under set_light_linking.

WHY THIS EXISTS, and it is not "for completeness". Before 2026-09-03 the ONLY place in this entire
addon that touched bpy.data.collections was a four-line private helper inside set_light_linking,
which creates an EMPTY collection when it cannot find one by name. Nothing else could make a
collection, list one, put an object in one, or take one out.

That is not merely a missing feature - it makes a SHIPPED op unable to reach its own success state.
Light linking restricts a light to the objects inside a receiver collection, so an empty receiver
collection means the light illuminates NOTHING AT ALL. set_light_linking is honest about it and
returns `litsNothing: true`. But the caller then had no typed way to fix it, because putting an
object into the collection was not expressible - the only escape was run_python, i.e. the
arbitrary-code switch a user may well have turned off. An op whose only reachable outcome is the
broken one is worse than an absent op.

=============================================================================
A COLLECTION THAT IS NOT LINKED INTO THE SCENE TREE DOES NOTHING
=============================================================================
bpy.data.collections.new() creates a collection that belongs to NO scene. Objects inside it are not
in the view layer, do not render, and do not appear in the outliner. Every field on it reads
perfectly - name, objects, children - and the whole thing is inert. This is the same shape
02_GOTCHAS calls "right-looking and inert", and it is the default outcome of the obvious API call.

So create_collection LINKS by default, list_collections reports `inScene` per collection, and an
orphan is called out rather than left to be discovered at render time.

=============================================================================
THE EXCLUDE CHECKBOX IS NOT ON THE COLLECTION
=============================================================================
The four visibility controls people mean when they say "hide a collection" live in two different
places and two of them are PER VIEW LAYER:

  collection.hide_viewport        global, the monitor icon. Same for every view layer.
  collection.hide_render          global, the camera icon.
  layer_collection.exclude        PER VIEW LAYER, the checkbox. Excluded means the objects are not
                                  even evaluated - they leave the depsgraph entirely.
  layer_collection.hide_viewport  PER VIEW LAYER, the EYE icon - and this is the one people click.

Writing `collection.hide_viewport` when the caller meant the eye is a silent wrong answer, because
both are real properties that accept the write. So set_collection_visibility takes them by their
distinct names, says which view layer a per-layer write landed on, and reads all four back.
"""
import bpy

from .ops_common import (MifOpError, exclude_rows, get_object, limit_rows, reject_unknown, remove_objects, take,
                         take_bool)

_CREATE_KEYS = {"name", "collection", "parent", "objects", "link", "colorTag"}
_LIST_KEYS = {"viewLayer", "withObjects", "limit", "nameExcludes"}
_LINK_KEYS = {"collection", "name", "scene", "sceneName", "objects", "object", "move"}
_UNLINK_KEYS = {"collection", "name", "scene", "sceneName", "objects", "object", "allowOrphans"}
_VIS_KEYS = {"collection", "name", "viewLayer", "hideViewport", "hideRender", "exclude",
             "hideInViewLayer", "indirectOnly", "holdout"}
_DELETE_KEYS = {"collection", "name", "deleteObjects", "reparentTo"}
_DUPLICATE_KEYS = {"collection", "name", "newName", "linked", "parent"}


def _find_collection(name, verb):
    """A collection by name, or a refusal that lists what exists."""
    coll = bpy.data.collections.get(str(name))
    if coll is None:
        known = sorted(c.name for c in bpy.data.collections)[:30]
        raise MifOpError("no collection named '%s'. Present: %s. NOTHING was changed."
                         % (name, ", ".join(known) if known else "<none>"))
    return coll


def _link_target(params, verb):
    """The collection to link into: a named collection, or a named SCENE's master collection.

    WHY `scene` EXISTS HERE. create_scene could make a scene and nothing could put anything in it.
    Measured on 4.4 before this was written: create_scene "MifPopScene", then
    link_objects{collection:"MifPopScene"} answers `no collection named 'MifPopScene'. Present:
    Collection` - because a scene's MASTER collection is not in bpy.data.collections, and there is
    no other way to address it. So a scene the bridge created could be named, rendered empty and
    deleted, and never filled, which is most of the reason create_scene exists.

    In Blender an object is linked to a COLLECTION, and a scene's master collection is
    scene.collection - so "link into scene X" is not a new concept, it is the spelling that was
    missing. Added to unlink_objects at the same time: a target one verb can reach and its opposite
    cannot is the asymmetry audit_family_asymmetry exists to catch.

    BOTH FORMS AT ONCE IS A REFUSAL, not a precedence rule. A caller who passes collection AND scene
    has two different targets in mind and silently picking one would link into a place they did not
    name.
    """
    coll_name = take(params, "collection", "name", kind=str)
    scene_name = take(params, "scene", "sceneName", kind=str)
    if coll_name is not None and scene_name is not None:
        raise MifOpError(
            "%s takes 'collection' OR 'scene', not both - they name different targets and picking "
            "one for you would link into a place you did not ask for. NOTHING was changed." % verb)
    if scene_name is not None:
        scene = bpy.data.scenes.get(str(scene_name))
        if scene is None:
            known = sorted(s.name for s in bpy.data.scenes)[:30]
            raise MifOpError("no scene named '%s'. Present: %s. NOTHING was changed."
                             % (scene_name, ", ".join(known) if known else "<none>"))
        return scene.collection
    if coll_name is None:
        raise MifOpError(
            "%s needs 'collection' (a collection name) or 'scene' (a scene name, for its master "
            "collection). NOTHING was changed." % verb)
    return _find_collection(coll_name, verb)


def _scene_root():
    return bpy.context.scene.collection


def _walk(coll, seen=None):
    """Every collection reachable from `coll`, itself included. Cycle-safe.

    Blender's own API will not let you build a cycle, but a collection CAN be linked under two
    parents at once, so a naive walk visits it twice and a naive depth count is wrong. The seen-set
    is what makes `inScene` a set-membership question rather than a traversal that might not
    terminate on data this addon did not create.
    """
    if seen is None:
        seen = set()
    if coll.name in seen:
        return seen
    seen.add(coll.name)
    for child in coll.children:
        _walk(child, seen)
    return seen


def _in_scene_names():
    """Names of every collection actually reachable from the scene's root collection."""
    return _walk(_scene_root()) - {_scene_root().name}


def _layer_collection(name, view_layer):
    """The per-view-layer LayerCollection for a collection, or None.

    THIS IS A DIFFERENT OBJECT FROM THE COLLECTION and it is where exclude and the eye icon live.
    It exists only for collections linked into that view layer, which is itself the answer to "why
    did excluding it do nothing" - an orphaned collection has no LayerCollection to exclude.
    """
    def rec(lc):
        if lc.collection.name == name:
            return lc
        for child in lc.children:
            got = rec(child)
            if got is not None:
                return got
        return None
    return rec(view_layer.layer_collection)


def _object_names(params, *keys):
    """A list of object names from either a list key or a single-object key."""
    raw = take(params, *keys)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, (list, tuple)):
        raise MifOpError("'%s' must be a list of object names or a single name, got %s"
                         % (keys[0], type(raw).__name__))
    return [str(x) for x in raw]


def _collections_of(obj):
    """Which collections hold this object, sorted. One object, scanning every collection.

    NO CALLERS LEFT IN THE ADDON as of 2026-09-09 - all five moved to _collections_index below -
    and it is kept deliberately rather than deleted. It is the single-object reference form the
    bulk index is VERIFIED AGAINST: the check recomputes this for every object and requires the
    index to agree, so deleting it would remove the oracle and leave the fast path checking only
    against itself. Grepped before saying that: it appears nowhere else in the repo.
    """
    return sorted(c.name for c in bpy.data.collections if obj.name in c.objects)


def _collections_index():
    """{object name: [collection names]} for the WHOLE blend, in one pass.

    _collections_of asks the question for one object by scanning every collection, and
    `name in c.objects` is a linear scan (ledger A12), so calling it once per object made
    op_list_collections quadratic in objects x collections. MEASURED on 4.4.0 headless with 8000
    objects across 8 collections: 0.026s at 1000 objects, 0.105s at 2000, 0.419s at 4000, 1.695s
    at 8000 - exactly four times the work for twice the objects, three doublings running - and the
    _collections_of calls alone accounted for 1.734s of that 1.695s total, which is all of it.

    THE ORDER OF THE VALUES IS NOT SORTED, unlike _collections_of, because every caller that loops
    uses the result as a set or for truthiness only. Anything wanting a sorted list should keep
    calling _collections_of; this is the bulk form, not a drop-in replacement.

    BUILD IT WHERE THE OLD CALL WAS, not earlier. Three of the call sites sit either side of a
    link, unlink or remove, and an index built before a mutation and read after it would answer
    about the scene as it used to be - which is the kind of "faster" that is simply wrong.
    """
    index = {}
    for coll in bpy.data.collections:
        cname = coll.name
        for oname in [o.name for o in coll.objects]:
            index.setdefault(oname, []).append(cname)
    return index


def op_create_collection(params):
    """Make a collection and LINK IT, because an unlinked one is invisible and renders nothing.

    bpy.data.collections.new() alone produces a collection in no scene: its objects are outside the
    view layer, outside the depsgraph and outside the render, while every field on it reads
    correctly. That is the default outcome of the obvious call and it is why `link` defaults to
    true here rather than mirroring the API.

    params:
      name / collection (str)  required
      parent (str)             collection to nest under. Default the scene's root collection.
      objects (list[str])      objects to link in at creation - the common case, and the one that
                               keeps a light-linking receiver from being born empty.
      link (bool)              default true. false makes a deliberately unlinked collection and the
                               response says inScene:false so it cannot be mistaken for a live one.
      colorTag (str)           COLOR_01..COLOR_08, or NONE.
    """
    reject_unknown(params, _CREATE_KEYS, "create_collection")
    name = take(params, "name", "collection", required=True, kind=str)
    if bpy.data.collections.get(name) is not None:
        raise MifOpError("a collection named '%s' already exists - link objects into it with "
                         "link_objects rather than creating a second one. NOTHING was changed."
                         % name)
    parent_name = take(params, "parent", kind=str)
    link = take_bool(params, "link", default=True)
    tag = take(params, "colorTag", kind=str)
    names = _object_names(params, "objects")
    # RESOLVE EVERY OBJECT BEFORE CREATING ANYTHING. A refusal must fire before a mutation, and a
    # typo in the fourth name would otherwise leave a half-populated collection behind.
    objs = [get_object(n) for n in names]

    parent = _find_collection(parent_name, "create_collection") if parent_name else _scene_root()
    if parent_name and link is False:
        raise MifOpError("pass a parent OR link:false, not both - an unlinked collection has no "
                         "parent by definition. NOTHING was changed.")

    coll = bpy.data.collections.new(name)
    if tag:
        try:
            coll.color_tag = str(tag).upper()
        except (TypeError, ValueError) as exc:
            bpy.data.collections.remove(coll)
            raise MifOpError("colorTag '%s' is not valid (%s) - use NONE or COLOR_01..COLOR_08. "
                             "The collection was removed again, NOTHING was changed." % (tag, exc))
    if link:
        parent.children.link(coll)
    for obj in objs:
        coll.objects.link(obj)

    # THE POSTCONDITION IS REACHABILITY, not existence. `bpy.data.collections.get(name)` would be
    # true for an orphan too, which is exactly the state this op exists to avoid producing by
    # accident, so it is measured by walking the scene tree instead.
    # THE NAME BLENDER GAVE IT, not the one that was asked for. Blender sanitises a name -
    # control characters removed, 63-character truncation - so the two differ more often than
    # "a clash got a .001 suffix". Checking the REQUESTED name made this postcondition report
    # a false failure for every adjusted name: the collection was linked and reachable, under
    # a slightly different string, and the op refused anyway AND left it behind. Found by the
    # matrix leak pass on its first run, not by reading.
    made_name = coll.name
    in_scene = made_name in _in_scene_names()
    if link and not in_scene:
        # AND IT IS REMOVED. This refusal used to leave the collection in the file while
        # saying it had failed - the colorTag path above always cleaned up and this one did
        # not, which is what per-site cleanup looks like when a second site appears.
        bpy.data.collections.remove(coll)
        raise MifOpError("created '%s' and linked it under '%s', but it is NOT reachable from the "
                         "scene collection afterwards - it would be invisible and render nothing. "
                         "It was removed again, NOTHING was changed."
                         % (made_name, parent.name))
    return {
        "ok": True,
        "collection": made_name,
        "requestedName": name,
        # BLENDER ADJUSTS A NAME SILENTLY. Anything looking this collection up by the string it
        # asked for needs to know it will not find it.
        #
        # `nameWasAdjusted` WAS EMITTED HERE AND IS GONE as of 2026-09-07. It was this op's private
        # spelling of the boolean ten others call nameWasSuffixed, and one boolean under three names
        # meant a caller testing the common one silently missed this op and create_action. Both
        # spellings shipped side by side from 2026-09-04 so nothing broke mid-version; this removes
        # the old one at the release boundary the decision always belonged to. Nothing in this repo
        # read it - no suite, no tool, no showcase - which is what made the removal decidable rather
        # than a guess.
        "nameWasSuffixed": made_name != name,
        "parent": parent.name if link else None,
        "inScene": in_scene,
        "objectCount": len(coll.objects),
        "objects": sorted(o.name for o in coll.objects),
        "colorTag": getattr(coll, "color_tag", None),
        "note": (None if link else
                 "link:false was requested, so this collection is in NO scene: its objects are "
                 "outside the view layer and the render, and every field on it still reads "
                 "correctly. Link it later by creating a parent relationship."),
    }


def op_list_collections(params):
    """The collection tree, with the four visibility states that are kept in two different places.

    Reports `inScene` per collection because an orphan is indistinguishable from a live collection
    by every other field, and reports the PER-VIEW-LAYER states separately from the global ones
    because "hidden" is four different properties and people mean the eye icon.

    params:
      viewLayer (str)      which view layer the per-layer states come from. Default the active one.
      withObjects (bool)   include each collection's object names. Default false - a scene with
                           thousands of objects makes this response enormous.
    """
    reject_unknown(params, _LIST_KEYS, "list_collections")
    with_objects = take_bool(params, "withObjects", default=False)
    vl_name = take(params, "viewLayer", kind=str)
    if vl_name:
        vl = bpy.context.scene.view_layers.get(vl_name)
        if vl is None:
            known = sorted(v.name for v in bpy.context.scene.view_layers)
            raise MifOpError("no view layer named '%s' in scene '%s'. Present: %s."
                             % (vl_name, bpy.context.scene.name, ", ".join(known)))
    else:
        vl = bpy.context.view_layer

    live = _in_scene_names()
    rows = []
    for coll in sorted(bpy.data.collections, key=lambda c: c.name):
        lc = _layer_collection(coll.name, vl)
        row = {
            "name": coll.name,
            "inScene": coll.name in live,
            "objectCount": len(coll.objects),
            "childCollections": sorted(c.name for c in coll.children),
            "colorTag": getattr(coll, "color_tag", None),
            # GLOBAL, the monitor and camera icons - the same for every view layer.
            "hideViewport": bool(coll.hide_viewport),
            "hideRender": bool(coll.hide_render),
            # PER VIEW LAYER. None means this collection is not in that view layer at all, which is
            # a different answer from "not excluded" and is the reason excluding an orphan does
            # nothing.
            "exclude": bool(lc.exclude) if lc else None,
            "hideInViewLayer": bool(lc.hide_viewport) if lc else None,
            "indirectOnly": bool(lc.indirect_only) if lc else None,
            "holdout": bool(lc.holdout) if lc else None,
        }
        if with_objects:
            row["objects"] = sorted(o.name for o in coll.objects)
        rows.append(row)

    orphans = [r["name"] for r in rows if not r["inScene"]]
    # OBJECTS IN NO COLLECTION AT ALL. They exist in bpy.data, they are in no scene, and they are
    # invisible everywhere - the object-level version of the same trap, and the state unlink_objects
    # refuses to create.
    homes = _collections_index()
    homeless = sorted(o.name for o in bpy.data.objects if not homes.get(o.name))
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "collection(s)")
    return dict({
        "ok": True,
        "viewLayer": vl.name,
        "sceneRoot": _scene_root().name,
        "count": matched,
        "collections": rows,
        "orphanCollections": orphans,
        "objectsInNoCollection": homeless,
        "note": ("exclude and hideInViewLayer are PER VIEW LAYER and are reported for '%s' only. "
                 "hideViewport and hideRender are global. A null in a per-layer field means the "
                 "collection is not in that view layer at all." % vl.name),
    }, **extra)


def op_link_objects(params):
    """Put objects into a collection - the operation that makes light linking usable at all.

    An object can be in MANY collections at once; linking adds, it does not move. Pass move:true to
    unlink it from every other collection, which is what a caller usually means by "put it in".

    params:
      collection / name (str)   required
      objects (list[str])       or `object` for one. Required.
      move (bool)               unlink from all OTHER collections first. Default false.
    """
    reject_unknown(params, _LINK_KEYS, "link_objects")
    coll = _link_target(params, "link_objects")
    names = _object_names(params, "objects", "object")
    if not names:
        raise MifOpError("'objects' is required - a list of object names, or 'object' for one. "
                         "NOTHING was changed.")
    objs = [get_object(n) for n in names]        # resolve all before mutating any
    # A NAME LISTED TWICE IS ONE OBJECT, and get_object hands back the same datablock both times.
    # The loop below linked it on the first pass and then found it present on the second, so a
    # fresh link came back as `alreadyPresent` with `linked` empty - a response contradicting its
    # own objectCountBefore/objectCount, which moved. Deduplicated AFTER the resolve above, so an
    # unknown name anywhere in the list is still refused before anything mutates.
    objs = list({o.name: o for o in objs}.values())
    move = take_bool(params, "move", default=False)

    before = len(coll.objects)
    # ONE PASS EACH, not one scan per object: `name in coll.objects` is a linear scan (ledger A12)
    # and so is the per-object `in c.objects` the move path used to do across every collection.
    in_coll = {o.name for o in coll.objects}
    root = _scene_root()
    homes = _collections_index() if move else {}
    by_name = {c.name: c for c in bpy.data.collections} if move else {}
    root_names = {o.name for o in root.objects} if move else set()
    # BUILT BEFORE THE LOOP AND STILL CORRECT INSIDE IT, which is the one thing worth checking
    # before hoisting an index past a mutation: unlinking X from C cannot change which collections
    # Y is in, and the links this loop makes are all into `coll`, which is filtered out below.
    # root_names is the exception - the loop DOES unlink from the root - so it is kept in step by
    # discarding the name at the point of the unlink rather than left to go stale.
    already, moved_from = [], {}
    for obj in objs:
        if move:
            others = [by_name[n] for n in homes.get(obj.name) or () if n != coll.name]
            for other in others:
                other.objects.unlink(obj)
            # THE SCENE ROOT IS A COLLECTION TOO and is not in bpy.data.collections, so a
            # move that ignored it would leave the object linked at the top level and the
            # "moved" claim would be false.
            if obj.name in root_names:
                root.objects.unlink(obj)
                root_names.discard(obj.name)
                others.append(root)
            if others:
                moved_from[obj.name] = sorted(c.name for c in others)
        if obj.name in in_coll:
            already.append(obj.name)
            continue
        coll.objects.link(obj)
        in_coll.add(obj.name)

    # MEASURED ON THE COLLECTION, not on the return of link(). Every requested object must be in
    # there afterwards or this op is reporting a success it did not achieve. RE-READ from
    # coll.objects rather than reusing in_coll above - in_coll is what this function BELIEVES it
    # did, and a postcondition checking a belief against itself checks nothing.
    final = {o.name for o in coll.objects}
    missing = [o.name for o in objs if o.name not in final]
    if missing:
        raise MifOpError("linked into '%s' but %d object(s) are not in it afterwards: %s. The "
                         "collection now holds %d object(s)."
                         % (coll.name, len(missing), ", ".join(missing[:10]), len(coll.objects)))
    return {
        "ok": True,
        "collection": coll.name,
        "linked": sorted(o.name for o in objs if o.name not in already),
        "alreadyPresent": sorted(already),
        "movedFrom": moved_from,
        "objectCountBefore": before,
        "objectCount": len(coll.objects),
        "inScene": coll.name in _in_scene_names(),
        # THE POINT OF THE WHOLE MODULE. If the collection is an orphan, filling it changes nothing
        # anybody can see, and that has to be said here rather than discovered at render time.
        "note": (None if coll.name in _in_scene_names() else
                 "this collection is NOT linked into the scene, so its objects are outside the "
                 "view layer and the render no matter what is in it. It is still a valid light "
                 "linking receiver or blocker, which is the one case where that is fine."),
    }


def op_unlink_objects(params):
    """Take objects out of a collection, refusing to strand one in no collection at all.

    AN OBJECT IN ZERO COLLECTIONS STILL EXISTS in bpy.data and is in NO scene: invisible in the
    viewport, absent from the render, and gone from the outliner. It is not deleted, so nothing
    warns, and it survives a save. That is the object-level twin of an orphaned collection and it is
    refused here rather than produced silently - pass allowOrphans to mean it.

    params:
      collection / name (str)   required
      objects (list[str])       or `object` for one. Required.
      allowOrphans (bool)       permit leaving an object in no collection. Default false.
    """
    reject_unknown(params, _UNLINK_KEYS, "unlink_objects")
    coll = _link_target(params, "unlink_objects")
    names = _object_names(params, "objects", "object")
    if not names:
        raise MifOpError("'objects' is required - a list of object names, or 'object' for one. "
                         "NOTHING was changed.")
    objs = [get_object(n) for n in names]
    # THE SAME DEDUP AS link_objects, and here it was worse than a wrong report. The unlink loop
    # below called coll.objects.unlink twice on one datablock, and Blender RAISES when you unlink
    # something the collection does not hold (measured on 4.4.0: RuntimeError "Object 'X' not in
    # collection 'Y'") - so the op unlinked the object, then threw a RuntimeError straight past
    # every "NOTHING was changed" refusal in this file, having changed something.
    objs = list({o.name: o for o in objs}.values())
    allow = take_bool(params, "allowOrphans", default=False)

    in_coll = {o.name for o in coll.objects}     # one pass; membership on it is a linear scan
    absent = [o.name for o in objs if o.name not in in_coll]
    if absent:
        raise MifOpError("%d object(s) are not in '%s' to begin with: %s. NOTHING was changed."
                         % (len(absent), coll.name, ", ".join(absent[:10])))

    # COMPUTED BEFORE THE UNLINK, on every object, so the refusal fires before any mutation rather
    # than partway through the list.
    would_strand = []
    root = _scene_root()
    index = _collections_index()                 # one pass, read below - nothing mutates until later
    root_names = {o.name for o in root.objects}
    for obj in objs:
        homes = set(index.get(obj.name) or ())
        if obj.name in root_names:
            homes.add(root.name)
        if homes - {coll.name}:
            continue
        would_strand.append(obj.name)
    if would_strand and not allow:
        raise MifOpError("this would leave %d object(s) in NO collection at all: %s. They would "
                         "still exist but be in no scene - invisible, unrendered and absent from "
                         "the outliner, with nothing to warn you. Link them somewhere else first, "
                         "delete them with delete_object, or pass allowOrphans:true to mean it. "
                         "NOTHING was changed."
                         % (len(would_strand), ", ".join(would_strand[:10])))

    before = len(coll.objects)
    for obj in objs:
        coll.objects.unlink(obj)
    # RE-READ after the loop, for the same reason link_objects re-reads: this postcondition exists
    # to catch an unlink that did not take, so it must ask the collection rather than the set this
    # function built from what it intended to do.
    remaining = {o.name for o in coll.objects}
    still = [o.name for o in objs if o.name in remaining]
    if still:
        raise MifOpError("unlinked from '%s' but %d object(s) are still in it: %s."
                         % (coll.name, len(still), ", ".join(still[:10])))
    # REBUILT HERE, after the unlink loop above - the pre-check's index is stale by exactly the
    # change this postcondition exists to measure.
    after = _collections_index()
    root_after = {o.name for o in root.objects}
    stranded = [o.name for o in objs
                if not after.get(o.name) and o.name not in root_after]
    return {
        "ok": True,
        "collection": coll.name,
        "unlinked": sorted(o.name for o in objs),
        "objectCountBefore": before,
        "objectCount": len(coll.objects),
        "nowInNoCollection": sorted(stranded),
        "note": (None if not stranded else
                 "%d object(s) are now in NO collection: they exist but are in no scene. This was "
                 "requested with allowOrphans." % len(stranded)),
    }


def op_set_collection_visibility(params):
    """The four different things "hide this collection" can mean, taken by their real names.

    Two are global properties on the collection and two are PER VIEW LAYER properties on a
    LayerCollection, and writing the wrong one is a silent no-op that reads back as a success on the
    property that WAS written. So each is a distinctly named parameter, the response says which view
    layer a per-layer write landed on, and all four are read back.

    EXCLUDE IS NOT HIDING. An excluded collection leaves the depsgraph entirely: its objects are not
    evaluated, so constraints targeting them, drivers reading them and modifiers depending on them
    all change behaviour. That is reported as a distinct field rather than folded in with the rest.

    params:
      collection / name (str)   required
      viewLayer (str)           which view layer the per-layer writes apply to. Default active.
      hideViewport (bool)       GLOBAL monitor icon
      hideRender (bool)         GLOBAL camera icon
      exclude (bool)            PER VIEW LAYER checkbox - removes it from evaluation entirely
      hideInViewLayer (bool)    PER VIEW LAYER eye icon - the one people mean
      indirectOnly (bool)       PER VIEW LAYER - contributes only bounce light
      holdout (bool)            PER VIEW LAYER - punches alpha through the render
    """
    reject_unknown(params, _VIS_KEYS, "set_collection_visibility")
    coll = _find_collection(take(params, "collection", "name", required=True, kind=str),
                            "set_collection_visibility")
    vl_name = take(params, "viewLayer", kind=str)
    if vl_name:
        vl = bpy.context.scene.view_layers.get(vl_name)
        if vl is None:
            known = sorted(v.name for v in bpy.context.scene.view_layers)
            raise MifOpError("no view layer named '%s' in scene '%s'. Present: %s. NOTHING was "
                             "changed." % (vl_name, bpy.context.scene.name, ", ".join(known)))
    else:
        vl = bpy.context.view_layer

    per_layer = {"exclude": "exclude", "hideInViewLayer": "hide_viewport",
                 "indirectOnly": "indirect_only", "holdout": "holdout"}
    # CHECKED AS BOOLEANS, AND THE CHECKED VALUE IS THE ONE WRITTEN. These went straight to setattr, and Blender
    # coerces - a dict became True, so a caller who sent garbage hid the collection from every render and was told it
    # had worked. take_bool refuses anything that is not a boolean, a number or a yes/no string. Until 2026-10-26 the
    # two global flags were checked and then written as bool(params[k]) - bool('false') is True, so 'false' hid the
    # collection and the read-back, comparing with the same bool(), agreed - and the four per-layer flags were not
    # checked at all.
    try:
        wanted_layer = {k: take_bool(params, k) for k in per_layer if params.get(k) is not None}
        wanted_global = {k: take_bool(params, k) for k in ("hideViewport", "hideRender")
                         if params.get(k) is not None}
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    if not wanted_layer and not wanted_global:
        raise MifOpError("nothing to do - pass at least one of hideViewport, hideRender, exclude, "
                         "hideInViewLayer, indirectOnly or holdout. NOTHING was changed.")

    lc = _layer_collection(coll.name, vl)
    if wanted_layer and lc is None:
        raise MifOpError("'%s' is not in view layer '%s' at all, so exclude, hideInViewLayer, "
                         "indirectOnly and holdout have nowhere to be written - those live on a "
                         "LayerCollection, which only exists for a collection linked into the "
                         "scene. Link it first. NOTHING was changed." % (coll.name, vl.name))

    if "hideViewport" in wanted_global:
        coll.hide_viewport = bool(wanted_global["hideViewport"])
    if "hideRender" in wanted_global:
        coll.hide_render = bool(wanted_global["hideRender"])
    for key, attr in per_layer.items():
        if key in wanted_layer:
            setattr(lc, attr, bool(wanted_layer[key]))

    lc = _layer_collection(coll.name, vl)     # re-fetch: excluding rebuilds the layer tree
    after = {
        "hideViewport": bool(coll.hide_viewport),
        "hideRender": bool(coll.hide_render),
        "exclude": bool(lc.exclude) if lc else None,
        "hideInViewLayer": bool(lc.hide_viewport) if lc else None,
        "indirectOnly": bool(lc.indirect_only) if lc else None,
        "holdout": bool(lc.holdout) if lc else None,
    }
    # EVERY REQUESTED WRITE VERIFIED INDIVIDUALLY. Four of these are on a different datablock from
    # the other two, and a version that silently wrote the collection-level property when the caller
    # meant the eye icon would read back as a perfect success on the wrong field.
    wrong = {k: (v, after.get(k)) for k, v in list(wanted_global.items()) + list(wanted_layer.items())
             if after.get(k) != bool(v)}
    if wrong:
        raise MifOpError("wrote %s to '%s' but it reads back as %s afterwards"
                         % ({k: v[0] for k, v in wrong.items()}, coll.name,
                            {k: v[1] for k, v in wrong.items()}))
    # INDIRECT ONLY IS A CYCLES FEATURE, and EEVEE renders it as if it were never set. Reported by the second seat
    # (mailbox id 75) and MEASURED 2026-09-16: an orange box in its own collection over a white floor, 160x120, flag off
    # vs on - EEVEE 0.0 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 alike, Cycles 40.9-41.5 (0-255 mean difference). Still
    # written - a scene is often set up under one engine and rendered with another - but said, the way set_light_shadow
    # says it for its engine-bound keys.
    engine = bpy.context.scene.render.engine
    inert = ["indirectOnly"] if (wanted_layer.get("indirectOnly") and "CYCLES" not in engine) else []
    return {
        "ok": True,
        "collection": coll.name,
        "viewLayer": vl.name,
        "perViewLayerWrites": sorted(wanted_layer),
        "globalWrites": sorted(wanted_global),
        "objectCount": len(coll.objects),
        "excludedFromEvaluation": bool(after["exclude"]),
        "note": ("exclude removes the collection from the depsgraph entirely for view layer '%s' - "
                 "its objects stop being evaluated, so constraints, drivers and modifiers that "
                 "depend on them change behaviour too." % vl.name) if after["exclude"] else None,
        "engine": engine,
        "inertUnderThisEngine": inert or None,
        "engineNote": ("indirectOnly is written, and this scene renders with %s, which does not read it - only Cycles "
                       "renders a collection as indirect-only." % engine) if inert else None,
        **after,
    }


def op_delete_collection(params):
    """Remove a collection, and say what happened to the objects that were in it.

    bpy.data.collections.remove() deletes the COLLECTION and leaves its objects alone - which sounds
    safe and is how objects end up in no collection at all: still in bpy.data, in no scene,
    invisible everywhere and surviving the save. So this decides that explicitly rather than by
    default. Objects that would be stranded are moved to the scene root unless reparentTo names
    somewhere else, or deleted outright if deleteObjects is set.

    Child collections are re-linked to the parent rather than orphaned, for the same reason.

    params:
      collection / name (str)   required
      deleteObjects (bool)      delete the objects too, instead of rehoming them. Default false.
      reparentTo (str)          where to rehome objects and child collections. Default scene root. Only objects
                                the deletion would strand are rehomed; objectsKeptElsewhere names the rest.
    """
    reject_unknown(params, _DELETE_KEYS, "delete_collection")
    name = take(params, "collection", "name", required=True, kind=str)
    coll = _find_collection(name, "delete_collection")
    if coll is _scene_root():
        raise MifOpError("that is the scene's root collection and cannot be deleted. NOTHING was "
                         "changed.")
    delete_objects = take_bool(params, "deleteObjects", default=False)
    to_name = take(params, "reparentTo", kind=str)
    target = _find_collection(to_name, "delete_collection") if to_name else _scene_root()
    if target is coll:
        raise MifOpError("reparentTo names the collection being deleted. NOTHING was changed.")

    members = list(coll.objects)
    children = list(coll.children)
    # WHICH OBJECTS WOULD BE STRANDED, computed before anything is removed. An object in another
    # collection as well is fine and must not be moved - doing so would silently reorganise a scene.
    index = _collections_index()                        # before any removal below
    root_names = {o.name for o in _scene_root().objects}   # was re-fetched per object, and scanned
    stranded = [o for o in members
                if not (set(index.get(o.name) or ()) - {coll.name}) and o.name not in root_names]
    # THE MEMBERS THAT ARE NOT STRANDED, AND WHERE THEY STAY. Second seat, mailbox id 135 (2): reparentTo "rehomed nothing"
    # - objectsRehomed [] - for an object link_objects had LINKED into a second collection (it links, it does not move,
    # by default). Right, and silent. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 through this op:
    # collection A holding Solo (only there), Shared (also in B) and Rooted (also in the scene root), deleted with
    # reparentTo Dest - Solo went to Dest, Shared stayed in B, Rooted stayed at the root, and the reply named only Solo
    # (scratchpad measure_delete_collection_reparent.py). So the others are named, with the homes they keep.
    kept = {} if delete_objects else {
        o.name: sorted((set(index.get(o.name) or ()) - {coll.name})
                       | ({"<scene root>"} if o.name in root_names else set()))
        for o in members if o not in stranded}

    deleted, rehomed = [], []
    joints, freed, not_freed, bake_note, joints_note = [], [], [], None, None
    if delete_objects:
        # THROUGH ops_scene._delete, the rigid-body-safe path delete_object and clear_scene use (2026-09-28): a bare
        # bpy.data.objects.remove frees a rigid body or joint without taking it out of Bullet's world (docs/22 B51, the
        # crash through delete_object on 5.2.1 and 4.2.17), and this was that bare remove - the same shape the 2026-09-21
        # pack-rule review found in import_mesh's clear. So a joint holding one of the collection's bodies goes with it
        # and is named, and a rigid body bake the delete leaves stale is freed and said. purge=False, as before.
        from .ops_scene import _delete, _freed_note, _joints_note   # runtime: op modules import ops_common, not each other
        deleted, _purged, freed, not_freed, joints = _delete(list(members), False)
        bake_note, joints_note = _freed_note(freed, not_freed), _joints_note(joints)
    else:
        for obj in stranded:
            target.objects.link(obj)
            rehomed.append(obj.name)
    relinked = []
    for child in children:
        if child.name not in {c.name for c in target.children}:
            target.children.link(child)
            relinked.append(child.name)

    bpy.data.collections.remove(coll)

    # POSTCONDITION ON THE THING THAT MATTERS, which is not that the collection is gone - it is that
    # nothing was stranded by its going. Both are checked.
    if bpy.data.collections.get(name) is not None:
        raise MifOpError("removed '%s' but it is still in bpy.data.collections afterwards." % name)
    # REBUILT after bpy.data.collections.remove above, for the same reason as unlink_objects'
    # postcondition: the pre-check's index describes the scene this op has just changed.
    after = _collections_index()
    root_after = {o.name for o in _scene_root().objects}
    homeless = sorted(o.name for o in bpy.data.objects
                      if not after.get(o.name) and o.name not in root_after)
    return {
        "ok": True,
        "deletedCollection": name,
        "objectsDeleted": sorted(deleted),
        "objectsRehomedTo": target.name if rehomed else None,
        "objectsRehomed": sorted(rehomed),
        "childCollectionsRelinked": sorted(relinked),
        "objectsInNoCollection": homeless,
        "objectsKeptElsewhere": kept,
        # deleteObjects' rigid body side, in delete_object's words - see the _delete call above.
        "jointsRemovedWithBodies": joints or None, "jointsNote": joints_note,
        "freedRigidBodyBake": freed or None, "rigidBodyBakeNotFreed": not_freed or None,
        "rigidBodyBakeNote": bake_note,
        "keptNote": (("%d object(s) were also in another collection or the scene root, so they stay there and were NOT "
                      "linked into '%s': %s. reparentTo rehomes only the objects this deletion would strand; to move "
                      "the others, link_objects with move:true."
                      % (len(kept), target.name,
                         "; ".join("%s (%s)" % (k, ", ".join(v)) for k, v in sorted(kept.items())[:10])
                         + (" ..." if len(kept) > 10 else "")))
                     if kept and to_name else None),
        "note": ("%d object(s) are in no collection after this. That is not something this op did "
                 "on purpose - check them." % len(homeless)) if homeless else None,
    }


def op_duplicate_collection(params):
    """Duplicate a collection with everything in it - its objects and its child collections - under a new name.

    THE GAP, 2026-10-04: a whole collection copied into a low-poly and a high-poly pair for a bake went through raw
    Python. Every object anywhere under the collection is copied once (duplicate_object's copy: its own data, or the
    same data with linked), every child collection is copied with the copies linked in the same places, and a copy
    whose parent was copied too is parented to that copy. An object under two child collections is copied once and
    linked into both copies.

    collection (or name) is the source. newName names the copy (default: the source's name, which Blender suffixes
    .001); child collections keep their names, suffixed by Blender - every name is echoed. parent puts the copy under
    that collection; by default it goes wherever the source is linked (the scene root for a collection linked nowhere).
    """
    reject_unknown(params, _DUPLICATE_KEYS, "duplicate_collection")
    from .ops_create import copy_object, copy_row   # local: op modules import ops_common, not each other, at load
    src = _find_collection(take(params, "collection", "name", required=True, kind=str), "duplicate_collection")
    new_name = take(params, "newName", kind=str)
    linked = take_bool(params, "linked", default=False)
    parent_name = take(params, "parent", kind=str)
    if parent_name is not None:
        parents = [_find_collection(parent_name, "duplicate_collection")]
    else:
        # ONE PASS OVER THE LINKS, comparing blocks: `name in c.children` scans the children per collection.
        parents = [c for c in bpy.data.collections for child in c.children if child == src]
        if src.name in _scene_root().children or not parents:
            parents.append(_scene_root())

    order, seen = [], set()

    def gather(coll):
        if coll.name in seen:
            return
        seen.add(coll.name)
        order.append(coll)
        for child in coll.children:
            gather(child)

    gather(src)
    sources, taken = [], set()
    for coll in order:
        for obj in coll.objects:
            if obj.name not in taken:
                taken.add(obj.name)
                sources.append(obj)
    obj_map, asked = {}, {}
    for obj in sources:
        obj_map[obj], asked[obj] = copy_object(obj, linked)
    for obj, new in obj_map.items():
        if obj.parent in obj_map:
            new.parent = obj_map[obj.parent]
            new.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    coll_map = {}
    for coll in order:
        made = bpy.data.collections.new(new_name if coll is src and new_name else coll.name)
        coll_map[coll] = made
        for obj in coll.objects:
            made.objects.link(obj_map[obj])
    for coll in order:
        for child in coll.children:
            coll_map[coll].children.link(coll_map[child])
    top = coll_map[src]
    for parent in parents:
        parent.children.link(top)
    bpy.context.view_layer.update()

    wanted = new_name or src.name
    return {
        "collection": top.name,
        "requestedName": wanted,
        "nameWasSuffixed": top.name != wanted,
        "source": src.name,
        "parents": [p.name for p in parents],
        "inScene": top.name in _in_scene_names(),
        "linked": linked,
        "collections": [{"source": c.name, "name": coll_map[c].name} for c in order],
        "objects": [copy_row(o, obj_map[o], asked[o], linked) for o in sources],
        "objectCount": len(obj_map),
        "materialsShared": True,
    }


_CI_KEYS = {"collection", "mode", "transforms", "points", "name", "target", "instanceOffset", "objects", "limit"}
_CI_MODES = ("ADD", "LIST", "REALIZE")


def _vec3(v, what, tail):
    if (not isinstance(v, (list, tuple)) or len(v) != 3
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
        raise MifOpError("%s is [x, y, z] numbers, got %r. %s" % (what, v, tail))
    return tuple(float(x) for x in v)


def _instance_boxes(names):
    """{empty name: (objects drawn, world box)} for the named collection-instance Empties, in ONE pass over the
    depsgraph's instances - an instance's object is a temporary valid only inside the loop, so the box is read there
    (vertices for a mesh, the evaluated bound box otherwise)."""
    import mathutils                                                     # noqa: PLC0415
    deps = bpy.context.evaluated_depsgraph_get()
    acc = {n: [0, [float("inf")] * 3, [float("-inf")] * 3] for n in names}
    for inst in deps.object_instances:
        if not inst.is_instance or inst.parent is None:
            continue
        a = acc.get(inst.parent.original.name)
        if a is None:
            continue
        a[0] += 1
        o = inst.object
        mw = inst.matrix_world
        pts = ([mw @ v.co for v in o.data.vertices] if o.type == "MESH" and o.data is not None
               else [mw @ mathutils.Vector(c) for c in o.bound_box])
        lo, hi = a[1], a[2]
        for p in pts:
            for i in range(3):
                lo[i] = min(lo[i], p[i])
                hi[i] = max(hi[i], p[i])
    out = {}
    for n, (count, lo, hi) in acc.items():
        box = None if lo[0] == float("inf") else {
            "min": [round(x, 6) for x in lo], "max": [round(x, 6) for x in hi],
            "size": [round(hi[i] - lo[i], 6) for i in range(3)]}
        out[n] = (count, box)
    return out


def _instance_row(empty, boxes):
    n, box = boxes.get(empty.name, (0, None))
    coll = empty.instance_collection
    return {"object": empty.name, "collection": coll.name if coll else None,
            "location": [round(x, 6) for x in empty.matrix_world.translation],
            "rotation": [round(x, 6) for x in empty.matrix_world.to_euler("XYZ")],
            "scale": [round(x, 6) for x in empty.matrix_world.to_scale()],
            "instanceOffset": [round(x, 6) for x in coll.instance_offset] if coll else None,
            "drawnObjects": n, "bboxWorld": box}


def op_add_collection_instance(params):
    """Instance a collection - an Empty with instance_type COLLECTION - at transforms or points, list them, or make them
    real.

    WHY (second 1.3 audit, 2026-10-25). Dressing a scene from a kit without copying its geometry 500 times; the Empty was
    made only inside append_from_file's link path. The data route (Object(instance_type='COLLECTION')) is the one
    object.collection_instance_add takes (5.2 bl_ui/space_view3d.py:2754), and it works headless.

    params:
      collection (str)       the collection instanced (ADD; LIST filters by it)
      mode                   ADD (default), LIST, REALIZE
      transforms (list)      ADD: [{location, rotation (radians, as transform_object), scale}] - one Empty each
      points (list)          ADD: [[x, y, z], ...] - one Empty each at a point (transforms OR points)
      name (str)             ADD: the Empties' name (default '<collection>_inst'; Blender suffixes .001)
      target (str)           ADD: the collection the Empties go in (default the scene's root collection)
      instanceOffset         ADD: [x, y, z] sets the collection's instance offset (it applies to every instance of it)
      objects (list)         REALIZE: instance Empties to make real (their objects copied in place, the Empty removed);
                             LIST: only these
      limit (int)            LIST / ADD reply rows (default 200)
    A target inside the instanced collection's own tree is REFUSED: Blender drops such a cycle silently and draws
    nothing. Each Empty's world box comes from what it actually draws.
    """
    reject_unknown(params, _CI_KEYS, "add_collection_instance")
    mode = str(take(params, "mode", default="ADD", kind=str)).upper()
    if mode not in _CI_MODES:
        raise MifOpError("mode is one of %s, got %r. NOTHING was changed." % (", ".join(_CI_MODES), mode))
    limit = params.get("limit", 200)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise MifOpError("limit is a whole number 1 or more, got %r. NOTHING was changed." % (limit,))
    tail = "NOTHING was changed."
    if mode == "LIST":
        cname = take(params, "collection", kind=str)
        only = set(_object_names(params, "objects"))
        coll = _find_collection(cname, "add_collection_instance") if cname else None
        picked = [o for o in bpy.data.objects
                  if o.instance_type == "COLLECTION" and o.instance_collection is not None
                  and (coll is None or o.instance_collection == coll) and (not only or o.name in only)]
        boxes = _instance_boxes([o.name for o in picked[:limit]])
        rows = [_instance_row(o, boxes) for o in picked]
        missing = sorted(only - {r["object"] for r in rows})
        if missing:
            raise MifOpError("not collection instances: %s. NOTHING was read." % ", ".join(missing))
        return {"mode": mode, "count": len(rows), "instances": rows[:limit], "truncated": len(rows) > limit}
    if mode == "REALIZE":
        names = _object_names(params, "objects")
        if not names:
            raise MifOpError("REALIZE needs objects: the instance Empties to make real. %s" % tail)
        empties = []
        for n in names:
            o = get_object(n)
            if o.instance_type != "COLLECTION" or o.instance_collection is None:
                raise MifOpError("'%s' instances no collection. %s" % (o.name, tail))
            empties.append(o)
        from .ops_common import select_only, selection_restore, selection_snapshot   # noqa: PLC0415
        before = set(bpy.data.objects.keys())
        made_by = {}
        # duplicates_make_real CLEARS the Empty's instance_collection (measured on 5.2.1), so the name is kept first.
        coll_of = {e.name: e.instance_collection.name for e in empties}
        snap = selection_snapshot()
        try:
            for e in empties:
                select_only([e])
                prior = set(bpy.data.objects.keys())
                bpy.ops.object.duplicates_make_real(use_base_parent=False, use_hierarchy=True)
                made_by[e.name] = sorted(set(bpy.data.objects.keys()) - prior)
        finally:
            selection_restore(snap)
        rows = []
        for e in empties:
            nm = e.name
            rows.append({"object": nm, "collection": coll_of[nm], "made": made_by[nm]})
            remove_objects([e])
        return {"mode": mode, "realized": rows,
                "objectsMade": len(set(bpy.data.objects.keys()) - before),
                "note": "each Empty's drawn objects are now real objects (meshes share their data with the "
                        "collection's), and the Empty is removed."}
    # ADD
    coll = _find_collection(take(params, "collection", required=True, kind=str), "add_collection_instance")
    tname = take(params, "target", kind=str)
    target = _find_collection(tname, "add_collection_instance") if tname else _scene_root()
    if tname and target.name in _walk(coll):
        raise MifOpError("'%s' is inside '%s' (or is it): an instance there instances itself, which Blender drops "
                         "silently and draws nothing. Pick a target outside it. %s" % (target.name, coll.name, tail))
    transforms = params.get("transforms")
    points = params.get("points")
    if (transforms is None) == (points is None):
        raise MifOpError("ADD takes transforms OR points (one Empty per entry). %s" % tail)
    import mathutils                                                     # noqa: PLC0415
    plan = []
    if points is not None:
        if not isinstance(points, list) or not points:
            raise MifOpError("points is a non-empty list of [x, y, z]. %s" % tail)
        for p in points:
            plan.append((_vec3(p, "a point", tail), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    else:
        if not isinstance(transforms, list) or not transforms:
            raise MifOpError("transforms is a non-empty list of {location, rotation, scale}. %s" % tail)
        for t in transforms:
            if not isinstance(t, dict) or any(k not in ("location", "rotation", "scale") for k in t):
                raise MifOpError("each transform is {location, rotation, scale}, got %r. %s" % (t, tail))
            sc = t.get("scale", (1.0, 1.0, 1.0))
            if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                sc = (sc, sc, sc)
            plan.append((_vec3(t.get("location", (0, 0, 0)), "location", tail),
                         _vec3(t.get("rotation", (0, 0, 0)), "rotation", tail), _vec3(sc, "scale", tail)))
    off = params.get("instanceOffset")
    if off is not None:
        coll.instance_offset = mathutils.Vector(_vec3(off, "instanceOffset", tail))
    base = take(params, "name", default="%s_inst" % coll.name, kind=str)
    made = []
    for loc, rot, scl in plan:
        e = bpy.data.objects.new(base, None)
        e.instance_type = "COLLECTION"
        e.instance_collection = coll
        e.location, e.rotation_euler, e.scale = loc, rot, scl
        target.objects.link(e)
        made.append(e)
    bpy.context.view_layer.update()
    boxes = _instance_boxes([e.name for e in made[:limit]])
    rows = [_instance_row(e, boxes) for e in made]
    empty_draws = [r["object"] for r in rows if not r["drawnObjects"]]
    # requestedName is the base every instance was made from; nameWasSuffixed says whether the FIRST one got it (the
    # rest are suffixed by design, and each is named in instances)
    out = {"mode": mode, "collection": coll.name, "target": target.name, "count": len(made),
           "requestedName": base, "nameWasSuffixed": bool(made) and made[0].name != base,
           "instances": rows[:limit], "truncated": len(rows) > limit,
           "instanceOffset": [round(x, 6) for x in coll.instance_offset]}
    if empty_draws:
        out["warning"] = ("%d instance(s) draw nothing - the collection is empty or excluded: %s"
                          % (len(empty_draws), ", ".join(empty_draws[:10])))
    return out


OPS = {
    "add_collection_instance": op_add_collection_instance,
    "duplicate_collection": op_duplicate_collection,
    "create_collection": op_create_collection,
    "list_collections": op_list_collections,
    "link_objects": op_link_objects,
    "unlink_objects": op_unlink_objects,
    "set_collection_visibility": op_set_collection_visibility,
    "delete_collection": op_delete_collection,
}
