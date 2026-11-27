"""Values between a caller, the wire and Studio, decided by the property's type from the catalog.

A caller writes friendly JSON ([1, 2, 3] or {"x": 1, "y": 2, "z": 3} for a Vector3, "Plastic" for an Enum.Material,
"Workspace.Part" for an Instance). encode() turns it into the wire form the plugin decodes, or refuses it with a reason
BEFORE anything is sent. decode() turns the plugin's read-back into the friendly form. same() is the read-back judgement.

Wire form: a primitive as itself; anything else {"$t": type, "v": ...}; an enum item
{"$t": "EnumItem", "enum": E, "v": name}; an instance {"$t": "Instance", "path": [names]}; nil {"$t": "nil"}. A type the
plugin cannot encode comes back {"$t": type, "str": text} and is never written.
"""
import math

_NUMS = {"float": "float", "double": "float", "int": "int", "int64": "int"}
_STRINGS = ("string", "ProtectedString", "ContentId", "BinaryString")
_VECTORS = {"Vector3": 3, "Vector2": 2, "Vector3int16": 3, "Vector2int16": 2, "UDim": 2, "UDim2": 4, "NumberRange": 2,
            "Color3": 3}
_KEYS = {"Vector3": "xyz", "Vector2": "xy", "Vector3int16": "xyz", "Vector2int16": "xy", "Color3": "rgb"}
_INT_VECTORS = ("Vector3int16", "Vector2int16")
REL_TOL = 1e-4
COLOR_TOL = 1.0 / 255


INT32 = (-2 ** 31, 2 ** 31 - 1)
INT64_EXACT = 2 ** 53            # past it a JSON number becomes a Luau double and loses its low digits


def _num(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:        # an int too large for a float (10**400): not a number this codec carries
        return False


def nonfinite(value):
    """"inf" or "-inf" for an infinite float or that word, else None (NaN stays refused). JSON cannot carry infinity, so
    it crosses as {"$t": "double", "v": word} (plugin 0.15.0, lane 4's L4-4): Roblox makes a pathfinding label
    impassable only at math.huge (RB410)."""
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, str) and value in ("inf", "-inf"):
        return value
    return None


# The components of each vector type that may be infinite: its floats (a UDim's Offset is a whole number, Color3 and the
# int16 vectors never are).
_NONFINITE_SLOTS = {"Vector3": (0, 1, 2), "Vector2": (0, 1), "NumberRange": (0, 1), "UDim": (0,), "UDim2": (0, 2)}


def _unwrap_double(x):
    """A typed double's word ({"$t": "double", "v": "inf"}), else x unchanged."""
    if isinstance(x, dict) and x.get("$t") == "double" and set(x) == {"$t", "v"}:
        return x["v"]
    return x


def split_path(path):
    """A dot path or a list of names -> a list of names, without a leading 'game'. None for a bad shape."""
    if isinstance(path, str):
        parts = [p for p in path.split(".")]
    elif isinstance(path, list) and all(isinstance(p, str) for p in path):
        parts = list(path)
    else:
        return None
    if parts and parts[0] == "game":
        parts = parts[1:]
    if any(p == "" for p in parts):
        return None
    return parts


def join_path(parts):
    """A list of names -> "A.B.C", or the list itself when a name holds a dot ("Cube.001"), so the answer can be sent
    back as a path. Not for a dict key, a set or ", ".join: use path_text, which is always a string."""
    if parts is None:
        return None
    return parts if any("." in p for p in parts) else ".".join(parts)


def path_text(parts):
    """A path as text for a message or a dict key, always a string: names joined by "." (a name that holds a dot is
    kept whole, so the text is for reading, not for splitting back). An empty path reads "game", None "(none)"."""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, (list, tuple)):
        return ".".join(str(p) for p in parts) if parts else "game"
    return "(none)" if parts is None else str(parts)


def _vecs(value, n, what):
    if isinstance(value, list) and len(value) == n and all(_num(x) for x in value):
        return list(value), None
    return None, "%s expects %d finite numbers, got %r" % (what, n, value)


def _keys(value, allowed, required, what):
    if not isinstance(value, dict):
        return "%s expects an object with %s, got %r" % (what, ", ".join(allowed), value)
    extra = sorted(set(value) - set(allowed))
    missing = sorted(set(required) - set(value))
    if extra or missing:
        return "%s: %s%s" % (what, ("unknown keys %s " % extra) if extra else "", ("missing %s" % missing) if missing else "")
    return None


def _enum_item(enum, name, enums):
    items = (enums or {}).get(enum)
    if not isinstance(name, str) or (items is not None and name.split(".")[-1] not in items):
        return None, "Enum.%s has no item %r; items: %s" % (enum, name, ", ".join((items or [])[:40]))
    return name.split(".")[-1], None


# Types this codec writes beyond the first twenty. Each wire form is what the plugin's decode builds.
def _encode_more(typ, value, enums):
    if typ == "NumberSequence":
        if not isinstance(value, list) or not 2 <= len(value) <= 20:
            return None, "NumberSequence expects 2-20 keys [time, value, envelope?], got %r" % (value,)
        keys = []
        for k in value:
            if not isinstance(k, list) or len(k) not in (2, 3) or not all(_num(x) for x in k):
                return None, "a NumberSequence key is [time, value, envelope?], got %r" % (k,)
            keys.append([k[0], k[1], k[2] if len(k) == 3 else 0])
        if keys[0][0] != 0 or keys[-1][0] != 1 or any(b[0] < a[0] for a, b in zip(keys, keys[1:])):
            return None, "NumberSequence times run 0 to 1, ascending"
        return {"$t": typ, "v": keys}, None
    if typ == "ColorSequence":
        if not isinstance(value, list) or not 2 <= len(value) <= 20 or not all(
                isinstance(k, list) and len(k) == 4 and all(_num(x) for x in k) for k in value):
            return None, "ColorSequence expects 2-20 keys [time, r, g, b], got %r" % (value,)
        if value[0][0] != 0 or value[-1][0] != 1 or any(b[0] < a[0] for a, b in zip(value, value[1:])):
            return None, "ColorSequence times run 0 to 1, ascending"
        return {"$t": typ, "v": [list(k) for k in value]}, None
    if typ == "Font":
        err = _keys(value, ("family", "weight", "style"), ("family",), "Font")
        if err:
            return None, err
        w, err = _enum_item("FontWeight", value.get("weight", "Regular"), enums)
        if err:
            return None, err
        s, err = _enum_item("FontStyle", value.get("style", "Normal"), enums)
        if err:
            return None, err
        if not isinstance(value["family"], str):
            return None, "Font family is a content id string, e.g. rbxasset://fonts/families/GothamSSm.json"
        return {"$t": typ, "family": value["family"], "weight": w, "style": s}, None
    if typ in ("Rect", "Region3", "Region3int16"):
        n = 2 if typ == "Rect" else 3
        err = _keys(value, ("min", "max"), ("min", "max"), typ)
        if err:
            return None, err
        lo, e1 = _vecs(value["min"], n, typ + " min")
        hi, e2 = _vecs(value["max"], n, typ + " max")
        if e1 or e2:
            return None, e1 or e2
        if typ == "Region3int16" and any(float(x) != int(x) for x in lo + hi):
            return None, "Region3int16 takes whole numbers"
        return {"$t": typ, "v": lo + hi}, None
    if typ == "Ray":
        err = _keys(value, ("origin", "direction"), ("origin", "direction"), "Ray")
        if err:
            return None, err
        o, e1 = _vecs(value["origin"], 3, "Ray origin")
        d, e2 = _vecs(value["direction"], 3, "Ray direction")
        return (None, e1 or e2) if (e1 or e2) else ({"$t": typ, "v": o + d}, None)
    if typ == "PhysicalProperties":
        if value is None:
            # BasePart.CustomPhysicalProperties reads nil until set, and nil puts the material's own back: a value the
            # sweep wrote is restored by sending nil (live 0.748, RB2442). The dump types it without a "?".
            return {"$t": "nil"}, None
        if isinstance(value, str):
            m, err = _enum_item("Material", value, enums)
            return (None, err) if err else ({"$t": typ, "material": m}, None)
        names = ("density", "friction", "elasticity", "frictionWeight", "elasticityWeight")
        err = _keys(value, names + ("acousticAbsorption",), names, "PhysicalProperties")
        if err or not all(_num(value[k]) for k in value):
            return None, err or "PhysicalProperties values are numbers"
        v = [value[k] for k in names] + ([value["acousticAbsorption"]] if "acousticAbsorption" in value else [])
        return {"$t": typ, "v": v}, None
    if typ in ("Faces", "Axes"):
        allowed = ("Right", "Top", "Back", "Left", "Bottom", "Front") + (("X", "Y", "Z") if typ == "Axes" else ())
        if not isinstance(value, list) or not all(isinstance(x, str) and x in allowed for x in value):
            return None, "%s expects a list of %s, got %r" % (typ, ", ".join(allowed), value)
        return {"$t": typ, "v": sorted(set(value))}, None
    if typ == "DateTime":
        if not _num(value) or float(value) != int(value):
            return None, "DateTime is Unix milliseconds, a whole number"
        return {"$t": typ, "v": int(value)}, None
    if typ == "TweenInfo":
        names = ("time", "easingStyle", "easingDirection", "repeatCount", "reverses", "delayTime")
        err = _keys(value, names, ("time",), "TweenInfo")
        if err:
            return None, err
        es, e1 = _enum_item("EasingStyle", value.get("easingStyle", "Quad"), enums)
        ed, e2 = _enum_item("EasingDirection", value.get("easingDirection", "Out"), enums)
        if e1 or e2:
            return None, e1 or e2
        out = {"$t": typ, "time": value["time"], "easingStyle": es, "easingDirection": ed,
               "repeatCount": value.get("repeatCount", 0), "reverses": bool(value.get("reverses", False)),
               "delayTime": value.get("delayTime", 0)}
        if not all(_num(out[k]) for k in ("time", "repeatCount", "delayTime")):
            return None, "TweenInfo time, repeatCount and delayTime are numbers"
        return out, None
    if typ == "Path2DControlPoint":
        err = _keys(value, ("position", "leftTangent", "rightTangent"), ("position",), typ)
        if err:
            return None, err
        parts = []
        for k in ("position", "leftTangent", "rightTangent"):
            v, e = _vecs(value.get(k, [0, 0, 0, 0]), 4, "%s %s (UDim2: xScale, xOffset, yScale, yOffset)" % (typ, k))
            if e:
                return None, e
            parts += v
        return {"$t": typ, "v": parts}, None
    if typ in ("FloatCurveKey", "RotationCurveKey", "ValueCurveKey"):
        err = _keys(value, ("time", "value", "interpolation", "leftTangent", "rightTangent"), ("time", "value"), typ)
        if err:
            return None, err
        mode, e = _enum_item("KeyInterpolationMode", value.get("interpolation", "Linear"), enums)
        if e:
            return None, e
        if typ == "FloatCurveKey" and not _num(value["value"]):
            return None, "FloatCurveKey value is a number"
        inner = value["value"]
        if typ == "RotationCurveKey":
            inner, e = encode("CFrame", inner, enums)
            if e:
                return None, e
        elif typ == "ValueCurveKey":
            inner, e = encode("Variant", inner, enums)
            if e:
                return None, e
        out = {"$t": typ, "time": value["time"], "value": inner, "interpolation": mode}
        for k in ("leftTangent", "rightTangent"):
            if k in value:
                if not _num(value[k]):
                    return None, "%s is a number" % k
                out[k] = value[k]
        return out, None
    if typ == "Content":
        if isinstance(value, str):
            return {"$t": "Content", "uri": value}, None
        if isinstance(value, dict) and set(value) == {"handle"} and isinstance(value["handle"], str):
            return {"$t": "Content", "handle": value["handle"]}, None
        return None, "Content is an asset uri string (rbxassetid://...) or {handle} for an object the plugin holds"
    if typ == "buffer":
        import base64
        if not isinstance(value, str):
            return None, "buffer is base64 text"
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError as exc:
            return None, "buffer is base64 text: %s" % exc
        if len(raw) > 1 << 20:
            return None, "a buffer is at most 1 MB a call"
        return {"$t": "buffer", "b64": value, "len": len(raw)}, None
    if typ == "User":
        return encode("int64", value, enums)
    if typ in ("Array", "Instances"):
        if not isinstance(value, list):
            return None, "%s expects a list" % typ
        out = []
        for x in value:
            w, e = encode("Class.Instance" if typ == "Instances" else "Variant", x, enums)
            if e:
                return None, e
            out.append(w)
        return {"$t": typ, "v": out}, None
    if typ in ("Dictionary", "Map"):
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            return None, "%s expects an object with string keys" % typ
        out = {}
        for k, x in value.items():
            w, e = encode("Variant", x, enums)
            if e:
                return None, "%s[%r]: %s" % (typ, k, e)
            out[k] = w
        return {"$t": "Dictionary", "v": out}, None
    if typ == "Variant":
        if value is None:
            return {"$t": "nil"}, None
        if isinstance(value, float) and math.isinf(value):
            return {"$t": "double", "v": nonfinite(value)}, None
        if isinstance(value, (bool, str)) or _num(value):
            return value, None
        if isinstance(value, dict) and isinstance(value.get("$t"), str):
            t = value["$t"]
            # An enum item inside a Variant, Array or Dictionary (Terrain:WriteVoxels' materials grid): the item says its
            # own enum, {"$t": "EnumItem", "enum": "Material", "v": "Grass"} or {"$t": "Enum.Material", "v": "Grass"}.
            if t == "EnumItem" or t.startswith("Enum."):
                enum = value.get("enum") if t == "EnumItem" else t[5:]
                extra = set(value) - ({"$t", "enum", "v"} if t == "EnumItem" else {"$t", "v"})
                if not isinstance(enum, str) or not enum or extra:
                    return None, ("an enum item in a Variant is {\"$t\": \"EnumItem\", \"enum\": \"Material\", \"v\": \"Grass\"} "
                                  "or {\"$t\": \"Enum.Material\", \"v\": \"Grass\"}, got %r" % (value,))
                if enums is not None and enum not in enums:
                    return None, "there is no Enum.%s in the catalog" % enum
                return encode("Enum." + enum, value.get("v"), enums)
            # An instance inside a Variant, Array or Dictionary (GeometryService:SubtractAsync's parts list): by path,
            # {"$t": "Instance", "path": "Workspace.Part"}, or an object the plugin holds, {"$t": "Object", "handle": id}.
            if t in ("Instance", "Object"):
                key = "path" if t == "Instance" else "handle"
                if set(value) != {"$t", key}:
                    return None, ("an instance in a Variant is {\"$t\": \"Instance\", \"path\": \"Workspace.Part\"} or "
                                  "{\"$t\": \"Object\", \"handle\": id}, got %r" % (value,))
                if t == "Object":
                    if not isinstance(value["handle"], str):
                        return None, "a handle is the id a call returned, a string"
                    return {"$t": "Object", "handle": value["handle"]}, None
                return encode("Class.Instance", value["path"], enums)
            inner = value.get("v", {k: v for k, v in value.items() if k != "$t"})
            if t in ("Variant", "Tuple") or t.startswith("Class."):
                return None, "a Variant's $t names a data type, e.g. Vector3 or EnumItem"
            return encode(t, inner, enums)
        if isinstance(value, list):
            return encode("Array", value, enums)
        if isinstance(value, dict):
            return encode("Dictionary", value, enums)
        return None, "a Variant cannot hold %r" % (value,)
    if typ in ("RaycastParams", "OverlapParams"):
        names = ("filterType", "filter", "ignoreWater", "collisionGroup", "respectCanCollide", "bruteForceAllSlow")
        if typ == "OverlapParams":
            names = ("filterType", "filter", "collisionGroup", "respectCanCollide", "bruteForceAllSlow", "maxParts")
        err = _keys(value, names, (), typ)
        if err:
            return None, err
        out = {"$t": typ}
        if "filterType" in value:
            out["filterType"], e = _enum_item("RaycastFilterType", value["filterType"], enums)
            if e:
                return None, e
        if "filter" in value:
            f, e = encode("Instances", value["filter"], enums)
            if e:
                return None, e
            out["filter"] = f["v"]
        for k in ("ignoreWater", "respectCanCollide", "bruteForceAllSlow"):
            if k in value:
                if not isinstance(value[k], bool):
                    return None, "%s is true or false" % k
                out[k] = value[k]
        if "collisionGroup" in value:
            if not isinstance(value["collisionGroup"], str):
                return None, "collisionGroup is a name"
            out["collisionGroup"] = value["collisionGroup"]
        if "maxParts" in value:
            if not _num(value["maxParts"]) or value["maxParts"] < 0:
                return None, "maxParts is a whole number, 0 for no limit"
            out["maxParts"] = int(value["maxParts"])
        return out, None
    if typ == "CatalogSearchParams":
        # Its fields are not in the API dump: the names are Roblox's datatype page's (read 2026-11-19), every enum one is
        # an enum the dump has, and plugin 0.16.0 reads each field back after building it.
        err = _keys(value, tuple(CATALOG_SEARCH_FIELDS), (), typ)
        if err:
            return None, err
        out = {"$t": typ}
        for k, x in value.items():
            kind = CATALOG_SEARCH_FIELDS[k]
            if kind == "string":
                if not isinstance(x, str):
                    return None, "%s is a string" % k
                out[k] = x
            elif kind == "bool":
                if not isinstance(x, bool):
                    return None, "%s is true or false" % k
                out[k] = x
            elif kind == "int":
                if not _num(x) or float(x) != int(x) or x < 0:
                    return None, "%s is a whole number, 0 or more" % k
                out[k] = int(x)
            elif kind.startswith("[]"):
                if not isinstance(x, list):
                    return None, "%s is a list of Enum.%s names" % (k, kind[2:])
                names = []
                for n in x:
                    item, e = _enum_item(kind[2:], n, enums)
                    if e:
                        return None, "%s: %s" % (k, e)
                    names.append(item)
                out[k] = names
            else:
                item, e = _enum_item(kind, x, enums)
                if e:
                    return None, "%s: %s" % (k, e)
                out[k] = item
        return out, None
    return None, NO_ENCODING


# CatalogSearchParams' fields, by kind: a primitive, an Enum's name, or "[]" + an Enum for a list of its items.
CATALOG_SEARCH_FIELDS = {
    "SearchKeyword": "string", "MinPrice": "int", "MaxPrice": "int", "SortType": "CatalogSortType",
    "SortAggregation": "CatalogSortAggregation", "CategoryFilter": "CatalogCategoryFilter",
    "SalesTypeFilter": "SalesTypeFilter", "BundleTypes": "[]BundleType", "AssetTypes": "[]AvatarAssetType",
    "IncludeOffSale": "bool", "CreatorName": "string", "CreatorType": "CreatorTypeFilter", "CreatorId": "int",
    "Limit": "int",
}


NO_ENCODING = object()

# Every type name the codec carries, and which way: "both" written and read back, "in" only sent
# (an argument), "out" only read, "read" read and never written, "refused" answered with its reason. The live suite
# holds one row per name here and fails when the two counts differ.
CODEC_TYPES = {
    "bool": "both", "float": "both", "double": "both", "int": "both", "int64": "both", "string": "both",
    "ProtectedString": "both", "ContentId": "both", "BinaryString": "both", "Vector3": "both", "Vector2": "both",
    "Vector3int16": "both", "Vector2int16": "both", "UDim": "both", "UDim2": "both", "NumberRange": "both",
    "Color3": "both", "CFrame": "both", "BrickColor": "both", "EnumItem": "both", "Instance": "both",
    "NumberSequence": "both", "ColorSequence": "both", "Font": "both", "Rect": "both", "Region3": "both",
    "Region3int16": "both", "Ray": "both", "PhysicalProperties": "both", "Faces": "both", "Axes": "both",
    "DateTime": "both", "TweenInfo": "both", "Path2DControlPoint": "both", "FloatCurveKey": "both",
    "RotationCurveKey": "both", "ValueCurveKey": "both", "Content": "both", "buffer": "both", "User": "in",
    "Array": "both", "Instances": "both", "Dictionary": "both", "Map": "out", "Variant": "both", "nullable": "both",
    "RaycastParams": "in", "OverlapParams": "in", "CatalogSearchParams": "in", "RaycastResult": "out",
    "Tuple": "out", "Object": "both", "PathWaypoint": "out", "EnumItem in Array": "in",
    "QFont": "refused", "QDir": "refused", "SecurityCapabilities": "read",
}
# Read-only or refused types: read back as {"$t", "str"} at most, never written.
REFUSED_TYPES = {
    "Function": "no wire form: the bridge sends no code",
    "RBXScriptConnection": "only returned by members that take a Function",
    "RBXScriptSignal": "rbx_watch subscribes to signals instead",
    "QFont": "Studio's own Luau cannot read it: 'Unable to get property Font, type QFont' (measured on 0.748)",
    "QDir": "Studio's own Luau cannot read it: 'Unable to get property PluginsDir, type QDir' (measured on 0.748)",
    "SecurityCapabilities": "sandboxing is not the bridge's to change; read only",
    "Secret": "a secret is never decoded or written", "SharedTable": "a runtime handle of an experience",
    "AdReward": "a runtime handle of an experience", "ClipEvaluator": "a runtime handle of an experience",
    "CollectionHandle": "a runtime handle of an experience",
    "DockWidgetPluginGuiInfo": "the plugin's own widget, not the place's",
}


def encode(typ, value, enums=None):
    """(wire, None) or (None, reason). typ is the catalog's type string."""
    if typ is None:
        return None, "the catalog gives this member no type"
    if isinstance(typ, str) and typ.endswith("?"):
        if value is None:
            return {"$t": "nil"}, None
        typ = typ[:-1]
    if typ in ("CoordinateFrame", "OptionalCoordinateFrame"):
        typ = "CFrame"
    base = typ.rstrip("?") if isinstance(typ, str) else typ
    if base in REFUSED_TYPES:
        return None, "type %s is refused: %s" % (base, REFUSED_TYPES[base])
    # A typed value given where its type is already known ({"$t": "buffer", "v": ...} for a buffer parameter): the
    # same value, unwrapped, so a caller can send one form everywhere.
    if isinstance(value, dict) and value.get("$t") == typ and typ not in ("Variant", "Array", "Dictionary", "Map"):
        value = value["v"] if set(value) == {"$t", "v"} else {k: x for k, x in value.items() if k != "$t"}
    more = _encode_more(typ, value, enums)
    if more[1] is not NO_ENCODING:
        return more
    if typ == "bool":
        return (value, None) if isinstance(value, bool) else (None, "bool expects true or false, got %r" % (value,))
    if typ in ("float", "double") and nonfinite(value) is not None:
        return {"$t": "double", "v": nonfinite(value)}, None   # JSON has no infinity: plugin 0.15.0 decodes this
    if typ in _NUMS:
        if not _num(value):
            return None, "%s expects a finite number, got %r" % (typ, value)
        if _NUMS[typ] == "int" and float(value) != int(value):
            return None, "%s expects a whole number, got %r" % (typ, value)
        if typ == "int" and not INT32[0] <= int(value) <= INT32[1]:
            return None, "int is 32-bit, %d to %d, got %r" % (INT32[0], INT32[1], value)
        if typ == "int64" and abs(int(value)) > INT64_EXACT:
            return None, ("int64 %r is past 2^53: the wire's number cannot hold it exactly, so it would be written with "
                          "other low digits; nothing was sent" % (value,))
        # A float property's value crosses as a float, so a whole number is never mistaken for an int in same().
        return (int(value) if _NUMS[typ] == "int" else float(value)), None
    if typ in _STRINGS:
        return (value, None) if isinstance(value, str) else (None, "%s expects a string, got %r" % (typ, value))
    if typ in _VECTORS:
        n, keys = _VECTORS[typ], _KEYS.get(typ)
        if isinstance(value, dict) and keys:
            extra = sorted(set(value) - set(keys))
            if extra or set(value) != set(keys):
                return None, "%s expects keys %s, got %s" % (typ, list(keys), sorted(value))
            value = [value[k] for k in keys]
        if isinstance(value, list) and len(value) == n and any(nonfinite(_unwrap_double(x)) for x in value):
            # An infinite component (UISizeConstraint.MaxSize defaults to (inf, inf), RB2282) crosses as a typed double,
            # as decode reads it, so a read value writes back (plugin 0.17.16 decodes one inside a vector).
            slots = _NONFINITE_SLOTS.get(typ, ())
            value = [_unwrap_double(x) for x in value]
            wrong = [i for i, x in enumerate(value) if nonfinite(x) and i not in slots]
            if not wrong and all(_num(x) or nonfinite(x) for x in value):
                return {"$t": typ, "v": [{"$t": "double", "v": nonfinite(x)} if nonfinite(x) else x for x in value]}, None
        if not isinstance(value, list) or len(value) != n or not all(_num(x) for x in value):
            return None, "%s expects %d finite numbers%s, got %r" % (typ, n, " or {%s}" % ", ".join(keys) if keys else "",
                                                                   value)
        if typ in _INT_VECTORS and any(float(x) != int(x) for x in value):
            return None, "%s expects whole numbers, got %r" % (typ, value)
        return {"$t": typ, "v": list(value)}, None
    if typ == "CFrame":
        if isinstance(value, dict):
            extra = sorted(set(value) - {"position", "rotation"})
            pos, rot = value.get("position"), value.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
            if extra or not isinstance(pos, list) or len(pos) != 3 or not isinstance(rot, list) or len(rot) != 9:
                return None, "CFrame expects {position: [x,y,z], rotation: [9 numbers, optional]} or 12 numbers, got %r" % (value,)
            value = pos + rot
        if not isinstance(value, list) or len(value) != 12 or not all(_num(x) for x in value):
            return None, "CFrame expects 12 finite numbers or {position, rotation}, got %r" % (value,)
        return {"$t": "CFrame", "v": list(value)}, None
    if typ == "BrickColor":
        if isinstance(value, str) or (_num(value) and float(value) == int(value)):
            return {"$t": "BrickColor", "v": value}, None
        return None, "BrickColor expects a name or a number, got %r" % (value,)
    if typ.startswith("Enum."):
        enum = typ[5:]
        items = (enums or {}).get(enum)
        if not isinstance(value, str):
            return None, "%s expects an item name, got %r" % (typ, value)
        name = value.split(".")[-1]
        if items is not None and name not in items:
            return None, "%s has no item %r; items: %s" % (typ, name, ", ".join(items[:40]))
        return {"$t": "EnumItem", "enum": enum, "v": name}, None
    if typ.startswith("Class."):
        if value is None:
            return {"$t": "nil"}, None
        if isinstance(value, dict) and isinstance(value.get("handle"), str):
            return {"$t": "Object", "handle": value["handle"]}, None
        parts = split_path(value)
        if parts is None:
            return None, "%s expects a path ('Workspace.Part' or a list of names) or null, got %r" % (typ, value)
        return {"$t": "Instance", "path": parts}, None
    return None, "type %s has no wire encoding yet: it can be read, not written" % typ


def decode(wire):
    """The plugin's wire value -> what a caller reads (and can send back to encode)."""
    if not isinstance(wire, dict) or "$t" not in wire:
        return wire
    t = wire["$t"]
    if t == "nil":
        return None
    if "str" in wire and "v" not in wire:
        return {"type": t, "text": wire["str"]}
    v = wire.get("v")
    if t in _NONFINITE_SLOTS and isinstance(v, list):
        v = [decode(x) if isinstance(x, dict) and x.get("$t") == "double" else x for x in v]   # inf in a vector
    if t in _KEYS and isinstance(v, list):
        return dict(zip(_KEYS[t], v))
    if t == "CFrame" and isinstance(v, list) and len(v) == 12:
        return {"position": v[:3], "rotation": v[3:]}
    if t == "Instance":
        return join_path(wire.get("path"))
    if t == "double":
        return float(v) if isinstance(v, (int, float, str)) and str(v) in ("inf", "-inf", "nan") or _num(v) else v
    if t == "Object":
        return {"handle": wire.get("handle"), "className": wire.get("className")}
    if t in ("Rect", "Region3", "Region3int16") and isinstance(v, list):
        h = len(v) // 2
        return {"min": v[:h], "max": v[h:]}
    if t == "Ray" and isinstance(v, list):
        return {"origin": v[:3], "direction": v[3:]}
    if t == "PhysicalProperties":
        if "material" in wire:
            return wire["material"]
        names = ("density", "friction", "elasticity", "frictionWeight", "elasticityWeight", "acousticAbsorption")
        return dict(zip(names, v or []))
    if t == "Path2DControlPoint" and isinstance(v, list):
        return {"position": v[0:4], "leftTangent": v[4:8], "rightTangent": v[8:12]}
    if t == "PathWaypoint":
        return {"position": decode(wire.get("position")), "action": decode(wire.get("action")), "label": wire.get("label")}
    if t in ("Font", "TweenInfo", "CatalogSearchParams"):
        return {k: x for k, x in wire.items() if k != "$t"}
    if t in ("FloatCurveKey", "RotationCurveKey", "ValueCurveKey"):
        out = {k: x for k, x in wire.items() if k != "$t"}
        out["value"] = decode(out.get("value"))
        return out
    if t == "Content":
        return wire.get("uri") if wire.get("uri") is not None else {"handle": wire.get("handle")}
    if t == "buffer":
        return {"base64": wire.get("b64"), "length": wire.get("len")}
    if t == "bytes":
        # a string of raw bytes, not UTF-8 (a digest from EncodingService:ComputeStringHash, RB2408): base64, and hex
        # for a short one so a digest reads as one (plugin 0.17.16)
        import base64
        out = {"bytes": True, "base64": wire.get("b64"), "length": wire.get("len")}
        try:
            raw = base64.b64decode(wire.get("b64") or "", validate=True)
        except ValueError:
            return out
        if len(raw) <= 4096:
            out["hex"] = raw.hex()
        return out
    if t in ("Array", "Instances", "Tuple") and isinstance(v, list):
        out = [decode(x) for x in v]
        if wire.get("total") is not None:
            return {"items": out, "total": wire["total"], "truncated": wire.get("truncated", False)}
        return out
    if t == "Dictionary" and isinstance(v, dict):
        return {k: decode(x) for k, x in v.items()}
    if t == "RaycastResult":
        return {k: decode(x) for k, x in wire.items() if k != "$t"}
    return v


def _close(a, b, tol):
    return _num(a) and _num(b) and abs(a - b) <= max(tol, tol * max(abs(a), abs(b)))


_LIST_TAGS = ("Array", "Instances", "Objects")


def _empty_list(w):
    return isinstance(w, dict) and w.get("$t") in _LIST_TAGS and w.get("v") == []


def same(sent, back, typ=None):
    """Does the read-back wire value hold what was sent? Numbers to REL_TOL, colors to 1/255, the rest exactly. An int
    or int64 is exact: typ the catalog type when the caller has it, else a sent int (encode sends a float property's
    value as a float, so an int on the wire is an integer property's): a tolerance of 1e-4 relative is 100 at one
    million and an asset id with a wrong digit."""
    if isinstance(sent, bool) or isinstance(back, bool):
        return sent is back
    base = typ.rstrip("?") if isinstance(typ, str) else None
    if _num(sent) and (base in ("int", "int64") or (base is None and isinstance(sent, int))):
        return _num(back) and back == sent
    if _num(sent):
        return _close(sent, back, REL_TOL)
    if not isinstance(sent, dict):
        return sent == back
    if _empty_list(sent) and _empty_list(back):
        # An empty list has no element to type it: the plugin encodes Studio's empty answer as an Array whatever the
        # member's list type (Selection:Get after Selection:Set({}) reads back {"$t": "Array", "v": []}, live 0.748,
        # RB1136). Two empty lists are the same; a non-empty one is still judged by its own type below.
        return True
    if not isinstance(back, dict) or back.get("$t") != sent.get("$t"):
        return False
    t = sent["$t"]
    if t == "Instance":
        return sent.get("path") == back.get("path")
    if t == "EnumItem":
        return sent.get("v") == back.get("v") and sent.get("enum") == back.get("enum")
    if t == "BrickColor":
        # A number and a name name the same BrickColor; the plugin reads back both.
        return sent.get("v") in (back.get("v"), back.get("number"))
    if t == "nil":
        return True
    sv, bv = sent.get("v"), back.get("v")
    if t == "Faces":
        # Studio lists faces in its own order (live, 0.748): compare as sets.
        return isinstance(bv, list) and set(sv or []) == set(bv)
    if t == "Axes":
        # An Axes made from X and Z reads back X, Z and the four faces on those axes (live, 0.748): compare the axes.
        def axes(names):
            names = set(names or [])
            return {a for a, faces in (("X", ("Right", "Left")), ("Y", ("Top", "Bottom")), ("Z", ("Back", "Front")))
                    if a in names or names & set(faces)}
        return isinstance(bv, list) and axes(sv) == axes(bv)
    if t == "PhysicalProperties" and isinstance(sv, list) and isinstance(bv, list):
        # Studio adds AcousticAbsorption (default 1) to a five-number PhysicalProperties (live, 0.748).
        return len(bv) >= len(sv) and all(_close(a, b, REL_TOL) for a, b in zip(sv, bv))
    if t == "PhysicalProperties" and "material" in sent:
        # A Material's defaults: the plugin reports what it built from them; checked against that (plugin 0.6.2+).
        return isinstance(back.get("intended"), list) and _same_any(back.get("intended"), bv)
    if t in ("Color3", "ColorSequence") or not isinstance(sv, list) or not all(_num(x) for x in sv):
        return _same_any(sent, back, COLOR_TOL if t in ("Color3", "ColorSequence") else None)
    if isinstance(bv, list) and len(sv) == len(bv) and all(_num(x) for x in bv):
        return all(_close(a, b, REL_TOL) for a, b in zip(sv, bv))
    return False


def _same_any(a, b, abs_tol=None):
    """Structural comparison of two wire values: numbers to REL_TOL (or abs_tol), lists and objects element by element.
    Studio stores sequences and fonts in floats and names, so this is the read-back rule for every composite type."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _num(a) and _num(b):
        return abs(a - b) <= abs_tol if abs_tol is not None else _close(a, b, REL_TOL)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_any(x, y, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("$t") != b.get("$t"):
            return False
        keys = set(a) - {"len"}
        return keys <= set(b) and all(_same_any(a[k], b[k], abs_tol) for k in keys)
    return a == b
