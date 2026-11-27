"""The Roblox catalog, loaded: every class and member of one Studio build, with the rules that decide what a plugin may use.

The catalog is roblox_catalog.json beside this file, written by gen_catalog.py from an API dump. The RULES live here, once:
gen_catalog counts with them and the tools refuse with them, so the coverage denominator and a refusal can never disagree.

A member row is [kind, type, readSecurity, writeSecurity, tags, params]:
    kind      "P" property, "F" function, "E" event, "C" callback
    type      a property's value type, a function's return type; "Enum.X" and "Class.X" where the dump's category says so
    security  the dump's own tags; a function, event or callback has one, stored as both
    tags      the dump's member tags (ReadOnly, Hidden, Deprecated, NotScriptable, Yields, ...)
    params    functions, events, callbacks: [[name, type, default or null], ...]; properties: []

the internal notes is the contract.
"""
import difflib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "roblox_catalog.json")

# The security tags a Studio plugin runs with. Everything else - RobloxScriptSecurity, RobloxSecurity, LocalUserSecurity,
# NotAccessibleSecurity, and any tag a later Studio invents - is refused by its own name.
REACHABLE_SECURITY = ("None", "PluginSecurity")
# Tags that take a member out of the coverage denominator. Only NotScriptable is also refused: Hidden members are in use
# (BasePart.Position is Hidden) and Deprecated ones still run.
NOT_COUNTED_TAGS = ("Deprecated", "Hidden", "NotScriptable")
KINDS = {"P": "Property", "F": "Function", "E": "Event", "C": "Callback"}
# Members that put a prompt, a file picker or a purchase dialog on the person's screen. Never called by the bridge: on
# 2026-11-18 PluginManager:ExportPlace(path) opened a native "Export Place" (OBJ) save dialog over Studio although a
# path was passed. The 62 plugin-reachable matches on 0.748 are listed by `rbx_describe_member` as opensDialog.
# PluginMenu:ShowAsync puts a context menu up and yields until the person dismisses it; SoundService's two curve editors
# open an editor window over Studio.
OPENS_DIALOG = re.compile(r"^(Prompt|Export(Place|Selection)$|ImportFbx|SaveSelectedToRoblox|OpenWikiPage|"
                          r"RequestScreenshotPermission|ShowAsync$|Open(Attenuation|Directional)CurveEditor$)")
# Plugin-reachable functions named Open* or Show* that OPENS_DIALOG does not refuse, each with why it may be called.
# The offline suite fails on a reachable Open*/Show* function that is in neither, so a new
# Studio's new opener is decided, not sent by default.
OPENS_DIALOG_ALLOWED = {
    "AdService.ShowRewardedVideoAdAsync": "an ad shown to a player of a running experience, not a Studio prompt",
    "AdService.ShowVideoAd": "deprecated; a video ad for a player of a running experience, not a Studio prompt",
    "CustomLog.Open": "opens a log the object writes to; no window",
    "JointsService.ShowPermissibleJoints": "draws joint previews in the 3D view; no window and nothing to dismiss",
    "MarketplaceService.OpenShop": "the in-experience shop for a player of a running experience, not a Studio prompt",
    "Plugin.OpenScript": "deprecated; opens a script in a Studio editor tab, nothing modal",
    "ScriptEditorService.OpenScriptDocumentAsync": "opens a script in a Studio editor tab, nothing modal",
    "SocialService.ShowSelfView": "the in-experience self view for a player of a running experience",
}
ROOT = "<<<ROOT>>>"


def type_name(t):
    """The dump's {Category, Name} as one string: Enum.X and Class.X keep their category, everything else is its name."""
    if isinstance(t, list):
        # A function returning several values lists one type per value (Model:GetBoundingBox -> CFrame, Vector3).
        return "Tuple(%s)" % ",".join(type_name(x) or "?" for x in t)
    if not isinstance(t, dict):
        return None
    cat, name = t.get("Category"), t.get("Name")
    return "%s.%s" % (cat, name) if cat in ("Enum", "Class") else name


def security_pair(sec):
    """(read, write) from the dump's Security, which is a string for functions and a {Read, Write} for properties."""
    if isinstance(sec, dict):
        return sec.get("Read"), sec.get("Write")
    return sec, sec


def is_reachable(row):
    """A plugin can use it: its read security (a function's only security) is one a plugin runs with."""
    return row[2] in REACHABLE_SECURITY


def is_counted(row):
    """In the coverage denominator: reachable, and none of Deprecated, Hidden, NotScriptable."""
    return is_reachable(row) and not (set(row[4]) & set(NOT_COUNTED_TAGS))


def compute_counts(classes, enums):
    """The numbers the internal notes and the release notes quote, from catalog rows - the same predicates the tools refuse with."""
    by_kind = {k: 0 for k in KINDS.values()}
    security = {}
    total = live = counted = 0
    counted_classes = set()
    for cname, c in classes.items():
        for row in c["m"].values():
            total += 1
            by_kind[KINDS[row[0]]] += 1
            if not (set(row[4]) & set(NOT_COUNTED_TAGS)):
                live += 1
            if is_counted(row):
                counted += 1
                counted_classes.add(cname)
            tags = (row[2], row[3]) if row[0] == "P" else (row[2],)
            for s in tags:
                security[s] = security.get(s, 0) + 1
    return {"classes": len(classes), "enums": len(enums), "enumItems": sum(len(v) for v in enums.values()),
            "members": total, "byKind": by_kind, "live": live,
            "reachable": counted, "reachableClasses": len(counted_classes),
            "securityTags": dict(sorted(security.items(), key=lambda kv: -kv[1])),
            "definitions": {"live": "not tagged %s" % "/".join(NOT_COUNTED_TAGS),
                            "reachable": "live, and read security (a function's security) in %s - the coverage denominator"
                                         % "/".join(REACHABLE_SECURITY),
                            "securityTags": "property read and write counted separately, one each for the rest"}}


class Catalog:
    def __init__(self, doc):
        self.doc = doc
        self.classes = doc.get("classes") or {}
        self.enums = doc.get("enums") or {}
        self.studio = doc.get("studio") or {}
        self._lower = {k.lower(): k for k in self.classes}

    # ---------------------------------------------------------------- lookup
    def chain(self, cls):
        """cls and its superclasses, nearest first. Unknown classes give []."""
        out = []
        while cls in self.classes and cls not in out:
            out.append(cls)
            cls = self.classes[cls].get("s")
        return out

    def is_a(self, cls, base):
        return base in self.chain(cls)

    def find_member(self, cls, name):
        """(owner class, row) for name on cls or a superclass, or (None, None)."""
        for k in self.chain(cls):
            row = self.classes[k]["m"].get(name)
            if row is not None:
                return k, row
        return None, None

    def members_of(self, cls):
        seen = {}
        for k in self.chain(cls):
            for n, row in self.classes[k]["m"].items():
                seen.setdefault(n, (k, row))
        return seen

    def class_did_you_mean(self, cls):
        if isinstance(cls, str) and cls.lower() in self._lower:
            return [self._lower[cls.lower()]]
        return difflib.get_close_matches(str(cls), list(self.classes), n=5, cutoff=0.6)

    def member_did_you_mean(self, cls, name):
        names = list(self.members_of(cls))
        low = {n.lower(): n for n in names}
        if str(name).lower() in low:
            return [low[str(name).lower()]]
        return difflib.get_close_matches(str(name), names, n=5, cutoff=0.6)

    # ---------------------------------------------------------------- answers
    def describe_row(self, owner, name, row):
        out = {"class": owner, "member": name, "kind": KINDS[row[0]], "type": row[1],
               "security": row[2] if row[0] != "P" else {"read": row[2], "write": row[3]},
               "tags": list(row[4]), "reachable": is_reachable(row), "counted": is_counted(row)}
        if row[0] == "P":
            out["writable"] = self.write_refusal(owner, name) is None
        else:
            out["params"] = [{"name": p[0], "type": p[1], "default": p[2]} for p in row[5]]
        if "Deprecated" in row[4]:
            out["deprecated"] = True
        return out

    def describe_class(self, cls):
        c = self.classes[cls]
        rows = self.members_of(cls)
        return {"class": cls, "superclasses": self.chain(cls)[1:], "tags": list(c.get("t") or []),
                "creatable": self.create_refusal(cls) is None,
                "members": len(rows), "reachable": sum(1 for _o, r in rows.values() if is_reachable(r)),
                "byKind": {v: sorted(n for n, (_o, r) in rows.items() if r[0] == k) for k, v in KINDS.items()}}

    def unknown_class(self, cls):
        return {"ok": False, "refusedRule": "unknown-class", "error": "no class %r in the catalog (Studio %s)"
                % (cls, self.studio.get("version") or "unstamped"), "didYouMean": self.class_did_you_mean(cls), "ran": False}

    def unknown_member(self, cls, name):
        return {"ok": False, "refusedRule": "unknown-member", "error": "%s has no member %r in the catalog (Studio %s). If "
                "this Studio is newer, regenerate the catalog: gen_catalog.py --studio-exe auto"
                % (cls, name, self.studio.get("version") or "unstamped"),
                "didYouMean": self.member_did_you_mean(cls, name), "ran": False}

    def read_refusal(self, cls, name, kind=None):
        """None when a plugin may read (or call) cls.name; else the refusal, naming the member's own tag."""
        if cls not in self.classes:
            return self.unknown_class(cls)
        owner, row = self.find_member(cls, name)
        if row is None:
            return self.unknown_member(cls, name)
        if kind and row[0] != kind:
            return {"ok": False, "refusedRule": "unknown-member", "ran": False,
                    "error": "%s.%s is a %s, not a %s" % (owner, name, KINDS[row[0]], KINDS[kind])}
        if not is_reachable(row):
            return {"ok": False, "refusedRule": "security", "securityTag": row[2], "ran": False,
                    "error": "%s.%s is %s: a Studio plugin cannot use it, and MifBridge does not try" % (owner, name, row[2])}
        if "NotScriptable" in row[4]:
            return {"ok": False, "refusedRule": "not-scriptable", "ran": False,
                    "error": "%s.%s is tagged NotScriptable: no script can reach it" % (owner, name)}
        if row[0] == "F" and OPENS_DIALOG.match(name):
            return {"ok": False, "refusedRule": "opens-dialog", "ran": False,
                    "error": "%s:%s opens a prompt or a file picker on the person's screen; the bridge never calls it"
                             % (owner, name)}
        return None

    def write_refusal(self, cls, name):
        """None when a plugin may write property cls.name."""
        refused = self.read_refusal(cls, name, kind="P")
        if refused:
            return refused
        owner, row = self.find_member(cls, name)
        if row[3] not in REACHABLE_SECURITY:
            return {"ok": False, "refusedRule": "security", "securityTag": row[3], "ran": False,
                    "error": "%s.%s can be read but its write is %s: a Studio plugin cannot set it" % (owner, name, row[3])}
        if "ReadOnly" in row[4]:
            return {"ok": False, "refusedRule": "read-only", "securityTag": "ReadOnly", "ran": False,
                    "error": "%s.%s is ReadOnly" % (owner, name)}
        return None

    def create_refusal(self, cls):
        if cls not in self.classes:
            return self.unknown_class(cls)
        tags = set(self.classes[cls].get("t") or [])
        bad = sorted(tags & {"NotCreatable", "Service"})
        if bad:
            return {"ok": False, "refusedRule": "not-creatable", "tags": bad, "ran": False,
                    "error": "%s is tagged %s: Instance.new cannot make it" % (cls, " and ".join(bad))}
        return None

    def property_type(self, cls, name):
        _o, row = self.find_member(cls, name)
        return row[1] if row and row[0] == "P" else None


_CACHE = {}


def load(path=None):
    """The catalog at path (default roblox_catalog.json beside this file), cached by path and mtime. Raises when absent."""
    path = path or os.environ.get("MIF_ROBLOX_CATALOG") or DEFAULT_PATH
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE.clear()
            _CACHE[key] = Catalog(json.load(fh))
    return _CACHE[key]
