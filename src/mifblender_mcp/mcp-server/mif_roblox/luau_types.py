"""Roblox's API and a place's tree as plain Luau types, for an analyzer that takes no definitions file.

luau-analyze (0.748 measured, 2026-11-22) has no option for a definitions file or a sourcemap: its options are a
formatter, a mode, a solver and a time trace. So rbx_script_analyze cannot hand it Roblox's own definitions. What it
can be handed is a module of ordinary table types, which any analyzer reads, one module per script:

    N<i>   an instance of the place looking down: the members of its class (from the catalog, roblox_catalog.json -
           the API dump of one Studio build), Parent as any, and one field per child. This is the sourcemap - names,
           classes and where each script sits - in the only form this analyzer can read.
    U<i>   the script, or an ancestor of it, looking up: the same node with Parent typed as its parent's node.
    S_<C>  a class with no children known: a service the place has not made yet, the answer of Instance.new.

The module returns typed values (game, workspace, script, Instance) that a one-line prefix binds over the script's
untyped globals, and a canary the caller reads to prove the typed pass worked on that script.

Measured on that build, and the reason for each choice here:
  - a key a sealed table type lacks is reported in strict mode only, by either solver: the typed pass is strict;
  - every class and every node of a place in one module answers "Code is too complex to typecheck", and 100 siblings
    under one parent lost findings without a word. So a script's module holds only what the script can name: a
    member or a child is written when its name is a word of the script's text. A name the script does not contain
    cannot be indexed by it, and a name it contains that Roblox and the place lack is exactly the finding;
  - overloads on a literal string choose by the literal (GetService("Players")), but a call with a string that is not
    a literal matches none and is reported. A script is given them only when every GetService, or every
    Instance.new, in its text is called with a literal the catalog has (literal_calls).

Typed narrowly on purpose, because a finding that is not a mistake costs more than one that is missed:
  - a property, parameter or return that is a number, string or boolean is typed; everything else is any, so
    FindFirstChild, WaitForChild and every member that answers an Instance give an untyped value;
  - every typed parameter is optional, so a wrong type and an argument too many are found, a missing one is not;
  - a node whose children were not all read (a limit, a depth) and the DataModel itself take any unknown key;
  - a child named like a member of its class, or twice under one parent, is not typed as a child.
"""
import re

MODULE = "mif_rbx_types"
CANARY = "__mif_absent"
PRIM = {"bool": "boolean", "int": "number", "int64": "number", "float": "number", "double": "number", "string": "string"}
KEYWORDS = frozenset("and break do else elseif end false for function if in local nil not or repeat return then true "
                     "until while continue export type typeof".split())
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SIGNAL = ("{ Connect: (self: any, f: (...any) -> ...any) -> any, ConnectParallel: (self: any, f: (...any) -> ...any) -> any, "
          "Once: (self: any, f: (...any) -> ...any) -> any, Wait: (self: any) -> ...any }")


def _prim(t):
    """The Luau type of a dump type that is a number, string or boolean (with its ?), else None."""
    if not isinstance(t, str):
        return None
    base = PRIM.get(t[:-1] if t.endswith("?") else t)
    return base and (base + "?" if t.endswith("?") else base)


def _key(name):
    """name as a table type's field key, or None when it cannot be written as one."""
    if _IDENT.match(name) and name not in KEYWORDS:
        return name
    if any(ord(ch) < 0x20 or ord(ch) > 0x7E for ch in name):
        return None
    return '["%s"]' % name.replace("\\", "\\\\").replace('"', '\\"')


def member_type(row):
    """One catalog row [kind, type, read, write, tags, params] as a Luau type."""
    kind, typ, params = row[0], row[1], row[5]
    if kind == "P":
        return _prim(typ) or "any"
    if kind == "E":
        return "Sig"
    if kind != "F":
        return "any"
    args = ["self: any"]
    for i, p in enumerate(params or []):
        ptype = p[1] if len(p) > 1 else None
        if isinstance(ptype, str) and ptype.startswith("Tuple"):
            args.append("...any")
            break
        prim = _prim(ptype)
        args.append("p%d: %s" % (i, (prim.rstrip("?") + "?") if prim else "any"))
    if typ in (None, "null", "void", ""):
        ret = "()"
    elif isinstance(typ, str) and typ.startswith("Tuple"):
        ret = "...any"
    elif typ in ("Instances", "Array", "Objects"):
        ret = "{any}"
    else:
        ret = (_prim(typ) or "any")
    return "(%s) -> %s" % (", ".join(args), ret)


def literal_calls(source, call, known):
    """The class names `call` is given in source, when every `call` there is called with a string literal that is in
    known; None when one is not (a variable, an alias of the function, a name the catalog lacks). `call` is
    "GetService" or "Instance.new". A script that never names it answers an empty set."""
    token = re.escape(call).replace(r"\.", r"\s*\.\s*")
    total = len(re.findall(r"\b%s\b" % token, source))
    lit = re.findall(r"\b%s\s*\(\s*([\"'])([A-Za-z0-9_]+)\1\s*[,)]" % token, source)
    names = {name for _q, name in lit}
    return names if total == len(lit) and all(n in known for n in names) else None


class Place:
    """A place's tree, indexed once: tree is the DataModel's node as the plugin's tree op answers it ({name, className,
    childCount, children, truncated}), each child the same shape."""

    def __init__(self, catalog, tree):
        self.catalog = catalog
        self.classes = catalog.classes
        self.nodes, self.by_path, self.ambiguous = [], {}, set()
        self._members = {}
        self.unknown_classes = []
        todo = [(tree, None, ())]
        while todo:
            node, parent, path = todo.pop()
            nid = len(self.nodes)
            kids = node.get("children")
            count = node.get("childCount") or 0
            rec = {"parent": parent, "cls": node.get("className"), "kids": [], "path": path,
                   "open": parent is None or bool(node.get("truncated"))
                   or (len(kids) < count if isinstance(kids, list) else count > 0)}
            self.nodes.append(rec)
            if path in self.by_path:
                self.ambiguous.add(path)
            self.by_path.setdefault(path, nid)
            if parent is not None:
                self.nodes[parent]["kids"].append((path[-1], nid))
            if rec["cls"] not in self.classes and rec["cls"] not in self.unknown_classes:
                self.unknown_classes.append(rec["cls"])
            for k in reversed(kids if isinstance(kids, list) else []):
                if isinstance(k, dict) and isinstance(k.get("name"), str):
                    todo.append((k, nid, path + (k["name"],)))
                else:
                    rec["open"] = True
        self.workspace = next((i for n, i in self.nodes[0]["kids"]
                               if self.nodes[i]["cls"] == "Workspace" and (n,) not in self.ambiguous), None)
        self.services = {}
        for _n, i in self.nodes[0]["kids"]:
            self.services.setdefault(self.nodes[i]["cls"], i)

    def members(self, cls):
        if cls not in self._members:
            self._members[cls] = self.catalog.members_of(cls)
        return self._members[cls]

    def info(self):
        return {"nodes": len(self.nodes), "unknownClasses": self.unknown_classes,
                "openNodes": sum(1 for r in self.nodes if r["open"] and r["cls"] in self.classes)}

    def describe(self, alias):
        """What a type alias of a module stands for, in words: N12 and U12 are node 12, S_Part the class, G0 game."""
        if alias.startswith("S_"):
            return "class %s" % alias[2:]
        if alias[1:].isdigit() and int(alias[1:]) < len(self.nodes):
            rec = self.nodes[int(alias[1:])]
            return "%s (class %s)" % (".".join(rec["path"]) or "game", rec["cls"])
        return alias

    def script_node(self, path):
        """The node of the script at path (names from the DataModel down), or None when it is not in the tree read or
        another instance has the same path."""
        key = tuple(path or ())
        return self.by_path.get(key) if key and key not in self.ambiguous else None

    def module(self, source, script_node=None):
        """(module text, bound) for one script: bound says which of GetService and Instance.new are typed by literal."""
        words = set(_WORD.findall(source))
        out = ["--!nonstrict", "type Sig = %s" % SIGNAL]
        done = {}

        def said(name):
            return name in words if _IDENT.match(name) else name in source

        def class_fields(cls, skip=()):
            return ["%s: %s" % (n, member_type(row)) for n, (_o, row) in sorted(self.members(cls).items())
                    if n in words and n != "Parent" and n not in skip and n not in KEYWORDS and _IDENT.match(n)]

        def node(nid, up, get_service=None):
            alias = "G0" if get_service else ("U%d" if up else "N%d") % nid
            if alias in done:
                return done[alias]
            rec = self.nodes[nid]
            if rec["cls"] not in self.classes:
                done[alias] = "any"
                return "any"
            done[alias] = alias
            own = self.members(rec["cls"])
            is_open = rec["open"]
            fields, seen = [], set()
            names = [n for n, _i in rec["kids"]]
            for name, kid in rec["kids"]:
                if name in seen or name in own or name == "Parent" or not said(name):
                    continue
                seen.add(name)
                fkey = _key(name)
                if fkey is None:
                    is_open = True
                elif names.count(name) > 1:
                    fields.append("%s: any" % fkey)
                else:
                    fields.append("%s: %s" % (fkey, node(kid, False)))
            parent = "any" if not up or rec["parent"] is None else node(rec["parent"], rec["parent"] != 0)
            fields = ["Parent: %s" % parent] + class_fields(rec["cls"], ("GetService",) if get_service else ()) + fields
            if get_service:
                fields.append("GetService: %s" % get_service)
            if is_open:
                fields.append("[string]: any")
            out.append("type %s = { %s }" % (alias, ", ".join(fields)))
            return alias

        def bare(cls):
            alias = "S_%s" % cls
            if alias not in done:
                done[alias] = alias
                out.append("type %s = { %s }" % (alias, ", ".join(["Parent: any"] + class_fields(cls))))
            return alias

        ret = ["%s = (nil :: any) :: { present: boolean }" % CANARY]
        root = node(0, False)
        services = literal_calls(source, "GetService", [c for c, v in self.classes.items() if "Service" in v.get("t", ())])
        if services and root != "any":
            # GetService by literal name answers the place's own service node, or the bare class
            root = node(0, False, " & ".join('((self: any, className: "%s") -> %s)'
                                             % (c, node(self.services[c], False) if c in self.services else bare(c))
                                             for c in sorted(services)))
        ret.append("game = (nil :: any) :: %s" % root)
        ret.append("workspace = (nil :: any) :: %s" % ("any" if self.workspace is None else node(self.workspace, False)))
        if script_node is not None:
            ret.append("script = (nil :: any) :: %s" % node(script_node, True))
        news = literal_calls(source, "Instance.new", [c for c, v in self.classes.items()
                                                      if "NotCreatable" not in v.get("t", ())])
        if news:
            ret.append("Instance = (nil :: any) :: { new: %s, [string]: any }"
                       % " & ".join('((className: "%s", parent: any?) -> %s)' % (c, bare(c)) for c in sorted(news)))
        out.append("return { %s }" % ", ".join(ret))
        return "\n".join(out) + "\n", {"script": script_node is not None, "services": bool(services),
                                       "new": bool(news)}


def prefix(module, bound):
    """The one-line prefix that binds a script's typed values over its untyped globals. module is the base name of
    the script's own type module."""
    names = ["game", "workspace"] + (["script"] if bound.get("script") else []) + (["Instance"] if bound.get("new") else [])
    return 'local __mif = require("./%s"); local %s = %s; ' % (module, ", ".join(names),
                                                               ", ".join("__mif.%s" % n for n in names))


def canary():
    """The line appended to a typed script: a member the canary's type lacks. The typed pass reports it, or the pass
    did not work on that script and the answer says so."""
    return "\nlocal _ = __mif.%s.absent" % CANARY
