"""MifBridge extensions - third parties add tools, tool packs and native endpoints without forking. docs/32_EXTENSIONS.md.

An extension is a LOCAL FOLDER: nothing here downloads, asks a registry or touches the network. It holds a manifest
(mifextension.json), a Python module whose register(api) adds MCP tools that compose existing endpoints and Blender ops,
and optionally native/<Plugin>/ - an Unreal plugin that registers endpoints through MifBridgeExtensionApi.h - and
blender/<AddOn>/ - a Blender add-on that registers ops into MifBlender's dispatcher (mifblender_api, docs/32 section 8).

    python mifext.py new <name> [--prefix p_] [--native] [--dir D]   write a working skeleton
    python mifext.py validate <folder> [--load]                      every check the loader makes, and the native half's
    python mifext.py install <folder> [--force]                      validate, then copy into the extensions directory
    python mifext.py list                                            what is installed, and what would fail to load
    python mifext.py remove <name>                                   move it to <extensions dir>/.removed/
    python mifext.py install-native <folder> --project <dir> [--force]   copy native/<Plugin>/ into <project>/Plugins/
    python mifext.py install-blender <folder> --addons <dir> [--force]   copy blender/<AddOn>/ into a Blender add-ons folder

The extensions directory is MIF_EXTENSIONS_DIR (several joined with the path separator; the first is where install
writes) or, unset, ~/.mifbridge/extensions. MIF_EXTENSIONS=off makes the MCP server load none.

Installing and removing are a PERSON's acts, on purpose: no MCP tool installs or removes an extension, because an
extension's module runs inside the MCP server with the person's privileges (an agent that could install code into its
own server would not be bounded by anything the server checks). The server's mif_extensions tool only reads.

Stdlib only: the MCP server imports this, and so does a person with a bare Python.
"""

import argparse
import ast
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
import types

API_VERSIONS = (1,)                 # the server-side api versions this module speaks
NATIVE_API_VERSION = 2              # MIFBRIDGE_EXTENSION_API_VERSION in Source/MifBridge/Public/MifBridgeExtensionApi.h
MANIFEST = "mifextension.json"
RESERVED_PREFIXES = ("bl_", "mif_", "kr_", "ue_", "ext_", "vfx_")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
PREFIX_RE = re.compile(r"^[a-z][a-z0-9]{1,15}_$")
TOOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PLUGIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,63}$")
BACKENDS = ("unreal", "blender", "any")
# A native plugin's module must load before MifBridge's route table goes live (MifBridge starts its server at
# PostEngineInit, and routes bind once): these phases all run before it.
NATIVE_PHASES = ("EarliestPossible", "PostConfigInit", "PostSplashScreen", "PreEarlyLoadingScreen", "PreLoadingScreen",
                 "PreDefault", "Default", "PostDefault")
HERE = os.path.dirname(os.path.abspath(__file__))


class ExtensionError(Exception):
    """An extension that cannot be loaded, with the reason a person can act on."""


def extension_dirs():
    raw = os.environ.get("MIF_EXTENSIONS_DIR", "").strip()
    if raw:
        return [os.path.abspath(os.path.expanduser(p)) for p in raw.split(os.pathsep) if p.strip()]
    return [os.path.join(os.path.expanduser("~"), ".mifbridge", "extensions")]


def loading_disabled():
    return os.environ.get("MIF_EXTENSIONS", "").strip().lower() in ("off", "0", "false", "no")


# ---------------------------------------------------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------------------------------------------------

def check_manifest(m):
    """Every error in a parsed manifest, as sentences. [] when it is sound."""
    errs = []
    if not isinstance(m, dict):
        return ["%s must hold a JSON object" % MANIFEST]
    known = {"apiVersion", "name", "version", "description", "prefix", "pack", "module", "tools", "native", "blender"}
    for k in sorted(set(m) - known):
        errs.append("unknown manifest key %r (known: %s)" % (k, ", ".join(sorted(known))))
    v = m.get("apiVersion")
    if not isinstance(v, int) or isinstance(v, bool) or v not in API_VERSIONS:
        errs.append("apiVersion %r is not one this MifBridge speaks (%s)" % (v, ", ".join(map(str, API_VERSIONS))))
    name = m.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errs.append("name %r must be lowercase: a letter, then 1-31 letters, digits or '_'" % (name,))
    for key in ("version", "description"):
        if not isinstance(m.get(key), str) or not m.get(key).strip():
            errs.append("%s is required, a non-empty string" % key)
    prefix = m.get("prefix")
    if not isinstance(prefix, str) or not PREFIX_RE.match(prefix):
        errs.append("prefix %r must be 2-16 lowercase letters or digits ending in '_' (e.g. 'acme_')" % (prefix,))
    elif prefix in RESERVED_PREFIXES:
        errs.append("prefix %r is reserved (%s)" % (prefix, ", ".join(RESERVED_PREFIXES)))
    pack = m.get("pack", {})
    if not isinstance(pack, dict):
        errs.append("pack must be an object")
        pack = {}
    for k in sorted(set(pack) - {"title", "backend", "rules"}):
        errs.append("unknown pack key %r (known: title, backend, rules)" % k)
    if "title" in pack and (not isinstance(pack["title"], str) or not pack["title"].strip()):
        errs.append("pack.title must be a non-empty string")
    if pack.get("backend", "unreal") not in BACKENDS:
        errs.append("pack.backend %r must be one of %s" % (pack.get("backend"), ", ".join(BACKENDS)))
    rules = pack.get("rules", [])
    if not isinstance(rules, list) or any(not isinstance(r, str) or not r.strip() or len(r) > 280 for r in rules) \
            or len(rules) > 10:
        errs.append("pack.rules must be a list of at most 10 non-empty strings of at most 280 characters")
    module = m.get("module", "tools.py")
    if not isinstance(module, str) or not module.endswith(".py") or os.path.isabs(module) or ".." in module.replace("\\", "/").split("/"):
        errs.append("module %r must be a relative .py path inside the extension folder" % (module,))
    tools = m.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
        errs.append("tools must be a list of names")
        tools = []
    for t in tools:
        if not TOOL_RE.match(t):
            errs.append("tool %r must be lowercase snake_case" % t)
        elif isinstance(prefix, str) and not t.startswith(prefix):
            errs.append("tool %r must start with the prefix %r" % (t, prefix))
    if len(set(tools)) != len(tools):
        errs.append("tools lists a name twice")
    native = m.get("native")
    endpoints = []
    if native is not None:
        if not isinstance(native, dict):
            errs.append("native must be an object {plugin, provider, endpoints}")
        else:
            for k in sorted(set(native) - {"plugin", "provider", "endpoints"}):
                errs.append("unknown native key %r (known: plugin, provider, endpoints)" % k)
            if not isinstance(native.get("plugin"), str) or not PLUGIN_RE.match(native.get("plugin") or ""):
                errs.append("native.plugin %r must be the plugin's folder and .uplugin name (letters and digits)"
                            % (native.get("plugin"),))
            if not isinstance(native.get("provider"), str) or not native.get("provider", "").strip():
                errs.append("native.provider is required: the Provider string the plugin registers with")
            endpoints = native.get("endpoints", [])
            if not isinstance(endpoints, list) or not endpoints or any(not isinstance(e, str) for e in endpoints):
                errs.append("native.endpoints must be a non-empty list of endpoint names")
                endpoints = []
            for e in endpoints:
                if not TOOL_RE.match(e):
                    errs.append("native endpoint %r must be lowercase snake_case" % e)
                elif isinstance(prefix, str) and not e.startswith(prefix):
                    errs.append("native endpoint %r must start with the prefix %r" % (e, prefix))
    blender = m.get("blender")
    bl_ops = []
    if blender is not None:
        if not isinstance(blender, dict):
            errs.append("blender must be an object {addon, provider, ops}")
        else:
            for k in sorted(set(blender) - {"addon", "provider", "ops"}):
                errs.append("unknown blender key %r (known: addon, provider, ops)" % k)
            if not isinstance(blender.get("addon"), str) or not PLUGIN_RE.match(blender.get("addon") or ""):
                errs.append("blender.addon %r must be the add-on's folder name under blender/ (letters and digits)"
                            % (blender.get("addon"),))
            if not isinstance(blender.get("provider"), str) or not blender.get("provider", "").strip():
                errs.append("blender.provider is required: the provider string the add-on registers its ops with")
            bl_ops = blender.get("ops", [])
            if not isinstance(bl_ops, list) or not bl_ops or any(not isinstance(e, str) for e in bl_ops):
                errs.append("blender.ops must be a non-empty list of op names")
                bl_ops = []
            for e in bl_ops:
                if not TOOL_RE.match(e):
                    errs.append("blender op %r must be lowercase snake_case" % e)
                elif isinstance(prefix, str) and not e.startswith(prefix):
                    errs.append("blender op %r must start with the prefix %r" % (e, prefix))
    if not tools and not endpoints and not bl_ops:
        errs.append("the extension declares nothing: no tools, no native endpoints and no Blender ops")
    return errs


def read_manifest(folder):
    """(manifest, errors). The manifest is None when the file is missing or not JSON."""
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        return None, ["no %s in %s" % (MANIFEST, folder)]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, ["%s does not parse: %s" % (MANIFEST, exc)]
    return m, check_manifest(m)


# ---------------------------------------------------------------------------------------------------------------------
# The api object a module's register(api) receives
# ---------------------------------------------------------------------------------------------------------------------

class Api:
    """The server-side extension api, version 1. register(api) uses it to add tools; the tools use it to call the
    editor and Blender. Calls go through the MCP server's own request path, so the editor's safety gate, write modes,
    change cap, journal and parameter checks apply exactly as to any built-in tool."""

    def __init__(self, manifest, transport):
        self.version = manifest["apiVersion"]
        self.extension = types.MappingProxyType(dict(manifest))
        self._prefix = manifest["prefix"]
        self._declared = set(manifest.get("tools") or [])
        self._transport = transport
        self._staged = {}
        self._open = True

    def tool(self, fn):
        """Register fn as an MCP tool, named after the function. Usable as a decorator; returns fn."""
        if not self._open:
            raise ExtensionError("api.tool(%s) after register() returned - register tools inside register(api)"
                                 % getattr(fn, "__name__", fn))
        if not callable(fn) or not getattr(fn, "__name__", ""):
            raise ExtensionError("api.tool takes a named function")
        name = fn.__name__
        if not name.startswith(self._prefix):
            raise ExtensionError("tool %r does not start with the extension's prefix %r" % (name, self._prefix))
        if name not in self._declared:
            raise ExtensionError("tool %r is not listed in the manifest's tools" % name)
        if name in self._staged:
            raise ExtensionError("tool %r is registered twice" % name)
        if not (fn.__doc__ or "").strip():
            raise ExtensionError("tool %r has no docstring - its first sentence is the tool's description" % name)
        sig = inspect.signature(fn)
        bare = [p.name for p in sig.parameters.values()
                if p.annotation is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if bare:
            raise ExtensionError("tool %r: annotate every parameter (the schema is built from them): %s"
                                 % (name, ", ".join(bare)))
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values()):
            raise ExtensionError("tool %r: *args and **kwargs cannot be described to a client - name each parameter"
                                 % name)
        self._staged[name] = fn
        return fn

    def call(self, endpoint, **body):
        """One editor endpoint, by its own (camelCase) keys. Returns the editor's reply as a dict."""
        if self._transport is None:
            return {"ok": False, "error": "api.call(%r): no MCP server is running this extension (validate --load "
                                          "runs register() only)" % endpoint}
        return self._transport.call(str(endpoint), dict(body))

    def blender(self, op, **params):
        """One MifBlender op, through the add-on's socket. Returns the add-on's reply as a dict."""
        if self._transport is None:
            return {"ok": False, "error": "api.blender(%r): no MCP server is running this extension" % op}
        return self._transport.blender(str(op), dict(params))

    @staticmethod
    def ok(result):
        """Failure is the PRESENCE of an `error` key, never the absence of `ok`."""
        return isinstance(result, dict) and "error" not in result


class Loaded:
    def __init__(self, manifest, folder, tools):
        self.manifest = manifest
        self.folder = folder
        self.tools = tools                      # name -> function

    @property
    def name(self):
        return self.manifest["name"]

    @property
    def pack_id(self):
        return "ext_" + self.manifest["name"]

    def row(self):
        m = self.manifest
        native = m.get("native") or {}
        return {"name": m["name"], "version": m["version"], "apiVersion": m["apiVersion"], "folder": self.folder,
                "pack": self.pack_id, "prefix": m["prefix"], "tools": sorted(self.tools),
                "nativePlugin": native.get("plugin"), "nativeEndpoints": list(native.get("endpoints") or []),
                "blenderAddon": (m.get("blender") or {}).get("addon"),
                "blenderOps": list((m.get("blender") or {}).get("ops") or []),
                "description": m["description"]}


def load_extension(folder, transport=None, taken=()):
    """Load one extension folder: manifest, module, register(api). Raises ExtensionError; registers nothing itself -
    the caller commits the returned tools, so a failure never leaves half an extension behind."""
    manifest, errs = read_manifest(folder)
    if errs:
        raise ExtensionError("; ".join(errs))
    module_path = os.path.join(folder, manifest.get("module", "tools.py"))
    tools_declared = manifest.get("tools") or []
    if not tools_declared:
        return Loaded(manifest, folder, {})         # a native-only extension: nothing to import
    if not os.path.isfile(module_path):
        raise ExtensionError("module %s is missing" % manifest.get("module", "tools.py"))
    clash = sorted(set(tools_declared) & set(taken))
    if clash:
        raise ExtensionError("tool name(s) already registered by MifBridge or another extension: %s" % ", ".join(clash))
    mod_name = "mifext_%s" % manifest["name"]
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ExtensionError("cannot import %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a third party's import error is a load failure, not a server crash
        raise ExtensionError("importing %s raised %s: %s" % (os.path.basename(module_path), type(exc).__name__, exc))
    declared_api = getattr(module, "MIF_EXTENSION_API", manifest["apiVersion"])
    if declared_api != manifest["apiVersion"]:
        raise ExtensionError("the module says MIF_EXTENSION_API = %r and the manifest apiVersion %r - make them agree"
                             % (declared_api, manifest["apiVersion"]))
    register = getattr(module, "register", None)
    if not callable(register):
        raise ExtensionError("%s defines no register(api)" % os.path.basename(module_path))
    api = Api(manifest, transport)
    try:
        register(api)
    except ExtensionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtensionError("register(api) raised %s: %s" % (type(exc).__name__, exc))
    finally:
        api._open = False
    missing = sorted(set(tools_declared) - set(api._staged))
    if missing:
        raise ExtensionError("the manifest lists tool(s) register(api) never registered: %s" % ", ".join(missing))
    return Loaded(manifest, folder, dict(api._staged))


def discover():
    """[(folder, error or None)] for every extension folder in every extensions directory, in load order."""
    out = []
    for d in extension_dirs():
        if not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            path = os.path.join(d, entry)
            if entry.startswith(".") or not os.path.isdir(path):
                continue
            out.append(path)
    return out


def load_all(transport=None, taken=()):
    """Load every installed extension. Returns {"loaded": [Loaded], "failed": [{folder, name?, error}], "dirs",
    "disabled"}. Never raises for one bad extension."""
    report = {"loaded": [], "failed": [], "dirs": extension_dirs(), "disabled": loading_disabled()}
    if report["disabled"]:
        return report
    taken = set(taken)
    names = {}
    for folder in discover():
        base = os.path.basename(folder)
        try:
            manifest, errs = read_manifest(folder)
            if manifest is not None and not errs and manifest["name"] != base:
                raise ExtensionError("the folder is %r and the manifest name %r - install it with mifext.py install, "
                                     "which names the folder after the extension" % (base, manifest["name"]))
            if manifest is not None and not errs and manifest["name"] in names:
                raise ExtensionError("an extension named %r already loaded from %s" % (manifest["name"],
                                                                                       names[manifest["name"]]))
            ext = load_extension(folder, transport, taken)
        except ExtensionError as exc:
            report["failed"].append({"folder": folder, "name": base, "error": str(exc)})
            continue
        names[ext.name] = folder
        taken.update(ext.tools)
        report["loaded"].append(ext)
    return report


# ---------------------------------------------------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------------------------------------------------

def _plugin_root():
    """MifBridge's own folder (the one holding MifBridge.uplugin), walking up from here - or None."""
    d = HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "MifBridge.uplugin")):
            return d
        d = os.path.dirname(d)
    return None


def builtin_endpoints():
    """MifBridge's built-in endpoint names, read from MIF_BIND in its source - or None when the source is not here."""
    root = _plugin_root()
    path = os.path.join(root, "Source", "MifBridge", "Private", "MifBridgeCommon.cpp") if root else None
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        names = set(re.findall(r"MIF_BIND\((\w+)\)", fh.read()))
    return names or None


def builtin_tools():
    """MifBridge's tool names, read from packs.json beside this file - or None."""
    try:
        with open(os.path.join(HERE, "packs.json"), "r", encoding="utf-8") as fh:
            packs = (json.load(fh) or {}).get("packs") or {}
    except (OSError, ValueError):
        return None
    names = {t for p in packs.values() if isinstance(p, dict) for t in (p.get("tools") or [])}
    return names or None


def _module_scan(path):
    """Static read of the module: {defs: {name: [unannotated params]}, tools: [decorated with api.tool], calls:
    [endpoint literals passed to api.call], register: bool, api_const}."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {"tools": [], "calls": [], "register": False, "api_const": None, "unannotated": {}}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "register":
                out["register"] = True
            for dec in node.decorator_list:
                if isinstance(dec, ast.Attribute) and dec.attr == "tool":
                    out["tools"].append(node.name)
                    bare = [a.arg for a in node.args.args + node.args.kwonlyargs if a.annotation is None]
                    if bare:
                        out["unannotated"][node.name] = bare
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("call", "tool"):
            if node.func.attr == "call" and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out["calls"].append(node.args[0].value)
            if node.func.attr == "tool" and node.args and isinstance(node.args[0], ast.Name):
                out["tools"].append(node.args[0].id)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MIF_EXTENSION_API"
                                                  for t in node.targets):
            if isinstance(node.value, ast.Constant):
                out["api_const"] = node.value.value
    return out


def _read_text_tree(folder, exts):
    out = {}
    for base, _dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(exts):
                p = os.path.join(base, f)
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    out[p] = fh.read()
    return out


def _validate_native(folder, manifest, errors, warnings, checked):
    native = manifest.get("native") or {}
    plugin = native.get("plugin")
    pdir = os.path.join(folder, "native", plugin or "")
    if not plugin or not os.path.isdir(pdir):
        errors.append("native.plugin %r: no folder native/%s" % (plugin, plugin))
        return
    up = os.path.join(pdir, plugin + ".uplugin")
    checked.append("native: %s.uplugin" % plugin)
    try:
        with open(up, "r", encoding="utf-8") as fh:
            desc = json.load(fh)
    except (OSError, ValueError) as exc:
        errors.append("native/%s/%s.uplugin is missing or not JSON: %s" % (plugin, plugin, exc))
        return
    deps = [p.get("Name") for p in (desc.get("Plugins") or []) if isinstance(p, dict) and p.get("Enabled", True)]
    if "MifBridge" not in deps:
        errors.append("%s.uplugin does not list MifBridge under Plugins - the extension API's DLL would not load first"
                      % plugin)
    modules = [m for m in (desc.get("Modules") or []) if isinstance(m, dict)]
    if not modules:
        errors.append("%s.uplugin declares no Modules" % plugin)
    for m in modules:
        phase = m.get("LoadingPhase", "Default")
        if phase not in NATIVE_PHASES:
            errors.append("module %s loads at %s; it must register before MifBridge's route table goes live - use "
                          "Default" % (m.get("Name"), phase))
        if m.get("Type") not in ("Editor", "UncookedOnly", "EditorNoCommandlet"):
            warnings.append("module %s is Type %r; MifBridge runs in the editor only, so 'Editor' is the usual choice"
                            % (m.get("Name"), m.get("Type")))
    src = os.path.join(pdir, "Source")
    builds = _read_text_tree(src, (".Build.cs",))
    checked.append("native: Build.cs dependencies")
    if not builds:
        errors.append("native/%s/Source has no .Build.cs" % plugin)
    for path, text in builds.items():
        if '"MifBridge"' not in text:
            errors.append("%s does not depend on the MifBridge module" % os.path.basename(path))
        if '"Json"' not in text:
            errors.append("%s does not depend on Json (the handler's In and Out are FJsonObjects)" % os.path.basename(path))
    code = _read_text_tree(src, (".cpp", ".h"))
    checked.append("native: registration source")
    text = "\n".join(code.values())
    if "RegisterExtensionEndpoint" not in text:
        errors.append("the native source never calls RegisterExtensionEndpoint (MifBridgeExtensionApi.h)")
    if re.search(r"\bRegisterExternalEndpoint\s*\(", text):
        warnings.append("the native source calls the unversioned RegisterExternalEndpoint: its endpoints get no declared "
                        "parameters, write-gate flag or change cap - use RegisterExtensionEndpoint")
    if "UnregisterExternalEndpoints" not in text:
        errors.append("the native source never calls UnregisterExternalEndpoints(Provider) - call it in ShutdownModule")
    provider = native.get("provider") or ""
    if provider and ('"%s"' % provider) not in text:
        errors.append("the provider string %r does not appear in the native source" % provider)
    literals = set(re.findall(r'"([a-z][a-z0-9_]*)"', text))
    for e in native.get("endpoints") or []:
        if e not in literals:
            errors.append("native endpoint %r is in the manifest and not in the native source" % e)
    prefix = manifest.get("prefix") or ""
    stray = sorted(lit for lit in literals
                   if prefix and lit.startswith(prefix) and lit not in (native.get("endpoints") or []))
    if stray:
        warnings.append("the native source names %s, which carry the prefix and are not in native.endpoints - list "
                        "every endpoint the plugin registers" % ", ".join(stray))


def validate(folder, load=False):
    """{"ok", "errors", "warnings", "checked", "notChecked"} for one extension folder."""
    folder = os.path.abspath(folder)
    errors, warnings, checked, not_checked = [], [], [], []
    manifest, errs = read_manifest(folder)
    errors.extend(errs)
    checked.append("manifest")
    if manifest is None:
        return {"ok": False, "errors": errors, "warnings": warnings, "checked": checked, "notChecked": not_checked}
    prefix = manifest.get("prefix") if isinstance(manifest.get("prefix"), str) else ""
    if os.path.basename(folder) != manifest.get("name"):
        warnings.append("the folder is %r and the extension %r; install names the installed folder after the extension"
                        % (os.path.basename(folder), manifest.get("name")))
    tools = builtin_tools()
    if tools is None:
        not_checked.append("prefix clash with MifBridge's tools (packs.json not found beside mifext.py)")
    else:
        checked.append("prefix clash with MifBridge's tools")
        hit = sorted(t for t in tools if prefix and t.startswith(prefix))
        if hit:
            errors.append("prefix %r is already used by MifBridge tool(s): %s" % (prefix, ", ".join(hit[:8])))
    endpoints = builtin_endpoints()
    if endpoints is None:
        not_checked.append("prefix clash with MifBridge's endpoints, and the endpoints api.call names (MifBridge's "
                           "Source is not beside this copy of mifext.py)")
    else:
        checked.append("prefix clash with MifBridge's endpoints")
        hit = sorted(e for e in endpoints if prefix and e.startswith(prefix))
        if hit:
            errors.append("prefix %r is already used by MifBridge endpoint(s): %s" % (prefix, ", ".join(hit[:8])))
    if manifest.get("tools"):
        module_path = os.path.join(folder, manifest.get("module", "tools.py"))
        if not os.path.isfile(module_path):
            errors.append("module %s is missing" % manifest.get("module", "tools.py"))
        else:
            checked.append("module (static read)")
            try:
                scan = _module_scan(module_path)
            except SyntaxError as exc:
                errors.append("module does not parse: %s" % exc)
                scan = None
            if scan:
                if not scan["register"]:
                    errors.append("module defines no register(api)")
                if scan["api_const"] is not None and scan["api_const"] != manifest.get("apiVersion"):
                    errors.append("MIF_EXTENSION_API = %r disagrees with the manifest apiVersion %r"
                                  % (scan["api_const"], manifest.get("apiVersion")))
                declared = set(manifest.get("tools") or [])
                found = set(scan["tools"])
                for t in sorted(declared - found):
                    errors.append("tool %r is in the manifest and no function is registered with api.tool under that name" % t)
                for t in sorted(found - declared):
                    errors.append("function %r is registered with api.tool and is not in the manifest's tools" % t)
                for t, bare in sorted(scan["unannotated"].items()):
                    errors.append("tool %r: annotate every parameter: %s" % (t, ", ".join(bare)))
                if endpoints is not None:
                    checked.append("endpoints named in api.call")
                    own = set((manifest.get("native") or {}).get("endpoints") or [])
                    for c in sorted(set(scan["calls"])):
                        if c not in endpoints and c not in own:
                            errors.append("api.call(%r): no MifBridge endpoint or native endpoint of this extension has "
                                          "that name" % c)
                not_checked.append("endpoints api.call receives from a variable (only literal names are read)")
            if load and not errors:
                checked.append("module (imported, register(api) run)")
                try:
                    load_extension(folder, None, tools or ())
                except ExtensionError as exc:
                    errors.append(str(exc))
            elif not load:
                not_checked.append("register(api) itself (run validate --load to import the module and run it)")
    if manifest.get("native"):
        _validate_native(folder, manifest, errors, warnings, checked)
        not_checked.append("that the native plugin compiles and registers live (build it and read self_audit, or "
                           "mif_extensions with an editor connected)")
    if manifest.get("blender"):
        _validate_blender(folder, manifest, errors, warnings, checked)
        not_checked.append("that the Blender add-on registers live (enable it after MifBlender and read ping's "
                           "extensionApi, or mif_extensions with Blender connected)")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked": checked, "notChecked": not_checked}


def _validate_blender(folder, manifest, errors, warnings, checked):
    """The Blender half's static checks: the add-on folder, bl_info, register_ops / unregister_ops through
    mifblender_api, the provider string, and each declared op named in the source."""
    bl = manifest.get("blender") or {}
    addon = bl.get("addon") or ""
    init = os.path.join(folder, "blender", addon, "__init__.py")
    checked.append("blender: %s/__init__.py" % addon)
    if not os.path.isfile(init):
        errors.append("blender.addon %r: no file blender/%s/__init__.py" % (addon, addon))
        return
    text = "\n".join(_read_text_tree(os.path.join(folder, "blender", addon), (".py",)).values())
    try:
        with open(init, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append("blender/%s/__init__.py does not parse: %s" % (addon, exc))
        return
    names = {getattr(t, "id", None) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets}
    if "bl_info" not in names:
        errors.append("blender/%s/__init__.py has no bl_info - Blender lists an add-on by it" % addon)
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fn in ("register", "unregister"):
        if fn not in defs:
            errors.append("blender/%s/__init__.py defines no %s()" % (addon, fn))
    checked.append("blender: registration source")
    if "mifblender_api" not in text:
        errors.append("the add-on never imports mifblender_api - the API MifBlender publishes when it is enabled")
    if "register_ops(" not in text:
        errors.append("the add-on never calls register_ops (mifblender_api)")
    if "unregister_ops(" not in text:
        errors.append("the add-on never calls unregister_ops(provider) - call it in unregister()")
    provider = bl.get("provider") or ""
    if provider and provider not in text:
        errors.append("the provider string %r does not appear in the add-on's source" % provider)
    for op in bl.get("ops") or []:
        if op not in text:
            errors.append("blender op %r is in the manifest and not in the add-on's source" % op)
    prefix = manifest.get("prefix") or ""
    stray = sorted(lit for lit in set(re.findall(r"[\"']([a-z][a-z0-9_]+)[\"']", text))
                   if prefix and lit.startswith(prefix) and lit != prefix and lit not in (bl.get("ops") or [])
                   and lit not in (manifest.get("tools") or []) and lit not in ((manifest.get("native") or {})
                                                                                 .get("endpoints") or []))
    if stray:
        warnings.append("the add-on names %s, which carry the prefix and are not in blender.ops - list them if they "
                        "are ops" % ", ".join(stray[:6]))


# ---------------------------------------------------------------------------------------------------------------------
# Install, list, remove, install-native
# ---------------------------------------------------------------------------------------------------------------------

def _move_aside(path, root):
    dest_dir = os.path.join(root, ".removed")
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.join(dest_dir, "%s-%s" % (os.path.basename(path), time.strftime("%Y%m%d-%H%M%S")))
    dest, n = base, 1
    while os.path.exists(dest):
        n += 1
        dest = "%s-%d" % (base, n)
    shutil.move(path, dest)
    return dest


_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "Binaries", "Intermediate", ".git")


def install(folder, force=False):
    """Validate, then copy into the first extensions directory. Returns {"ok", "installedAt", ...}."""
    folder = os.path.abspath(folder)
    v = validate(folder)
    if not v["ok"]:
        return {"ok": False, "error": "not installed - the extension does not validate", "validation": v}
    manifest, _ = read_manifest(folder)
    root = extension_dirs()[0]
    dest = os.path.join(root, manifest["name"])
    if os.path.normcase(dest) == os.path.normcase(folder):
        return {"ok": False, "error": "%s is already the installed copy" % folder}
    replaced = None
    if os.path.exists(dest):
        if not force:
            return {"ok": False, "error": "%r is installed already at %s - pass --force to replace it (the installed "
                                         "copy is moved to .removed/, not deleted)" % (manifest["name"], dest)}
        replaced = _move_aside(dest, root)
    os.makedirs(root, exist_ok=True)
    shutil.copytree(folder, dest, ignore=_COPY_IGNORE)
    out = {"ok": True, "installedAt": dest, "warnings": v["warnings"],
           "note": "restart the MCP server (your client's MCP reconnect) to load it; its tools are the pack ext_%s. "
                   "The module runs inside the MCP server with your privileges." % manifest["name"]}
    if replaced:
        out["replacedMovedTo"] = replaced
    if manifest.get("native"):
        out["nativeNote"] = ("the native half is not installed by this: run install-native with --project <your "
                             "project folder>, then build the project")
    return out


def list_installed():
    rows, failed = [], []
    for folder in discover():
        m, errs = read_manifest(folder)
        if errs:
            failed.append({"folder": folder, "error": "; ".join(errs)})
            continue
        native = m.get("native") or {}
        rows.append({"name": m["name"], "version": m["version"], "folder": folder, "tools": m.get("tools") or [],
                     "nativeEndpoints": native.get("endpoints") or [], "nativePlugin": native.get("plugin")})
    return {"ok": True, "dirs": extension_dirs(), "extensions": rows, "invalid": failed,
            "loadingDisabled": loading_disabled()}


def remove(name):
    for root in extension_dirs():
        path = os.path.join(root, name)
        if os.path.isdir(path) and not name.startswith("."):
            moved = _move_aside(path, root)
            return {"ok": True, "removed": name, "movedTo": moved,
                    "note": "restart the MCP server to unload its tools. A native plugin installed into a project "
                            "stays there - disable or delete it in that project."}
    return {"ok": False, "error": "no installed extension named %r in %s" % (name, ", ".join(extension_dirs()))}


def install_native(folder, project, force=False):
    folder, project = os.path.abspath(folder), os.path.abspath(project)
    manifest, errs = read_manifest(folder)
    if errs:
        return {"ok": False, "error": "; ".join(errs)}
    native = manifest.get("native")
    if not native:
        return {"ok": False, "error": "%s has no native half" % manifest["name"]}
    if not any(f.endswith(".uproject") for f in os.listdir(project)):
        return {"ok": False, "error": "%s holds no .uproject - pass the project's folder" % project}
    src = os.path.join(folder, "native", native["plugin"])
    if not os.path.isdir(src):
        return {"ok": False, "error": "no folder native/%s" % native["plugin"]}
    dest = os.path.join(project, "Plugins", native["plugin"])
    replaced = None
    if os.path.exists(dest):
        if not force:
            return {"ok": False, "error": "%s exists - pass --force to replace it (moved to %s, not deleted)"
                                         % (dest, os.path.join(project, "Plugins", ".removed"))}
        replaced = _move_aside(dest, os.path.join(project, "Plugins"))
    shutil.copytree(src, dest, ignore=_COPY_IGNORE)
    out = {"ok": True, "installedAt": dest,
           "note": "build the project (the editor's next build compiles it), then self_audit lists %s's endpoints "
                   "under externalProviders" % native["provider"]}
    if replaced:
        out["replacedMovedTo"] = replaced
    return out


def install_blender(folder, addons, force=False):
    """Copy blender/<AddOn>/ into a Blender add-ons folder (scripts/addons). It does not enable it: the person enables
    it in Blender's preferences, after MifBlender."""
    folder, addons = os.path.abspath(folder), os.path.abspath(addons)
    manifest, errs = read_manifest(folder)
    if errs:
        return {"ok": False, "error": "; ".join(errs)}
    bl = manifest.get("blender")
    if not bl:
        return {"ok": False, "error": "%s has no Blender half" % manifest["name"]}
    if not os.path.isdir(addons):
        return {"ok": False, "error": "%s is not a folder - pass Blender's scripts/addons folder (Preferences > File "
                                      "Paths > Scripts, then addons)" % addons}
    src = os.path.join(folder, "blender", bl["addon"])
    if not os.path.isdir(src):
        return {"ok": False, "error": "no folder blender/%s" % bl["addon"]}
    dest = os.path.join(addons, bl["addon"])
    replaced = None
    if os.path.exists(dest):
        if not force:
            return {"ok": False, "error": "%s exists - pass --force to replace it (moved to %s, not deleted)"
                                         % (dest, os.path.join(addons, ".removed"))}
        replaced = _move_aside(dest, addons)
    shutil.copytree(src, dest, ignore=_COPY_IGNORE)
    out = {"ok": True, "installedAt": dest,
           "note": "enable it in Blender (Preferences > Add-ons) after MifBlender; ping's extensionApi then lists "
                   "%s's ops" % bl["provider"]}
    if replaced:
        out["replacedMovedTo"] = replaced
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The scaffold
# ---------------------------------------------------------------------------------------------------------------------

_TOOLS_TEMPLATE = '''"""{title} - a MifBridge extension (docs/32_EXTENSIONS.md in MifBridge).

register(api) runs once when the MCP server starts. Each tool is an MCP tool in the pack ext_{name}; its parameters
must be annotated, and its docstring's first sentence is its description.
"""

MIF_EXTENSION_API = 1


def register(api):
    @api.tool
    def {prefix}level_summary(class_filter: str = "") -> dict:
        "Count the open level's actors by class - one call that composes list_level_actors."
        res = api.call("list_level_actors", classFilter=class_filter or None, limit=1000)
        if not api.ok(res):
            return res
        counts = {{}}
        for actor in res.get("actors") or []:
            cls = actor.get("class") or "?"
            counts[cls] = counts.get(cls, 0) + 1
        return {{"ok": True, "actors": sum(counts.values()), "byClass": dict(sorted(counts.items()))}}
'''

_NATIVE_TOOL_TEMPLATE = '''

    @api.tool
    def {prefix}hello(name: str = "") -> dict:
        "Ask the extension's native endpoint {prefix}hello for a greeting and the editor's world name."
        return api.call("{prefix}hello", name=name or None)
'''

_UPLUGIN_TEMPLATE = '''{{
	"FileVersion": 3,
	"Version": 1,
	"VersionName": "0.1.0",
	"FriendlyName": "{plugin}",
	"Description": "A MifBridge extension: native endpoints registered through MifBridge's extension API.",
	"Category": "Editor",
	"CanContainContent": false,
	"Modules": [
		{{
			"Name": "{plugin}",
			"Type": "Editor",
			"LoadingPhase": "Default"
		}}
	],
	"Plugins": [
		{{
			"Name": "MifBridge",
			"Enabled": true
		}}
	]
}}
'''

_BUILD_TEMPLATE = '''using UnrealBuildTool;

public class {plugin} : ModuleRules
{{
	public {plugin}(ReadOnlyTargetRules Target) : base(Target)
	{{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		PrivateDependencyModuleNames.AddRange(new string[] {{ "Core", "CoreUObject", "Engine", "UnrealEd", "Json", "MifBridge" }});
	}}
}}
'''

_MODULE_TEMPLATE = '''// {plugin} - a MifBridge extension. Registers its endpoints through MifBridge's versioned extension API
// (MifBridgeExtensionApi.h); the bridge checks their parameters, gates and counts them like its own.
#include "Modules/ModuleManager.h"
#include "Dom/JsonObject.h"
#include "Editor.h"
#include "Engine/World.h"
#include "MifBridgeExtensionApi.h"
#include "MifBridgeParams.h"

static const TCHAR* const G{plugin}Provider = TEXT("{plugin}");

class F{plugin}Module : public IModuleInterface
{{
public:
	virtual void StartupModule() override
	{{
		using namespace MifBridge;
		FExtensionEndpointDesc Hello;
		Hello.Name = TEXT("{prefix}hello");
		Hello.Provider = G{plugin}Provider;
		Hello.Summary = TEXT("A greeting and the editor world's name - the scaffold's sample endpoint.");
		Hello.Bucket = EEndpointBucket::ReadOnly;
		Hello.Params = {{ {{ TEXT("name"), EExtensionParamType::String, false, TEXT("who to greet") }} }};
		Hello.Handler = [](const TSharedRef<FJsonObject>& In, const TSharedRef<FJsonObject>& Out)
		{{
			const FString Who = JStr(In, TEXT("name"), TEXT("world"));
			Out->SetStringField(TEXT("greeting"), FString::Printf(TEXT("hello, %s"), *Who));
			UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
			Out->SetStringField(TEXT("world"), World ? World->GetName() : FString());
		}};
		FString Error;
		if (!RegisterExtensionEndpoint(MoveTemp(Hello), &Error))
		{{
			UE_LOG(LogTemp, Error, TEXT("{plugin}: %s"), *Error);
		}}
	}}

	virtual void ShutdownModule() override
	{{
		MifBridge::UnregisterExternalEndpoints(G{plugin}Provider);
	}}
}};

IMPLEMENT_MODULE(F{plugin}Module, {plugin})
'''

_README_TEMPLATE = '''# {title}

A MifBridge extension (MifBridge docs/32_EXTENSIONS.md).

- `mifextension.json` - the manifest: name, prefix, the tools `tools.py` registers{native_line}
- `tools.py` - `register(api)`: each `@api.tool` function is an MCP tool in the pack `ext_{name}`

Check it: `python <MifBridge>/tools/mcp-server/mifext.py validate .`
Install it: `python <MifBridge>/tools/mcp-server/mifext.py install .`, then restart the MCP server.
{native_steps}'''


def _camel(name):
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def scaffold(name, dest_dir=".", prefix=None, native=False):
    """Write a working skeleton at <dest_dir>/<name>. Returns {"ok", "folder", "files"}."""
    if not NAME_RE.match(name or ""):
        return {"ok": False, "error": "name %r must be lowercase: a letter, then 1-31 letters, digits or '_'" % name}
    prefix = prefix or (name.split("_")[0][:15] + "_")
    if not PREFIX_RE.match(prefix) or prefix in RESERVED_PREFIXES:
        return {"ok": False, "error": "prefix %r must be 2-16 lowercase letters or digits ending in '_', and not one of "
                                      "%s" % (prefix, ", ".join(RESERVED_PREFIXES))}
    folder = os.path.abspath(os.path.join(dest_dir, name))
    if os.path.exists(folder):
        return {"ok": False, "error": "%s exists already" % folder}
    title = " ".join(p.capitalize() for p in name.split("_"))
    plugin = _camel(name) + "Ext"
    manifest = {"apiVersion": 1, "name": name, "version": "0.1.0",
                "description": "%s: tools that compose MifBridge's endpoints." % title, "prefix": prefix,
                "pack": {"title": title, "backend": "unreal", "rules": []}, "module": "tools.py",
                "tools": [prefix + "level_summary"]}
    files = {}
    tools_py = _TOOLS_TEMPLATE.format(title=title, name=name, prefix=prefix)
    if native:
        manifest["tools"].append(prefix + "hello")
        manifest["native"] = {"plugin": plugin, "provider": plugin, "endpoints": [prefix + "hello"]}
        tools_py += _NATIVE_TOOL_TEMPLATE.format(prefix=prefix)
        base = os.path.join("native", plugin)
        files[os.path.join(base, plugin + ".uplugin")] = _UPLUGIN_TEMPLATE.format(plugin=plugin)
        files[os.path.join(base, "Source", plugin, plugin + ".Build.cs")] = _BUILD_TEMPLATE.format(plugin=plugin)
        files[os.path.join(base, "Source", plugin, "Private", plugin + "Module.cpp")] = \
            _MODULE_TEMPLATE.format(plugin=plugin, prefix=prefix)
    files[MANIFEST] = json.dumps(manifest, indent=2) + "\n"
    files["tools.py"] = tools_py
    files["README.md"] = _README_TEMPLATE.format(
        title=title, name=name,
        native_line=(", and the native plugin's endpoints" if native else ""),
        native_steps=("Native half: `mifext.py install-native . --project <your project folder>`, then build the "
                      "project. `native/%s/` is a normal Unreal editor plugin that depends on MifBridge.\n" % plugin)
        if native else "")
    for rel, text in files.items():
        path = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return {"ok": True, "folder": folder, "files": sorted(files)}


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def _print(result):
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mifext.py", description="MifBridge extensions: new, validate, install, list, "
                                 "remove, install-native, install-blender. Local folders only - nothing is downloaded.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="write a working extension skeleton")
    p.add_argument("name")
    p.add_argument("--prefix")
    p.add_argument("--native", action="store_true", help="also write an Unreal plugin with one native endpoint")
    p.add_argument("--dir", default=".")
    p = sub.add_parser("validate", help="check an extension folder")
    p.add_argument("folder")
    p.add_argument("--load", action="store_true", help="also import the module and run register(api)")
    p = sub.add_parser("install", help="validate, then copy into the extensions directory")
    p.add_argument("folder")
    p.add_argument("--force", action="store_true")
    sub.add_parser("list", help="the installed extensions")
    p = sub.add_parser("remove", help="move an installed extension to .removed/")
    p.add_argument("name")
    p = sub.add_parser("install-native", help="copy native/<Plugin>/ into a project's Plugins/")
    p.add_argument("folder")
    p.add_argument("--project", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("install-blender", help="copy blender/<AddOn>/ into a Blender add-ons folder")
    p.add_argument("folder")
    p.add_argument("--addons", required=True, help="Blender's scripts/addons folder")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "new":
        return _print(scaffold(a.name, a.dir, a.prefix, a.native))
    if a.cmd == "validate":
        return _print(validate(a.folder, a.load))
    if a.cmd == "install":
        return _print(install(a.folder, a.force))
    if a.cmd == "list":
        return _print(list_installed())
    if a.cmd == "remove":
        return _print(remove(a.name))
    if a.cmd == "install-blender":
        return _print(install_blender(a.folder, a.addons, a.force))
    return _print(install_native(a.folder, a.project, a.force))


if __name__ == "__main__":
    sys.exit(main())
