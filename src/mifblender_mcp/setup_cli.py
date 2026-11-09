"""`mifblender-mcp setup`: connect AI apps to the server, and install and enable the add-on in Blender.

It shows what it will do and asks first. It merges into a client's config and never replaces it, keeps a backup, and refuses a
file it cannot parse. It never stops a process, and it makes no network call. Every step reports what it verified; a step
that could not be checked says so. The token is never printed.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile

from mifblender_mcp import __version__

SERVER_NAME = "mifblender"
PACKAGE = "mifblender-mcp"
ADDON_MODULE = "MifBlender"
HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.join(HERE, "blender-addon", ADDON_MODULE)
JSON_CLIENTS = ("claude-desktop", "cursor", "windsurf", "vscode")
ALL_CLIENTS = ("claude-code",) + JSON_CLIENTS + ("codex",)
RESULT_MARK = "MIFBLENDER_RESULT "
HIDE = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW


def _home():
    return os.path.expanduser("~")


def client_config_path(client):
    """Where this client keeps its user-level MCP config."""
    home, appdata = _home(), os.environ.get("APPDATA", "")
    mac = sys.platform == "darwin"
    if client == "claude-desktop":
        if os.name == "nt":
            return os.path.join(appdata, "Claude", "claude_desktop_config.json")
        base = os.path.join(home, "Library", "Application Support") if mac else os.path.join(home, ".config")
        return os.path.join(base, "Claude", "claude_desktop_config.json")
    if client == "cursor":
        return os.path.join(home, ".cursor", "mcp.json")
    if client == "windsurf":
        return os.path.join(home, ".codeium", "windsurf", "mcp_config.json")
    if client == "vscode":
        if os.name == "nt":
            return os.path.join(appdata, "Code", "User", "mcp.json")
        base = os.path.join(home, "Library", "Application Support") if mac else os.path.join(home, ".config")
        return os.path.join(base, "Code", "User", "mcp.json")
    if client == "codex":
        return os.path.join(home, ".codex", "config.toml")
    raise ValueError(client)


def client_present(client):
    if client == "claude-code":
        return shutil.which("claude") is not None
    if client == "codex" and shutil.which("codex"):
        return True
    if client == "vscode" and shutil.which("code"):
        return True
    return os.path.isdir(os.path.dirname(client_config_path(client)))


def server_command():
    """(command, args) a client starts. uvx's absolute path, because GUI clients often lack the user's PATH; pinned, so the
    server cannot move ahead of the add-on this setup installs."""
    uvx = shutil.which("uvx")
    if uvx:
        return uvx.replace("\\", "/"), ["%s@%s" % (PACKAGE, __version__)]
    return sys.executable.replace("\\", "/"), ["-m", "mifblender_mcp"]


def _backup(path):
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak-mifblender")


def _write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-mifblender"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _read_json_config(path):
    """(dict) or raises ValueError naming why it will not be touched."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        try:
            data = json.load(fh)
        except ValueError as e:
            raise ValueError("%s is not plain JSON (%s), so it was left alone" % (path, e))
    if not isinstance(data, dict):
        raise ValueError("%s is not a JSON object, so it was left alone" % path)
    return data


def _json_key(client):
    return "servers" if client == "vscode" else "mcpServers"


def json_entry(client, command, args, token):
    entry = {"command": command, "args": args}
    if client == "vscode":
        entry = {"type": "stdio", **entry}
    if token:
        entry["env"] = {"MIF_BLENDER_TOKEN": token}
    return entry


TOML_BLOCK = re.compile(r"(?ms)^\[mcp_servers\.%s(\.[^\]]+)?\]\s*?$.*?(?=^\[|\Z)" % SERVER_NAME)


def _toml_str(s):
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def toml_block(command, args, token):
    lines = ["[mcp_servers.%s]" % SERVER_NAME, "command = %s" % _toml_str(command),
             "args = [%s]" % ", ".join(_toml_str(a) for a in args)]
    if token:
        lines += ["", "[mcp_servers.%s.env]" % SERVER_NAME, "MIF_BLENDER_TOKEN = %s" % _toml_str(token)]
    return "\n".join(lines) + "\n"


def read_existing_token(client):
    """A token an earlier setup wrote, so a re-run keeps a working pair."""
    if client == "claude-code":
        return None
    path = client_config_path(client)
    try:
        if client == "codex":
            with open(path, encoding="utf-8") as fh:
                m = re.search(r'(?m)^\[mcp_servers\.%s\.env\]\s*\n(?:.*\n)*?MIF_BLENDER_TOKEN = "([^"]+)"' % SERVER_NAME, fh.read())
            return m.group(1) if m else None
        return ((_read_json_config(path).get(_json_key(client)) or {}).get(SERVER_NAME) or {}).get("env", {}).get("MIF_BLENDER_TOKEN")
    except (OSError, ValueError, AttributeError):
        return None


def connect_client(client, token):
    """Write this client's entry. Returns (ok, message), verified by reading it back."""
    command, args = server_command()
    if client == "claude-code":
        claude = shutil.which("claude")
        if not claude:
            return False, "the claude command is not on PATH"
        subprocess.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True, text=True, **HIDE)
        cmd = [claude, "mcp", "add", "--scope", "user", SERVER_NAME]
        if token:
            cmd += ["-e", "MIF_BLENDER_TOKEN=%s" % token]
        r = subprocess.run(cmd + ["--", command] + args, capture_output=True, text=True, **HIDE)
        if r.returncode:
            return False, "claude mcp add failed with exit code %d" % r.returncode
        got = subprocess.run([claude, "mcp", "get", SERVER_NAME], capture_output=True, text=True, **HIDE)
        return (got.returncode == 0), ("registered, read back" if got.returncode == 0 else "registered, but not read back")
    path = client_config_path(client)
    if client == "codex":
        old = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                old = fh.read()
        text = TOML_BLOCK.sub("", old).rstrip()
        text = (text + "\n\n" if text else "") + toml_block(command, args, token)
        _backup(path)
        _write_atomic(path, text)
        with open(path, encoding="utf-8") as fh:
            return ("[mcp_servers.%s]" % SERVER_NAME in fh.read()), path
    try:
        data = _read_json_config(path)
    except ValueError as e:
        return False, str(e)
    key = _json_key(client)
    if not isinstance(data.get(key, {}), dict):
        return False, "%s: %s is not an object, so it was left alone" % (path, key)
    data.setdefault(key, {})[SERVER_NAME] = json_entry(client, command, args, token)
    _backup(path)
    _write_atomic(path, json.dumps(data, indent=2) + "\n")
    return SERVER_NAME in _read_json_config(path).get(key, {}), path


def disconnect_client(client):
    if client == "claude-code":
        claude = shutil.which("claude")
        if not claude:
            return False, "the claude command is not on PATH"
        r = subprocess.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True, text=True, **HIDE)
        return r.returncode == 0, "removed" if r.returncode == 0 else "no entry removed"
    path = client_config_path(client)
    if not os.path.exists(path):
        return True, "no config"
    if client == "codex":
        with open(path, encoding="utf-8") as fh:
            old = fh.read()
        _backup(path)
        _write_atomic(path, TOML_BLOCK.sub("", old).rstrip() + "\n")
        return True, path
    try:
        data = _read_json_config(path)
    except ValueError as e:
        return False, str(e)
    if SERVER_NAME in (data.get(_json_key(client)) or {}):
        del data[_json_key(client)][SERVER_NAME]
        _backup(path)
        _write_atomic(path, json.dumps(data, indent=2) + "\n")
    return True, path


# ---------------------------------------------------------------------------------------------------- Blender


def find_blenders():
    """Every Blender executable found, newest name first."""
    found = []
    if os.name == "nt":
        roots = [os.environ.get(k, "") for k in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        for r in filter(None, roots):
            found += glob.glob(os.path.join(r, "Blender Foundation", "Blender*", "blender.exe"))
            found += glob.glob(os.path.join(r, "Programs", "Blender Foundation", "Blender*", "blender.exe"))
            found += glob.glob(os.path.join(r, "Steam", "steamapps", "common", "Blender", "blender.exe"))
    elif sys.platform == "darwin":
        found += glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender")
        found += glob.glob(os.path.join(_home(), "Applications", "Blender*.app", "Contents", "MacOS", "Blender"))
    else:
        found += glob.glob("/opt/blender*/blender") + glob.glob(os.path.join(_home(), "blender*", "blender"))
        found += glob.glob("/snap/bin/blender")
    on_path = shutil.which("blender")
    if on_path:
        found.append(on_path)
    seen, out = set(), []
    for p in found:
        key = os.path.normcase(os.path.realpath(p))
        if key not in seen and os.path.isfile(p):
            seen.add(key)
            out.append(p)
    return sorted(out, reverse=True)


def blender_version(exe):
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60, **HIDE)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Blender (\d+\.\d+(?:\.\d+)?)", r.stdout or "")
    return m.group(1) if m else None


def running_blenders():
    """PIDs of Blender processes. One that exits later saves its preferences over what this setup enabled."""
    try:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq blender.exe", "/NH", "/FO", "CSV"],
                               capture_output=True, text=True, timeout=30, **HIDE)
            return [int(m) for m in re.findall(r'^"blender\.exe","(\d+)"', r.stdout, re.M | re.I)]
        r = subprocess.run(["pgrep", "-ix", "blender"], capture_output=True, text=True, timeout=30)
        return [int(x) for x in r.stdout.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def addon_source_hash(d=ADDON_DIR):
    """The add-on's own fold: sha256 over each .py's name and bytes, 16 hex characters."""
    h = hashlib.sha256()
    for f in sorted(f for f in os.listdir(d) if f.endswith(".py")):
        h.update(f.encode("utf-8"))
        with open(os.path.join(d, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def build_addon_zip(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r, dirs, fs in os.walk(ADDON_DIR):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(fs):
                if not f.endswith((".pyc", ".pyo")):
                    p = os.path.join(r, f)
                    zf.write(p, os.path.join(ADDON_MODULE, os.path.relpath(p, ADDON_DIR)).replace(os.sep, "/"))
    return path


BLENDER_SCRIPT = r'''
import json, os, sys
import addon_utils, bpy
action, mod = os.environ["MIFBLENDER_ACTION"], os.environ["MIFBLENDER_MODULE"]
out = {"blender": bpy.app.version_string, "action": action}
try:
    if action == "install":
        addon_utils.disable(mod, default_set=True)
        bpy.ops.preferences.addon_install(overwrite=True, filepath=os.environ["MIFBLENDER_ZIP"])
        addon_utils.modules_refresh()
        addon_utils.enable(mod, default_set=True, persistent=True)
        token = os.environ.get("MIFBLENDER_TOKEN")
        if token:
            bpy.context.preferences.addons[mod].preferences.token = token
        bpy.ops.wm.save_userpref()
    elif action == "remove":
        # Not preferences.addon_remove: it ends in context.area.tag_redraw(), and a background Blender has no area, so it
        # raises before the preferences are saved. A folder that is a link is someone's development setup, so it is left
        # alone: deleting through it would delete what it points at.
        import shutil, stat
        addon_utils.disable(mod, default_set=True)
        path = os.path.join(bpy.utils.user_resource("SCRIPTS", path="addons"), mod)
        if os.path.lexists(path):
            attrs = getattr(os.lstat(path), "st_file_attributes", 0)
            if os.path.islink(path) or attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT if os.name == "nt" else os.path.islink(path):
                raise RuntimeError("%s is a link, so it was left alone" % path)
            shutil.rmtree(path)
        addon_utils.modules_refresh()
        bpy.ops.wm.save_userpref()
    enabled = mod in bpy.context.preferences.addons
    out["enabled"] = enabled
    if enabled:
        m = sys.modules.get(mod)
        out["version"] = ".".join(map(str, m.bl_info["version"]))
        out["sourceHash"] = m.source_hash()[:16] if m.source_hash() else None
        out["tokenSet"] = bpy.context.preferences.addons[mod].preferences.token not in ("", "dev")
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print("%s%s" % (os.environ["MIFBLENDER_MARK"], json.dumps(out)))
'''


def run_in_blender(exe, action, addon_zip=None, token=None):
    """Run the install, remove or check script in a background Blender with the user's own preferences. Never
    --factory-startup: saving preferences from one would write factory settings over the user's."""
    with tempfile.TemporaryDirectory(prefix="mifblender_setup_") as tmp:
        script = os.path.join(tmp, "mifblender_setup_script.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(BLENDER_SCRIPT)
        env = dict(os.environ, MIFBLENDER_ACTION=action, MIFBLENDER_MODULE=ADDON_MODULE, MIFBLENDER_MARK=RESULT_MARK)
        if addon_zip:
            env["MIFBLENDER_ZIP"] = addon_zip
        if token:
            env["MIFBLENDER_TOKEN"] = token
        try:
            r = subprocess.run([exe, "--background", "--python-exit-code", "1", "--python", script],
                               capture_output=True, text=True, timeout=300, env=env, cwd=tmp, **HIDE)
        except (OSError, subprocess.SubprocessError) as e:
            return {"error": "could not run Blender: %s" % e}
    for line in (r.stdout or "").splitlines():
        if line.startswith(RESULT_MARK):
            return json.loads(line[len(RESULT_MARK):])
    return {"error": "Blender gave no result (exit code %s)" % r.returncode}


# ---------------------------------------------------------------------------------------------------- the command


def _ask(question, yes):
    if yes:
        return True
    try:
        return input("%s [y/N] " % question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _choose_blenders(found, wanted, yes):
    """Never assumes which Blender is meant. Returns (exes, questions)."""
    chosen = []
    for w in wanted or []:
        if os.path.isfile(w):
            chosen.append(w)
            continue
        hits = [e for e in found if blender_version(e) and blender_version(e).startswith(w)]
        if not hits:
            raise SystemExit("mifblender setup: no Blender found for %r. Pass the path to its executable." % w)
        chosen += hits
    if chosen:
        return chosen, []
    if len(found) == 1 and yes:
        return found, []
    question = {"id": "blender", "ask": "Which Blender should get the add-on?", "candidates": found}
    if yes or not sys.stdin.isatty():
        return [], [question]
    if not found:
        print("No Blender was found. Run setup again with --blender <path to its executable>.")
        return [], []
    print("Blender installs found:")
    for i, e in enumerate(found, 1):
        print("  %d. %s (%s)" % (i, e, blender_version(e) or "version unknown"))
    reply = input("Install into which? Numbers separated by commas, 'all', or Enter for none: ").strip().lower()
    if reply == "all":
        return list(found), []
    picks = [int(x) for x in re.findall(r"\d+", reply) if 1 <= int(x) <= len(found)]
    return [found[i - 1] for i in picks], []


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="mifblender-mcp setup", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", help="comma-separated: %s. Default: the ones found on this computer" % ",".join(ALL_CLIENTS))
    ap.add_argument("--blender", action="append", metavar="PATH|VERSION", help="a Blender to install into; repeat for several")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen and change nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--no-addon", action="store_true", help="only connect the AI apps")
    ap.add_argument("--no-clients", action="store_true", help="only install the add-on")
    ap.add_argument("--check", action="store_true", help="report what is set up, and change nothing")
    ap.add_argument("--upgrade", action="store_true", help="bring the add-on and every client entry to this version")
    ap.add_argument("--uninstall", action="store_true", help="remove the client entries and the add-on")
    ap.add_argument("--json", action="store_true", help="print the plan and any questions as JSON, and change nothing")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv or [])
    if a.clients:
        clients = [c.strip() for c in a.clients.split(",") if c.strip()]
        unknown = sorted(set(clients) - set(ALL_CLIENTS))
        if unknown:
            print("mifblender setup: unknown client(s): %s. Known: %s" % (", ".join(unknown), ", ".join(ALL_CLIENTS)))
            return 2
    else:
        clients = [c for c in ALL_CLIENTS if client_present(c)]
    if a.no_clients:
        clients = []
    found = [] if a.no_addon else find_blenders()
    non_interactive_plan = a.json or a.dry_run or a.check
    blenders, questions = ([], []) if a.no_addon else _choose_blenders(found, a.blender, a.yes or non_interactive_plan)
    pids = [] if a.no_addon else running_blenders()

    if a.json or a.dry_run:
        plan = {"version": __version__, "clients": clients, "blenders": blenders, "runningBlenders": pids,
                "uninstall": a.uninstall, "questions": questions, "changed": False}
        print(json.dumps(plan, indent=2) if a.json else "Would %s: clients %s; Blender %s. Nothing was changed." % (
            "remove" if a.uninstall else "set up", clients or "none", blenders or "none"))
        return 0

    if a.check:
        rc = 0
        for c in clients:
            present = c != "claude-code" and os.path.exists(client_config_path(c)) and SERVER_NAME in open(
                client_config_path(c), encoding="utf-8-sig").read()
            print("%-15s %s" % (c, "entry found" if present else "not checked here" if c == "claude-code" else "no entry"))
        for exe in blenders or found:
            r = run_in_blender(exe, "check")
            same = r.get("sourceHash") == addon_source_hash()
            print("%s: enabled %s, version %s, matches this package: %s%s" % (
                r.get("blender", exe), r.get("enabled"), r.get("version"), same, " ERROR " + r["error"] if r.get("error") else ""))
            rc = rc or (0 if r.get("enabled") and same else 1)
        return rc

    if questions:
        print("Which Blender? Pass --blender <path>. Found: %s" % (", ".join(found) or "none"))
        return 2
    if pids is None and blenders:
        print("Could not tell whether Blender is running, so the add-on was not installed. Close Blender, and use --no-addon "
              "to skip this step.")
        return 1
    if pids and blenders:
        print("Blender is running (process %s). Close it and run setup again: one that exits later saves its preferences over "
              "this install. Nothing was changed." % ", ".join(map(str, pids)))
        return 1

    verb = "Remove from" if a.uninstall else "Set up"
    print("%s:\n  AI apps: %s\n  Blender: %s" % (verb, ", ".join(clients) or "none", ", ".join(blenders) or "none"))
    if not a.uninstall:
        print("Note: run_python is on by default and lets the assistant run any Python inside Blender. Turn it off in the "
              "add-on's preferences (Allow run_python) to limit it to the typed tools.")
    if not _ask("Continue?", a.yes):
        print("Nothing was changed.")
        return 1

    rc = 0
    if a.uninstall:
        for c in clients:
            ok, msg = disconnect_client(c)
            print("%-15s %s %s" % (c, "removed" if ok else "FAILED", msg))
            rc = rc or (0 if ok else 1)
        for exe in blenders:
            r = run_in_blender(exe, "remove")
            ok = not r.get("error") and not r.get("enabled")
            print("%s: %s" % (r.get("blender", exe), "add-on removed" if ok else "FAILED %s" % r.get("error")))
            rc = rc or (0 if ok else 1)
        return rc

    token = next(filter(None, (read_existing_token(c) for c in clients)), None)
    if not token and blenders:
        token = secrets.token_urlsafe(24)
    if blenders:
        with tempfile.TemporaryDirectory(prefix="mifblender_addon_") as tmp:
            zp = build_addon_zip(os.path.join(tmp, "MifBlender-%s.zip" % __version__))
            want = addon_source_hash()
            for exe in blenders:
                r = run_in_blender(exe, "install", zp, token)
                ok = r.get("enabled") and r.get("sourceHash") == want and not r.get("error")
                print("%s: %s" % (r.get("blender", exe), "add-on %s enabled and verified" % r.get("version") if ok
                                  else "FAILED %s" % (r.get("error") or "the installed add-on does not match this package")))
                rc = rc or (0 if ok else 1)
    for c in clients:
        ok, msg = connect_client(c, token)
        print("%-15s %s %s" % (c, "connected" if ok else "FAILED", msg))
        rc = rc or (0 if ok else 1)
    if not rc:
        print("Done. Restart your AI app, open Blender, and ask it to call bl_status.")
        if token:
            print("A private token links them; it is stored in the add-on's preferences and each app's config, and not shown.")
    return rc
