"""The per-install token: a random secret in a per-user file that MifBlender and the MCP server both read.

WHY (1.4, FOUND 2026-11-11 in a code review of 1.3.0). The add-on shipped with the token "dev", the value every doc and
sample named, with run_python on, so any program on the machine could run Python in Blender out of the box. Now a GUI
Blender's server, with the Token mode preference left on Per-install, uses a random token made on first start and kept
in this file; the MCP server reads the same file when its environment names no token, so nothing needs configuring.

THE FILE. MIF_BLENDER_TOKEN_FILE, else ~/.mifbridge/blender_token: one line, the token, created with owner-only
permissions where the OS has them. tools/mcp-server/server.py's _bl_token_file() resolves the same path the same way -
the two must stay in sync (the token suite's BT100 checks they agree).

WHAT KEEPS "dev". A server started with an explicit token (serve_forever(token=...), start_server(token=...)), a
Custom token preference, and a headless Blender served by a script (blender -b ... serve_forever()) - a command someone
typed, never an install's default. The MCP server tries the file's token and then "dev", so it reaches both kinds.
"""
import os
import secrets

TOKEN_FILE_ENV = "MIF_BLENDER_TOKEN_FILE"
LEGACY_TOKEN = "dev"
_MIN_LEN = 16


def token_path():
    raw = (os.environ.get(TOKEN_FILE_ENV) or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.join(os.path.expanduser("~"), ".mifbridge", "blender_token")


def _valid(tok):
    return isinstance(tok, str) and len(tok) >= _MIN_LEN and all(" " < c <= "~" for c in tok)


def read_token():
    """The file's token, or None when there is no usable one."""
    try:
        with open(token_path(), "r", encoding="utf-8") as fh:
            tok = fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return tok if _valid(tok) else None


def read_or_create():
    """(token, error). Reads the file, or makes it with a new random token. error is a sentence when neither worked;
    the token is then a random one for this session only, so nothing can connect with a guessed value."""
    tok = read_token()
    if tok:
        return tok, None
    path = token_path()
    new = secrets.token_urlsafe(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):              # present but unusable (empty, damaged): replaced
            os.remove(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new + "\n")
    except FileExistsError:                   # another Blender made it first: use theirs
        tok = read_token()
        if tok:
            return tok, None
        return new, "the token file %s exists but cannot be read" % path
    except OSError as exc:
        return new, "could not write the token file %s (%s)" % (path, exc)
    return new, None
