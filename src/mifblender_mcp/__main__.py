"""`mifblender-mcp`: start the MCP server, or `mifblender-mcp setup` to connect it.

With no arguments this IS the server an MCP client starts: stdout belongs to the protocol, so nothing here prints to it.
The server is a script that imports the modules beside it, so it runs from its own folder on sys.path, as `python server.py`
would, and finds the add-on it compares against in the sibling blender-addon folder.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(HERE, "mcp-server")


def run_server():
    server = os.path.join(SERVER_DIR, "server.py")
    sys.path.insert(0, SERVER_DIR)
    sys.argv = [server]
    runpy.run_path(server, run_name="__main__")


def main():
    args = sys.argv[1:]
    if args and args[0] == "setup":
        from mifblender_mcp import setup_cli
        return setup_cli.main(args[1:])
    if args and args[0] in ("--version", "-V"):
        from mifblender_mcp import __version__
        print(__version__)
        return 0
    if args and args[0] in ("--help", "-h", "help"):
        print("mifblender-mcp          start the MCP server (your AI app does this)\n"
              "mifblender-mcp setup    connect your AI apps and install the Blender add-on\n"
              "mifblender-mcp setup --help    every setup option")
        return 0
    if args:
        print("mifblender-mcp: unknown argument %r. Try --help." % args[0], file=sys.stderr)
        return 2
    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
