"""UEFN (Unreal Editor for Fortnite) - MifBridge's UEFN backend, preview in 1.4.0.

    epic      a client for Epic's UEFN MCP (Streamable HTTP): the bootstrap that installs the listener, and status
    backend   what the uefn_* tools do: status and setup guidance, install, one listener op with the ownership check,
              the Blender hand-off
    listener/ the files that run INSIDE UEFN, loaded there by path: mifbridge_uefn_listener.py (the listener),
              mifbridge_uefn_panel.py (the panel), mifbridge_uefn_win32.py (process and window helpers, also used here)

Standard library only.
"""
