"""Roblox Studio - MifBridge's Roblox backend, on the `roblox` branch.

    gen_catalog  an API-Dump.json (or an installed Studio's -API) -> roblox_catalog.json, stamped with the Studio build
    catalog      the catalog loaded: describe a member, check a plugin may use it, did-you-mean
    values       caller values <-> wire values by the catalog's type, and the read-back comparison
    daemon       the loopback daemon the Studio plugin long-polls (run alone or as a thread)
    tools        the rbx_* tools, TOOLS and PACK for server.py

The plugin that runs inside Studio is tools/roblox-plugin/MifBridge.server.lua. Standard library only.
"""
