"""Generate mif_uefn/uefn_catalog.json - every tool of Epic's UEFN MCP with MifBridge's additions - from the toolsets'
own schemas, stamped with the UEFN build they came from. Re-run it when UEFN updates.

    python tools/mcp-server/mif_uefn/gen_catalog.py --live [--port 8000]          # from a running UEFN
    python tools/mcp-server/mif_uefn/gen_catalog.py --from-dir DIR --uefn BUILD   # from saved describe_toolset answers

--live asks Epic's MCP for list_toolsets, then describe_toolset for each, one call at a time (each ~0.3-1 s), and reads
the build from the engine version through the listener when it is installed, else takes --uefn. A saved folder holds one
<toolset>.json per toolset, each the describe_toolset reply ({"result": {"content": [{"text": "<json>"}]}}).

What it adds per tool - mode (read/write, corrected), getter, deny - is mifbridge_uefn_catalog.py's, so the rules live
in one place. It refuses to write a catalog with no tools, and prints the counts the release notes quote.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mif_uefn import epic as E  # noqa: E402
from mif_uefn.listener import mifbridge_uefn_catalog as C  # noqa: E402

OUT = os.path.join(HERE, "uefn_catalog.json")


def _toolset_doc(reply):
    res = reply.get("result", reply) if isinstance(reply, dict) else {}
    text = "".join(c.get("text", "") for c in res.get("content") or [] if isinstance(c, dict))
    return json.loads(text) if text else (res.get("structuredContent") or {})


def from_dir(path):
    out = []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".json") or fn.startswith(("list_toolsets", "_")):
            continue
        with open(os.path.join(path, fn), encoding="utf-8-sig") as fh:
            reply = json.load(fh)
        try:
            doc = _toolset_doc(reply)
        except ValueError:
            continue
        if doc.get("tools"):
            out.append(doc)
    return out


def from_live(port, mcp=None):
    """Every toolset's describe_toolset document from Epic's UEFN MCP over HTTP (mcp: an EpicMCP, or a stand-in)."""
    m = mcp or E.EpicMCP(port)
    m.initialize()
    listed = m.rpc("tools/call", {"name": "list_toolsets", "arguments": {}}, timeout=60) or {}
    text = "".join(c.get("text", "") for c in listed.get("content") or [])
    names = re.findall(r"(?m)^- ([\w\.]+):", text)
    out = []
    for n in names:
        r = m.rpc("tools/call", {"name": "describe_toolset", "arguments": {"toolset_name": n}}, timeout=60) or {}
        doc = _toolset_doc({"result": r})
        if doc.get("tools"):
            out.append(doc)
    return out


def from_catalog(path):
    """The describe_toolset documents back out of a catalog this script wrote, so MifBridge's additions (mode, getter,
    deny, verify) can be re-derived after a rule changes here, with no UEFN running. Epic's own data - names,
    annotations, descriptions, schemas, the build stamp - is kept exactly."""
    with open(path, encoding="utf-8") as fh:
        cat = json.load(fh)
    docs = {}
    for full, r in cat["tools"].items():
        ts = r["toolset"]
        doc = docs.setdefault(ts, {"name": ts, "description": (cat.get("toolsets") or {}).get(ts, ""), "tools": []})
        doc["tools"].append({"name": full, "annotations": {"accessMode": r.get("annotated")},
                             "description": r.get("description") or "", "inputSchema": r.get("inputSchema"),
                             "outputSchema": r.get("outputSchema")})
    return list(docs.values()), cat.get("uefn"), cat.get("generated")


def build(docs, uefn):
    tools, toolsets = {}, {}
    for doc in docs:
        ts = doc.get("name")
        toolsets[ts] = (doc.get("description") or "").strip()
        for t in doc["tools"]:
            full = t["name"]
            ann = (t.get("annotations") or {}).get("accessMode")
            desc = (t.get("description") or "").strip()
            tools[full] = {"toolset": ts, "name": full.rsplit(".", 1)[-1], "annotated": ann,
                           "mode": C.corrected_mode(full, ann), "summary": desc.split("\n\n")[0][:300],
                           "description": desc, "inputSchema": t.get("inputSchema") or {"type": "object"},
                           "outputSchema": t.get("outputSchema")}
    for full, row in tools.items():
        row["getter"] = C.find_getter(full, tools)
        row["deny"] = C.deny_reason(full)
        row["verify"] = C.verify_route(row)
    writes = [r for r in tools.values() if r["mode"] == "write" and not r["deny"]]
    counts = {"toolsets": len(toolsets), "tools": len(tools),
              "denied": sum(1 for r in tools.values() if r["deny"]),
              "read": sum(1 for r in tools.values() if r["mode"] == "read" and not r["deny"]),
              "write": len(writes), "writesWithGetter": sum(1 for r in writes if r["getter"]),
              "writesWithoutGetter": sum(1 for r in writes if not r["getter"]),
              "writeVerify": {k: sum(1 for r in writes if r["verify"] == k) for k in ("getter", "target", "result", "asset", "none")},
              "correctedToRead": sum(1 for r in tools.values() if r["mode"] == "read" and r["annotated"] != "read")}
    return {"_source": "Generated by tools/mcp-server/mif_uefn/gen_catalog.py from Epic's UEFN MCP describe_toolset "
                       "answers. Do not edit by hand - re-run it when UEFN updates.",
            "uefn": uefn, "generated": time.strftime("%Y-%m-%d"), "counts": counts, "toolsets": toolsets,
            "tools": dict(sorted(tools.items()))}


def write_catalog(cat, path):
    """Write a catalog one tool per line (compact, and a diff after a UEFN update shows which tools changed), then read
    it back: it must parse as the JSON it is and hold the same tools. Written beside the target and moved into place,
    so a failed write never leaves half a catalog where the listener loads one."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("{\n")
        for k in ("_source", "uefn", "generated", "counts", "toolsets"):
            fh.write("%s:%s,\n" % (json.dumps(k), json.dumps(cat[k], separators=(",", ":"), ensure_ascii=False)))
        fh.write('"tools":{\n')
        rows = list(cat["tools"].items())
        for i, (k, v) in enumerate(rows):
            fh.write("%s:%s%s\n" % (json.dumps(k), json.dumps(v, separators=(",", ":"), ensure_ascii=False),
                                     "," if i < len(rows) - 1 else ""))
        fh.write("}}\n")
    with open(tmp, encoding="utf-8") as fh:
        back = json.load(fh)
    if set(back.get("tools") or {}) != set(cat["tools"]):
        raise ValueError("the catalog written to %s does not read back with the same tools" % tmp)
    os.replace(tmp, path)
    return back


EPIC_FIELDS = ("toolset", "annotated", "description", "inputSchema", "outputSchema")


def diff_catalogs(old, new):
    """Tools added, removed and changed from old to new (catalog dicts). changed names the Epic fields that differ and
    whether the corrected mode moved; MifBridge's own additions are derived from those, so they are not compared."""
    ot, nt = (old or {}).get("tools") or {}, (new or {}).get("tools") or {}
    changed = []
    for k in sorted(set(ot) & set(nt)):
        fields = [f for f in EPIC_FIELDS
                  if json.dumps(ot[k].get(f), sort_keys=True) != json.dumps(nt[k].get(f), sort_keys=True)]
        if fields:
            row = {"tool": k, "fields": fields}
            if ot[k].get("mode") != nt[k].get("mode"):
                row["mode"] = [ot[k].get("mode"), nt[k].get("mode")]
            changed.append(row)
    return {"from": (old or {}).get("uefn"), "to": (new or {}).get("uefn"),
            "added": sorted(set(nt) - set(ot)), "removed": sorted(set(ot) - set(nt)), "changed": changed}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--from-dir")
    ap.add_argument("--uefn", help="the UEFN build, e.g. ++Fortnite+Release-43.10-CL-60458657")
    ap.add_argument("--rederive", action="store_true",
                    help="re-derive MifBridge's additions on the shipped catalog after a rule change; keeps its build "
                         "stamp and date")
    a = ap.parse_args(argv)
    generated = None
    if a.rederive:
        docs, a.uefn, generated = from_catalog(OUT)
    else:
        docs = from_live(a.port) if a.live else from_dir(a.from_dir) if a.from_dir else None
    if docs is None:
        ap.error("--live, --from-dir or --rederive")
    if not a.uefn:
        ap.error("--uefn BUILD is required: the catalog is stamped with the UEFN it was read from")
    cat = build(docs, a.uefn)
    if generated:
        cat["generated"] = generated
    if not cat["tools"]:
        print("REFUSING: no tools were read - nothing written")
        return 1
    write_catalog(cat, OUT)
    print(json.dumps(cat["counts"]))
    print("wrote %s (%d bytes)" % (OUT, os.path.getsize(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
