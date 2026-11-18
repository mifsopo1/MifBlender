"""Epic's UEFN MCP, as a client: Streamable HTTP on 127.0.0.1:<port>/mcp, standard library only.

Used for two things only - installing MifBridge's listener (execute_python) and the status checks that need no
listener (is the server on, is Python ready for the project). Everyday work goes through the listener.

Measured on UEFN 43.10: initialize answers protocol 2025-06-18 with an EMPTY serverInfo and no
instance _meta, so nothing here takes the editor's identity from an answer; tools/list holds three meta-tools
(list_toolsets, describe_toolset, call_tool) and a toolset's tool is reached through call_tool with toolset_name,
tool_name and arguments. A reply may be JSON or an SSE stream. Calls are serialized on the editor's game thread, and
Epic lists hitching or hanging tool calls as a known issue, so every call has a timeout and the error says which of
DOWN (refused) or BUSY (no answer in time) it was.
"""
import json
import socket
import time
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"


class EpicError(Exception):
    """kind: 'down' (nothing answered), 'busy' (connected, no reply in time), 'http' (an HTTP error status),
    'protocol' (an answer that is not a JSON-RPC result), 'tool' (the tool reported an error)."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


class EpicMCP:
    def __init__(self, port=8000, host="127.0.0.1", timeout=65.0, opener=None):
        self.url = "http://%s:%d/mcp" % (host, int(port))
        self.timeout = timeout
        self.session = None
        self.proto = PROTOCOL
        self._id = 0
        self._open = opener or urllib.request.urlopen

    # ------------------------------------------------------------ transport
    def _post(self, body, timeout=None):
        data = json.dumps(body).encode("utf-8")
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        if body.get("method") != "initialize":
            h["MCP-Protocol-Version"] = self.proto
        req = urllib.request.Request(self.url, data=data, headers=h, method="POST")
        try:
            with self._open(req, timeout=timeout or self.timeout) as r:
                headers, raw = dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            raise EpicError("http", "Epic's UEFN MCP answered HTTP %s at %s" % (e.code, self.url))
        except (socket.timeout, TimeoutError):
            raise EpicError("busy", "Epic's UEFN MCP at %s accepted the request and did not answer in %.0f s - UEFN "
                                    "may be hitching (a known UEFN issue) or busy with another call"
                            % (self.url, timeout or self.timeout))
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise EpicError("busy", "Epic's UEFN MCP at %s did not answer in %.0f s" % (self.url,
                                                                                         timeout or self.timeout))
            raise EpicError("down", "nothing answered at %s (%s)" % (self.url, reason))
        except (ConnectionError, OSError) as e:
            raise EpicError("down", "nothing answered at %s (%s)" % (self.url, e))
        sid = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        if sid:
            self.session = sid
        return parse(raw, headers.get("Content-Type") or headers.get("content-type") or "")

    def rpc(self, method, params=None, timeout=None):
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params
        for m in self._post(body, timeout=timeout):
            if isinstance(m, dict) and m.get("id") == self._id:
                if m.get("error"):
                    raise EpicError("protocol", "%s: %s" % (method, json.dumps(m["error"])[:400]))
                return m.get("result")
        raise EpicError("protocol", "%s: no JSON-RPC result in the reply" % method)

    def notify(self, method):
        self._post({"jsonrpc": "2.0", "method": method})

    def initialize(self):
        self.session = None
        res = self.rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                      "clientInfo": {"name": "mifbridge-uefn", "version": "1"}}) or {}
        self.proto = res.get("protocolVersion") or PROTOCOL
        self.notify("notifications/initialized")
        return res

    # ------------------------------------------------------------ tools
    def call(self, toolset, tool, arguments=None, timeout=None):
        """One toolset tool through the call_tool meta-tool. Returns (text, structured) where structured is the JSON
        of the text when it parses. Raises EpicError('tool') when the tool reports isError."""
        if self.session is None:
            self.initialize()
        t0 = time.perf_counter()
        res = self.rpc("tools/call", {"name": "call_tool", "arguments": {
            "toolset_name": toolset, "tool_name": tool, "arguments": arguments or {}}}, timeout=timeout) or {}
        text = "".join(c.get("text", "") for c in res.get("content") or [] if isinstance(c, dict)
                       and c.get("type") == "text")
        if res.get("isError"):
            raise EpicError("tool", "%s.%s: %s" % (toolset, tool, text[:600]))
        try:
            structured = json.loads(text) if text else None
        except ValueError:
            structured = None
        self.last_ms = round((time.perf_counter() - t0) * 1000, 1)
        return text, structured


def parse(raw, ctype):
    """JSON-RPC messages in a reply body: plain JSON, or the data lines of an SSE stream."""
    if not raw:
        return []
    txt = raw.decode("utf-8", "replace")
    if "text/event-stream" in ctype:
        out, buf = [], []
        for line in txt.splitlines() + [""]:
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
            elif not line.strip() and buf:
                try:
                    out.append(json.loads("\n".join(buf)))
                except ValueError:
                    pass
                buf = []
        return out
    try:
        msg = json.loads(txt)
    except ValueError:
        return []
    return msg if isinstance(msg, list) else [msg]


def return_value(structured):
    """A toolset tool's JSON answer carries its value under returnValue."""
    return structured.get("returnValue") if isinstance(structured, dict) else structured
