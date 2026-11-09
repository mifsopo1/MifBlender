#!/usr/bin/env python3
"""MifBlender MCP server - the Blender tools of MifBridge, for any MCP client, with no Unreal Engine.

GENERATED from MifBridge's tools/mcp-server/server.py by tools/build_blender_mcp.py: the same code, less every Unreal
tool and the Unreal transport. Do not edit it here; a change belongs in MifBridge's server.py.

bl_* tools reach the MifBlender add-on over a loopback socket (length-prefixed JSON). find_tools, mif_call, mif_help,
mif_enable_pack, mif_list_packs and run_python work as they do in MifBridge.

Config (environment):
  MIF_BLENDER_HOST  default 127.0.0.1
  MIF_BLENDER_PORT  default 8792   (the add-on's port; NOT 9876, the third-party blender-mcp add-on's)
  MIF_BLENDER_TOKEN default "dev"  (must match the add-on's token)
  MIF_BLENDER_CONNECT_TIMEOUT default 3    (seconds)
  MIF_BLENDER_PROBE_TIMEOUT   default 5    (seconds; bl_status only)
  MIF_BLENDER_TIMEOUT         default 180  (seconds; geometry work)
  MIF_TOOL_PACKS    default "lean"; "blender" loads every Blender tool up front

Run:  python server.py [--debug]
"""

import argparse
import asyncio
import base64
import contextvars
import difflib
import functools
import itertools
import json

import os
import socket
import struct

import sys

import threading
import time
import zlib
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    from mcp.server.lowlevel.server import NotificationOptions   # see _SlimFastMCP.__init__
except ImportError:  # an SDK that moved it: the server still runs, and says what it could not declare
    NotificationOptions = None


def _locate_project_dir():
    """(folder, how) of the Unreal project this server works for, or (None, why not). Never raises."""
    named = (os.environ.get("MIF_PROJECT_DIR") or "").strip()
    if named:
        return (os.path.abspath(named), "MIF_PROJECT_DIR") if os.path.isdir(named) else (
            None, "MIF_PROJECT_DIR=%r is not a folder" % named)
    try:
        here = os.path.abspath(os.getcwd())
    except OSError:
        return None, "no working directory"
    while True:
        try:
            if any(n.lower().endswith(".uproject") for n in os.listdir(here)):
                return here, "the .uproject above the working directory"
        except OSError:
            pass
        up = os.path.dirname(here)
        if up == here:
            return None, "no .uproject in the working directory or above it"
        here = up

BASE = ""   # no Unreal Editor in the Blender-only server

TOKEN = os.environ.get("MIF_BRIDGE_TOKEN", "dev")
# Self-reported caller tag for the Activity panel (X-Mif-Agent header) - set MIF_AGENT before
# launching this server so the panel can tell "claude", "gpt", "gemini", etc. apart when several
# agents share one editor. Unset, the client's own handshake name is used instead - see _agent_tag.
AGENT = os.environ.get("MIF_AGENT", "")


def _agent_tag() -> str:
    """Who is calling, for the Activity panel: MIF_AGENT when it is set, else the name the MCP client
    gave for itself in its handshake (clientInfo.name - "claude-code", "claude-ai", "codex-mcp-client").

    SELF-REPORTED EITHER WAY, like the header it fills. Until 2026-09-21 only MIF_AGENT counted, so a
    client whose config did not set it arrived untagged: a user ran self_audit from Claude and
    from Codex, and only the Codex row said who it was. The handshake name is not a guess - it is what
    the client sent about itself - so it closes that gap with no configuration, and MIF_AGENT still wins
    for anyone who wants their own label. Printable ASCII only, because it travels as an HTTP header.
    """
    name = AGENT
    if not name:
        try:
            name = mcp.get_context().session.client_params.clientInfo.name or ""
        except Exception:  # noqa: BLE001 - outside a request, or an SDK that moved it: no tag, never a failed call
            name = ""
    return "".join(ch for ch in str(name) if " " <= ch <= "~").strip()[:32]
try:
    TIMEOUT = float(os.environ.get("MIF_BRIDGE_TIMEOUT", "30"))
except ValueError:
    TIMEOUT = 30.0


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


BLENDER_HOST = os.environ.get("MIF_BLENDER_HOST", "127.0.0.1")
try:
    BLENDER_PORT = int(os.environ.get("MIF_BLENDER_PORT", "8792"))
except ValueError:
    BLENDER_PORT = 8792
BLENDER_TOKEN = os.environ.get("MIF_BLENDER_TOKEN", TOKEN)
# GENERATIVE TOOLS ARE ON BY DEFAULT, AND EACH PROJECT CHOOSES (2026-10-01, reversing the same day's
# non-generative default). The bl_gen_* tools are the only ones that run a trained model (Flux and Hunyuan3D-2, through
# a ComfyUI the user runs). MIF_GENERATIVE=off here switches them off for every project this server touches; the
# add-on preference allow_generative and each project's own mode (mif_project_mode) are the other two switches - see
# _generative_refusal beside the tools. MIF_ALLOW_GENERATIVE, the opt-in of the non-generative default, is still read
# when MIF_GENERATIVE is unset, so a config that set it keeps meaning what it said. A module global, not a constant, so
# tools/test_generative_mode.py can switch it.
_OFF_WORDS = ("0", "false", "no", "off", "non-generative")


def _env_generative() -> bool:
    raw = os.environ.get("MIF_GENERATIVE", "").strip() or os.environ.get("MIF_ALLOW_GENERATIVE", "").strip()
    return raw.lower() not in _OFF_WORDS


_GENERATIVE = _env_generative()

# THREE Blender timeouts, deliberately, because "Blender is not running",
# "Blender is wedged" and "Blender is chewing a 200k-tri bevel" are three
# different waits and one number cannot serve all of them:
#   CONNECT (3s)   bounds reaching the listening socket at all. MEASURED on
#                  this box: a closed 127.0.0.1 port returns WSAECONNREFUSED in
#                  ~2.0s, not instantly - Windows retries the SYN first. 3s is
#                  chosen to sit just ABOVE that, so "Blender is not running"
#                  surfaces as the accurate "actively refused" message with the
#                  install steps; drop it to 1s and the same case reports a
#                  misleading connect TIMEOUT instead. Anything longer just adds
#                  dead wait to the commonest failure.
#   PROBE   (5s)   read timeout for bl_status ONLY. A ping does no bpy work, so
#                  5s of silence means the main thread is blocked (modal
#                  operator, file browser, render). This is the number that
#                  makes mif_mesh_roundtrip's step-0 health check cheap instead
#                  of a 180s hang, and it is why that step runs BEFORE Unreal is
#                  asked to write anything.
#   WORK  (180s)   read timeout for every other op. An FBX import/export or a
#                  bevel runs as ONE long main-thread frame in Blender; it
#                  cannot be polled and it must not be cut off half-applied.
BLENDER_CONNECT_TIMEOUT = _envf("MIF_BLENDER_CONNECT_TIMEOUT", 3.0)
BLENDER_PROBE_TIMEOUT = _envf("MIF_BLENDER_PROBE_TIMEOUT", 5.0)
BLENDER_TIMEOUT = _envf("MIF_BLENDER_TIMEOUT", 180.0)

DEBUG = False

# ---------------------------------------------------------------------------
# WHAT A CLIENT IS SENT, SLIMMED - added 2026-09-11.
#
# Every tool's name, description and input schema sit in the model's context on every turn, called
# or not. Measured through FastMCP's own list_tools - the exact payload a client receives - the 651
# tools came to 497,196 characters, about 124k tokens, before any work began. Two things in that
# carried nothing:
#   * 3,566 "title" annotations (84k chars) that FastMCP generates by capitalising each parameter
#     name ("path" -> "Path") and the argument model's name ("set_retarget_rigsArguments").
#   * anyOf [X, {"type": "null"}] wrappers around optional parameters. Arguments are validated
#     against this server's own pydantic model, not against the advertised schema, so a client that
#     still sends null is accepted - the wrapper only cost context.
# And one thing worth keeping but not on every turn: descriptions over _DESC_BUDGET, nearly all of
# them bl_* tools written after the 2026-08-30 extraction into tool_help.json (see below). Those go
# out as their lead sentence plus a mif_help pointer, and mif_help serves the full registered text.
# Nothing is deleted, and the summary is DERIVED rather than hand-kept, so a new tool cannot drift
# back out of budget the way 180 of them did after the one-off extraction.
#
# PARAMETER NAMES ARE NOT ANNOTATIONS. 51 tools take a parameter literally called type, default or
# properties. Keys inside a "properties" (or "$defs") map are parameter names and are never touched;
# only a schema's own annotation keys are. tools/test_mcp_footprint.py asserts every parameter and
# every required list survives, and ratchets the total so this cannot quietly grow back.
# ---------------------------------------------------------------------------
import re as _re

_DESC_BUDGET = 240


def _slim_schema(node, names_map=False):
    if isinstance(node, dict):
        if names_map:
            return {k: _slim_schema(v) for k, v in node.items()}
        out = {}
        for k, v in node.items():
            # An empty-string or null default says "optional, unset" - which the parameter's absence
            # from `required` already says. 21k chars of nothing across the list, measured 2026-09-11.
            if k == "title" or (k == "default" and (v == "" or v is None)):
                continue
            out[k] = _slim_schema(v, names_map=(k in ("properties", "$defs", "definitions")))
        alts = out.get("anyOf")
        if isinstance(alts, list) and len(alts) == 2 and {"type": "null"} in alts:
            other = alts[0] if alts[1] == {"type": "null"} else alts[1]
            if isinstance(other, dict):
                merged = {k: v for k, v in out.items() if k != "anyOf"}
                for k, v in other.items():
                    merged.setdefault(k, v)
                out = merged
        return out
    if isinstance(node, list):
        return [_slim_schema(x) for x in node]
    return node


# NOT EVERY PERIOD ENDS A SENTENCE, and _lead asks where one ends twice: for a long text's first sentence, and whether
# a short text stops mid-sentence. AN ELLIPSIS DOES NOT END ONE: "at a /Game/... path." was led as "at a /Game/...", so
# create_blueprint's served text and find_tools row never said what it made (docs/TOKEN_OPTIMIZATION.md 2.10). NOR DOES
# AN ABBREVIATION, or Niagara's User. namespace. Measured 2026-09-27, five leads stopped at one - "Solve a limb of an
# AnimSequence at every key so its tip (bone, e.g." (solve_anim_two_bone_ik, add_actor_bound_event,
# set_niagara_static_switch), "Create a User." and "Set a User." (add_/set_niagara_user_parameter) - and six texts the
# 2026-08-30 extraction had cut at "(e.g." or "/ ..." were sent whole, as if finished, with no pointer to the rest
# (add_variable, move_node, remove_node, refresh_node, preview_composite_widget, describe_property). A dotted run
# (e.g., i.e., a.k.a.) never ends a sentence here, nor does a listed word. etc. can end one, but a sentence it does
# end only runs on to the next end - cut at a clause if long - where one it does not end stops mid-thought.
_NOT_A_SENTENCE_END = ("etc", "vs", "cf", "incl", "approx", "User")
_NO_END_HERE = r"(?<!\.\.)(?<!\b[A-Za-z]\.[A-Za-z])" + "".join(r"(?<!\b%s)" % w for w in _NOT_A_SENTENCE_END)
_FIRST_SENTENCE = _re.compile(r"(.+?" + _NO_END_HERE + r"[.!?])(?:\s|$)", _re.S)
_ENDS_FINISHED = _re.compile(r"(?:" + _NO_END_HERE + r"[.!?]|[)\"'`\]])$")


def _lead(name: str, desc: str) -> str:
    desc = (desc or "").strip()
    # A DESCRIPTION THAT STOPS MID-SENTENCE IS A TRIMMED ONE. The 2026-08-30 extraction cut 42 tools' own text at
    # 240 characters mid-sentence and moved the rest to tool_help.json, and at <= 240 they were sent as they
    # stood - "... that folds in object scale)," - with no pointer to the notes that finish the sentence
    # (found 2026-09-21 by the pack-rule review). Such a text is led like any long one, pointer included.
    # Stopping at "(e.g." is stopping mid-sentence too (_NO_END_HERE).
    cut_off = bool(desc) and not _ENDS_FINISHED.search(desc) and name in (_tool_help() or {})
    if len(desc) <= _DESC_BUDGET and not cut_off:
        return desc
    m = _FIRST_SENTENCE.match(desc)
    lead = " ".join((m.group(1) if m else desc).split())
    if len(lead) > _DESC_BUDGET - 45:
        # AT A CLAUSE, NOT MID-WORD: "... outside the depsgr..." told an agent less than "... outside the view
        # layer...". The last clause break past 60% of the room, else the last space.
        room = lead[:_DESC_BUDGET - 48]
        cut = max(room.rfind(" - "), room.rfind("; "), room.rfind(", "))
        if cut < 0.6 * len(room):
            cut = room.rfind(" ")
        lead = (room[:cut] if cut > 0 else room).rstrip(" ,;-.") + "..."
    return "%s Full notes: mif_help('%s')." % (lead, name)


def _compact_blocks(blocks):
    out = []
    for b in blocks:
        text = getattr(b, "text", None)
        if getattr(b, "type", None) == "text" and isinstance(text, str) and text[:1] in ("{", "["):
            try:
                b = b.model_copy(update={"text": json.dumps(json.loads(text), separators=(",", ":"),
                                                            ensure_ascii=False)})
            except ValueError:
                pass
        out.append(b)
    return out


# LOADED WITHOUT A SEARCH. Claude Code defers every MCP tool behind its ToolSearch unless the tool's _meta carries
# "anthropic/alwaysLoad" (code.claude.com/docs/en/mcp.md), so a session on core - the default since e4e86226 - could
# not call even these without a lookup first. the benchmark's round three measured it: every Blender run spent its first two
# requests on ToolSearch -> find_tools -> mif_call to reach bl_run_python, and find_tools' answers were 52.6% of what
# those runs carried (docs/TOKEN_OPTIMIZATION.md 2.6). These are what a default session reaches for first, so they come
# up front: 1,195 characters, ~528 tokens a request (measured), against a lookup request of 5-8k. Everything else stays
# deferred, and a client that loads every tool up front anyway ignores the key.
# ue_run_python joined them after round four (2026-09-23): all five as-shipped U1 runs spent two
# requests asking find_tools for an Unreal Python tool before any work (docs/TOKEN_OPTIMIZATION.md 2.7).
# After round five the two Python tools became one, run_python (2.9): one schema a request, not two.
_ALWAYS_LOAD = ("find_tools", "mif_call", "run_python")


def _served_description(name: str, desc: str) -> str:
    """The description a client is sent: _lead's, less the mif_help pointer on an always-loaded tool.

    AN ALWAYS-LOADED TOOL IS RE-READ ON EVERY REQUEST OF EVERY SESSION, and rule 1 of the instructions already says
    mif_help('<tool>') has each tool's traps, so " Full notes: mif_help('bl_run_python')." bought nothing there for
    40 characters a request, on each of the two Python tools (round four, docs/TOKEN_OPTIMIZATION.md 2.8). A lead cut
    mid-sentence keeps its pointer: a cut text always says where the rest is."""
    served = _lead(name, desc)
    lead, pointer, _ = served.partition(" Full notes: mif_help(")
    if pointer and name in _ALWAYS_LOAD and not lead.endswith("..."):
        return lead
    return served


class _SlimFastMCP(FastMCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # THE HANDSHAKE MUST ADMIT THE TOOL LIST CAN CHANGE, because since tool packs (2026-09-11) it
        # does: mif_enable_pack registers tools mid-session and sends tools/list_changed. FastMCP never
        # declares that - its stdio runner calls create_initialization_options() with no options, and
        # NotificationOptions() defaults tools_changed to False (mcp 1.27.2) - so every client was told
        # at initialize that this server's tool list never changes, and then told that it had. A client
        # honouring the handshake may ignore the notification; one user's pack tools never
        # appeared in Claude or Codex, however often he reloaded. tools/audit_mcp_capabilities.py reads
        # the handshake and fails while it says false.
        #
        # WRAPPED, NOT COPIED: create_initialization_options still builds everything else exactly as
        # the SDK does; only the one flag changes. Private SDK state, read defensively like
        # call_tool's _tool_manager below - a renamed attribute costs the declaration, not the server.
        low = getattr(self, "_mcp_server", None)
        make = getattr(low, "create_initialization_options", None)
        if NotificationOptions is None or make is None:
            print("[mif] could not declare tools.listChanged - mif_enable_pack's refresh may be ignored",
                  file=sys.stderr)
            return

        def _declare_tools_changed(notification_options=None, experimental_capabilities=None):
            opts = notification_options or NotificationOptions()
            opts.tools_changed = True
            return make(notification_options=opts, experimental_capabilities=experimental_capabilities)

        low.create_initialization_options = _declare_tools_changed

    async def list_tools(self):
        out = []
        for t in await super().list_tools():
            update = {"description": _served_description(t.name, t.description or ""),
                      "inputSchema": _slim_schema(t.inputSchema or {})}
            if t.name in _ALWAYS_LOAD:
                update["meta"] = dict(t.meta or {}, **{"anthropic/alwaysLoad": True})
            out.append(t.model_copy(update=update))
        return out

    # RESULTS, TOO - and these are paid on every CALL, not every turn. FastMCP serialises a dict result
    # with json.dumps(indent=2) and ASCII-escapes anything non-ASCII; measured on 60 real result records,
    # compact JSON is 85% of that size. Same data, re-serialised: the JSON text is parsed and dumped
    # compactly with non-ASCII kept as characters. Anything that is not JSON passes through untouched.
    async def call_tool(self, name, arguments):
        # Pydantic's default model policy ignores extra fields. For an MCP write tool that turns a
        # typo into a successful call with Blender/Unreal defaults: width=0.0005 accidentally sent
        # beside settings used to add a Bevel at Blender's 0.1 default and report ok:true. Validate
        # against the same registered schema advertised to clients before FastMCP can discard keys.
        # PRIVATE FASTMCP STATE, READ DEFENSIVELY. The registered schema is reachable only as _tool_manager._tools,
        # and a FastMCP upgrade that renamed either would otherwise raise AttributeError on EVERY tool call - the
        # whole server down to protect one check. So the check is skipped instead, and said so once on stderr
        # (stdout carries the JSON-RPC stream).
        tools = getattr(getattr(self, "_tool_manager", None), "_tools", None)
        if isinstance(tools, dict):
            tool = tools.get(name)
        else:
            tool = None
            if not getattr(self, "_unknown_arg_check_warned", False):
                self._unknown_arg_check_warned = True
                print("[mif] the unknown-argument check is OFF: this FastMCP no longer exposes _tool_manager._tools",
                      file=sys.stderr)
        schema = (tool.parameters or {}) if tool is not None else {}
        accepted = set((schema.get("properties") or {}).keys())
        # A DEPRECATED SPELLING IS RENAMED FIRST (_PARAM_ALIASES): it is a documented name, so it is not unknown
        arguments, deprecated, both = _deprecated_renames(name, arguments)
        if both:
            raise ValueError(_both_spellings_error(name, both))
        # THE RESPONSE CONTROLS (_select, _omit, _page, _cursor, _wait) are this server's, never the tool's: taken off
        # before the unknown-argument check, and refused here, before anything runs, when they cannot be used.
        controls, arguments = _rc_split(arguments)
        why = _rc_check(controls) if controls else None
        if why:
            raise ValueError("%s: %s. Nothing was called." % (name, why))
        # A PICTURE IS AN IMAGE BLOCK (_with_pictures). Set before the tool runs, because bl_render_still asks the add-on for
        # a different preview when it will become a block; inside mif_call's args the control is still there to read.
        inner = arguments.get("args") if name == "mif_call" else None
        _IMAGE_MODE.set((inner if isinstance(inner, dict) else controls).get("_image", True))
        unknown = sorted(set((arguments or {}).keys()) - accepted) if tool is not None else []
        if unknown:
            raise ValueError(
                "%s received unknown parameter(s): %s. Accepted parameters: %s"
                % (name, ", ".join(unknown), ", ".join(sorted(accepted)) or "(none)"))
        if controls and tool is not None:
            ctx = self.get_context()
            raw = await _rc_run(name, controls, lambda: self._tool_manager.call_tool(
                name, arguments, context=ctx, convert_result=False), ctx)
            result = tool.fn_metadata.convert_result(raw)
        else:
            result = await super().call_tool(name, arguments)
        # THE PROJECT'S MODE, ONCE: a short note on the first reply from a project whose mode is not chosen yet, and the
        # bl_gen_* tools hidden or brought back when a mode changes. Nothing is added once a mode is set
        # (_project_mode_after_call), so a set project pays for no text here.
        note, changed = _project_mode_after_call()
        # THE VERSION HANDSHAKE, ONCE PER EDITOR PROCESS: a note only when this server and the plugin or add-on disagree.
        note = "\n".join(n for n in (_handshake_after_call(), note) if n) or None
        _ue_buckets_after_call()
        if deprecated:
            note = "\n".join(n for n in (_deprecation_note(name, deprecated), note) if n)
        if changed:
            try:
                await self.get_context().session.send_tool_list_changed()
            except Exception:  # noqa: BLE001 - a client that cannot be told still gets the refusals
                pass
        if isinstance(result, tuple) and result and isinstance(result[0], (list, tuple)):
            return (_compact_blocks(_with_pictures(name, arguments, list(result[0])) + _note_blocks(note)),) + tuple(result[1:])
        if isinstance(result, (list, tuple)):
            return _compact_blocks(_with_pictures(name, arguments, list(result)) + _note_blocks(note))
        return result

mcp = _SlimFastMCP(
    "mifblender",
    instructions="MifBlender drives a LIVE Blender through the MifBlender add-on (the bl_* tools).\n1. A description says what a tool does and how to call it; mif_help('<tool>') has its traps - read it when a description is not enough, before a destructive step new to you, or after a refusal you do not understand.\n2. FAILURE IS THE PRESENCE OF `error`, NEVER THE ABSENCE OF `ok`: ok:true can carry warnings, notes, partial results or a measured zero.\n3. TOOLS COME IN PACKS; a tool you cannot see is not missing. find_tools(<words>) finds it and its pack; mif_call('<tool>', {...}) calls any tool with its rules. mif_call('mif_enable_pack', {'pack': p}) puts a whole pack in every later request; mif_help('<pack>') has its rules.\nbl_describe_op('<op>') gives the add-on's real parameters, which outrank this server; bl_status says which Blender answered.",
)

# ---------------------------------------------------------------------------
# Tool help, moved out of the tool descriptions themselves.
#
# WHY THIS EXISTS. Every MCP tool's name, description and parameter schema sit in the model's
# context on EVERY turn, whether the tool is called or not. At 450 tools those descriptions came to
# 289,944 characters - roughly 72,000 tokens - spent before any work began. The detail was worth
# writing and is not worth re-reading 450 times a turn, so the lead sentence stays inline and the
# rest is served from tool_help.json on demand. Nothing was deleted: the sidecar holds the FULL
# original text, and the extraction asserted the surviving lead still matched it.
# ---------------------------------------------------------------------------
_TOOL_HELP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_help.json")

_TOOL_HELP_CACHE = None


def _tool_help():
    global _TOOL_HELP_CACHE
    if _TOOL_HELP_CACHE is None:
        try:
            with open(_TOOL_HELP_PATH, "r", encoding="utf-8") as fh:
                _TOOL_HELP_CACHE = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            _TOOL_HELP_CACHE = {"__error__": str(exc)}
    return _TOOL_HELP_CACHE


def _registered_description(name: str) -> str:
    """A tool's description exactly as it declared it - FastMCP's registry, else the function's docstring."""
    try:
        return (getattr(mcp, "_tool_manager")._tools[name].description or "").strip()
    except Exception:  # noqa: BLE001
        return ((globals().get(name) or (lambda: None)).__doc__ or "").strip()


_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+(?=[A-Z(`'\"])")
# A DESCRIPTION'S POINTER TO ITS OWN HELP IS NOT A SENTENCE THE HELP CAN LACK. 78 descriptions end 'Call
# mif_help("<tool>") first.' - an instruction to read the very entry mif_help serves - and _says_everything demanded
# the entry repeat it, so 50 of them were served as OLDER NOTES (three for that sentence alone), while 27 carried
# the pointer inside their own help to get past the check. Measured 2026-09-28. Only the tool's OWN name is dropped: a
# pointer to another tool's help would be content (none exists today). It is dropped AFTER the split, from the sentence
# that holds it, so a lowercase sentence the split glued onto it ("first. rotation_mode (...) is applied") is still
# asked for, and is not glued to the sentence before it instead. test_mcp_footprint F108 splits the same way.
_OWN_HELP_POINTER = r"""\b(?:Call|Read|See) mif_help\((["'])%s\1\)(?: first| for [^.]*)?\."""


def _help_sentences(text: str, name: str = "") -> list:
    """The sentences of a description, whitespace-normalised. Fragments of 25 characters or fewer are
    skipped - "e.g." and "(see below)." split badly and say nothing an entry could be missing - and so is the
    description's pointer to `name`'s own help."""
    flat = " ".join((text or "").split())
    pointer = _re.compile(_OWN_HELP_POINTER % _re.escape(name)) if name else None
    out = []
    for piece in _SENTENCE_SPLIT.split(flat):
        piece = " ".join((pointer.sub(" ", piece) if pointer else piece).split())
        if len(piece) > 25:
            out.append(piece)
    return out


def _says_everything(notes: str, description: str, name: str = "") -> bool:
    """True when every sentence of `description` is somewhere in `notes` - i.e. serving the notes alone
    loses nothing the tool itself declares. tools/test_mcp_footprint.py F108 asserts the consequence."""
    flat = " ".join((notes or "").split())
    return all(s in flat for s in _help_sentences(description, name))


def _log(*args):
    # NEVER write to stdout: it carries the MCP stdio JSON-RPC stream.
    if DEBUG:
        print("[mif-bridge]", *args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# VERSION HANDSHAKE (docs/00_ARCHITECTURE.md "Version handshake"). Until it, nothing said at connect time that this server
# and the editor's plugin - or Blender's add-on - speak the same surface: a plugin built before a tool failed that tool
# its own way (a 404 here, an unrecognised key there, read as a typo), one call at a time. Now each side is read ONCE:
#   - Unreal: every reply carries X-Mif-Protocol and X-Mif-Instance (the editor's pid). A reply from a pid this session has
#     not checked - the first editor, or one restarted or rebuilt on the same port - makes the next tool call's tail read
#     self_audit {summaryOnly, includeSurface}: protocolVersion, build stamp, and every built-in endpoint's accepted keys.
#     A DLL older than the handshake refuses includeSurface and is read again with includeEndpoints (names only).
#   - Blender: a fresh socket (first call, or a reconnect after a drop) makes the tail read ping, whose protocolVersion,
#     addonVersion, ops, addonPath and sourceHash the add-on already reports. A ping a tool made itself is read in place.
# Both are compared with what THIS file sends - every _post / _blender literal and its keyword names, parsed from this
# file once (_hs_my_surface) - so what an older side lacks is measured, never inferred from a version number. When the two
# disagree, one note says so on that reply (once per process and verdict): which side is older, what it costs, how to
# update. self_audit and bl_status carry the full record as `handshake`. Nothing is added to a reply while they agree.
# The protocol numbers move only for a change the surface diff cannot see (MIF_PROTOCOL_VERSION, MifBridgeHandlers.h;
# PROTOCOL_VERSION, MifBlender/server.py); tools/test_handshake.py holds each pair equal.
MIF_UE_PROTOCOL = 1
MIF_BL_PROTOCOL = 1
_SERVER_PATH = os.path.abspath(__file__)


def _read_server_file():
    """(text, sha256[:16]) of server.py as it is on disk now, or (None, None) when it cannot be read."""
    try:
        with open(_SERVER_PATH, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None, None
    import hashlib
    return raw.decode("utf-8", "replace"), hashlib.sha256(raw).hexdigest()[:16]


# WHAT THIS PROCESS IS RUNNING, taken once at import (R27, 2026-10-19). After a plugin update the editor had ten endpoints the
# connected server's tool list did not know - the checkout was pulled and the editor rebuilt, but the MCP server process
# was the one started before the pull. Every check read server.py and MifBridge.uplugin from DISK, which by then was the
# new release, so the server called itself current. The handshake's surface and self_audit's bridge now read this
# snapshot - the code actually loaded - and compare it with the file on disk now.
_SERVER_SOURCE, _SERVER_SHA = _read_server_file()

_SIBLING_ADDON = os.path.join(os.path.dirname(os.path.dirname(_SERVER_PATH)), "blender-addon", "MifBlender")
_HS = {"unreal": None, "blender": None}          # backend -> the last handshake record (see _hs_record)
_HS_DUE = set()                                  # backends whose handshake the current tool call's tail runs
_HS_UE_SEEN = {"instance": None, "protocol": None, "replies": 0}   # headers of the last Unreal reply, and a count of them
_HS_NOTED = set()                                # (backend, instance, verdict) already said this session
_HS_RETRY_AFTER = {}                             # backend -> monotonic time before which a failed read is not retried
_HS_SURFACE = None
_HS_BUSY = False
_HS_LIST = 8                                     # names a note spells out before "+N more"
_HS_TIMEOUT = 10.0                               # the handshake's own read: self_audit summaryOnly answers in milliseconds
_HS_RETRY_SECONDS = 60.0
_BL_CONNECTS = 0                                 # sockets _bl_connect has opened: a new one is a (re)connect
# Endpoints another plugin registers through MifBridge's external registry (tools/parity_check.py UE_EXTERNAL_PREFIXES):
# absent when that plugin is not installed, whatever MifBridge's age, so they are left out of the comparison - and a
# call to one where the plugin is not loaded is answered by _external_absent, which names the provider, rather than by
# "rebuild MifBridge", which would send the reader to the wrong fix.
_EXTERNAL_PROVIDERS = {"kr_": "MifKismetReconstructor"}
_HS_BL_DISPATCHED = ("batch",)                   # ops the add-on's dispatcher answers outside its op table
_HS_EXTERNAL_PREFIXES = tuple(_EXTERNAL_PROVIDERS)


def _hs_my_surface():
    """{"unreal": {endpoint: {keys}}, "blender": {op: {keys}}} - what this file's tools send, read from its own source.
    A key behind a ** splat is not knowable statically and is left out (the endpoint still counts)."""
    global _HS_SURFACE
    if _HS_SURFACE is None:
        import ast
        out = {"unreal": {}, "blender": {}}
        try:
            # The text this process LOADED (_SERVER_SOURCE), not the file now: a pull after the server started would
            # otherwise make the handshake compare the editor with tools this process does not have.
            tree = ast.parse(_SERVER_SOURCE) if _SERVER_SOURCE is not None else None
        except (SyntaxError, ValueError):
            tree = None
        for node in (ast.walk(tree) if tree is not None else ()):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("_post", "_blender") and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                continue
            if node.func.id == "_post" and node.args[0].value.startswith(_HS_EXTERNAL_PREFIXES):
                continue
            keys = out["unreal" if node.func.id == "_post" else "blender"].setdefault(node.args[0].value, set())
            keys.update(k.arg for k in node.keywords if k.arg and not k.arg.startswith("_"))
        _HS_SURFACE = out
    return _HS_SURFACE


def _hs_compare(mine: dict, theirs, their_keys: bool) -> dict:
    """What each side has that the other lacks. `theirs` is {name: [keys] | None} (keys known) or a list of names."""
    names = set(theirs or ())
    missing = sorted(set(mine) - names)
    extra = sorted(names - set(mine))
    params = []
    if their_keys:
        for name in sorted(set(mine) & names):
            keys = theirs.get(name)
            if not isinstance(keys, list):
                continue                          # no harvested row on that side: its keys are unknown, not none
            low = {str(k).lower() for k in keys}
            params += ["%s.%s" % (name, k) for k in sorted(mine[name]) if k.lower() not in low]
    return {"missingEndpoints": missing, "missingParams": params, "extraEndpoints": extra}


def _hs_verdict(server_protocol: int, their_protocol: int, diff: dict):
    """Which side is older: "plugin", "server", "diverged" (each lacks something of the other's), or None (they agree).
    A protocol difference decides it; at the same protocol the measured surface does."""
    if their_protocol < server_protocol:
        return "plugin"
    if their_protocol > server_protocol:
        return "server"
    lacks = bool(diff["missingEndpoints"] or diff["missingParams"])
    if lacks and diff["extraEndpoints"]:
        return "diverged"
    return "plugin" if lacks else ("server" if diff["extraEndpoints"] else None)


def _hs_names(items) -> str:
    items = list(items)
    shown = ", ".join(items[:_HS_LIST])
    return shown + (" (+%d more)" % (len(items) - _HS_LIST) if len(items) > _HS_LIST else "")


def _hs_record(backend: str, instance, their_protocol: int, built, diff: dict, extra: dict = None) -> dict:
    side = "plugin" if backend == "unreal" else "add-on"
    verdict = _hs_verdict(MIF_UE_PROTOCOL if backend == "unreal" else MIF_BL_PROTOCOL, their_protocol, diff)
    rec = dict({"backend": backend, "instance": instance, "verdict": verdict, "agree": verdict is None,
                "serverProtocol": MIF_UE_PROTOCOL if backend == "unreal" else MIF_BL_PROTOCOL,
                "%sProtocol" % ("plugin" if backend == "unreal" else "addon"): their_protocol,
                "%sBuilt" % ("plugin" if backend == "unreal" else "addon"): built,
                "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S")}, **diff, **(extra or {}))
    if verdict is None:
        return rec
    what = "Unreal editor's MifBridge plugin" if backend == "unreal" else "MifBlender add-on Blender loaded"
    where = BASE if backend == "unreal" else "%s:%s" % (BLENDER_HOST, BLENDER_PORT)
    head = {"plugin": "the %s (%s) is OLDER than this MCP server" % (what, where),
            "server": "this MCP server is OLDER than the %s (%s)" % (what, where),
            "diverged": "this MCP server and the %s (%s) come from different builds" % (what, where)}[verdict]
    proto = "%s protocol %s%s, server protocol %s" % (side, their_protocol,
                                                      " (predates the handshake)" if not their_protocol else "",
                                                      rec["serverProtocol"])
    costs = []
    if diff["missingEndpoints"]:
        costs.append("%d %s this server calls %s missing there, so the tools that use them fail: %s"
                     % (len(diff["missingEndpoints"]), "endpoints" if backend == "unreal" else "ops",
                        "are" if len(diff["missingEndpoints"]) > 1 else "is", _hs_names(diff["missingEndpoints"])))
    if diff["missingParams"]:
        costs.append("%d parameter%s this server sends %s refused there: %s"
                     % (len(diff["missingParams"]), "s" if len(diff["missingParams"]) > 1 else "",
                        "are" if len(diff["missingParams"]) > 1 else "is", _hs_names(diff["missingParams"])))
    if diff["extraEndpoints"]:
        costs.append("%d %s there that this server has no tool for: %s"
                     % (len(diff["extraEndpoints"]), "endpoints" if backend == "unreal" else "ops",
                        _hs_names(diff["extraEndpoints"])))
    if backend == "unreal":
        update = {"plugin": "Rebuild MifBridge from the checkout this server runs from (or install the release that "
                            "matches it) and restart the editor.",
                  "server": "Update this MCP server to the plugin's release (pull the checkout at %s, or install the "
                            "matching release) and restart it - in Claude Code, /mcp then reconnect."
                            % os.path.dirname(os.path.dirname(os.path.dirname(_SERVER_PATH))),
                  "diverged": "Bring both to one release: rebuild the plugin from this server's checkout and restart "
                              "the editor."}[verdict]
    else:
        update = {"plugin": "Update the add-on Blender loads (%s) from %s (or install the matching release's add-on), "
                            "then restart Blender." % (rec.get("addonPath") or "its scripts/addons/MifBlender",
                                                       _SIBLING_ADDON),
                  "server": "Update this MCP server to the add-on's release (pull the checkout at %s) and restart it."
                            % os.path.dirname(os.path.dirname(os.path.dirname(_SERVER_PATH))),
                  "diverged": "Bring both to one release: install the add-on from %s and restart Blender."
                              % _SIBLING_ADDON}[verdict]
    rec["costs"] = costs or ["no endpoint or parameter difference was measured; the protocols differ"]
    rec["update"] = update
    rec["note"] = "MifBridge version check: %s (%s%s). %s. %s Full list: %s." % (
        head, proto, ", built %s" % built if built else "", "; ".join(rec["costs"]), update,
        "self_audit's handshake field" if backend == "unreal" else "bl_status's handshake field")
    return rec


def _hs_unreal():
    """Read the editor's protocol and surface and record the verdict. None when it did not answer.

    ONCE PER EDITOR PROCESS, whatever it answered (decided 2026-10-11). An editor that answers without the fields the
    verdict needs - a build older than self_audit's endpointCount, or one that refuses the read - gets a record that
    says so, so it is not read again on later calls; only an editor that did not answer at all is tried again
    (_HS_RETRY_SECONDS)."""
    replies = _HS_UE_SEEN.get("replies", 0)
    res = _post("self_audit", summaryOnly=True, includeSurface=True, _timeout=_HS_TIMEOUT)
    surface = res.get("surface") if isinstance(res, dict) else None
    if isinstance(res, dict) and res.get("ok") is False and "includeSurface" in str(res.get("error")):
        res = _post("self_audit", summaryOnly=True, includeEndpoints=True, _timeout=_HS_TIMEOUT)   # before the handshake
    if not isinstance(res, dict) or res.get("ok") is False or "endpointCount" not in res:
        if _HS_UE_SEEN.get("replies", 0) == replies:
            return None                                    # no answer: busy or gone, so it is tried again later
        return {"backend": "unreal", "instance": _HS_UE_SEEN.get("instance"), "verdict": None, "agree": None,
                "unavailable": "this editor's self_audit answered without endpointCount (%s), so its surface could "
                               "not be compared; it is not read again for this editor process"
                               % (str(res.get("error") or "no such field")[:160] if isinstance(res, dict)
                                  else "not an object")}
    built = " ".join(str(res[k]) for k in ("buildDate", "buildTime") if res.get(k)) or None
    if isinstance(surface, dict):
        diff = _hs_compare(_hs_my_surface()["unreal"], surface, True)
    else:
        diff = _hs_compare(_hs_my_surface()["unreal"], res.get("endpoints") or [], False)
    try:
        proto = int(res.get("protocolVersion") or 0)
    except (TypeError, ValueError):
        proto = 0
    return _hs_record("unreal", _HS_UE_SEEN.get("instance"), proto, built, diff,
                      {"engineVersion": res.get("engineVersion"), "paramsCompared": isinstance(surface, dict),
                       "pluginUpdate": res.get("update") if isinstance(res.get("update"), dict) else None})


def _hs_sibling_addon():
    """(version list, source hash) of the add-on beside this server, or (None, None) when there is none."""
    init = os.path.join(_SIBLING_ADDON, "__init__.py")
    try:
        with open(init, "r", encoding="utf-8") as fh:
            m = _re.search(r'"version"\s*:\s*\(([\d,\s]+)\)', fh.read())
        version = [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else None
        import hashlib
        h = hashlib.sha256()                                      # the same fold as MifBlender.source_hash
        for f in sorted(f for f in os.listdir(_SIBLING_ADDON) if f.endswith(".py")):
            h.update(f.encode("utf-8"))
            with open(os.path.join(_SIBLING_ADDON, f), "rb") as fh:
                h.update(fh.read())
        return version, h.hexdigest()[:16]
    except OSError:
        return None, None


def _hs_blender_from_ping(data):
    """Record the Blender verdict from a ping reply (the add-on's ping already carries every field the handshake reads)."""
    if not isinstance(data, dict) or not data.get("pong"):
        return None
    ops = data.get("ops") if isinstance(data.get("ops"), list) else []
    # THE DISPATCHER'S OWN OPS ARE NOT IN THE OP TABLE the ping lists: batch is answered by the add-on's server._execute
    # (parity_check.SPECIAL_DISPATCH, held equal by tools/test_handshake.py H111). Until 2026-10-11 every current add-on
    # was called OLDER, missing "batch", on the first Blender reply of every session (found live on 5.0.1).
    ops = list(ops) + [o for o in _HS_BL_DISPATCHED if o not in ops]
    diff = _hs_compare(_hs_my_surface()["blender"], ops, False)
    try:
        proto = int(data.get("protocolVersion") or 0)
    except (TypeError, ValueError):
        proto = 0
    sib_version, sib_hash = _hs_sibling_addon()
    version = data.get("addonVersion")
    rec = _hs_record("blender", data.get("pid"), proto, ".".join(str(v) for v in version) if version else None, diff,
                     {"addonPath": data.get("addonPath"), "sourceHash": data.get("sourceHash"),
                      "siblingAddon": _SIBLING_ADDON if sib_version else None,
                      "siblingAddonVersion": sib_version, "siblingSourceHash": sib_hash,
                      "sameSourceAsSibling": (data.get("sourceHash") == sib_hash) if sib_hash and data.get("sourceHash")
                      else None, "connection": _BL_CONNECTS})
    _HS["blender"] = rec
    _HS_DUE.discard("blender")
    return rec


def _hs_seen_blender(op, data):
    """Every Blender reply passes here (from _blender). A ping is read in place; any other reply on a socket the last
    handshake was not taken on - the first call, or a reconnect after a drop - makes the handshake due."""
    if op == "ping" and isinstance(data, dict) and data.get("pong") and isinstance(data.get("ops"), list):
        _hs_blender_from_ping(data)
    elif op == "ping" and isinstance(data, dict) and data.get("pong"):
        # ONCE PER SOCKET, whatever it answered (decided 2026-10-11). An add-on older than the ops list in its ping
        # answers without it, and the handshake used to ping again after EVERY call for the rest of the session.
        _HS["blender"] = {"backend": "blender", "instance": data.get("pid"), "verdict": None, "agree": None,
                          "connection": _BL_CONNECTS,
                          "unavailable": "this add-on's ping lists no ops, so its surface could not be compared; it "
                                         "is not read again on this connection"}
        _HS_DUE.discard("blender")
    elif (_HS.get("blender") or {}).get("connection") != _BL_CONNECTS:
        _HS_DUE.add("blender")


def _handshake_after_call():
    """The tail of every tool call: run a due handshake, then return the note of any verdict not yet said for that
    process (a ping a tool made itself is read in place, so its verdict is said here too), else None. Near free when
    nothing is due and nothing is unsaid: a few lookups."""
    global _HS_BUSY
    if _HS_BUSY:
        return None
    _HS_BUSY = True
    notes = []
    try:
        now = time.monotonic()
        for backend in sorted(_HS_DUE):
            if _HS_RETRY_AFTER.get(backend, 0) > now:
                continue
            if backend == "unreal":
                rec = _hs_unreal()
                if rec is not None:
                    _HS["unreal"] = rec
                    _HS_DUE.discard("unreal")
            else:
                res = _blender("ping", _timeout=BLENDER_PROBE_TIMEOUT, _lock_timeout=BLENDER_PROBE_TIMEOUT)
                rec = _HS.get("blender") if isinstance(res, dict) and res.get("pong") else None   # _hs_seen_blender read it
            if rec is None:
                _HS_RETRY_AFTER[backend] = now + _HS_RETRY_SECONDS   # busy or gone: not again on every call
        for backend in ("unreal", "blender"):
            rec = _HS.get(backend) or {}
            mark = (backend, rec.get("instance"), rec.get("verdict"), len(rec.get("missingEndpoints") or ()),
                    len(rec.get("missingParams") or ()))
            if rec.get("verdict") and mark not in _HS_NOTED:
                _HS_NOTED.add(mark)
                notes.append(rec["note"])
            # A NEWER RELEASE THE PLUGIN'S UPDATE CHECK HAS SEEN (self_audit `update`, 2026-10-11): one line, once per
            # editor process, read from the same self_audit the handshake already makes - no request of its own.
            upd = rec.get("pluginUpdate")
            if isinstance(upd, dict) and upd.get("newerVersion") and ("update", rec.get("instance")) not in _HS_NOTED:
                _HS_NOTED.add(("update", rec.get("instance")))
                notes.append("MifBridge %s is available (this editor has %s)%s." % (
                    upd.get("newerVersion"), upd.get("pluginVersion") or "an older version",
                    " - what is new: %s" % upd["notes"] if upd.get("notes") else ""))
    finally:
        _HS_BUSY = False
    return "\n".join(notes) or None


def _hs_report(backend: str) -> dict:
    """What self_audit / bl_status add: the last handshake record, or why there is none yet."""
    rec = _HS.get(backend)
    if rec:
        return rec
    return {"backend": backend, "verdict": None, "agree": None,
            "pending": "not read yet - the handshake runs at the end of the first call that reaches it"}

# Each endpoint's (bucket, writeGated, access), read once a session from self_audit's rows (_ue_buckets_after_call), keyed
# by base url. A timeout says a call's outcome is unknown unless the endpoint is known not to write: its access is read or
# observe (the plugin's safety table, docs/34_SAFETY_CLASSIFICATION.md). A plugin older than the access column is judged as
# before - readOnly bucket and off the write gate's list - though that bucket is about the undo stack and holds compile.
_UE_BUCKETS = {}
_UE_BUCKETS_FOR = {}          # base url -> the editor instance (X-Mif-Instance) its buckets were read from
_UE_BUCKETS_WANTED = set()


def _ue_buckets_after_call():
    """Read every endpoint's bucket and write-gate flag once, after the first call an editor answered (the project-mode
    read's pattern): one self_audit, so a later timeout can say whether the call that timed out writes."""
    base = BASE
    if base not in _UE_BUCKETS_WANTED:
        return
    _UE_BUCKETS_WANTED.discard(base)
    replies = _HS_UE_SEEN.get("replies", 0)
    res = _post("self_audit", summaryOnly=True, includeEndpointDetails=True)
    if _HS_UE_SEEN.get("replies", 0) == replies:
        return                           # no answer: the next call an editor answers asks again
    # ONCE PER EDITOR PROCESS (decided 2026-10-11). A plugin older than endpointDetails answers without them, and this
    # read used to be repeated after EVERY call for as long as the session lasted. An answer without rows is recorded as
    # none - every timeout then says its outcome is unknown, the safe reading - until an editor with another pid answers.
    rows = res.get("endpointDetails") if isinstance(res, dict) else None
    _UE_BUCKETS[base] = {r["name"]: (r.get("bucket") or "unknown", bool(r.get("writeGated")),
                                     r.get("access") if isinstance(r.get("access"), str) else None)
                         for r in (rows if isinstance(rows, list) else [])
                         if isinstance(r, dict) and isinstance(r.get("name"), str)}
    _UE_BUCKETS_FOR[base] = _HS_UE_SEEN.get("instance")
    _UE_BUCKETS_WANTED.discard(base)     # the read's own reply asked for a read, before the line above recorded it


def _post(endpoint: str, /, _timeout: float = None, _rid: str = None, **payload) -> dict:
    """No Unreal Editor in the Blender-only server: every Unreal request is answered "not part of MifBlender"."""
    return {"ok": False, "error": "%s is an Unreal Engine call; this Blender-only MifBlender server has no Unreal "
                                  "connection (MifBridge adds it: https://mifshowcase.store/mifbridge)" % endpoint}

_BL_HDR_FMT = ">I"
_BL_HDR_SIZE = 4
_BL_MAX_FRAME = 64 * 1024 * 1024

_BL_SOCK = None
_BL_LOCK = threading.Lock()

# What the lock is currently held FOR, as (op name, time.monotonic() at acquire).
# Written under the lock, read WITHOUT it by a caller that failed to acquire -
# which is the whole point: "another op is in flight and has been for 96s" is
# itself the diagnosis a probe is asking for, and it is unobtainable if the probe
# has to take the lock to learn it. A torn read here is a slightly wrong number
# in an error message, never a wrong answer.
_BL_INFLIGHT = None


# WHERE EACH MIFBLENDER LISTENS (2026-10-12). A MifBlender on its own port preference (:8802) was unreachable to this
# server, which read MIF_BLENDER_PORT once at start: every bl_* call failed against a Blender that was running and ready.
# The add-on now writes each listener into a per-user file (MifBlender/registry.py; the path rule below is the SAME rule
# as registry.listeners_path - test_bl_listeners checks they agree), bl_status lists the live ones and switches to one by
# port or pid, and an unreachable default port names the live add-on when there is exactly one. Never picked silently.
def _bl_listeners_path():
    override = (os.environ.get("MIF_BLENDER_LISTENERS") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "MifBridge", "Blender", "listeners.json")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Logs", "MifBridge", "Blender", "listeners.json")
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(home, ".local", "state")
    return os.path.join(base, "mifbridge", "blender", "listeners.json")


def _bl_pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _bl_registered(alive_only=True):
    """The add-on's listener entries (registry.py), each with alive: its pid is running. Unreadable file: []."""
    try:
        with open(_bl_listeners_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    rows = []
    for r in (data.get("listeners") if isinstance(data, dict) else None) or []:
        if not isinstance(r, dict) or not isinstance(r.get("port"), int):
            continue
        row = dict(r, alive=_bl_pid_alive(r.get("pid")))
        if row["alive"] or not alive_only:
            rows.append(row)
    return sorted(rows, key=lambda r: r["port"])


def _bl_probe(port: int, timeout: float = 2.0) -> dict:
    """One ping on its own socket to 127.0.0.1:port - the handshake that proves a listener is a MifBlender that takes our
    token. Never touches the session's socket. {"ok": False, "error"} when it does not answer."""
    frame = json.dumps({"endpoint": "ping", "token": BLENDER_TOKEN, "params": {}}).encode("utf-8")
    try:
        with socket.create_connection((BLENDER_HOST, int(port)), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(struct.pack(_BL_HDR_FMT, len(frame)) + frame)
            (length,) = struct.unpack(_BL_HDR_FMT, _bl_recv_exact(s, _BL_HDR_SIZE))
            if length > _BL_MAX_FRAME:
                return {"ok": False, "error": "frame too large"}
            return json.loads(_bl_recv_exact(s, length).decode("utf-8"))
    except (OSError, ValueError, ConnectionError) as exc:
        return {"ok": False, "error": str(exc)}


def _bl_listener_row(r, probe=None):
    row = {k: r.get(k) for k in ("port", "pid", "blenderVersion", "addonVersion", "blendFile", "background", "since")}
    if probe is not None:
        row["answers"] = "pong" in probe
        if "pong" in probe and probe.get("pid") not in (None, r.get("pid")):
            row["answers"] = False
            row["note"] = "another process answers on this port (pid %s)" % probe.get("pid")
        elif "pong" not in probe:
            row["note"] = str(probe.get("error"))[:160]
    row["current"] = r.get("port") == BLENDER_PORT
    return row


def _bl_unreachable(exc) -> dict:
    out = {"ok": False, "error": (
        f"Blender backend unreachable at {BLENDER_HOST}:{BLENDER_PORT} ({exc}). "
        "Start Blender, enable the MifBlender addon (Edit > Preferences > Add-ons), and check "
        "View3D > N-panel > MifBridge that the server is listening. Nothing in the UE plugin or "
        "this server can start it for you. Override the address with MIF_BLENDER_HOST / "
        "MIF_BLENDER_PORT if the addon is on a non-default port.")}
    # A LIVE MIFBLENDER ELSEWHERE IS NAMED, never switched to by itself: pid checked here, the handshake by bl_status.
    others = [r for r in _bl_registered() if r.get("port") != BLENDER_PORT]
    if len(others) == 1:
        o = others[0]
        out["listeners"] = [_bl_listener_row(o)]
        out["error"] = (f"Blender backend unreachable at {BLENDER_HOST}:{BLENDER_PORT} ({exc}) - but a MifBlender is listening on "
                        f":{o['port']} (pid {o.get('pid')}, Blender {o.get('blenderVersion')}, "
                        f"{'file ' + o['blendFile'] if o.get('blendFile') else 'unsaved file'}). "
                        f"bl_status(port={o['port']}) checks it and switches this session to it.")
    elif len(others) > 1:
        out["listeners"] = [_bl_listener_row(o) for o in others]
        out["error"] += (" %d MifBlenders are listening on other ports (listeners) - pick one with bl_status(port=...); "
                         "none is chosen for you." % len(others))
    return out


def _bl_close():
    """Drop the cached socket so the NEXT call reconnects. Never raises."""
    global _BL_SOCK
    if _BL_SOCK is not None:
        try:
            _BL_SOCK.close()
        except OSError:
            pass
        _BL_SOCK = None


def _bl_connect():
    """Return the cached socket, or open one. Raises OSError on failure."""
    global _BL_SOCK, _BL_CONNECTS
    if _BL_SOCK is not None:
        return _BL_SOCK
    _BL_CONNECTS += 1                    # a new socket: the version handshake reads this Blender again (_hs_seen_blender)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(BLENDER_CONNECT_TIMEOUT)
    sock.connect((BLENDER_HOST, BLENDER_PORT))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _BL_SOCK = sock
    return sock


def _bl_recv_exact(sock, count: int) -> bytes:
    buf = b""
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            raise ConnectionError("Blender closed the connection mid-frame")
        buf += chunk
    return buf


def _bl_busy_error(op: str, waited: float) -> dict:
    """The answer when the transport lock could not be taken in time.

    Not a timeout dressed up as one: this call never reached Blender, and saying
    WHICH op is holding the line and for HOW LONG is the diagnosis a health probe
    was asking for in the first place.
    """
    held = _BL_INFLIGHT
    if held:
        other, since = held
        detail = (f"'{other}' has held the Blender transport for "
                  f"{time.monotonic() - since:.1f}s")
    else:
        detail = ("another Blender call holds the transport (it finished between the "
                  "failed acquire and this message, so it was only just blocking)")
    return {"ok": False, "error": (
        f"did not send '{op}': {detail}, and this call waited {waited:.1f}s for it. "
        "The connection to Blender is a single serialised socket, so one long op "
        "(an FBX import/export, a bevel on a dense mesh) blocks the rest. Nothing "
        "was sent and nothing was changed. Either wait for it - a geometry op runs "
        f"as ONE main-thread frame and cannot be polled - or, if it has been going "
        "far longer than the work should take, Blender's main thread is wedged on a "
        "modal operator or an open file browser and needs clearing by hand.")}


def _blender(op: str, /, _timeout: float = None, _lock_timeout: float = None, _job_timeout: float = None,
             **params) -> dict:
    """Send one framed op to the MifBlender addon and return its JSON verbatim.

    Mirrors _post: unset (None) params are dropped, and every failure is a
    {"ok": False, "error": ...} dict rather than an exception.

    `op` IS POSITIONAL-ONLY because an op can take a parameter called `op`. describe_op does, and
    until 2026-09-22 bl_describe_op's `_blender("describe_op", op=op)` raised "got multiple values
    for argument 'op'" on EVERY call through MCP - since the tool was added on 2026-09-08, while the
    server instructions and every Blender pack prompt named it as the authority on an op's
    parameters. Its tests spoke to the add-on directly, so none of them went through this function.
    The benchmark caught it: 5 agent calls, 5 failures (docs/TOKEN_OPTIMIZATION.md).

    _lock_timeout bounds the wait for the transport lock (NOT the read). Omit it
    and the call queues behind whatever is in flight, which is right for real
    work. Pass it for a PROBE: bl_status exists to answer "is Blender wedged" in
    seconds, and it could not do that while blocking on the same lock for the
    full 180s work timeout - the one tool you reach for when Blender is stuck was
    the one tool made unavailable by Blender being stuck.
    """
    global _BL_INFLIGHT
    refused = _generative_refusal(op, params)
    if refused:
        return refused        # before the socket: a refused generative call never reaches Blender, let alone ComfyUI
    body = {k: v for k, v in params.items() if v is not None}
    frame = {"endpoint": op, "token": BLENDER_TOKEN, "params": body}
    agent = _agent_tag()
    if agent:
        # The same caller tag the UE side gets as X-Mif-Agent. A top-level key beside endpoint and token,
        # so an add-on that does not read it ignores it: every MifBlender reads only endpoint, params,
        # token and readBack from the frame.
        frame["agent"] = agent
    if _job_timeout is not None:
        # an interactive Blender's main-thread wait for THIS call (the add-on's jobTimeout, default 150 s): a call known
        # to be long sends one under its own read timeout, so the add-on does not answer "timed out" while it works
        frame["jobTimeout"] = float(_job_timeout)
    read_timeout = BLENDER_TIMEOUT if _timeout is None else _timeout
    _log("bl->", op, body)
    # THE CALLER'S DEADLINE (1.4, FOUND 2026-11-07): the wall-clock second this call stops waiting for its reply. A
    # background Blender serves one connection at a time, so a request can sit in its accept backlog behind another
    # client's socket past this read's timeout - and it used to run then, on a session its caller had changed since
    # being told "timed out". An add-on that reads it refuses such a request and runs nothing; an older one ignores it.
    # Stamped AFTER the transport lock below, so a wait for the lock does not count against the op.

    try:
        payload = json.dumps(frame).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"params for Blender op '{op}' are not JSON-serialisable: {exc}"}
    if len(payload) > _BL_MAX_FRAME:
        return {"ok": False, "error": (
            f"request for Blender op '{op}' is {len(payload)} bytes, over the {_BL_MAX_FRAME}-byte "
            "frame cap. Pass a file path rather than inline geometry.")}
    header = struct.pack(_BL_HDR_FMT, len(payload))

    # Bounded acquire ONLY when the caller asked for one. threading.Lock.acquire
    # rejects a timeout together with blocking=False, so the two forms are kept
    # apart rather than collapsed.
    waited = time.monotonic()
    if _lock_timeout is None:
        _BL_LOCK.acquire()
    elif not _BL_LOCK.acquire(timeout=_lock_timeout):
        return _bl_busy_error(op, time.monotonic() - waited)
    _BL_INFLIGHT = (op, time.monotonic())
    try:
        frame["deadline"] = round(time.time() + read_timeout, 3)
        payload = json.dumps(frame).encode("utf-8")
        header = struct.pack(_BL_HDR_FMT, len(payload))
        # The retry covers the SEND phase only, and only on a socket we reused.
        # A send that fails on a keep-alive socket means the frame never landed
        # whole, so the op cannot have run: reconnecting and resending is safe.
        # A failure after the send is NEVER retried - the op may already have
        # mutated the scene, and re-running a bevel would bevel twice.
        sock = None
        for attempt in (0, 1):
            reused = _BL_SOCK is not None
            try:
                sock = _bl_connect()
            except OSError as exc:
                _bl_close()
                return _bl_unreachable(exc)
            try:
                sock.settimeout(BLENDER_CONNECT_TIMEOUT)
                sock.sendall(header + payload)
            except OSError as exc:
                _bl_close()
                if reused and attempt == 0:
                    _log("bl--", op, "stale keep-alive socket, reconnecting")
                    continue
                return _bl_unreachable(exc)
            break

        try:
            sock.settimeout(read_timeout)
            (length,) = struct.unpack(_BL_HDR_FMT, _bl_recv_exact(sock, _BL_HDR_SIZE))
            if length > _BL_MAX_FRAME:
                raise ValueError(f"frame header claims {length} bytes, over the {_BL_MAX_FRAME} cap")
            data = json.loads(_bl_recv_exact(sock, length).decode("utf-8"))
        except socket.timeout:
            _bl_close()
            return {"ok": False, "error": (
                f"Blender read timeout after {read_timeout}s on '{op}'. NOT retried: the operation "
                "may have run to completion. Blender's main thread is blocked - a modal operator, "
                "an open file browser or a render will do it. Dismiss it and call bl_status. Raise "
                "MIF_BLENDER_TIMEOUT for a genuinely long geometry op. If it had not started by now, it never "
                "will: the add-on refuses a request that reaches it after this deadline.")}
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _bl_close()
            return {"ok": False, "error": (
                f"Blender transport error on '{op}': {exc}. The connection was dropped; the next "
                "call reconnects. Check Blender's system console for an addon traceback.")}
    finally:
        _BL_INFLIGHT = None
        _BL_LOCK.release()

    if not isinstance(data, dict):
        return {"ok": False, "error": f"Blender returned a non-object response for '{op}': {data!r:.300}"}
    _log("bl<-", op, data)
    _bl_mode_seen(op, data)            # the open .blend's mode: read once per file (_project_mode_after_call)
    _hs_seen_blender(op, data)         # the version handshake: a ping is read in place, a fresh socket makes it due
    _gen_license_after(op, body, data)  # the license block: which models this call ran, and their terms
    _note_failed_call("blender", op, body, data)     # mif_report_draft drafts from these
    return data

# AN OP INSIDE A BATCH TAKES THE NAMES ITS OWN TOOL ADVERTISES. Each tool renames its parameters on the way out -
# add_tree_widget's blueprint_id reaches Unreal as blueprintId, bl_apply_transform's object_name reaches Blender as
# object - but batch and bl_batch forwarded their ops verbatim, so an agent writing an op the way the tool list had just
# taught it was refused: 13 failed batch calls in the benchmark's UMG and material runs, and bl_batch's apply_transform in the
# bracket runs, each costing a whole request (docs/TOKEN_OPTIMIZATION.md). The map is READ OFF THE TOOL'S OWN CALL to
# _post / _blender, never written by hand, so it cannot drift from what the tool itself sends. Only a name its tool
# renames is changed: a backend name (the camelCase Unreal reads) passes untouched, and anything the backend does not
# accept is still refused there, by name.
_OP_PARAM_NAMES = {}
# THE SOURCE THE MAP IS READ FROM IS THE ONE THIS PROCESS RUNS, kept from import. inspect.getsource reads the file on
# disk at the line numbers the running code was compiled with, so after an install or update replaced server.py under a
# running server it parsed whatever sat at those lines now: every map came out empty and batch sent each op verbatim,
# refused op by op on 'unrecognised parameters blueprint_id, widget_name' (found in use 2026-11-06, a server started
# hours before the install it outlived).
try:
    with open(__file__, encoding="utf-8") as _fh:
        _OWN_SOURCE_LINES = _fh.read().splitlines(True)
except (OSError, NameError, UnicodeDecodeError):
    _OWN_SOURCE_LINES = None


def _source_of(fn) -> str:
    """fn's source as this process loaded it: from the copy kept at import for a function defined in this file, else
    inspect.getsource. Raises what inspect.getsource raises when there is none."""
    import inspect
    code = getattr(inspect.unwrap(fn), "__code__", None) if fn is not None else None
    if _OWN_SOURCE_LINES is not None and code is not None and code.co_filename == __file__:
        return "".join(inspect.getblock(_OWN_SOURCE_LINES[code.co_firstlineno - 1:]))
    return inspect.getsource(fn)


def _op_param_names(backend: str, op: str) -> dict:
    """{tool parameter: backend key} for the tool that wraps `op` - the tool named `op` for Unreal, `bl_<op>` for
    Blender - read off that tool's single call to its transport. Empty when there is no such tool, or it posts `op`
    more than once (no one map would be right), or its source cannot be read."""
    key = (backend, op)
    if key in _OP_PARAM_NAMES:
        return _OP_PARAM_NAMES[key]
    import ast
    import textwrap
    names = {}
    tool = ("bl_" + op) if backend == "blender" else op
    fn = globals().get(tool) or _UNLOADED.get(tool)
    try:
        node = ast.parse(textwrap.dedent(_source_of(fn))).body[0]
        params = {a.arg for a in node.args.args + node.args.kwonlyargs}
    except (TypeError, OSError, SyntaxError, IndexError, AttributeError):
        node, params = None, set()
    transport = "_blender" if backend == "blender" else "_post"
    calls = [] if node is None else [
        n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == transport and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == op]
    if len(calls) == 1:
        pairs = []
        for kw in calls[0].keywords:
            if kw.arg is not None:
                pairs.append((kw.arg, kw.value))
            elif isinstance(kw.value, ast.Dict):          # **{"class": cls or None}
                pairs += [(k.value, v) for k, v in zip(kw.value.keys, kw.value.values)
                          if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        for backend_key, value in pairs:
            used = {n.id for n in ast.walk(value) if isinstance(n, ast.Name) and n.id in params}
            if len(used) == 1:                            # a value built from two parameters maps neither
                names[used.pop()] = backend_key
    _OP_PARAM_NAMES[key] = names
    return names


_BL_TOOL_OPS = {}


def _bl_tool_op(tool: str):
    """The one add-on op the bl_ tool `tool` posts, read off its `_blender` calls - None when there is no such tool,
    it posts no op or several, or its source cannot be read. bl_batch uses it for an entry that names the tool;
    cached like _op_param_names, since a walk cycle's batch can name the same tool ninety times."""
    if tool in _BL_TOOL_OPS:
        return _BL_TOOL_OPS[tool]
    import ast
    import textwrap
    fn = globals().get(tool) or _UNLOADED.get(tool)
    try:
        node = ast.parse(textwrap.dedent(_source_of(fn))).body[0]
    except (TypeError, OSError, SyntaxError, IndexError):
        node = None
    ops = set() if node is None else {
        n.args[0].value for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_blender" and n.args and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)}
    _BL_TOOL_OPS[tool] = ops.pop() if len(ops) == 1 else None
    return _BL_TOOL_OPS[tool]


# A BATCH REPLY THE AGENT CANNOT SEE IS WORSE THAN A SHORT ONE. Each op's full read-back rides in results[], and a
# read-back can list the whole action (bl_set_keyframe reports every F-curve on it), so the benchmark's walk cycles sent
# bl_batch replies of 67,686 to 253,721 characters - 11 of them in 5 of 6 runs. Claude Code refuses a tool result over
# 25,000 tokens and shows the agent an error instead, so it saw NOTHING of those batches, failures included, and spent
# requests reading the state back (docs/TOKEN_OPTIMIZATION.md). Over the line, each op that SUCCEEDED keeps its names,
# counts and flags (scalars, one level of nesting deep) and each op that failed keeps everything; under it, nothing
# changes. ~10k tokens - Claude Code's own warning line - at the 2.265 characters a token measured for this JSON.
_BATCH_SHOW_CHARS = 24000

# A NOTE REPEATED ON EVERY OP IS SAID ONCE. Some ops attach an explanatory note to every result - set_keyframe's
# valueNote, set_bone_pose's constrainedNote, transform_object's bakedNote - so a batch of ninety keyframes carried the
# same sentence ninety times: 37,207 characters over the 105 the benchmark runs on 11baf1c7 (docs/TOKEN_OPTIMIZATION.md
# 2.5). A note - a string field named *Note, 40 characters or more - that repeats word for word across the succeeded
# ops of one op type moves to sharedNotes once, with the rows it came from. A failed op keeps everything, a note on one
# op stays where it is, and only *Note fields move, so every value an op returned is still on its own row.
_NOTE_KEY = _re.compile(r"(?:^n|N)ote$")


def _hoist_batch_notes(res):
    """A batch reply with each note that repeats across succeeded ops of one op type said once, in sharedNotes;
    returned as it was when nothing repeats."""
    rows = res.get("results") if isinstance(res, dict) else None
    if not isinstance(rows, list):
        return res

    def fields(row):                    # bl_batch nests an op's fields under result; batch puts them on the row
        return row["result"] if isinstance(row.get("result"), dict) else row

    seen = {}
    for pos, row in enumerate(rows):
        if isinstance(row, dict) and row.get("ok") is True:
            for k, v in fields(row).items():
                if isinstance(v, str) and len(v) >= 40 and _NOTE_KEY.search(k):
                    seen.setdefault((row.get("op"), k, v), []).append(pos)
    shared = [(key, positions) for key, positions in seen.items() if len(positions) > 1]
    if not shared:
        return res
    rows = list(rows)
    for (_op, k, _v), positions in shared:
        for p in positions:
            row = dict(rows[p])
            if isinstance(row.get("result"), dict):
                row["result"] = {kk: vv for kk, vv in row["result"].items() if kk != k}
            else:
                row.pop(k, None)
            rows[p] = row
    out = dict(res, results=rows)
    out["sharedNotes"] = [{"op": op, "key": k, "rows": [rows[p].get("index", p) for p in positions], "text": v}
                          for (op, k, v), positions in shared]
    return out


# ONLY THE FAILURES, WHEN ASKED. A bulk write the agent verifies with a read anyway - ninety keyframes checked with
# bl_evaluate_at_frame - does not need every op's read-back, yet 11baf1c7's walk cycles carried them all: batch replies
# were 40% of what the tool-search runs re-read (docs/TOKEN_OPTIMIZATION.md 2.5). results='errors' keeps each failed op
# whole and each succeeded op as {index, op, ok}, and says how many read-backs it left out. The default keeps them all.
_BATCH_RESULTS = (None, "", "all", "errors")


def _batch_results_refusal(results):
    """The refusal for a `results` value batch/bl_batch does not know, or None - given before anything runs."""
    if results in _BATCH_RESULTS:
        return None
    return {"ok": False, "endpoint": "batch",
            "error": "results must be 'all' (the default) or 'errors', got %r. NOTHING was run." % (results,)}


def _errors_only(res):
    """A batch reply with each succeeded op reduced to {index, op, ok}; failed ops untouched; the omission counted."""
    rows = res.get("results") if isinstance(res, dict) else None
    if not isinstance(rows, list):
        return res
    kept, omitted = [], 0
    for row in rows:
        if isinstance(row, dict) and row.get("ok") is True:
            kept.append({k: row[k] for k in ("index", "op", "ok") if k in row})
            omitted += 1
        else:
            kept.append(row)
    out = dict(res, results=kept)
    out["resultsOmitted"] = omitted
    out["resultsOmittedNote"] = ("results='errors': each op that succeeded is listed without its read-back and each "
                                 "op that failed is whole. Leave results at its default when you need what an op "
                                 "returned.")
    return out


def _fit_batch_reply(res):
    """A batch reply (bl_batch's or batch's) that fits a client: repeated notes said once (_hoist_batch_notes), then
    untouched when it is under _BATCH_SHOW_CHARS, else with each succeeded op cut to its scalar fields and the cut
    said, in resultsCut and resultsCutNote."""
    res = _hoist_batch_notes(res)
    rows = res.get("results") if isinstance(res, dict) else None
    if not isinstance(rows, list):
        return res
    size = len(json.dumps(res, separators=(",", ":"), ensure_ascii=False, default=str))
    if size <= _BATCH_SHOW_CHARS:
        return res

    # A NESTED VALUE IS KEPT WHOLE OR LEFT OUT, NEVER IN PART (R60, 2026-10-25). A set_property row's `typed` read-back of a
    # brush kept its short scalars and lost its texture path (over 64 characters) and its sized members, so the cut reply read
    # as a brush with no resource - DrawAs Box, bHasUObject false, ResourceName None - and a session concluded the texture had
    # been dropped when it had been written. A part of a value is not a smaller value; it is a different one.
    def keep(v):
        return isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 64)

    def scalars(d, nested):
        out = {}
        for k, v in d.items():
            if keep(v):
                out[k] = v
            elif isinstance(v, dict) and nested and v and all(x is None or keep(x) for x in v.values()):
                out[k] = v
        return out

    kept, cut = [], 0
    for row in rows:
        if isinstance(row, dict) and row.get("ok") is True:
            slim = {k: row[k] for k in ("index", "op", "ok", "elapsedMs") if k in row}
            if isinstance(row.get("result"), dict):                  # bl_batch: {index, op, ok, result, elapsedMs}
                slim["result"] = scalars(row["result"], True)
            else:                                                    # batch: the op's own fields beside index/op/ok
                slim.update(scalars({k: v for k, v in row.items() if k not in slim}, True))
            kept.append(slim)
            cut += 1
        else:
            kept.append(row)
    out = dict(res, results=kept)
    out["resultsCut"] = cut
    out["resultsCutNote"] = ("the full reply was %d characters, more than a client shows (Claude Code refuses a tool "
                             "result over 25,000 tokens), so each op that succeeded keeps only its names, counts and "
                             "flags - a read-back such as typed only when it fits whole, never in part - and each op "
                             "that failed keeps everything. Read state back with a read tool."
                             % size)
    return out


def _backend_op_params(backend: str, op, params: dict, renamed: dict = None) -> dict:
    """`params` with each name its tool renames changed to the key the backend reads. A key whose backend name is
    already present is left as it is, so the backend refuses the pair by name rather than one silently winning. A key
    that is neither the tool's name nor the backend's goes through _arg_alias against both (sourceNode -> srcNode);
    each such rename is added to `renamed` when one is passed."""
    names = _op_param_names(backend, op) if isinstance(op, str) else {}
    backend_keys = set(names.values())
    if backend == "unreal" and isinstance(op, str) and op in _PARAM_ALIASES:
        params, gone, _ = _deprecated_renames(op, params)       # a deprecated spelling is a documented name, not a guess
        if renamed is not None:
            renamed.update(gone)
    out = {}
    for k, v in params.items():
        dst = names.get(k)
        if dst is None and k not in backend_keys and names:
            guess = _arg_alias(k, set(names) | backend_keys)
            if guess is not None:
                dst = names.get(guess, guess)
                if renamed is not None and dst not in params:
                    renamed[k] = dst
        out[dst if dst and dst != k and dst not in params else k] = v
    return out


# ONE SPELLING PER PARAMETER (spec, 2026-10-08). The same concept had four names across the asset,
# property and actor tools - an asset was `path` in 183 tools, `asset_path` in 27, `asset` in 20 and `object_path` in 11,
# a Blueprint `blueprint_id` in 70 and `blueprint` in 3 - so an agent that had just used one tool guessed the next one's
# name wrong. The most-used spelling wins: `path` for the asset or object a tool acts on, `blueprint_id` for a Blueprint.
# Each renamed tool declares its OLD names here, once, with @_renamed_params under its @mcp.tool(), and they keep
# working as deprecated aliases everywhere a call can come in - a direct MCP call (_SlimFastMCP.call_tool), mif_call and
# a batch op (_backend_op_params) - mapped before the unknown-parameter check, in docs-first mode too (a documented name
# is not a guess). No alias is a schema parameter: the tool list a client re-reads every turn carries one name, and
# mif_help(tool) lists the old ones under deprecatedParams. A call that gives an old name AND its new one is refused by
# name rather than one silently winning. The wire keys the plugin reads (assetPath, objectPath, blueprint) are unchanged.
_PARAM_ALIASES = {}          # {tool: {deprecated name: canonical name}}

# ONE CONCEPT, TWO SPELLINGS, ACROSS TOOLS (spec 1.3 second audit, 2026-10-25). The plugin reads actor and actorPath as one
# key wherever an endpoint declares one (MifKeyAliasesOf), but this server refused the other spelling before the plugin saw
# it: 40 Unreal tools take actor and 56 take actor_path, so a call shaped like the last tool's was refused here. Each pair
# below is accepted on a tool whose schema has the second name and not the first - both ways for the actor keys, one way for
# the rest (graph is a Blueprint graph's id where graph_id is declared, but a PCG graph's ASSET where graph is, so graph_id is
# never read as graph). Not a schema parameter and not deprecated: the tool keeps one documented name, and the reply names
# the spelling it was read as. bl_ tools keep their own conventions and are not touched.
_SPELLING_PAIRS = (("actor", "actor_path"), ("actor_path", "actor"), ("actors", "actor_paths"), ("actor_paths", "actors"),
                   ("property", "property_path"), ("graph", "graph_id"), ("limit", "top"), ("confirm", "apply"))
# The Sequencer and clip tools named one thing several ways (spec, the second audit's B5-W3): the sequence is path in 40+
# tools but sequence in these three, and a clip's output is output_path in most and new_path in mirror_anim_sequence. Same
# rule: accepted, renamed before the call, the documented name kept.
_TOOL_SPELLINGS = {
    "set_sequencer_time": {"path": "sequence"},
    "place_sequence_actor": {"path": "sequence"},
    "sequence_onscreen": {"path": "sequence"},
    "mirror_anim_sequence": {"output_path": "new_path"},
    "reverse_anim_sequence": {"new_path": "output_path"},
    "bake_locomotion": {"new_path": "output_path"},
    "plan_side_steps": {"new_path": "output_path"},
}
_PAIR_ALIAS_CACHE = {}


def _pair_aliases(tool: str) -> dict:
    """{alias: declared name} for `tool` from _SPELLING_PAIRS and its registered schema; {} for a bl_ tool or an unknown one."""
    if tool in _PAIR_ALIAS_CACHE:
        return _PAIR_ALIAS_CACHE[tool]
    found = {}
    tools = getattr(getattr(globals().get("mcp"), "_tool_manager", None), "_tools", None)
    entry = tools.get(tool) if isinstance(tools, dict) and not tool.startswith("bl_") else None
    if entry is not None:
        declared = set(((getattr(entry, "parameters", None) or {}).get("properties") or {}).keys())
        for alias, name in _SPELLING_PAIRS + tuple((_TOOL_SPELLINGS.get(tool) or {}).items()):
            if name in declared and alias not in declared and alias not in found:
                found[alias] = name
        _PAIR_ALIAS_CACHE[tool] = found
    return found


def _deprecated_renames(tool: str, args: dict):
    """(args with each deprecated name of `tool` renamed to its canonical one, {old: new} renamed, [(old, new)] given
    both ways). A name given both ways is left as it is, for the caller to refuse. The cross-tool spellings of
    _SPELLING_PAIRS are renamed here too, after the tool's own deprecated names."""
    aliases = dict(_pair_aliases(tool), **(_PARAM_ALIASES.get(tool) or {}))
    if not aliases or not isinstance(args, dict):
        return args, {}, []
    out, renamed, both = dict(args), {}, []
    for old, new in aliases.items():
        if old not in out:
            continue
        if new in out:
            both.append((old, new))
            continue
        out[new] = out.pop(old)
        renamed[old] = new
    return out, renamed, both


def _both_spellings_error(tool: str, both) -> str:
    """The refusal for a call that names one parameter twice, by its old and its new spelling."""
    return ("%s received %s: the old name is a deprecated spelling of the new one, so the call names one parameter "
            "twice. Pass %s only. Nothing was sent."
            % (tool, "; ".join("%s and %s" % (old, new) for old, new in both),
               " and ".join(new for _, new in both)))


def _deprecation_note(tool: str, renamed: dict) -> str:
    """One line teaching the canonical names a call used the old spellings of."""
    own = _PARAM_ALIASES.get(tool) or {}
    old = {k: v for k, v in (renamed or {}).items() if k in own}
    other = {k: v for k, v in (renamed or {}).items() if k not in own}
    lines = []
    if old:
        lines.append("%s: %s deprecated; use %s. The old name still works." % (
            tool, ", ".join("%s is" % k for k in old), ", ".join(old.values())))
    if other:
        lines.append("%s: %s; this tool documents the second spelling." % (
            tool, ", ".join("%s was read as %s" % kv for kv in other.items())))
    return "\n".join(lines)


# A BLIND CALL GUESSES A NAME IN A FEW PREDICTABLE WAYS, and round eight counted them. On the lean default an agent calls
# held-back tools through mif_call without their schemas, and the as-shipped Blueprint-graph runs spent failed calls on
# connect_pins(source_node=...) for src_node, add_variable_get(variable=) for var, mif_help(topic=) for tool and batch ops
# spelled sourceNode/targetNode (docs/TOKEN_OPTIMIZATION.md 2.12). A guess is mapped only when exactly one accepted name
# fits one of three rules, and every mapping is reported, so the agent learns the name; anything ambiguous is still
# refused by name. create_mesh_boolean(path=) stays refused: path fits target_path, tool_path and output_path alike.
_ALIAS_WORDS = (("destination", "dst"), ("source", "src"), ("target", "dst"),
                ("Destination", "Dst"), ("Source", "Src"), ("Target", "Dst"))
_ALIAS_SUFFIXES = ("_path", "_name", "_id", "Path", "Name", "Id")

# DOCS FIRST, BESIDE THE MAPPING (docs/TOKEN_OPTIMIZATION.md 2.13). Guessed names come from agents not following the documentation (2026-09-27), and round eight agreed: every guessed name was a tool the
# run had never looked up, every wrong type a tool whose find_tools row gave names without types. With MIF_DOCS_FIRST=1 no
# guess is mapped (_arg_alias answers None; mif_call swaps no object/object_name), a call that does not fit is refused
# unsent with the tool's call shape (_docs_refusal), find_tools rows type every parameter that is not a string, and
# mif_call's description asks for the lookup first (_apply_docs_first). Off by default, so round nine can compare the two
# on one build. A module global, not a constant, so tools/test_mcp_packs.py can switch it.
_DOCS_FIRST = os.environ.get("MIF_DOCS_FIRST", "").strip().lower() in ("1", "true", "yes", "on")
# mif_call's description in docs-first mode: whole under _DESC_BUDGET, so it is sent as it stands, never cut to a lead
_DOCS_FIRST_MIF_CALL = ("Call any MifBridge tool by name, loaded or not (no refresh needed); args holds its own "
                        "parameters. Read them first in its find_tools row or mif_help(tool): a call that does not match "
                        "is refused unsent, with its call shape.")


def _schema_type(prop) -> str:
    """One parameter's type from its JSON schema, in a word: its `type`, or its anyOf's non-null types joined by |."""
    if not isinstance(prop, dict):
        return "any"
    kind = prop.get("type")
    if isinstance(kind, str):
        return kind
    if isinstance(kind, list):
        return "|".join(k for k in kind if k != "null") or "null"
    kinds = [p.get("type") for p in prop.get("anyOf") or [] if isinstance(p, dict)]
    return "|".join(k for k in dict.fromkeys(kinds) if isinstance(k, str) and k != "null") or "any"


def _call_shape(schema) -> dict:
    """What a call must look like, from a tool's JSON schema: {param: "type, required"} for a parameter without a default
    and {param: "type = default"} for the rest - the documentation a docs-first refusal hands back."""
    schema = schema if isinstance(schema, dict) else {}
    required = set(schema.get("required") or [])
    shape = {}
    for key, prop in (schema.get("properties") or {}).items():
        kind = _schema_type(prop)
        if key in required:
            shape[key] = kind + ", required"
        elif isinstance(prop, dict) and "default" in prop:
            shape[key] = "%s = %s" % (kind, json.dumps(prop["default"], default=str))
        else:
            shape[key] = kind
    return shape


def _typed_params(schema) -> list:
    """A find_tools row's params in docs-first mode: each name, as name:type when its type is not a plain string - the
    types round eight's wrong-type calls lacked. A string stays bare, so a row grows only where a type can be wrong."""
    out = []
    for key, prop in ((schema if isinstance(schema, dict) else {}).get("properties") or {}).items():
        kind = _schema_type(prop)
        out.append(key if kind == "string" else "%s:%s" % (key, kind))
    return out


def _schema_of(name: str) -> dict:
    """A tool's JSON schema: the registered one, or for a held-back tool the one FastMCP would build on registering it;
    {} when neither can be read."""
    registry = getattr(getattr(mcp, "_tool_manager", None), "_tools", None) or {}
    if name in registry:
        return registry[name].parameters or {}
    fn = _UNLOADED.get(name)
    if fn is None:
        return {}
    try:
        from mcp.server.fastmcp.tools.base import Tool as _FastTool
        return _FastTool.from_function(fn).parameters or {}
    except Exception:  # noqa: BLE001 - a row without types is still a row
        return {}


def _type_problems(tool_obj, args: dict) -> list:
    """Each argument that does not fit its parameter's type, as "param: what is wrong" - checked by the tool's own
    argument model the way FastMCP checks before it calls, so nothing runs here. [] when every argument fits."""
    meta = getattr(tool_obj, "fn_metadata", None)
    model = getattr(meta, "arg_model", None)
    if model is None:
        return []
    try:
        parsed = meta.pre_parse_json(dict(args)) if hasattr(meta, "pre_parse_json") else dict(args)
        model.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - pydantic's ValidationError, read through its own errors()
        try:
            errors = exc.errors()
        except Exception:  # noqa: BLE001
            return [str(exc).splitlines()[0]]
        # one line per parameter: a union reports each branch it tried (location.dict[any,any], location.list[any]),
        # which reads as "a valid dictionary / a valid list" under the one name the call used
        per_param = {}
        for err in errors:
            loc = err.get("loc") or ("?",)
            per_param.setdefault(str(loc[0]), []).append(str(err.get("msg")))
        return ["%s: %s" % (param, " / ".join(dict.fromkeys(msgs))) for param, msgs in per_param.items()]
    return []


def _docs_refusal(name: str, error: str, **extra) -> dict:
    """A docs-first refusal (docs/TOKEN_OPTIMIZATION.md 2.13): nothing was sent, and the tool's documentation comes with
    it - its call shape and its lead sentence - so the next call is written from that, not from the next guess."""
    out = {"ok": False, "tool": name, "sent": False,
           "error": error + " Nothing was sent: write the call from callShape.",
           "callShape": _call_shape(_schema_of(name))}
    about = _lead(name, _registered_description(name))
    if about:
        out["about"] = about
    out.update(extra)
    return out


def _apply_docs_first() -> str:
    """At startup, in docs-first mode: mif_call's sent description asks for the lookup first. Says what it did."""
    if not _DOCS_FIRST:
        return "off (guessed names are mapped where one accepted name fits)"
    tool = (getattr(getattr(mcp, "_tool_manager", None), "_tools", None) or {}).get("mif_call")
    if tool is not None:
        tool.description = _DOCS_FIRST_MIF_CALL
    return "on (no guess is mapped; a call that does not fit is refused unsent with its call shape)"


def _did_you_mean(key: str, accepted) -> list:
    """The accepted names a refused `key` most likely meant, best first (three at most), so a refusal asks "you said
    `path` - did you mean `target_path` or `tool_path`?" instead of leaving the agent a whole list to guess from
    again. Wanted 2026-09-27: "you said X, did you mean Y or Z", so there is no repeated guessing. In order: _arg_alias's
    abbreviations (source to src) and prefixes, which the mapping takes only when exactly ONE fits; the same name in
    another case (graphId for graph_id); a whole word of an accepted name (path for target_path, name for
    name_contains); a near spelling (grpah_id). Empty when nothing is close: then the full list is the honest answer."""
    accepted = [a for a in (accepted or ()) if isinstance(a, str)]
    if not isinstance(key, str) or not key or not accepted:
        return []
    out = []

    def add(name):
        if name in accepted and name != key and name not in out:
            out.append(name)
    for long_word, short in _ALIAS_WORDS:
        if long_word in key:
            add(key.replace(long_word, short))
    flat = key.replace("_", "").lower()
    for a in accepted:
        if a.replace("_", "").lower() == flat:
            add(a)                                            # graphId -> graph_id, tool_Path -> tool_path
    for a in accepted:
        if min(len(a), len(key)) < 3:
            continue
        if key.startswith(a) and key[len(a):len(a) + 1].isalpha() and key[len(a):len(a) + 1].islower():
            add(a)                                            # variable -> var
        elif a.startswith(key) and a[len(key):] in _ALIAS_SUFFIXES:
            add(a)                                            # property -> property_path
        elif a.startswith(key + "_") or a.endswith("_" + key):
            add(a)                                            # path -> target_path, tool_path; name -> name_contains
    for a in difflib.get_close_matches(key, accepted, n=3, cutoff=0.75):
        add(a)                                                # grpah_id -> graph_id
    return out[:3]


def _unknown_params_refusal(name: str, unknown, params) -> tuple:
    """(error text, didYouMean) for a call naming parameters the tool does not take: each unknown name with the accepted
    names it most likely meant (_did_you_mean), then the full list. didYouMean is {} when nothing is close."""
    params = sorted(params)
    ask = {k: _did_you_mean(k, params) for k in unknown}
    ask = {k: v for k, v in ask.items() if v}
    error = "%s received unknown parameter(s): %s." % (name, ", ".join(unknown))
    if ask:
        error += " Did you mean: %s?" % "; ".join("%s -> %s" % (k, " or ".join(v)) for k, v in ask.items())
    error += " Accepted parameters: %s" % (", ".join(params) or "(none)")
    return error, ask


def _arg_alias(key: str, accepted, single: bool = False):
    """The one accepted name a guessed `key` stands for, or None: (1) the standard abbreviations - source to src, target
    and destination to dst (source_node -> src_node, targetPin -> dstPin); (2) a unique prefix across a neutral seam -
    the guess spells out an abbreviation (variable -> var) or drops a _path/_name/_id suffix (property -> property_path),
    never a filter's (name is not name_contains); (3) with `single`, the tool's only parameter. Always None in docs-first
    mode, which maps no guess (docs/TOKEN_OPTIMIZATION.md 2.13)."""
    if _DOCS_FIRST:
        return None
    accepted = set(accepted or ())
    if not isinstance(key, str) or key in accepted:
        return None
    for long_word, short in _ALIAS_WORDS:
        if long_word in key:
            cand = key.replace(long_word, short)
            if cand in accepted:
                return cand
    fits = set()
    for a in accepted:
        if min(len(a), len(key)) < 3:
            continue
        if key.startswith(a) and key[len(a):len(a) + 1].isalpha() and key[len(a):len(a) + 1].islower():
            fits.add(a)                                  # variable -> var: the guess spells the name out
        elif a.startswith(key) and a[len(key):] in _ALIAS_SUFFIXES:
            fits.add(a)                                  # property -> property_path: only a neutral suffix
    if len(fits) == 1:
        return fits.pop()
    if single and len(accepted) == 1:
        return next(iter(accepted))
    return None


def _preview_commit_refusal(preview_id, **request):
    """THE COMMIT HALF OF A BULK PREVIEW (docs/15_SAFETY_GATE_AND_JOURNAL.md section 5). A preview_id commits exactly the
    request that was previewed, so it travels ALONE - the endpoint refuses anything sent beside it, and this refuses it
    first, naming it, rather than sending it or dropping it: a request parameter dropped here would commit the preview
    while the caller believed their change went with it. `request` is every other parameter the wrapper takes; a value
    that is unset (None, False, "", empty) counts as not given. None when the commit may go; each wrapper then posts
    its OWN literal endpoint name, so parity_check can hold every call site to the plugin."""
    given = sorted(k for k, v in request.items() if v not in (None, False, "", [], {}))
    if not given:
        return None
    return {"ok": False, "committed": False, "nothingWritten": True, "previewId": preview_id,
            "error": "a commit takes preview_id alone, and %s was also given. The request was fixed when it was "
                     "previewed - to change it, preview again with dry_run=True. Nothing was sent." % ", ".join(given)}


# ONE ANSWER TO "IS THE BRIDGE CURRENT?" (spec "Extend self_audit with bridge health and staleness", 2026-10-14). The three
# pieces each knew their own version - this server its checkout's MifBridge.uplugin, the editor its loaded descriptor
# (self_audit plugin.version), the add-on its bl_info - and the handshake compared what they call, but nothing said
# whether they are one release, nor that the editor runs a binary older than the source beside it. bridge says both:
# each version, inStep, and stale - one line per reason, each with what to do.
def _server_plugin_version():
    """(VersionName, path) of the MifBridge.uplugin this server belongs to - beside tools/ in a checkout, four folders up
    in a store package (Content/Python/MifBridge/mcp-server) - or (None, None)."""
    d = os.path.dirname(_SERVER_PATH)
    for _ in range(6):
        d = os.path.dirname(d)
        path = os.path.join(d, "MifBridge.uplugin")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as fh:
                    return json.load(fh).get("VersionName"), path
            except (OSError, ValueError):
                return None, path
    return None, None

_SERVER_LOADED_VERSION = []        # [VersionName] of the uplugin when this process first read it (see _server_loaded_version)


def _server_loaded_version():
    """The release this PROCESS belongs to: the uplugin's version at the first read, which is at import (below)."""
    if not _SERVER_LOADED_VERSION:
        _SERVER_LOADED_VERSION.append(_server_plugin_version()[0])
    return _SERVER_LOADED_VERSION[0]


_server_loaded_version()           # at import, so a pull later in the session cannot move it

_FIND_STOP = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "with", "from", "into", "by", "at",
              "my", "some", "this", "that", "tool", "tools"}


# THE VERBS AN AGENT SAYS AND THE ONES THE TOOLS ARE NAMED WITH. "add keyframe" matched bl_add_fcurve_modifier and
# not bl_set_keyframe, whose name says set; "place actor" matched attach_actor and not spawn_actor_in_level. Only
# for words of a query of two or more - a one-word search stays the literal substring it always was.
_FIND_SYNONYMS = {
    "add": ("set", "create", "insert"), "insert": ("add", "set"), "create": ("add", "make"), "make": ("create",),
    "place": ("spawn", "add"), "put": ("spawn", "set"), "remove": ("delete", "clear"), "delete": ("remove", "clear"),
    "reduce": ("decimate",), "cut": ("boolean", "bisect"), "hole": ("boolean",), "round": ("bevel",),
}


# QUESTIONS WHOSE WORDS ARE NOT IN THE ANSWERING TOOL'S NAME OR DESCRIPTION, found 2026-10-11: an agent driving a game
# in Play-in-Editor asked for "bone world positions" and "a screenshot of what the player sees", got nothing it
# recognised, and fell back to run_python - read_component_pose and capture_viewport{world:"pie"} answer both. A row
# is (every one of these words, at least one of these, whether it also needs a word naming the game, the tools that
# answer, best first); a query that holds them ranks those tools ahead of every substring match. Words are matched by
# their stems (_find_forms).
_FIND_INTENTS = (
    (("bone",), ("world", "position", "location", "pie", "live", "where", "pose", "space"), False,
     ("read_component_pose",)),
    (("socket",), ("world", "position", "location", "pie", "live", "where"), False, ("read_component_pose",)),
    ((), ("screenshot", "see", "capture", "picture", "image", "render"), True,
     ("capture_viewport", "ui_scenario_capture")),
    ((), ("camera", "fov", "view"), True, ("read_player_camera",)),
    (("component",), ("visible", "visibility", "active", "hidden", "live", "state"), True, ("list_components",)),
    # a MetaHuman Performance is made, solved and exported by the three mocap tools, whose names say mocap, not performance
    (("metahuman", "performance"), ("export", "skeleton", "retarget", "animation"), False, ("export_mocap_animation",)),
    (("metahuman", "performance"), ("create", "make", "new", "start", "run", "pipeline", "process", "solve", "export",
                                    "skeleton", "retarget", "animation"), False,
     ("start_mocap_performance", "mocap_performance_status", "export_mocap_animation")),
    # "plan my mod", "steps to make a game", "roadmap": the grounded plan (1.3, docs/30_ROADMAP.md), whose name says neither
    ((), ("roadmap",), False, ("mif_roadmap",)),
    (("mod",), ("plan", "step", "steps", "start", "how"), False, ("mif_roadmap",)),
    ((), ("plan", "step", "steps"), True, ("mif_roadmap",)),
)
_FIND_INTENT_GAME = ("player", "game", "pie", "play")   # what "needs the game" means: the game, not the editor


def _find_intent_tools(words):
    """The tools _FIND_INTENTS pins for a query - an ordered list, possibly empty."""
    forms = set()
    for w in words:
        forms |= _find_forms(w)
    out = []
    for every, some, needs_game, tools in _FIND_INTENTS:
        if not all(w in forms for w in every) or not any(w in forms for w in some):
            continue
        if needs_game and not any(w in forms for w in _FIND_INTENT_GAME):
            continue
        out += [t for t in tools if t not in out]
    return out


def _find_forms(word):
    """A query word, its plain-English stems ("edges" -> "edge", "bodies" -> "body") and the tools' own verbs for it."""
    out = set()
    for w in (word,) + _FIND_SYNONYMS.get(word, ()):
        out.add(w)
        for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                out.add(w[:-len(suffix)] + repl)
    return out


def _find_rank(q, words, name, desc):
    """How a tool matches a query, as a sort key - lower is better - or None.

    (0,) the phrase is in the name; (1,) in the description - the only two classes a one-word query has, as
    before. For more than one word, (2, -k): every word is somewhere, k of them in the name. A tool that has
    only SOME of the words is not a match here - the fallback that offered them returned 38 tools for a nonsense
    keyword by matching "not" and "real". When nothing matches, find_tools asks _find_partial (words in the NAME)
    and, failing that, says which single words do match."""
    n, d = name.lower(), desc.lower()
    if q in n:
        return (0,)
    if q in d:
        return (1,)
    if len(words) < 2:
        return None
    in_name = 0
    for w in words:
        forms = _find_forms(w)
        if any(f in n for f in forms):
            in_name += 1
        elif not any(f in d for f in forms):
            return None
    return (2, -in_name)


def _find_partial(words, name, desc):
    """A sort key for a tool that has two or more query words as whole tokens of its NAME, else None: most words in the
    name first, then most of the rest in its description, then the shorter name. A token matches a word's form exactly
    or starts with it ("widget" finds `widgets`, "set" not `assets`). Only asked when no tool has every word."""
    tokens = name.lower().split("_")
    in_name = [w for w in words
               if any(tok == f or (len(f) >= 3 and tok.startswith(f)) for f in _find_forms(w) for tok in tokens)]
    if len(in_name) < 2:
        return None
    d = desc.lower()
    elsewhere = sum(1 for w in words if w not in in_name and any(f in d for f in _find_forms(w)))
    return (-len(in_name), -elsewhere, len(name))


def _find_summary(name, desc):
    """A find_tools row's summary: the tool's served lead - the sentence the tool list itself shows, whole - instead
    of its first 200 characters, which cut mid-word ("... THE ADDON'S DEFAU...") and averaged 178 characters to the
    lead's 128 over the 716 rows the benchmark's agents were shown (docs/TOKEN_OPTIMIZATION.md). The mif_help pointer is
    left off: every tool answers mif_help."""
    return _lead(name, " ".join((desc or "").split())).split(" Full notes: mif_help(")[0]


def _find_pack_hint(words):
    """The packs whose work a search names - [{pack, title, about}], at most two, else None. A search of two words or
    more names a pack when every word (its stems and the tools' verbs, _find_forms) is in the pack's title or
    description; one word, when it is in the title. More than two such packs says nothing, so none are given.

    WHY: a search by the job's own words finds the tools whose descriptions repeat them, and those are rarely the ones
    that do the job. 8 of round nine's 10 Blueprint-graph runs opened with find_tools('blueprint graph node'): 18 rows,
    every one a reader or an analyser, none that makes a node - then a second and third search of 8-15k characters
    (docs/TOKEN_OPTIMIZATION.md 2.15). The pack's description says how its work is done (gen_packs.py DESCRIPTIONS).

    A SEARCH TO MAKE A MODEL names the Blender pack that makes it, first and beyond the two (_find_makes_packs)."""
    words = [w for w in words if w]
    if not words:
        return None
    makes = _find_makes_packs(words)
    found = []
    for pid, p in (_packs() or {}).items():
        if pid in ("__error__", "core", "misc") or not isinstance(p, dict) or pid in makes:
            continue
        if p.get("makes") and _find_names_any(words, _FIND_UNREAL_MAKES):
            continue   # a search for what Unreal makes better never names a Blender making pack
        title = (p.get("title") or "").lower()
        if len(words) == 1:
            # ONE WORD MATCHES A WHOLE WORD OF THE TITLE, by its own stems only: the verbs' synonyms made "add_" name
            # "Core Assets" through "set".
            stems = _find_forms(words[0]) - set(_FIND_SYNONYMS.get(words[0], ()))
            ok = any(tok.startswith(s) for s in stems for tok in _re.split(r"[^a-z0-9]+", title) if tok and len(s) >= 3)
        else:
            text = title + " " + (p.get("description") or "").lower()
            ok = all(any(f in text for f in _find_forms(w)) for w in words)
        if ok:
            found.append((pid, p))
    if len(found) > 2:
        found = []
    found = [(pid, (_packs() or {}).get(pid) or {}) for pid in makes] + found
    if not found:
        return None
    # SAID ONCE A SESSION: the description in full the first time a pack is named, its name and title after. The
    # hand-off is said every time: it is the step a search for the model never finds by itself.
    rows = []
    for pid, p in found:
        row = {"pack": pid, "title": p.get("title")}
        if pid not in _PACK_HINTS_SENT:
            row["about"] = p.get("description")
            _PACK_HINTS_SENT.add(pid)
        if pid in makes:
            row["then"] = "mif_send_to_unreal(objects, destination)"
        rows.append(row)
    return rows


_PACK_HINTS_SENT = set()

# BLENDER IS THE GO-TO FOR A MODEL OR AN ASSET (decided 2026-09-30), and the search an agent makes for one says so in
# its own words: "make a rock prop", "create a static mesh", "build a wall". Those found the Unreal tools whose
# descriptions repeat the words - create_procedural_mesh, make_prop_blueprint - or nothing at all, and never the
# Blender builders or the hand-off. A search with one of these making words and one of a Blender pack's makes
# (packs.json, from blender_packs.MAKES) names that pack, unless it also names what Unreal makes better.
_FIND_MAKE_VERBS = {"make", "create", "build", "model", "modelling", "generate", "sculpt", "craft", "design"}
# WHAT UNREAL MAKES BETTER, OR IS NOT A MESH AT ALL: its ground, painted instances and procedural scatter, and the
# Unreal assets that share a noun with a model ("behavior tree", "state tree", "data asset", "widget tree").
_FIND_UNREAL_MAKES = {"landscape", "terrain", "heightmap", "foliage", "pcg", "water", "river", "lake", "ocean",
                      "navmesh", "blueprint", "widget", "umg", "niagara", "particle", "material", "texture", "sound",
                      "audio", "metasound", "data", "datatable", "behavior", "behaviour", "state", "blackboard",
                      "animation", "montage", "sequence", "snapshot"}


def _find_names_any(words, vocabulary):
    """True when a query word, or one of its plain-English stems, is in vocabulary. Stems only - the verbs' synonyms
    (_FIND_SYNONYMS) would make "make" a "create" and so on, which is not what a vocabulary of nouns means."""
    return any(f in vocabulary for w in words for f in _find_forms(w) - set(_FIND_SYNONYMS.get(w, ())) | {w})


def _find_makes_packs(words):
    """The Blender packs a search asks to make something in - [pack id] - or []. See _FIND_MAKE_VERBS."""
    if not _find_names_any(words, _FIND_MAKE_VERBS) or _find_names_any(words, _FIND_UNREAL_MAKES):
        return []
    return [pid for pid, p in (_packs() or {}).items()
            if isinstance(p, dict) and p.get("makes") and _find_names_any(words, set(p["makes"]))]


# WHICH TOOLS CAN RUN NOW (spec "Extend find_tools to say which tools are available now", 2026-10-14). A search answered
# names and parameters only, so an agent with no editor open, or an editor whose plugin predates a tool, picked a tool
# and learned from its failure that it could not run. find_tools now says what each backend is doing (`now`), what each
# full row needs (`needs`), and, for a row that cannot run here, why (`unavailable`): no editor or Blender answering,
# the editor's plugin lacking the tool's endpoint (the version handshake's measurement), or an engine older than the
# tool's own description names. What a tool needs is READ, never written by hand: its pack's backend (packs.json), the
# _post / _blender calls in its own body and the helpers it calls by name (this file's AST), and its description's
# "UE 5.N+". A tool whose backend is its argument's choice (run_python's app) needs either.
_AVAIL_TTL = 20.0                 # seconds a backend's state is reused before find_tools looks again
# ONLY A SERVING SERVER ASKS. A suite imports this file and calls find_tools hundreds of times with no editor of its own,
# and whatever listens on the default ports is somebody else's: main() sets this before mcp.run(), and until then a
# search says nothing about availability (tools/test_find_tools.py sets it for its AV rows, with every backend stubbed).
_SERVING = False
_AVAIL_STATE = {"at": None, "now": None}
_TOOL_NEEDS = None                # {function: {"needs", "ownNeeds", "endpoints", "ops"}} from this file's own source
_TOOL_NEEDS_EITHER = {"run_python": ("unreal", "blender")}        # the backend is the caller's choice (app)
_TOOL_NEEDS_NONE = {"mif_call", "mif_editor", "mif_help", "mif_enable_pack", "mif_list_packs", "find_tools",
                    "mif_python_report"}
_MIN_UE_RE = _re.compile(r"\bUE 5\.(\d+)\+")


def _tool_needs_table():
    """{function name: {"needs", "ownNeeds", "endpoints", "ops"}} for every function in this file: what its own body
    calls (_post / _blender literals) and, transitively, what the functions and the module-level tables of function
    names it refers to call. Parsed once."""
    global _TOOL_NEEDS
    if _TOOL_NEEDS is not None:
        return _TOOL_NEEDS
    import ast
    try:
        with open(_SERVER_PATH, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        _TOOL_NEEDS = {}
        return _TOOL_NEEDS
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tables = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            names = {c.value for c in ast.walk(n.value) if isinstance(c, ast.Constant) and c.value in funcs}
            if names:
                tables[n.targets[0].id] = names
    own, refs = {}, {}
    for name, fn in funcs.items():
        rec = {"needs": set(), "optional": set(), "endpoints": set(), "ops": set()}
        r = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("_post", "_blender"):
                lit = n.args[0].value if n.args and isinstance(n.args[0], ast.Constant) else None
                if n.func.id == "_post":
                    rec["needs"].add("unreal")
                    if isinstance(lit, str):
                        rec["endpoints"].add(lit)
                else:
                    rec["needs"].add("blender")
                    if isinstance(lit, str):
                        rec["ops"].add(lit)
            elif isinstance(n, ast.Name) and n.id in ("_post", "_blender"):
                # HANDED ON, not called: mif_synth_audio passes _post for its optional import, gameplay_recipe for the
                # build its dry run does without - the tool runs without that backend, and part of it uses it.
                rec["optional"].add("unreal" if n.id == "_post" else "blender")
            elif isinstance(n, ast.Name) and n.id != name:
                if n.id in funcs:
                    r.add(n.id)
                elif n.id in tables:
                    r |= tables[n.id]
        own[name], refs[name] = rec, r
    out = {}
    for name in funcs:
        seen, stack = {name}, [name]
        rec = {"needs": set(), "optional": set(), "endpoints": set(), "ops": set()}
        while stack:
            cur = stack.pop()
            for k in rec:
                rec[k] |= own[cur][k]
            for g in refs.get(cur, ()):
                if g not in seen:
                    seen.add(g)
                    stack.append(g)
        rec["ownNeeds"] = set(own[name]["needs"])
        out[name] = rec
    _TOOL_NEEDS = out
    return out


def _tool_needs(name, desc=""):
    """(needs, optional, endpoints, ops, minimum UE minor or None) for one tool. needs is a tuple of backends the tool
    needs all of, except where _TOOL_NEEDS_EITHER says the caller picks one; optional, the backends part of it uses."""
    m = _MIN_UE_RE.search(desc or "")
    minor = int(m.group(1)) if m else None
    if name in _TOOL_NEEDS_NONE:
        return (), (), set(), set(), None
    if name in _TOOL_NEEDS_EITHER:
        return _TOOL_NEEDS_EITHER[name], (), set(), set(), None
    rec = _tool_needs_table().get(name) or {"needs": set(), "optional": set(), "endpoints": set(), "ops": set(),
                                            "ownNeeds": set()}
    backend = None
    for p in (_packs() or {}).values():
        if isinstance(p, dict) and name in (p.get("tools") or []) and p.get("backend") in ("ue", "blender"):
            backend = "unreal" if p["backend"] == "ue" else "blender"
            break
    if not rec["needs"]:
        needs = set()           # nothing it calls reaches a backend: an offline tool, whatever pack carries it
    elif backend:
        # A pack's backend is the tool's; the other backend counts only when the tool's OWN body calls it - a helper's
        # optional read (the project mode a generative Blender tool checks) is not what the tool needs to run.
        needs = {backend} | (rec["ownNeeds"] - {backend})
    else:
        needs = set(rec["needs"])
    if minor is not None and needs:
        needs.add("unreal")
    order = ("unreal", "blender")
    return (tuple(b for b in order if b in needs), tuple(b for b in order if b in rec["optional"] and b not in needs),
            rec["endpoints"], rec["ops"], minor)


def _port_accepts(host, port, timeout=0.3):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def _backends_now():
    """{"unreal": {...}, "blender": {...}}: Unreal is never connected here; Blender's answer is reused for _AVAIL_TTL
    seconds, and a port nothing listens on is not pinged."""
    now = time.monotonic()
    if _AVAIL_STATE["now"] is not None and now - _AVAIL_STATE["at"] < _AVAIL_TTL:
        return _AVAIL_STATE["now"]
    ue = {"connected": False, "at": "not part of MifBlender"}
    bl = {"connected": False, "at": "%s:%s" % (BLENDER_HOST, BLENDER_PORT)}
    if _port_accepts(BLENDER_HOST, BLENDER_PORT):
        res = _blender("ping", _timeout=BLENDER_PROBE_TIMEOUT, _lock_timeout=BLENDER_PROBE_TIMEOUT)
        if isinstance(res, dict) and res.get("pong"):
            bl.update(connected=True, blenderVersion=res.get("blenderVersion"))
    state = {"unreal": ue, "blender": bl}
    _AVAIL_STATE.update(at=now, now=state)
    return state


def _state_engine_minor(state):
    m = _re.match(r"5\.(\d+)", str((state.get("unreal") or {}).get("engineVersion") or ""))
    return int(m.group(1)) if m else None


def _tool_unavailable(name, desc, state):
    """(needs, optional, why it cannot run now or None) for one tool against the backends' state."""
    needs, optional, endpoints, ops, minor = _tool_needs(name, desc)
    if not needs:
        return needs, optional, None
    up = [b for b in needs if (state.get(b) or {}).get("connected")]
    if name in _TOOL_NEEDS_EITHER:
        return needs, optional, None if up else "neither an Unreal editor (%s) nor Blender (%s) answers" % (
            state["unreal"]["at"], state["blender"]["at"])
    down = [b for b in needs if b not in up]
    if down:
        return needs, optional, "; ".join("%s at %s" % ("no Unreal editor answers" if b == "unreal" else "Blender does not answer",
                                                              state[b]["at"]) for b in down)
    if "unreal" in needs:
        have = _state_engine_minor(state)
        if minor is not None and have is not None and have < minor:
            return needs, optional, "needs UE 5.%d+, the editor runs %s" % (minor, state["unreal"].get("engineVersion"))
        lack = sorted(set(endpoints) & set((_HS.get("unreal") or {}).get("missingEndpoints") or ()))
        if lack:
            return needs, optional, "the editor's MifBridge plugin is older than this server: it has no %s" % ", ".join(lack)
    if "blender" in needs:
        lack = sorted(set(ops) & set((_HS.get("blender") or {}).get("missingEndpoints") or ()))
        if lack:
            return needs, optional, "the add-on Blender loaded is older than this server: it has no op %s" % ", ".join(lack)
    return needs, optional, None


def _backends_line(state):
    """The short `now` field of a find_tools answer."""
    ue, bl = state["unreal"], state["blender"]
    out = {"unreal": ("connected, UE %s%s" % (ue.get("engineVersion") or "?",
                                              ", %s" % ue["project"] if ue.get("project") else ""))
           if ue.get("connected") else "not answering at %s" % ue["at"],
           "blender": ("connected, Blender %s" % (bl.get("blenderVersion") or "?")) if bl.get("connected")
           else "not answering at %s" % bl["at"]}
    hs = _HS.get("unreal") or {}
    if ue.get("connected") and hs.get("verdict"):
        out["unrealPlugin"] = "%s older (version handshake): %d endpoint(s) this server calls are missing there" % (
            "the plugin is" if hs["verdict"] == "plugin" else "this server is" if hs["verdict"] == "server"
            else "each side is", len(hs.get("missingEndpoints") or ()))
    return out


_FIND_FULL_ROWS = 15     # find_tools' default limit: a default answer is unchanged


@mcp.tool()
def find_tools(keyword: str, limit: int = 15) -> dict:
    "Search every MifBridge tool, loaded here or not, by a keyword or a few words: each row gives the tool's summary and parameters, and an unloaded tool's pack."
    return _find_tools(keyword, limit, True)


def _find_tools(keyword, limit=15, probe=False):
    """find_tools' search. probe: say what each full row needs and whether it can run now (find_tools itself); a caller
    that only wants the ranking (mif_python_report) passes False and no backend is asked anything."""
    q = (keyword or "").strip().lower()
    if not q:
        return {"ok": False, "error": "keyword is required - a substring to search tool names/descriptions for."}
    try:
        tools = mcp._tool_manager.list_tools()
    except Exception as e:
        return {"ok": False, "error": "could not read this server's own tool registry: %s" % e}
    # A PHRASE WAS ONE SUBSTRING until 2026-09-21, so "bevel edges" found nothing - not even bl_bevel_edges - and
    # neither did "spawn actor", "add keyframe", "export fbx" or "compile blueprint": 21 of the 23 two-word agent
    # queries measured came back empty, which an agent in a one-pack session reads as "MifBridge cannot do that". The
    # whole phrase still ranks first; after it come the tools that match EVERY word (stems and the tools' own verbs
    # counted); and when nothing does, the answer says how many tools each word finds alone.
    words = [w for w in _re.split(r"[\s_\-/,.;:]+", q) if w and w not in _FIND_STOP]
    pinned = _find_intent_tools(words)

    def _rank(name, desc):
        return (-1, pinned.index(name)) if name in pinned else _find_rank(q, words, name, desc)
    ranked = []
    candidates = []     # every candidate, for the partial match below when no tool has every word
    for t in tools:
        name = t.name or ""
        desc = t.description or ""
        if t.name == "find_tools" or (t.name in _GEN_TOOLS and _gen_block()):
            continue   # the search tool itself is never the point; a gen tool is not offered while generation is off
        candidates.append((name, desc, t, False, None))
        rank = _rank(name, desc)
        if rank is None:
            continue
        try:
            params = list((t.parameters or {}).get("properties", {}).keys())
        except Exception:
            params = []
        row = {
            "name": name,
            "summary": _find_summary(name, desc),
            "params": params,
        }
        ranked.append((rank, row))

    # HELD-BACK TOOLS ARE STILL FOUND. With MIF_TOOL_PACKS limiting what is loaded, a tool that is not
    # registered would otherwise be invisible here - an agent searching "material" in a world-only
    # session would conclude no such tool exists. They come back with loaded:false and the pack that
    # carries them, so the next step is mif_enable_pack rather than a wrong conclusion.
    pack_of = {}
    for pid, p in (_packs() or {}).items():
        if pid != "__error__":
            for tn in p.get("tools") or []:
                pack_of.setdefault(tn, pid)
    for name, fn in sorted(_UNLOADED.items()):
        if name in _GEN_TOOLS and _gen_block():
            continue
        desc = getattr(fn, "__doc__", "") or ""
        candidates.append((name, desc, fn, True, pack_of.get(name)))
        rank = _rank(name, desc)
        if rank is None:
            continue
        code = getattr(fn, "__code__", None)
        row = {"name": name, "summary": _find_summary(name, desc),
               "params": list(code.co_varnames[:code.co_argcount]) if code else [],
               "loaded": False, "pack": pack_of.get(name)}
        ranked.append((rank, row))

    # TIES GO TO THE NAMES THAT CARRY THE WORDS. "blender mesh" is in the description of 77 tools, all rank (1,), and
    # they came back alphabetically from bl_add_cloth (benchmark round ten: every A3 run asked it at limit 40); within a
    # rank a tool whose name holds more of the words - bl_mesh_quality, bl_mesh_stats - now comes first.
    kept = sorted(ranked, key=lambda rr: (rr[0], -sum(1 for w in words if any(f in rr[1]["name"].lower()
                                                                                 for f in _find_forms(w))),
                                          rr[1]["name"]))
    # WHEN NO TOOL HAS EVERY WORD: THE TOOLS WITH MOST OF THEM IN THEIR NAMES, decided after round six
    # (docs/TOKEN_OPTIMIZATION.md 2.10). A search naming a whole job - "create widget blueprint set text property
    # compile save" - found nothing, and every as-shipped UMG run then listed all 36 widget tools, 10.9k characters
    # re-read on ~8 requests. The fallback removed on 2026-09-21 took any word anywhere and gave 38 tools for a
    # nonsense keyword; this one needs two query words or more as whole tokens of a tool's NAME, so a nonsense keyword
    # still finds nothing. Rows say how many words matched, and the answer says they are partial.
    partial = False
    # A PINNED TOOL (_FIND_INTENTS) IS NOT A MATCH OF EVERY WORD: the partial fallback still runs after it.
    pinned_kept = [kr for kr in kept if kr[0][0] == -1]
    if len(pinned_kept) == len(kept) and len(words) > 1:
        uniq = list(dict.fromkeys(words))
        scored = []
        for name, desc, obj, is_held, pack in candidates:
            key = _find_partial(uniq, name, desc)
            if key is None or name in pinned:
                continue
            if is_held:
                code = getattr(obj, "__code__", None)
                row = {"name": name, "summary": _find_summary(name, desc),
                       "params": list(code.co_varnames[:code.co_argcount]) if code else [],
                       "loaded": False, "pack": pack}
            else:
                try:
                    params = list((obj.parameters or {}).get("properties", {}).keys())
                except Exception:
                    params = []
                row = {"name": name, "summary": _find_summary(name, desc), "params": params}
            row["matched"] = -key[0]
            scored.append((key, row))
        scored = sorted(scored, key=lambda kr: (kr[0], kr[1]["name"]))
        partial = bool(scored)
        kept = pinned_kept + scored
    results = [row for _, row in kept][:max(1, limit)]
    # PAST THE FIRST _FIND_FULL_ROWS A ROW IS ITS NAME. A broad search at limit 40 was 14,737 characters, every row a
    # summary and its parameters, and it stayed in context for the rest of the run: benchmark round ten's Python-off
    # arm asked limit 30 or more in most runs, and those answers were replayed as 1.5 million tokens over A3, A5, A8 and
    # X1 (docs/TOKEN_OPTIMIZATION.md 2.18). The best-ranked rows keep everything; the rest are names to pick from,
    # with what a held-back one needs (loaded, pack) - mif_help('<tool>') or a narrower search gives the rest.
    for row in results[_FIND_FULL_ROWS:]:
        for key in ("summary", "params"):
            row.pop(key, None)
    # WHAT EACH FULL ROW NEEDS, AND WHETHER IT CAN RUN NOW (_backends_now above): `needs` names the backends, and a row
    # that cannot run here says why in `unavailable`. Names-only rows stay names.
    probe = probe and _SERVING
    state = _backends_now() if probe else None
    descs = {t.name: t.description or "" for t in tools}
    for row in (results[:_FIND_FULL_ROWS] if probe else ()):
        desc = descs.get(row["name"]) or getattr(_UNLOADED.get(row["name"]), "__doc__", "") or ""
        needs, optional, why = _tool_unavailable(row["name"], desc, state)
        if needs:
            row["needs"] = " or ".join(needs) if row["name"] in _TOOL_NEEDS_EITHER else "+".join(needs)
        elif optional:
            row["needs"] = "%s for part of it" % "+".join(optional)
        if why:
            row["unavailable"] = why
    if _DOCS_FIRST:
        # the rows kept, typed where a type can be wrong (docs/TOKEN_OPTIMIZATION.md 2.13); only these, not every match
        for row in results[:_FIND_FULL_ROWS]:
            row["params"] = _typed_params(_schema_of(row["name"])) or row.get("params") or []
    truncated = len(kept) > len(results)
    out = {
        "ok": True,
        "keyword": keyword,
        "count": len(results),
        "matched": len(kept),
        "results": results,
    }
    if probe:
        out["now"] = _backends_line(state)
    held = [r for r in results if r.get("loaded") is False]
    if held:
        out["notLoaded"] = len(held)
        out["enable"] = sorted({r["pack"] for r in held if r.get("pack")})
        loads, _ = _expand_packs(out["enable"])
        if loads != out["enable"]:
            out["enableLoads"] = loads
        # SAID ONCE, SHORTLY: this note rides on every find_tools reply with a held-back tool - ~330 characters each
        # time until 2026-09-22 (docs/TOKEN_OPTIMIZATION.md).
        # mif_call LEADS (docs/TOKEN_OPTIMIZATION.md 2.15): enabling a pack for a few of its tools costs every later
        # request the whole pack's names, and round nine's agents enabled `systems` and called through mif_call anyway.
        # mif_enable_pack as the mif_call that reaches it: on the lean default it is no tool of the client's (2.15).
        out["enableNote"] = ("%d not loaded: mif_call(tool, args) calls one, loading nothing, with its rules; "
                             "mif_call('mif_enable_pack', {'pack': <one of enable>}) puts the whole pack%s in every "
                             "later request."
                             % (len(held), " (and what it requires: enableLoads)" if loads != out["enable"] else ""))
    hint = _find_pack_hint(words)
    if hint:
        out["packs"] = hint
    if len(results) > _FIND_FULL_ROWS:
        out["namesOnlyFrom"] = _FIND_FULL_ROWS + 1
    if truncated:
        out["truncated"] = True
        out["note"] = ("%d total matches, showing %d - narrow the keyword or raise limit to see the rest."
                        % (len(kept), len(results)))
    if partial:
        out["partial"] = True
        out["note"] = ("no tool has every word of %r; these have the most of them in their names (matched), best first"
                       "%s." % (keyword, "; %d in all - raise limit for the rest" % len(kept) if truncated else ""))
    if not results:
        out["note"] = ("no tool name or description contains %r. Try a different word for the same "
                        "concept, or self_audit for the full endpoint list this server wraps." % keyword)
        if len(words) > 1:
            # WHICH WORD TO KEEP: each word's count alone, from the same registry and the held-back tools.
            pool = [(t.name or "", t.description or "") for t in tools if t.name != "find_tools"] + [
                (n, getattr(fn, "__doc__", "") or "") for n, fn in _UNLOADED.items()]
            out["wordCounts"] = {w: sum(1 for n, d in pool if _find_rank(w, [w], n, d) is not None) for w in words}
            out["note"] = ("no tool matches every word of %r. Each word alone: %s - search the one that names the "
                           "thing." % (keyword, ", ".join("find_tools(%r) %d" % kv for kv in out["wordCounts"].items())))
    return out


def _audio(module):
    """mif_audio.<module>, imported on first use with the server's folder on sys.path (as _vfx does)."""
    import importlib
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("mif_audio.%s" % module)

# ONE ALWAYS-LOADED PYTHON TOOL FOR BOTH APPLICATIONS, decided after round five (docs/TOKEN_OPTIMIZATION.md
# 2.9): bl_run_python and ue_run_python each cost every request of every session a schema, and one tool with an app
# switch costs one - 3,982 to 3,871 tokens a request, measured (tools/probe_first_request.py). It dispatches to those
# two, which stay callable by name, so each application keeps its own gate: the add-on's preference for Blender,
# exec_console's for Unreal.
# KNOWN SLIPS IN AN AGENT'S OWN PYTHON, answered with the fix in the reply that failed - nothing rides on a request
# that did not. benchmark round six: Blender 5's Action has no fcurves (4 failed calls, most on the walk cycle), and
# Unreal's Package has no is_dirty (3, as in round five). Both fixes read on the installs (docs/TOKEN_OPTIMIZATION.md
# 2.10): bpy_extras.anim_utils.action_get_channelbag_for_slot is in Blender 5.2's scripts; the dirty-package calls are
# BlueprintCallable on UEditorLoadingAndSavingUtils (FileHelpers.h, 5.8).
_PYTHON_ERROR_HINTS = (
    ("blender", "'Action' object has no attribute 'fcurves'",
     "Blender 5 keeps an Action's F-curves in its layers: from bpy_extras.anim_utils import "
     "action_get_channelbag_for_slot; action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves, with ad = "
     "ob.animation_data."),
    ("unreal", "'Package' object has no attribute 'is_dirty'",
     "unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages() and get_dirty_map_packages() list the dirty "
     "packages, and save_dirty_packages(True, True) saves them."),
)


def _python_hint(app, reply):
    if isinstance(reply, dict) and reply.get("ok") is False:
        error = str(reply.get("error") or "")
        for where, slip, hint in _PYTHON_ERROR_HINTS:
            if where == app and slip in error:
                return dict(reply, hint=hint)
    return reply

_RUN_PYTHON_APPS = {"blender": "bl_run_python"}


@mcp.tool()
def run_python(app: str = "blender", code: str = "", file: str = "", return_locals: bool = False) -> dict:
    "Run Python in Blender (bpy, bmesh, math imported): code or a .py file; stdout, stderr and any `result` you assign come back. Runs on Blender's main thread and is refused while the MifBlender add-on's preferences do not allow it. The escape hatch for what MifBlender has no typed tool for - prefer a typed tool whenever one exists, because those check their parameters and this does not."
    name = (app or "").strip().lower()
    if name == "unreal":
        return {"ok": False, "error": "this is the Blender-only MifBlender server; Unreal Python needs MifBridge "
                                      "(https://mifshowcase.store/mifbridge). Nothing was run."}
    target = _RUN_PYTHON_APPS.get(name)
    if target is None:
        return {"ok": False, "error": "app is 'blender', not %r - nothing was run." % app}
    return globals()[target](code=code, file=file, return_locals=return_locals)

# THE SESSION'S RAW PYTHON, AND THE TOOL THAT WOULD REPLACE EACH (spec "Report the session's raw-Python use and the tool
# that would replace each (mif_python_report)", 2026-10-14). Every run_python / ue_run_python / bl_run_python call is
# recorded here as it returns - which application, whether it worked, its size and the API calls it made (read from its
# own syntax tree, never run) - and mif_python_report names, per call, the typed tools whose names and descriptions
# carry its words (find_tools' own ranking). A call no tool answers is a gap to file: escape-hatch use, turned into
# the list of what to build. Kept for this server process, newest last, at most _PY_USES_MAX runs.
_PY_USES = []
_PY_USES_MAX = 200
_PY_SKIP_CALLS = {"print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "sorted", "round",
                  "min", "max", "abs", "sum", "isinstance", "getattr", "setattr", "hasattr", "enumerate", "zip", "open",
                  "repr", "type", "any", "all", "format", "join", "split", "strip", "append", "extend", "items", "keys",
                  "values", "get", "update", "pop", "startswith", "endswith", "lower", "upper", "replace", "add",
                  "unreal.log", "unreal.log_warning", "unreal.log_error", "unreal.Vector", "unreal.Rotator",
                  "unreal.LinearColor", "unreal.Color", "unreal.Transform", "unreal.Name", "unreal.Text",
                  "unreal.Vector2D", "unreal.Quat", "mathutils.Vector", "mathutils.Matrix", "mathutils.Euler",
                  "math.radians", "math.degrees", "math.sqrt", "math.sin", "math.cos", "result"}


def _py_calls(code):
    """The API calls a script makes, in order of first use: dotted names rooted at unreal / bpy / bmesh / mathutils, and
    the snake_case methods it calls on anything else (actor.set_actor_label). Read from the syntax tree; a script that
    does not parse is read with a pattern instead."""
    import ast
    out = []
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        found = _re.findall(r"\b((?:unreal|bpy|bmesh)(?:\.\w+)+)\s*\(", code or "")
        return [c for c in dict.fromkeys(found) if c not in _PY_SKIP_CALLS]
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        parts, f = [], n.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        parts.reverse()
        if not parts:
            continue
        if parts[0] in ("unreal", "bpy", "bmesh", "mathutils") and len(parts) > 1:
            name = ".".join(parts)
        elif len(parts) > 1 and "_" in parts[-1].strip("_"):
            name = "." + parts[-1]
        else:
            continue
        if name not in _PY_SKIP_CALLS and parts[-1] not in _PY_SKIP_CALLS:
            out.append(name)
    return list(dict.fromkeys(out))


def _py_record(app, code, file, reply):
    """Record one raw-Python run (called by ue_run_python and bl_run_python as they return)."""
    try:
        text = code
        if file and not code:
            try:
                with open(os.path.abspath(file), "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                text = ""
        ok = isinstance(reply, dict) and reply.get("ok") is not False
        _PY_USES.append({"app": app, "at": time.strftime("%H:%M:%S"), "ok": ok,
                         "source": "file %s" % file if file and not code else "code",
                         "lines": len((text or "").splitlines()), "calls": _py_calls(text)[:40],
                         "error": None if ok else str((reply or {}).get("error") or "")[:200]})
        del _PY_USES[:-_PY_USES_MAX]
    except Exception:       # noqa: BLE001 - the record is a by-product; it never fails the run it describes
        pass


class _MifToolError(Exception):
    """A composing tool's own refusal, raised so it cannot be ignored and caught
    at the tool boundary so it is never seen by the MCP client as a traceback.
    Every tool that raises it returns {"ok": False, "error": str(exc)}."""


# Boundary edges along Y, both sides - the two long edges of a road/sidewalk
# tile. Defined once so bl_* and mif_mesh_roundtrip cannot drift apart, and
# spelled the way the ADDON spells it (`boundaryOnly`, ops_mesh._BEVEL_KEYS) so
# the flattener is a rename of nothing.
_MIF_DEFAULT_EDGE_SELECTOR = {"boundaryOnly": True, "axis": "Y", "side": "both",
                              "tolerance": 1e-4}

# The addon's flat selector vocabulary, verbatim. Kept as a tuple rather than a
# set because these are also the kwargs order at every call site below, and a
# reader should be able to line the two up.
_BL_SELECTOR_KEYS = ("boundaryOnly", "axis", "side", "tolerance",
                     "minAngleDeg", "maxAngleDeg", "edgeIndices", "allEdges")

# The ONE historical misspelling, translated rather than silently dropped.
_BL_SELECTOR_ALIASES = {"boundary": "boundaryOnly"}


def _bl_selector(selector: dict) -> dict:
    """Flatten a nested selector into the addon's flat keys. Never returns None.

    Returns every key in _BL_SELECTOR_KEYS, None where unset - _blender() drops
    the Nones, so the addon sees only what the caller actually asked for.
    """
    if selector is None:
        selector = dict(_MIF_DEFAULT_EDGE_SELECTOR)
    if not isinstance(selector, dict):
        raise _MifToolError(
            f"selector must be an object, got {type(selector).__name__}. It is a small "
            f"declarative predicate, not a script: accepted keys are "
            f"{', '.join(_BL_SELECTOR_KEYS)}.")

    out = {key: None for key in _BL_SELECTOR_KEYS}
    unknown = []
    for key, value in selector.items():
        canon = _BL_SELECTOR_ALIASES.get(key, key)
        if canon not in out:
            unknown.append(key)
            continue
        out[canon] = value
    if unknown:
        raise _MifToolError(
            f"selector has unknown key(s) {', '.join(sorted(unknown))}. Accepted: "
            f"{', '.join(_BL_SELECTOR_KEYS)} (plus 'boundary' as an alias for "
            f"'boundaryOnly'). Refused here rather than at the addon, which would "
            f"reject the whole call after a round trip.")
    if all(value is None for value in out.values()):
        raise _MifToolError(
            "selector is empty and the addon refuses to guess which edges to touch. "
            "Give it boundaryOnly / axis+side / minAngleDeg / maxAngleDeg / edgeIndices, "
            "or allEdges:true to really mean every edge. Omit the argument entirely for "
            f"the default road-tile predicate {_MIF_DEFAULT_EDGE_SELECTOR}.")
    return out


def _bl_preserve_axes(preserve_x: bool):
    """preserve_x -> the addon's preserveAxes/assertAxes pair, or None.

    BOTH, never one. Asserting an axis without preserving it is not a
    configuration, it is a guaranteed failure: the assert measures exactly the
    drift the preserve exists to remove.
    """
    return ["X"] if preserve_x else None


@mcp.tool()
def bl_status(echo: str = None, list_ops: bool = False, port: int = None, pid: int = None) -> dict:
    "Health probe for the Blender backend, and the FIRST thing to call before any bl_* work. echo is returned verbatim - proof the ANSWER came from this call. list_ops=True adds every op name (opCount is always there). port or pid switches this session to that running MifBlender (listeners lists them)."
    global BLENDER_PORT
    switched = None
    if port is not None or pid is not None:
        # SWITCH, ONLY WHEN ASKED AND ONLY TO ONE THAT ANSWERS: by pid through the add-on's listener file, or by port.
        if pid is not None:
            hit = [r for r in _bl_registered() if r.get("pid") == pid]
            if not hit:
                return {"ok": False, "error": "no running MifBlender registered with pid %s" % pid,
                        "listeners": [_bl_listener_row(r) for r in _bl_registered()]}
            if port is not None and hit[0]["port"] != port:
                return {"ok": False, "error": "pid %s listens on :%s, not :%s - pass one of them" % (pid, hit[0]["port"], port)}
            port = hit[0]["port"]
        probe = _bl_probe(port)
        if "pong" not in probe:
            return {"ok": False, "error": ("nothing that answers as a MifBlender on :%s (%s) - this session stays on :%s. "
                                           "bl_status() with no port lists the live ones." % (port, probe.get("error"), BLENDER_PORT)),
                    "listeners": [_bl_listener_row(r) for r in _bl_registered()]}
        with _BL_LOCK:
            _bl_close()
            switched = {"from": BLENDER_PORT, "to": int(port), "pid": probe.get("pid")}
            BLENDER_PORT = int(port)
    # echo is the one ping parameter no tool could send. It is not decoration: a health probe whose
    # answer might be stale or from a different backend is worth exactly nothing, and a value handed
    # back verbatim is the cheapest proof that the response belongs to THIS request. The Blender
    # suites have used it directly over the socket since they were written; the MCP tool could not.
    res = _blender("ping", echo=echo, _timeout=BLENDER_PROBE_TIMEOUT,
                   _lock_timeout=BLENDER_PROBE_TIMEOUT)
    # THE OP LIST IS 3,300 OF THE REPLY'S ~4,200 CHARACTERS, and this is the call to make before any Blender work, so an
    # agent that did as told carried ~200 op names in every later request of the session (the benchmark, 2026-09-22,
    # docs/TOKEN_OPTIMIZATION.md). The count stays; the names are one flag away, because the add-on's describe_op
    # points at ping for the full list. The add-on's own ping, which the suites and mif_mesh_roundtrip read, is unchanged.
    if not list_ops and isinstance(res, dict) and isinstance(res.get("ops"), list):
        res = dict(res)
        res["opCount"] = len(res.pop("ops"))
    # EVERY SWITCH OF THE GENERATIVE MODE: the add-on's preference and the open file's mode (its ping's
    # allowGenerative and generativeMode - _blender has noted the mode), this server's MIF_GENERATIVE, and the Unreal
    # project's mode when this session has read it. generativeAllowed is true only when none holds the bl_gen_* tools
    # off, as a call needs.
    if isinstance(res, dict) and "pong" in res:
        addon = res.get("allowGenerative", res.get("generativeAllowed"))   # an add-on before 2026-10-01 says only this
        by = {"server": _GENERATIVE, "addon": addon}
        for backend, key in (("blender", "blenderFile"), ("unreal", "unrealProject")):
            state = _unreal_mode_state() if backend == "unreal" else _PM.get(backend)
            if state:
                by[key] = state.get("mode")
        res = dict(res, generativeAllowed=bool(res.get("generativeAllowed")) and _gen_block() is None,
                   generativeAllowedBy=by, handshake=_hs_report("blender"))   # its ping was read in place
    if isinstance(res, dict):
        if switched:
            res = dict(res, switched=switched, port=BLENDER_PORT)
        # THE OTHER LIVE MIFBLENDERS, each handshaken: listed whenever there is one besides the current port, so a second
        # Blender is never a surprise - and never chosen without a port or pid.
        others = [r for r in _bl_registered() if r.get("port") != BLENDER_PORT]
        if others:
            res = dict(res, listeners=[_bl_listener_row(r, _bl_probe(r["port"])) for r in others],
                       listenersNote="other running MifBlenders; bl_status(port=...) or bl_status(pid=...) switches this session")
    return res


@mcp.tool()
def bl_list_previews(preview_id: str = None, offset: int = None, limit: int = None) -> dict:
    "The bulk previews this Blender holds, newest first - id, op, age, expiry, change counts, spent or not. With preview_id, that preview's changes a page at a time (offset, limit up to 1000). Read-only."
    return _blender("list_previews", previewId=preview_id, offset=offset, limit=limit)


@mcp.tool()
def bl_call_log(limit: int = None, disk: bool = None) -> dict:
    "The calls this Blender ran through MifBlender, newest first - op, parameter keys and short values, ok or the error, milliseconds - so you can check what really went through the bridge. From memory by default, the last 64. disk=True reads the journal file instead: every Blender that wrote it and the sessions before a restart, with `unfinished` - a call whose start has no end, the one a crashed Blender died in - and each session's end: closed with its reason, restarted, or died (no shutdown record, and a later Blender came up with its pid). The file is per user (%LOCALAPPDATA%\\MifBridge\\Blender\\journal.jsonl on Windows, `path` in the response) and each call is written before it runs; an add-on preference turns the file off. bl_run_python code is never written, only its size."
    # MifBlender/journal.py writes the file and keeps the ring; ops_scene.op_call_log reads both back. Asked for by a
    # Discord user (2026-09-21) whose agent had sometimes reached Blender by some other route.
    return _blender("call_log", limit=limit, disk=disk)


@mcp.tool()
def bl_scene_info(detail: bool = False, limit: int = None, name_excludes: list = None) -> dict:
    "Summary of the current Blender scene: objectCount, objectsByType, objects[] (names and types; detail:true swaps in a full object_info each), activeObject, selectedObjects, collections, blendFile, and unitSettings. Read-only."
    return _blender("scene_info", detail=detail or None, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_list_objects(object_type: str = "", pattern: str = None,
                    detail: bool = False, limit: int = None, name_excludes: list = None, near: dict = None) -> dict:
    "List objects in the Blender scene with their types. object_type filters to one Blender type ('MESH', 'EMPTY', 'ARMATURE', ...); omit it for everything. pattern filters by NAME and is echoed back in filteredBy, so a caller can tell an empty result from a filter that matched nothing. detail returns more per object. near {point, radius, by ORIGIN|BOUNDS} keeps those within the radius, nearest first, with distance. Read-only."
    # pattern and detail were accepted by the addon and sent by nothing (ops_scene.py:66-67), so
    # listing a busy scene meant retrieving everything and filtering client-side. The addon reports
    # `filteredBy: {type, pattern}` precisely so a caller can tell "nothing matched" from "no filter
    # was applied" - which is unanswerable if the filter cannot be set.
    # detail, NOT `detail or None`. _blender drops only None (v is not None), so `or None` turns an
    # explicit False into ABSENT - harmless while the addon's default is False, and a silent drop the
    # day that default changes. The parameter is `bool = False` and so is never None anyway, which
    # makes the guard pure downside. The `or None` on object_type above is different: that one is a
    # STRING whose empty value genuinely means "no filter".
    return _blender("list_objects", type=object_type or None, pattern=pattern,
                    detail=detail, limit=limit, nameExcludes=name_excludes, near=near)


@mcp.tool()
def bl_object_info(object_name: str) -> dict:
    "Measurements for one Blender object, under an 'object' key: boundsLocalMinBU/MaxBU/SizeBU and boundsLocalSizeUU (the local bbox, already converted - use this one, do NOT multiply dimensionsBU yourself, that folds in object scale),"
    return _blender("object_info", object=object_name)


@mcp.tool()
def bl_list_bones(object_name: str, name_contains: str = "", limit: int = None,
                  name_excludes: list = None, posed: bool = None) -> dict:
    "The REST-POSE bone hierarchy of a Blender ARMATURE object; posed=True adds where each bone stands now (head, tail and matrix, world and armature space)."
    return _blender("list_bones", object=object_name, nameContains=name_contains or None,
                    limit=limit, nameExcludes=name_excludes, posed=posed or None)


@mcp.tool()
def bl_list_shape_keys(object_name: str, limit: int = None, name_excludes: list = None) -> dict:
    "Shape keys on a Blender mesh object - Blender's name for what Unreal calls morph targets (compare against UE's list_morph_targets on the same character)."
    return _blender("list_shape_keys", object=object_name, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_list_vertex_groups(object_name: str, limit: int = None, name_excludes: list = None,
                          region: dict = None) -> dict:
    "Vertex groups on a Blender mesh (a skinned mesh's, one per deforming bone, named after it); region {box | sphere} adds each one's mean weight there."
    return _blender("list_vertex_groups", object=object_name, limit=limit, nameExcludes=name_excludes,
                    region=region)


@mcp.tool()
def bl_list_modifiers(object_name: str, limit: int = None, name_excludes: list = None,
                      rna: bool = None) -> dict:
    "The modifier stack on a Blender object - a mesh, curve, text, surface or lattice - in EVALUATION ORDER (top to bottom in the Modifier Properties panel, which is also application order) - answers 'what will bl_export_mesh actually produce' before spending an export to find out. rna=True adds every readable RNA property of each modifier by identifier - what bl_add_modifier and bl_set_modifier can write. A FLUID's rna also carries its live role's settings (domain_settings, flow_settings or effector_settings) - where bl_set_modifier writes them - without the simulation grids, whose getter crashes Blender on a domain that has not simulated."
    return _blender("list_modifiers", object=object_name, limit=limit, nameExcludes=name_excludes, rna=rna)


@mcp.tool()
def bl_import_mesh(file: str, clear_scene: bool = True,
                   use_custom_normals: bool = None,
                   import_animation: bool = None, clear_scope: str = None) -> dict:
    "Import an FBX or glTF/GLB file into Blender and report what arrived. Those two only - OBJ and everything else are hard-refused, because FBX and glTF are the formats whose axis and unit round trip is verified: glTF because its SPEC fixes the convention (+Y up, metres) and FBX because it carries its own metadata in the file. UE's OBJ exporter swaps Y/Z, de-indexes and writes no normals, and the file cannot tell you it did. NOTE for glTF: it has no shared-vertex-with-split-normals concept, so vertices are de-indexed per corner and a cube's 8 come back as 24 - the geometry is identical, so compare DIMENSIONS rather than vertex counts across a round trip. use_custom_normals is an FBX option and is refused for glTF rather than silently ignored. ANIMATION: the FBX importer is told use_anim:false by default, so an animated FBX arrives as a static mesh - pass import_animation:true to keep the action. glTF has NO animation option on any supported build and always imports them, so import_animation:false is refused for glTF rather than accepted and ignored. The response reports animationImported and actionsCreated, measured from what arrived rather than from what was asked for. CLEAR_SCENE DEFAULTS TRUE HERE and deletes the objects of the scene being imported into (clear_scope 'scene', the default) - an object also linked into another scene is kept and named in sharedKept; clear_scope 'file' deletes EVERY object in the file, every scene's. It runs after every parameter has been checked and just before the import; pass clear_scene=False to import into the scene as it is. The response lists what it deleted: clearedCount, clearedObjects, and clearedFromOtherScenes for the ones that were not in the active scene. Rigid bodies leave the physics world before they are deleted, as bl_clear_scene's do, and a rigid body bake the clear leaves stale is freed and named (freedRigidBodyBake, rigidBodyBakeNote)."
    # use_custom_normals reads the FBX's authored normals instead of letting Blender recompute
    # them. The addon has always accepted it (ops_mesh.py:177-178) and nothing sent it, so a mesh
    # whose normals were authored deliberately - hard edges, a normal-map bake target - came in with
    # Blender's own. The export half of this pair gained useTspace the same night, for the same
    # reason: what survives the round trip is what Unreal ends up rendering.
    # None means unset; _blender drops it and the addon's default stands.
    # import_animation reaches the FBX importer's use_anim, which was PINNED False with no
    # parameter able to touch it until 2026-09-04 - so the same call dropped every FBX
    # animation and kept every glTF one, silently. Measured on 4.4: an animated cube written
    # to both came back with no action from FBX and with one from glTF.
    return _blender("import_mesh", file=file, clearScene=clear_scene,
                    useCustomNormals=use_custom_normals,
                    importAnimation=import_animation, clearScope=clear_scope)


@mcp.tool()
def bl_select_edges(object_name: str, selector: dict = None, max_reported: int = 512, seed_edge: int = None,
                    loop: bool = None, ring: bool = None, trait: str = None, stop_at: list = None) -> dict:
    "Resolve an edge selector against a mesh and report what it matches, WITHOUT modifying anything. seed_edge with loop=true or ring=true walks the edge loop or ring through one edge instead. trait NON_MANIFOLD, INTERIOR, LOOSE, BOUNDARY or DEGENERATE; max_reported 0 lists all."
    if trait is not None and selector is None and seed_edge is None and stop_at is None:
        return _blender("select_edges", object=object_name, trait=trait, maxReported=max_reported)
    if seed_edge is not None or loop or ring or stop_at is not None:
        if selector is not None:
            return {"ok": False, "error": "seed_edge selects on its own - drop selector."}
        return _blender("select_edges", object=object_name, seedEdge=seed_edge, loop=loop, ring=ring,
                        stopAt=stop_at, maxReported=max_reported)
    try:
        sel = _bl_selector(selector)
    except _MifToolError as exc:
        return {"ok": False, "error": str(exc)}
    return _blender("select_edges", object=object_name,
                    boundaryOnly=sel["boundaryOnly"], axis=sel["axis"], side=sel["side"],
                    tolerance=sel["tolerance"], minAngleDeg=sel["minAngleDeg"],
                    maxAngleDeg=sel["maxAngleDeg"], edgeIndices=sel["edgeIndices"],
                    allEdges=sel["allEdges"], maxReported=max_reported, trait=trait)


@mcp.tool()
def bl_transform_selection(object_name: str = None, component_type: str = None, indices: list = None,
                           box_min: list = None, box_max: list = None,
                           operation: str = "TRANSLATE", vector: list = None,
                           pivot: list = None, space: str = "LOCAL",
                           dry_run: bool = False, max_reported: int = 512,
                           shape_key: str = None, angle: float = None, axis: str | list = None,
                           falloff: dict = None, mask: str | dict = None, distance: float = None,
                           factor: float = None, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Translate, scale, rotate (angle about axis X, Y, Z or [x,y,z]), SHRINK_FATTEN (distance along normals) or TO_SPHERE (factor 0-1) selected vertices, edges or faces of a Blender mesh in place, with optional proportional falloff and a vertex-group mask. Edge and face selections transform the union of their vertices. Select with explicit indices, or use box_min/box_max for vertices. box_min, box_max, vector and pivot are interpreted in the requested local or world space. dry_run predicts and reports the exact affected coordinates without writing the mesh: on a dry run `changed` is false and `wouldChange` carries the prediction. WITH SHAPE KEYS and no shape_key, the edit moves the base mesh and the reference key, and every relative key by the same offset (measured on every build; shapeKeysNote says so). shape_key reshapes ONE key relative to its own vertices instead - selection, pivot and bounds are measured on that key, only it is written, and the reference key is refused. ROTATE takes angle (degrees) and axis (X, Y, Z or [x,y,z]) about pivot. falloff {type, radius, connectedOnly, projectAxis, seed} moves the vertices within radius of the selection too, by Blender's proportional-editing curves; mask (a vertex group, or {group, invert}) scales each vertex's move by its weight. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, component_type=component_type,
                                        indices=indices, box_min=box_min, box_max=box_max, vector=vector, pivot=pivot,
                                        dry_run=dry_run, shape_key=shape_key, angle=angle, axis=axis, falloff=falloff,
                                        mask=mask, distance=distance, factor=factor,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("transform_selection", previewId=preview_id))
    return _blender("transform_selection", object=object_name, componentType=component_type,
                    indices=indices, boxMin=box_min, boxMax=box_max,
                    operation=operation, vector=vector, pivot=pivot, space=space,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds, maxReported=max_reported,
                    shapeKey=shape_key,
                    angle=angle, axis=axis, falloff=falloff, mask=mask, distance=distance, factor=factor)


@mcp.tool()
def bl_rename_object(object: str, to: str, rename_data: bool = None,
                     allow_collision: bool = None) -> dict:
    "Rename a Blender object, REFUSING a name collision instead of letting Blender resolve it - because Blender resolves it in OPPOSITE DIRECTIONS across the 3.6/4.x line. Measured: with an existing 'Alpha', on 4.2/4.4/5.0 the incumbent keeps it and the renamer becomes 'Alpha.001'; on 3.6 whichever of the two names sorts first keeps the name - 'Zulu' renamed to 'Alpha' TAKES it and silently renames the INCUMBENT to 'Alpha.001', while 'Aardvark' renamed to 'Mango' becomes 'Mango.001' (3.6.23). So the same script renames a different object depending on the Blender running it, and on 3.6 it corrupts an object nobody asked it to touch. THE DATA NAME DOES NOT FOLLOW - obj.data.name keeps its old value, and a mesh named after the old object survives into the FBX - so rename_data defaults on. Pointers survive a rename (modifier targets, constraint targets, driver variables); STRING references do not, such as an exporter convention like UCX_<name>_00. The addon had bl_rename_bones and no way to rename an object."
    return _blender("rename_object", object=object, to=to, renameData=rename_data,
                    allowCollision=allow_collision)


@mcp.tool()
def bl_set_vertex_color(object: str, name: str = None, color: list = None, domain: str = None,
                        data_type: str = None, faces: list = None, make_active: bool = None,
                        make_render: bool = None, vertex_values: Any = None, falloff: Any = None,
                        base_color: Any = None, blend: str = None) -> dict:
    "Write a Blender colour attribute - what games use for masks, wear and blend weights, and which the addon could not touch at all. Uses color_attributes, which exists on every supported build, rather than the legacy vertex_colors. Defaults to BYTE_COLOR on the CORNER domain: that is what Blender's own legacy call produced and what survives an FBX round trip. THE NAME CAN FAIL SILENTLY - color_attributes.new() with an over-long name TRUNCATES on 3.6 and RETURNS None on 4.2/4.4/5.0, neither raising - so the length is refused up front and the None guarded anyway. The stored colour is READ BACK rather than echoed: BYTE_COLOR quantises to 8 bits per channel, so asking for 0.25 stores 0.250158, and `quantised` says so instead of pretending the request was the result. A value per vertex makes a smooth mask: vertex_values [0-1, ...] (one per vertex in index order) or [[index, value], ...] (only those vertices) or falloff {points: [[x,y,z], ...] world, radius, inner (default 0), type SMOOTH | SPHERE | ROOT | INVERSE_SQUARE | SHARP | LINEAR | CONSTANT} - 1 within inner, 0 at radius - mixes base_color (default black) toward color by each vertex's value, and on CORNER every corner takes its vertex's value, so the mask has no steps at face borders; blend REPLACE (default), OVER (mixed from the stored value) or MAX (the larger per channel), and the reply's values gives count, min, max, mean and aboveZero."
    return _blender("set_vertex_color", object=object, name=name, color=color, domain=domain,
                    dataType=data_type, faces=faces, makeActive=make_active,
                    makeRender=make_render, vertexValues=vertex_values, falloff=falloff, baseColor=base_color,
                    blend=blend)


@mcp.tool()
def bl_mesh_stats(object: str, evaluated: bool = None, attribute_stats: list | bool = None,
                  frame: int = None, components: bool = None, attribute_values: dict = None,
                  allow_simulation: bool = None, instances: Any = None) -> dict:
    "Vertex/edge/face/triangle counts and a world bounding box for a Blender mesh, plus the EVALUATED counts after modifiers - a Subsurf mesh has a base count and a rendered count differing by an order of magnitude, and both are reported rather than one chosen for you. THE BOX IS COMPUTED FROM VERTICES, NEVER FROM obj.bound_box, which is CACHED AND STALE: measured on all four builds, move a vertex to z=50 and bound_box still reports the old extent - mesh.update() does not refresh it, only view_layer.update() does. A box read off bound_box after any edit is quietly wrong, which is the worst kind of wrong for a number people position things against. Also guards the case where evaluated_get silently returns the UNevaluated object, which happens when the object is not in the active view layer's depsgraph. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2). ATTRIBUTES: `attributes` lists the base mesh's named attributes and evaluated.attributes the evaluated mesh's, with evaluated.attributesAddedByModifiers naming what the stack added - an Ocean's foam layer, a Store Named Attribute's output - which is what a material's Attribute node reads BY NAME. A CURVE, TEXT or SURFACE object is measured by its evaluated mesh (base is null - splines are not vertices), so what a bevel, an extrude, a resolution or a node group builds on a curve is readable. attribute_stats - a LIST of names, ['foam', ...], or true for every named attribute - reads what those attributes HOLD - min, max and mean per component (colours linear, as a shader reads them), off the evaluated mesh when there is one (attributeStatsFrom says which): a name that exists can still hold nothing useful, as an Ocean's foam under a JONSWAP spectrum does (0.093-0.105 everywhere, so a ColorRamp starting at 0.35 renders black). An unknown name is refused naming the attributes that exist. colorAttributes lists the colour attributes, and activeColorAttribute and renderColorAttribute say which one is active and which one renders - bl_set_vertex_color's make_active and make_render, which can differ. With frame=N it measures at that frame and puts the scene frame back; a cloth, soft body or rigid body cache that is unbaked, outdated or ends before N is refused rather than read. A mesh with shape keys beyond its basis is evaluated like a modifier stack - the keys deform with no modifier, and the evaluated box is where the keyed mesh is. With instances=true it lists each instance the object draws, its source and world transform."
    return _blender("mesh_stats", object=object, evaluated=evaluated, attributeStats=attribute_stats,
                    frame=frame, components=components, attributeValues=attribute_values,
                    allowSimulation=allow_simulation, instances=instances)


@mcp.tool()
def bl_mesh_quality(object: str = None, uv_layer: str = None, texel_density_for=None, conventions: bool = None,
                    kind: str = None, module_cm: float = None, boundary_loops: bool = None, bone: str = None,
                    armature: str = None, poles: bool = None, joint_rings: dict = None,
                    facing_against=None, flipped_against: str = None, collection: str = None, thickness=None,
                    thickness_min_cm: float = None, game_ready: bool | dict = None) -> dict:
    "Measure the objectively checkable things that get an asset REJECTED from a store: ngons, non-manifold edges, loose verts and edges, zero-area faces, unapplied scale and rotation, empty material slots, UV loops outside the 0-1 tile, and texel density spread across faces. Returns every NUMBER plus `concerns`, which lists only defects with a defensible objective threshold - it says NOTHING about whether the asset looks good and does not pretend to. READ notMeasured BEFORE TRUSTING A CLEAN REPORT: a mesh with no UV layer gets no UV verdict at all, which is an absence of data rather than a pass, and an empty concerns list beside a skipped check is not a clean bill of health. Measures the mesh AS AUTHORED, not the evaluated result, because that is what exports and what a store inspects - a subdivision modifier would make every ngon and density figure describe something the file does not contain. texel_density_for takes a size or [w, h]. game_ready {triangleBudget, allowNgons, uvOneTile, allowFloating, allowWaste} adds gameReady: pass or fail, each reason a measured number with its limit and the call that fixes it, or `choices` where only you can tell: floating parts, and triangles over the size class's number, the budget when no triangleBudget is given. Parts 2% or more of the longest side away, slivers under 10 micrometers, near-zero tangents and wasted faces fail it: 90% flat with no painted colors, or half out of any ray's reach."
    return _blender("mesh_quality", object=object, uvLayer=uv_layer,
                    texelDensityFor=texel_density_for, conventions=conventions, kind=kind, module=module_cm,
                    boundaryLoops=boundary_loops, bone=bone, armature=armature, poles=poles, jointRings=joint_rings,
                    facingAgainst=facing_against, flippedAgainst=flipped_against, collection=collection,
                    thickness=thickness, thicknessMinCm=thickness_min_cm, gameReady=game_ready)


@mcp.tool()
def bl_recipe_game_ready(object: str, apply_transform: bool = None, unwrap: bool = None,
                         force_unwrap: bool = None, uv_method: str = None, full: bool = None, kind: str = None,
                         pivot: str = None, lightmap: bool = None, collision: str = None, per_part: bool = None,
                         max_hulls: int = None, ue_name: str = None, module_cm: float = None) -> dict:
    "THE BORING PIPELINE, BANKED: apply transforms, ensure a UV layer, then MEASURE the result with the same battery bl_mesh_quality exposes. Composes existing ops rather than reimplementing them, so every guard they carry - shared mesh data, linked libraries, edit mode - still applies; it does nothing you could not do by hand, it does it in the right ORDER and then checks. Order is the whole value: unwrapping before the scale is baked lays UVs out against the unscaled mesh, so a non-uniformly scaled object ends up with a texel density that is wrong the moment scale is applied. forceUnwrap defaults FALSE because re-unwrapping a layout somebody made by hand is destructive and silent. If a step fails half way it reports leftBehind saying what already happened and is NOT undone - a partial composition that stayed quiet would be the worst outcome. Read quality.notMeasured before treating concernCount:0 as a clean bill of health. THE FULL PASS for Unreal (full=true, or each key alone): pivot by kind, a Lightmap UV layer last, UCX_ hulls, an SM_ name, then bl_mesh_quality's conventions. Call mif_help(\"bl_recipe_game_ready\") first."
    return _blender("recipe_game_ready", object=object, applyTransform=apply_transform,
                    unwrap=unwrap, forceUnwrap=force_unwrap, uvMethod=uv_method, full=full, kind=kind, pivot=pivot,
                    lightmap=lightmap, collision=collision, perPart=per_part, maxHulls=max_hulls, ueName=ue_name,
                    module=module_cm)


@mcp.tool()
def bl_create_collision_hull(object: str, index: int = None, name: str = None,
                             prefix: str = None, world_space: bool = None,
                             max_vertices: int = None, collection: str = None, shape: str = None,
                             per_part: bool = None, max_hulls: int = None, vertex_group: str = None,
                             vertex_indices: list = None, selected: bool = None) -> dict:
    "Build a convex hull collision MESH named for Unreal's UCX_<RenderMesh>_## convention. NOT bl_add_collision, which is a rigid-body PHYSICS setting evaluated by Blender's own simulation - same word, unrelated job, and the addon had that one and not this. THE RECIPE EVERY TUTORIAL GIVES IS WRONG AND LOOKS RIGHT: bm.from_mesh(source) then convex_hull then delete geom_interior returns, on Suzanne, the SAME 66 vertices as a correct hull plus 51 non-manifold edges, 4 boundary edges, Euler 16 instead of 2, 22 convexity violations and an 8% inflated volume, with nothing raised - measured on 4.4.0 and 5.0.1. This builds from POINTS ONLY and then AUDITS: closed, manifold, no loose verts, Euler 2, positive volume, and genuine convexity tested by checking every vertex against every face plane. A hull failing any of those is DELETED and refused, because a broken hull imports, looks like collision, and things fall through it. Degenerate input produces silent garbage rather than an exception - collinear points give an empty mesh, coplanar points a zero-volume sheet - which is why the audit refuses instead of warning. max_vertices simplifies to a hull that still ENCLOSES every point - the exact hull's own face planes, each added where the simplified hull sticks out most - never by decimating (a decimated hull is no longer convex: 24 violations at ratio 0.1) or by merging points (merged points lie inside, so the hull shrank and the mesh poked through it - 12 cm on a bl_build_rock rock at 40 vertices). Each hulls[] row's sourceOutsideM says how far any point lies outside. A name collision is REFUSED, not uniquified - Blender's .001 suffix survives FBX export and makes the engine parse the base name wrongly. The UCX naming convention itself is engine-side and is NOT verified here."
    return _blender("create_collision_hull", object=object, index=index, name=name, prefix=prefix,
                    worldSpace=world_space, maxVertices=max_vertices, collection=collection, shape=shape,
                    perPart=per_part, maxHulls=max_hulls, vertexGroup=vertex_group, vertexIndices=vertex_indices,
                    selected=selected)


@mcp.tool()
def bl_build_tree(name: str = None, preset: str = None, seed: int = None, height: float = None,
                  trunk_radius: float = None, levels: int = None, crown_shape: str = None,
                  leaf_shape: str = None, leaf_size: float = None, leaf_count: int = None,
                  leaf_color: list = None, leaf_color2: list = None, bark_color: list = None,
                  collision: bool = None, location: list = None, rotation: list = None,
                  collection: str = None, leaf_clusters: bool = None, leaves_per_cluster: int = None) -> dict:
    "Build a tree in Blender with real geometry leaves - one object with a bark slot and a leaf slot, its pivot at the trunk's base, the same tree for the same seed. Call mif_help(\"bl_build_tree\") first."
    return _blender("build_tree", name=name, preset=preset, seed=seed, height=height, trunkRadius=trunk_radius,
                    levels=levels, crownShape=crown_shape, leafShape=leaf_shape, leafSize=leaf_size,
                    leafCount=leaf_count, leafColor=leaf_color, leafColor2=leaf_color2, barkColor=bark_color,
                    collision=collision, location=location, rotation=rotation, collection=collection,
                    leafClusters=leaf_clusters, leavesPerCluster=leaves_per_cluster)

@mcp.tool()
def bl_build_kit_piece(kind: str = None, opening: str = None, module: float = None, length: float = None,
                       height: float = None, width: float = None, thickness: float = None,
                       opening_width: float = None, opening_height: float = None, sill_height: float = None,
                       opening_offset: float = None, pivot: str = None, bevel: float = None,
                       bevel_segments: int = None, trim: bool = None, uv_tile_cm: float = None, color: list = None,
                       material: str = None, trim_color: list = None, trim_material: str = None,
                       grid_cm: float = None, collision: bool = None, name: str = None, location: list = None,
                       rotation: list = None, collection: str = None, slat_pitch: float = None,
                       hood_radius: float = None, hardware: bool = None, louver: bool = None,
                       min_thickness_cm: float = None) -> dict:
    "Build a modular kit piece in Blender - a wall (with a door or window opening), a floor, a pillar, or a roll-up shutter or door leaf for a wall's opening - on a module grid, with bevels, trim, a lightmap UV layer and UCX_ hulls. Call mif_help(\"bl_build_kit_piece\") first."
    return _blender("build_kit_piece", kind=kind, opening=opening, module=module, length=length, height=height,
                    width=width, thickness=thickness, openingWidth=opening_width, openingHeight=opening_height,
                    sillHeight=sill_height, openingOffset=opening_offset, pivot=pivot, bevel=bevel,
                    bevelSegments=bevel_segments, trim=trim, uvTileCm=uv_tile_cm, color=color, material=material,
                    trimColor=trim_color, trimMaterial=trim_material, gridCm=grid_cm, collision=collision,
                    name=name, location=location, rotation=rotation, collection=collection, slatPitch=slat_pitch,
                    hoodRadius=hood_radius, hardware=hardware, louver=louver, minThicknessCm=min_thickness_cm)

@mcp.tool()
def bl_set_vertex_color_by_rule(object: str, rule: str, name: str = None, channel: str = None, range: list = None,
                                invert: bool = None, combine: str = None, seed: int = None, element: str = None,
                                distance: float = None, rays: int = None, space: str = None,
                                make_active: bool = None, dirt: dict = None, group: str = None,
                                adjust: dict = None) -> dict:
    "Write a 0-1 mask into a Blender mesh's color attribute by a rule - up (faces that look up), height, cavity or random per part - for a material to blend moss, dirt, grime or variation with. Also rules dirt, fromWeight and adjust. Call mif_help(\"bl_set_vertex_color_by_rule\") first."
    return _blender("set_vertex_color_by_rule", object=object, rule=rule, name=name, channel=channel, range=range,
                    invert=invert, combine=combine, seed=seed, element=element, distance=distance, rays=rays,
                    space=space, makeActive=make_active, dirt=dirt, group=group, adjust=adjust)

@mcp.tool()
def bl_build_lod_chain(object: str, ratios: list = None, file: str = None, overwrite: bool = None) -> dict:
    "Make hand LODs for a Blender mesh that Unreal's FBX importer reads as its LODs - decimated copies under an Empty whose fbx_type is LodGroup, the object itself LOD0 - and optionally write them to an FBX. Call mif_help(\"bl_build_lod_chain\") first."
    return _blender("build_lod_chain", object=object, ratios=ratios, file=file, overwrite=overwrite)


@mcp.tool()
def bl_build_rock(name: str = None, kind: str = None, seed: int = None, size: list = None, triangles: int = None,
                  sink: float = None, weathering: float = None, strata: bool = None, stone_color: list = None,
                  stone_color2: list = None, moss: float = None, collision: bool = None, location: list = None,
                  rotation: list = None, collection: str = None) -> dict:
    "Build a rock in Blender - a boulder, a slab or a cliff face - as one mesh with its own stone material, sized in centimeters, its pivot at the bottom center, the same rock for the same seed. Call mif_help(\"bl_build_rock\") first."
    return _blender("build_rock", name=name, kind=kind, seed=seed, size=size, triangles=triangles, sink=sink,
                    weathering=weathering, strata=strata, stoneColor=stone_color, stoneColor2=stone_color2,
                    moss=moss, collision=collision, location=location, rotation=rotation, collection=collection)


@mcp.tool()
def bl_build_plant(name: str = None, kind: str = None, seed: int = None, height: float = None, width: float = None,
                   count: int = None, bare: float = None, weeds: float = None, weed_mix: list = None,
                   flowers: float = None, length: float = None, ends: str = None, shag: float = None,
                   leaf_length: float = None, leaf_color: list = None, leaf_color2: list = None,
                   location: list = None, rotation: list = None, collection: str = None) -> dict:
    "Build ground cover in Blender - a grass tuft, a fern, a broadleaf plant, a bush, a tile of lawn or a piece of clipped hedge - as one mesh of real geometry blades and leaves, sized in centimeters, its pivot at the base, the same plant for the same seed. Call mif_help(\"bl_build_plant\") first."
    return _blender("build_plant", name=name, kind=kind, seed=seed, height=height, width=width, count=count,
                    bare=bare, weeds=weeds, weedMix=weed_mix, flowers=flowers, length=length, ends=ends, shag=shag,
                    leafLength=leaf_length, leafColor=leaf_color, leafColor2=leaf_color2, location=location,
                    rotation=rotation, collection=collection)


@mcp.tool()
def bl_uv_unwrap(object_name: str = None, method: str = "SMART", uv_layer: str = None,
                 angle_limit_deg: float = None, island_margin: float = 0.02,
                 replace: bool = False, dry_run: bool = False,
                 mark_seams: dict = None, clear_seams: bool = False,
                 uv_pack: bool = False, pack_margin: float = None,
                 uv_transform: dict = None, correct_aspect: bool = None, meters_per_tile: float = None,
                 texel_density: float = None, texture_size: int = None, box_space: str = None,
                 fill_holes: bool = None, active_face: int = None, keep_bounds: bool = None,
                 preview_id: str = None, preview_ttl_seconds: int = None, seam_rules: list = None,
                 pack_fill: float = None) -> dict:
    "Generate a UV layer on a Blender mesh - and THIS IS ALSO HOW YOU MAKE A SECOND UV CHANNEL FOR AN UNREAL LIGHTMAP: method='LIGHTMAP' with uv_layer='Lightmap' creates the channel, packs non-overlapping islands into 0-1, leaves the base UVs untouched and leaves active_render on the base colour layer. Verified on Blender 3.6, 4.2, 4.4 and 5.0. THE DEFAULT FAILURE OF A SECOND CHANNEL IS WRITING INTO THE FIRST ONE - uv_layers.new() does not make the new layer active on any Blender, and every UV operator writes to the ACTIVE layer, so a lightmap pass silently repacks the base colour UVs while the layer you asked for stays empty, and it is found at bake time. This op sets active correctly AND PROVES IT: otherLayersUnchanged and layersClobbered are measured by fingerprinting every other layer before and after, not asserted - activeLayer only reports what was intended and cannot disagree with itself. activeRenderLayer says which layer the renderer samples for textures, which is not the same as the active one. THREE METHODS: SMART (cuts its own seams, for a prop that has none), LIGHTMAP (the above), ANGLE (respects seams you marked - warns if the mesh has none, because it then flattens as one unusable island). With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since. With seam_rules it marks seams by rule before an ANGLE, CONFORMAL or MINIMUM_STRETCH unwrap (refused with the methods that cut their own or ignore seams): {rule: head, bone (default the one named head), armature (default the mesh's Armature modifier or parent), back (default [0,1,0])} rings the head where its skin weights meet the neck's - under the chin, round the nape - and runs a seam up the back of the skull to the crown; {rule: garment, prefer: inner | back | [x,y,z], preferWeight, midPlane {point, normal}, splitAtMidPlane} cuts every piece open from loop to loop along its preferred side (inner: the inside of a sleeve or leg) until it is one disc, gives a closed piece one slit, and with splitAtMidPlane halves it at the mid-plane first. With pack_fill (0-1) it packs to a fill target: every island scaled to one texel density, then packed at smaller margins until the layout uses that share of the tile, keeping the fullest; packFill gives target, achieved, met and every attempt. Every unwrap's layout gives fill, islands and texel density (pixels per metre at 1024) with the spread between islands - on a test character's head the automatic unwrap was 70 islands at 56% fill, and the head and garment rules 4 islands at 75%."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, uv_layer=uv_layer, seam_rules=seam_rules,
                                        pack_fill=pack_fill,
                                        angle_limit_deg=angle_limit_deg, replace=replace, dry_run=dry_run,
                                        mark_seams=mark_seams, clear_seams=clear_seams, uv_pack=uv_pack,
                                        pack_margin=pack_margin, uv_transform=uv_transform,
                                        correct_aspect=correct_aspect, meters_per_tile=meters_per_tile,
                                        texel_density=texel_density, texture_size=texture_size, box_space=box_space,
                                        fill_holes=fill_holes, active_face=active_face, keep_bounds=keep_bounds,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("uv_unwrap", previewId=preview_id))
    return _blender("uv_unwrap", object=object_name, method=method, uvLayer=uv_layer,
                    angleLimitDeg=angle_limit_deg, islandMargin=island_margin,
                    replace=replace, dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds,
                    markSeams=mark_seams, clearSeams=clear_seams or None,
                    uvPack=uv_pack or None, packMargin=pack_margin,
                    uvTransform=uv_transform,
                    # correctAspect scales the unwrap by the material's texture aspect ratio. The
                    # addon takes it (ops_mesh.py:1502) and nothing sent it, so a non-square texture
                    # got a UV layout stretched against it with no way to ask otherwise.
                    correctAspect=correct_aspect,
                    # method BOX: the world size of one tile (or a density on a texture size), and the space it
                    # projects in - OBJECT, or WORLD with the location added.
                    metersPerTile=meters_per_tile, texelDensity=texel_density, textureSize=texture_size,
                    boxSpace=box_space, fillHoles=fill_holes, activeFace=active_face, keepBounds=keep_bounds,
                    seamRules=seam_rules, packFill=pack_fill)


@mcp.tool()
def bl_decimate_mesh(object_name: str = None, ratio: float = None, target_tris: int = None,
                     mode: str = "COLLAPSE", angle_limit: float = None,
                     iterations: int = None, dry_run: bool = False, preview_id: str = None,
                     preview_ttl_seconds: int = None, keep_ngons: bool = None) -> dict:
    "Reduce a mesh's triangle count in Blender - the LOD edit a game pipeline wants most, and where analyze_skeletal_split's triangle counts finally have somewhere to go. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since. A DISSOLVE's merged n-gons are split into triangles unless keep_ngons."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, ratio=ratio, target_tris=target_tris,
                                        angle_limit=angle_limit, iterations=iterations, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("decimate_mesh", previewId=preview_id))
    return _blender("decimate_mesh", object=object_name, ratio=ratio, targetTris=target_tris,
                    mode=mode, angleLimit=angle_limit, iterations=iterations, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds, keepNgons=keep_ngons)


@mcp.tool()
def bl_apply_transform(object_name: str, location: bool = None, rotation: bool = None,
                       scale: bool = None, fix_normals: bool = None, with_meshes: bool = None) -> dict:
    "Bake an object's loc/rot/scale into its MESH DATA in Blender. Name a channel and only named ones apply; name none and all three do. An ARMATURE is applied with its skinned meshes (with_meshes, default true)."
    return _blender("apply_transform", object=object_name, location=location,
                    rotation=rotation, scale=scale, fixNormals=fix_normals, withMeshes=with_meshes)


@mcp.tool()
def bl_set_origin(object_name: str, mode: str = None, location: list = None, corner: str = None,
                  side: str = None) -> dict:
    "Move a Blender object's ORIGIN without moving its geometry in the world. The origin is what Unreal rotates and places the mesh around, it is baked into the FBX, and it CANNOT be fixed on the Unreal side - so it has to be right before export. mode defaults to 'geometry'; a location with no mode means 'point' (the response says modeInferred), and a location beside any other mode is refused rather than ignored."
    return _blender("set_origin", object=object_name, mode=mode, location=location, corner=corner, side=side)


@mcp.tool()
def bl_clean_mesh(object_name: str, merge_distance: float = None, remove_loose: bool = False,
                  dissolve_degenerate: bool | float = False, triangulate: bool = False,
                  recalc_normals: bool = False, force: bool = False, space: str = None,
                  min_face_area: float = None, remove_unused_slots: bool = None, fill_holes_up_to: int = None,
                  despike_distance: float = None, near_boundary_rings: int = None, smooth_loop_near: list = None,
                  smooth_loop_iterations: int = None, quadrify: dict = None, flip_normals=None,
                  relax_crumples=None, open_collapsed_uvs: bool | str = None, keep_custom_normals: bool = None,
                  split_non_manifold: bool = None) -> dict:
    "The cleanup pass an imported or edited mesh needs before it goes back to Unreal. FIVE INDEPENDENT STEPS, run in the only correct order: merge first (so loose/degenerate detection sees merged topology), then remove loose, then dissolve"
    return _blender("clean_mesh", object=object_name, mergeDistance=merge_distance,
                    removeLoose=remove_loose, dissolveDegenerate=dissolve_degenerate,
                    triangulate=triangulate, recalcNormals=recalc_normals, force=force, space=space,
                    minFaceArea=min_face_area, removeUnusedSlots=remove_unused_slots, fillHolesUpTo=fill_holes_up_to,
                    despikeDistance=despike_distance, nearBoundaryRings=near_boundary_rings,
                    smoothLoopNear=smooth_loop_near, smoothLoopIterations=smooth_loop_iterations, quadrify=quadrify,
                    flipNormals=flip_normals, relaxCrumples=relax_crumples, openCollapsedUvs=open_collapsed_uvs,
                    keepCustomNormals=keep_custom_normals, splitNonManifold=split_non_manifold)


@mcp.tool()
def bl_remove_mesh_attributes(object: str, names: list = None, kinds: list = None) -> dict:
    "Take attributes off a Blender mesh: names (bl_mesh_stats lists them; a UV map too) or kinds CUSTOM_NORMALS, SHARP_EDGE, SHARP_FACE (every face then shades smooth), COLOR. Replies what was removed, what was not there, and what remains."
    return _blender("remove_mesh_attributes", object=object, names=names, kinds=kinds)


@mcp.tool()
def bl_transfer_face_materials(source: str, target: str, max_distance: float = None) -> dict:
    "Give each face of a Blender target mesh the material of the nearest face of a source mesh, in world space - a high poly made from a low poly, or a remeshed garment, gets its panel materials back. Slots are matched by material name and added when missing; max_distance leaves far faces alone."
    return _blender("transfer_face_materials", source=source, target=target, maxDistance=max_distance)


@mcp.tool()
def bl_set_bone_rest(object: str, bones: list = None, apply_pose_as_rest: bool = None, mirror: dict = None,
                     recalc_roll: dict = None) -> dict:
    "Move existing Blender bones' REST positions - bones [{name, head?, tail?, roll?}] in the armature's space, or apply_pose_as_rest - read back from the bones. Skinned meshes are not updated and are named; a proportion variant that keeps the bone names."
    return _blender("set_bone_rest", object=object, bones=bones, applyPoseAsRest=apply_pose_as_rest, mirror=mirror,
                    recalcRoll=recalc_roll)


@mcp.tool()
def bl_set_bone_properties(object: str = None, bones: list = None, name_prefix: str = None, all: bool = None,
                           use_deform: bool = None, parent: str = None, clear_parent: bool = None,
                           connected: bool = None, bbone_segments: int = None, inherit_scale: str = None,
                           dry_run: bool = None, custom_shape: str = None,
                           custom_shape_translation: list = None, custom_shape_rotation: list = None,
                           custom_shape_scale: list = None, color: dict = None, color_palette: str = None,
                           preview_id: str = None) -> dict:
    "Write existing Blender bones' properties - use_deform, parent (or clear_parent), connected, bbone_segments, inherit_scale - on bones by name, name_prefix or all. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    # NO preview_ttl_seconds HERE, unlike the other previewing tools: blender_animation sits at test_mcp_packs P110's
    # 27,750-character ceiling, and the parameter would cross it. The preview keeps the add-on's 600 s default.
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, bones=bones, name_prefix=name_prefix, all=all,
                                        use_deform=use_deform, parent=parent, clear_parent=clear_parent,
                                        connected=connected, bbone_segments=bbone_segments,
                                        inherit_scale=inherit_scale, dry_run=dry_run, custom_shape=custom_shape,
                                        custom_shape_translation=custom_shape_translation,
                                        custom_shape_rotation=custom_shape_rotation,
                                        custom_shape_scale=custom_shape_scale, color=color,
                                        color_palette=color_palette)
                or _blender("set_bone_properties", previewId=preview_id))
    return _blender("set_bone_properties", object=object, bones=bones, namePrefix=name_prefix, all=all,
                    useDeform=use_deform, parent=parent, clearParent=clear_parent, connected=connected,
                    bboneSegments=bbone_segments, inheritScale=inherit_scale, dryRun=dry_run or None,
                    customShape=custom_shape, customShapeTranslation=custom_shape_translation,
                    customShapeRotation=custom_shape_rotation, customShapeScale=custom_shape_scale, color=color,
                    colorPalette=color_palette)


@mcp.tool()
def bl_add_control_rig(object: str, preset: str = None, chains: list = None, switch_property: str = None,
                       switch_direction: str = None, fk_prefix: str = None, ik_prefix: str = None,
                       target_parent: str = None) -> dict:
    "Add FK and IK control bones over an existing Blender deform skeleton, its names kept: per limb an FK copy, an IK copy with target and pole, two Copy Transforms on each deform bone and a driven FK/IK switch property. Reports the controls, constraints, switches and how far IK sits from the rest."
    return _blender("add_control_rig", object=object, preset=preset, chains=chains, switchProperty=switch_property,
                    switchDirection=switch_direction, fkPrefix=fk_prefix, ikPrefix=ik_prefix,
                    targetParent=target_parent)


@mcp.tool()
def bl_create_armature_from_skeleton(bones: list, name: str = None, unit_scale: float = None, height_m: float = None,
                                     leaf_length: float = None, collection: str = None, location: list = None) -> dict:
    "Build a Blender armature from an Unreal skeleton's bones - list_bones with include_transforms."
    return _blender("create_armature_from_skeleton", bones=bones, name=name, unitScale=unit_scale, heightM=height_m,
                    leafLength=leaf_length, collection=collection, location=location)


@mcp.tool()
def bl_fit_armature(object: str, landmarks: dict, symmetry: str = None, pair: list = None) -> dict:
    "Move a Blender armature's bones onto a character from landmarks {bone: [x, y, z]} (armature space), keeping names and hierarchy; unnamed bones follow their parent, symmetry 'X' fills the other side. Reports each bone's displacement and lengths."
    return _blender("fit_armature", object=object, landmarks=landmarks, symmetry=symmetry, pair=pair)


@mcp.tool()
def bl_extract_deform_skeleton(rig: str, name: str = None, disconnect: bool = None, actions: list = None,
                               clear_constraints: bool = None, suffix: str = None,
                               allow_simulation: bool = None) -> dict:
    "Make a deform-only copy of a Blender control rig (each deform bone under its nearest deform ancestor) and bake the rig's actions onto it through Copy Transforms - what a game engine keeps."
    return _blender("extract_deform_skeleton", rig=rig, name=name, disconnect=disconnect, actions=actions,
                    clearConstraints=clear_constraints, suffix=suffix, allowSimulation=allow_simulation)


@mcp.tool()
def bl_bind_modifier(object: str, modifier: str, unbind: bool = None) -> dict:
    "Bind a Blender deform modifier that does nothing until bound - Surface Deform, Mesh Deform, Corrective Smooth (rest source Bind), Laplacian Deform - or unbind it."
    return _blender("bind_modifier", object=object, modifier=modifier, unbind=unbind)


@mcp.tool()
def bl_vertex_groups_from_bones(object: str, groups: dict, replace: bool = None) -> dict:
    "Make Blender vertex groups from bone weights: groups {name: {bones: [...], threshold (default 0.5), value 'one' or 'sum'}} - each holds the vertices whose summed weight on those bones reaches the threshold. A region mask (hood, cuff, pin rim) from a skinned mesh."
    return _blender("vertex_groups_from_bones", object=object, groups=groups, replace=replace)


@mcp.tool()
def bl_delete_mesh_elements(object: str, elements: str = None, indices: list = None, vertex_group: str = None,
                            min_weight: float = None, loose_islands_under: int = None, occluded_by: dict = None,
                            disagree_with: dict = None) -> dict:
    "Delete part of a Blender mesh: elements VERTS, EDGES, FACES or ONLY_FACES, chosen by ONE of indices (from bl_select_faces or bl_vertex_info), vertex_group with min_weight (default 0.5), or loose_islands_under N (every face-connected island with fewer faces), occluded_by {meshes, reach, margin} or disagree_with {reference, maxDistance, minDot}. Counts before and after; a selector that matches nothing is refused."
    return _blender("delete_mesh_elements", object=object, elements=elements, indices=indices,
                    vertexGroup=vertex_group, minWeight=min_weight, looseIslandsUnder=loose_islands_under,
                    occludedBy=occluded_by, disagreeWith=disagree_with)


@mcp.tool()
def bl_push_out_of(object: str = None, collider: str = None, offset: float = None, side: str = None,
                   reach: float = None, iterations: int = None, smooth: int = None, cover_distance: float = None,
                   skip_facing_collider: bool = None, skip_slots: list = None, inside_box: list = None,
                   dry_run: bool = None, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Push a Blender mesh out of another to a clearance (offset, world units, default 5 mm): a garment off the body or the layer under it. side 'garment' (default) moves each vertex; 'collider' pushes the garment faces a body bump pokes through between vertices. Penetration is measured before and after. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, collider=collider, offset=offset, side=side,
                                        reach=reach, iterations=iterations, smooth=smooth,
                                        cover_distance=cover_distance, skip_facing_collider=skip_facing_collider,
                                        skip_slots=skip_slots, inside_box=inside_box, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("push_out_of", previewId=preview_id))
    return _blender("push_out_of", object=object, collider=collider, offset=offset, side=side,
                    reach=reach, iterations=iterations, smooth=smooth, coverDistance=cover_distance,
                    skipFacingCollider=skip_facing_collider, skipSlots=skip_slots, insideBox=inside_box,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_fold_boundary_under(object: str, thickness: float = None, depth: float = None, loop_near: list = None,
                           straighten: int = None, group: str = None) -> dict:
    "Fold a single-sheet Blender mesh's open rim under itself - edge-only thickness for a hem, cuff or opening: a wall thickness down (default 5 mm) and a lip depth back under (15 mm), world units. loop_near [x,y,z] folds one loop; straighten smooths the rim first."
    return _blender("fold_boundary_under", object=object, thickness=thickness, depth=depth, loopNear=loop_near,
                    straighten=straighten, group=group)


@mcp.tool()
def bl_reload_image(image: str = None, images: list = None, all: bool = None, discard_unsaved: bool = None) -> dict:
    "Re-read Blender image datablocks from their files after a bake or another tool rewrote them - a re-written file changes nothing in the session until reloaded. Refuses a packed image, one with no file, a missing file, and one with unsaved pixels unless discard_unsaved."
    return _blender("reload_image", image=image, images=images, all=all, discardUnsaved=discard_unsaved)


@mcp.tool()
def bl_set_vertex_weights(object: str, group: str = None, vertices: list = None, weight: float = None,
                          weights: list = None, mode: str = None, create: bool = None,
                          remove: bool = None, by_axis: dict = None, all_vertices: bool = None,
                          exclusive: bool = None, by_radius: dict = None, mirror_axis: str = None,
                          mirror_pair: list = None, tolerance: float = None, factor: float = None,
                          repeat: int = None, expand: bool = None, delete_groups=None, armature: str = None,
                          lock: bool = None, groups: Any = None, steps: int = None, threshold: float = None,
                          keep_single: bool = None, offset: float = None, gain: float = None,
                          auto_assign: bool = None, auto_remove: bool = None) -> dict:
    "Create a Blender vertex group and put weights in it - which NOTHING in the addon could do. BY POSITION TOO: by_axis {axis X|Y|Z, from, to, weightFrom (default 1), weightTo (default 0), space WORLD|LOCAL} gives every vertex (or just `vertices`) a weight from where it sits - weightFrom at `from`, weightTo at `to`, linear between, clamped beyond - which is how a soft body's base is pinned under a loose top, since no op returns vertex positions to weight by index. bl_list_vertex_groups, bl_normalize_weights and bl_transfer_weights all operate on vertex groups and nothing created one, so a rig imported with weights could be normalised and a rig built here could not be weighted at all. THE POSTCONDITION IS THE WEIGHTS, NOT THE GROUP: a group that exists with every weight at zero deforms nothing and reads back identically to a working one in bl_list_vertex_groups, so this reports verticesWithNonZeroWeight - the number that decides whether anything actually moves - and says so outright when every weight written was zero. Membership and a zero weight are DIFFERENT states and are distinguished: vg.weight() raises for a vertex outside the group rather than returning 0, so a vertex not in the group at all cannot be normalised or transferred. A duplicate group name gives the NEW group a .001 suffix and leaves the incumbent alone, uniformly on every build - unlike object renaming, which reverses across the 3.6/4.x line - and it is reported rather than refused. all_vertices weights every vertex; exclusive then takes them out of every other group (otherGroupsCleared names each) - a rigid piece on one bone."
    return _blender("set_vertex_weights", object=object, group=group, vertices=vertices,
                    weight=weight, weights=weights, mode=mode, create=create, remove=remove,
                    byAxis=by_axis, allVertices=all_vertices, exclusive=exclusive, byRadius=by_radius,
                    mirrorAxis=mirror_axis, mirrorPair=mirror_pair, tolerance=tolerance, factor=factor,
                    repeat=repeat, expand=expand, deleteGroups=delete_groups, armature=armature, lock=lock,
                    groups=groups, steps=steps, threshold=threshold, keepSingle=keep_single, offset=offset, gain=gain,
                    autoAssign=auto_assign, autoRemove=auto_remove)


@mcp.tool()
def bl_add_shape_key(object: str, name: str = None, from_mix: bool = None, value: float = None,
                     slider_min: float = None, slider_max: float = None, from_object: str = None,
                     vertex_group: str = None, relative_key: str = None) -> dict:
    "Add a Blender shape key - the other thing several ops consumed and nothing produced. bl_set_shape_key drives a key's value and bl_list_shape_keys reports them; nothing could CREATE one, so a mesh without shape keys could never be given any. THE FIRST KEY IS THE BASIS AND IT IS NOT A SHAPE: Blender makes the first key added to a mesh the rest position and stores every later key RELATIVE to it (verified on 3.6, 4.4 and 5.0), so a caller adding one key to a bare mesh gets something that holds the shape the mesh already has and moves nothing. basisCreated reports that separately rather than letting it look like the key you asked for. from_mix captures the current evaluated mix instead of the rest shape - how a corrective key is made - and is OFF by default, because a key silently capturing whatever the sliders happened to be set to is a surprise rather than a convenience. from_object is Join as Shapes: the key takes another mesh's evaluated vertex positions (same vertex count, refused otherwise), carried from that object's world space into this one's; a mesh with no keys gets a Basis first so the named key is the shape. maxOffsetFromBasis says how far it moves."
    return _blender("add_shape_key", object=object, name=name, fromMix=from_mix, value=value,
                    sliderMin=slider_min, sliderMax=slider_max, fromObject=from_object, vertexGroup=vertex_group,
                    relativeKey=relative_key)


@mcp.tool()
def bl_normalize_weights(object_name: str, max_influences: int = None,
                         normalize: bool = True, groups: list = None, scope: str = None,
                         respect_locked: bool = None) -> dict:
    "Make every vertex's bone weights sum to 1 in Blender, and cap how many bones influence one vertex."
    return _blender("normalize_weights", object=object_name, maxInfluences=max_influences,
                    normalize=normalize, groups=groups, scope=scope, respectLocked=respect_locked)


@mcp.tool()
def bl_transfer_weights(source: str, destination: str,
                        mapping: str = "POLYINTERP_NEAREST", only_groups: list = None,
                        exclude_groups: list = None, destination_mask: str = None) -> dict:
    "Copy vertex weights from one Blender mesh onto another by proximity - the op a retopology or LOD pass needs."
    return _blender("transfer_weights", source=source, destination=destination, mapping=mapping,
                    onlyGroups=only_groups, excludeGroups=exclude_groups, destinationMask=destination_mask)


@mcp.tool()
def bl_purge_orphans(confirm: bool = False, recursive: bool = False, only: list = None,
                     clear_fake_user: bool = None, dry_run: bool = False, preview_id: str = None,
                     preview_ttl_seconds: int = None) -> dict:
    "Delete every Blender datablock with no users and no fake user, NOW, without saving - and report what went, by type and by name. This is the ACT half of a report bl_file_info has always given: its purgeNote says a save deletes unlinked datablocks, and until 2026-09-08 nothing could act on that without either writing the file or clearing the scene. It is also why there is no bl_delete_material or bl_delete_image: Blender's lifecycle is unlink-then-purge, so a datablock with users cannot be removed without breaking what uses it and one without users costs nothing until the file is written - per-type deleters would be the wrong shape and this is the Outliner's own Purge. confirm:true is REQUIRED, matching bl_destroy_collection, because an unlinked material somebody spent an hour on looks exactly like scratch from here. With dry_run=True it previews what would go and returns a preview_id, no confirm needed; preview_id alone commits exactly that list in place of confirm. recursive repeats until nothing more is unused, reaching datablocks orphaned only because the first pass removed their last user - a deep clean, and more than a caller tidying up usually expects, so it is off by default. Counts are a before-and-after difference rather than the purge call's own return, because that return varies by build. ONLY SCOPES IT: only=['materials:Rust', 'node_groups:Leaves'] (the 'type:name' shape bl_file_info's orphans and purgedNames print, or a bare name when one datablock carries it) removes exactly those and leaves every other orphan - the file-wide pass takes a material made a moment ago and not yet assigned. Still unlink-then-purge: a named datablock with users, a fake user, a missing or an ambiguous name refuses the whole call, naming what uses it and how to unlink it; namesFreed confirms each name is free again (a re-create gets it back, not '.001'). This is how a material, node group, texture, world, font or action made through the bridge is removed on its own. clear_fake_user=true (only with `only`) clears the fake user on the NAMED datablocks and purges them - bl_create_action sets one by default, so without it an action can never be removed; without `only` it is refused."
    if preview_id:
        return (_preview_commit_refusal(preview_id, confirm=confirm, recursive=recursive, only=only,
                                        clear_fake_user=clear_fake_user, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("purge_orphans", previewId=preview_id))
    return _blender("purge_orphans", confirm=confirm, recursive=recursive, only=only,
                    clearFakeUser=clear_fake_user, dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_list_scenes(limit: int = None, name_excludes: list = None) -> dict:
    "Every Blender scene in the file, by name, with its object count, frame range, user count and a per-row current flag - the THIRD collection found created, consumed and not listable, and the first with no refusal to read the names out of either. bl_create_scene makes one and several ops take a scene NAME, but bl_scene_info reports the CURRENT scene and never touches bpy.data.scenes, so a caller in a multi-scene file had no way to learn what the others were called. The current flag rides on each row rather than sitting in a separate field, because 'what scenes are there' and 'which am I in' are asked together and answering them apart invites pairing them wrongly."
    return _blender("list_scenes", limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_list_worlds(limit: int = None, name_excludes: list = None) -> dict:
    "Every Blender world datablock in the file, by name, with usesNodes, user count, fake-user flag and a per-row current flag showing which the active scene uses - the FOURTH collection of this shape. bl_create_world makes one and ops take a world NAME, and the only enumeration of world names anywhere in the addon was inside bl_world_info's own refusal, so a caller discovered worlds by deliberately asking for one that does not exist. users and fakeUser are reported because a world with neither is one the next save deletes, and somebody choosing between worlds is exactly who needs that before choosing."
    return _blender("list_worlds", limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_list_images(source: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "Every Blender image datablock in the file, by name, with its source, pixel size, channel count, filepath, packed flag, user count and fake-user flag - the second collection found created, consumed and not listable, by re-running the question that produced bl_list_node_groups. Images are made by bl_create_texture and bl_bake_texture and loaded for HDRIs and renders; bl_create_texture and bl_set_material_texture both take an image NAME; clear_scene purges them and bl_file_info counts them. Nothing listed them: the only two places in the whole addon that enumerated image names were inside REFUSALS, capped at twenty-five, so a caller discovered images by deliberately asking for one that does not exist. Neither asymmetry audit catches this shape, and both are right to pass - images have a creator and are reached by a caller-supplied key, so what is missing is a third thing they do not ask about. sizePx and users are reported because a 4096 image is a different proposition from a 64 one when choosing what to plug into a material, and an image with no users and no fake user is one the next save deletes. Pass source to filter by Blender's own image source enum (FILE, GENERATED, VIEWER); omit it for all of them."
    return _blender("list_images", source=source, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_create_image(name: str, width: int, height: int, color: list = None, alpha: bool = None,
                    float_buffer: bool = None, generated_type: str = None, colorspace: str = None,
                    fake_user: bool = None, material: str = None, input: str = None) -> dict:
    "Make a blank Blender image datablock - a canvas filled with one colour (alpha, float, UV/colour grid, colour space), kept by a fake user, optionally wired into a material input. bl_save_image writes it to disk."
    return _blender("create_image", name=name, width=width, height=height, color=color, alpha=alpha, float=float_buffer,
                    generatedType=generated_type, colorspace=colorspace, fakeUser=fake_user, material=material,
                    input=input)


@mcp.tool()
def bl_save_image(image: str, filepath: str, format: str = None, color_depth: int = None, quality: int = None,
                  overwrite: bool = None, repoint: bool = None) -> dict:
    "Write a Blender image datablock to disk (PNG 8/16-bit, JPEG quality, EXR half/full, any still format of the build) and point it at the file so a painted canvas survives a reopen; repoint=false writes a copy."
    return _blender("save_image", image=image, filepath=filepath, format=format, colorDepth=color_depth,
                    quality=quality, overwrite=overwrite, repoint=repoint)


@mcp.tool()
def bl_derive_image_map(image: str, kind: str, name: str = None, strength: float = None, blur: float = None,
                        contrast: float = None, invert: bool = None, flip_green: bool = None, value_range: list = None,
                        filepath: str = None, colorspace: str = None) -> dict:
    "Derive a HEIGHT, NORMAL or SPECULAR map from one Blender colour image (luminance as height, blur and contrast, strength and green-up/down for normals, a 0-0.5 range for specular) as a new image; filepath also saves it."
    return _blender("derive_image_map", image=image, kind=kind, name=name, strength=strength, blur=blur,
                    contrast=contrast, invert=invert, flipGreen=flip_green, range=value_range, filepath=filepath,
                    colorspace=colorspace)


@mcp.tool()
def bl_image_stats(image: str = None, file: str = None, normal: bool = None, tiling: bool = None, pixels: list = None,
                   uvs: list = None, tile: int = None) -> dict:
    "Describe a Blender image's pixels - per channel min, max, mean, stddev, NaN and Inf counts, share at 0 and 1, alpha coverage; a normal-map check, a tiling check and samples at pixels or UVs. Catches a black bake, an inverted normal or a NaN EXR. See mif_help."
    return _blender("image_stats", image=image, file=file, normal=normal, tiling=tiling, pixels=pixels, uvs=uvs,
                    tile=tile)


@mcp.tool()
def bl_resize_image(image: str, width: int, height: int, filepath: str = None, format: str = None,
                    colorspace: str = None, overwrite: bool = None, pack: bool = None) -> dict:
    "Resize a Blender image read from a file to an exact width and height and re-encode it (filepath, format PNG, TARGA, OPEN_EXR, JPEG); pack packs the file. Replies sizes before and after, and the written file read back."
    return _blender("resize_image", image=image, width=width, height=height, filepath=filepath, format=format,
                    colorspace=colorspace, overwrite=overwrite, pack=pack)


@mcp.tool()
def bl_create_udim_image(name: str, tiles: dict, filepath: str = None, overwrite: bool = None, colorspace: str = None,
                         fake_user: bool = None, material: str = None, input: str = None) -> dict:
    "Load tile files {1001: file, 1002: file} as one tiled (UDIM) Blender image, optionally wired to a material input; filepath with <UDIM> copies them into one pattern. Replies each tile's size."
    return _blender("create_udim_image", name=name, tiles=tiles, filepath=filepath, overwrite=overwrite,
                    colorspace=colorspace, fakeUser=fake_user, material=material, input=input)


@mcp.tool()
def bl_paint_texture(layers: list, image: str = None, material: str = None, input: str = None, object_name: str = None,
                     uv_map: str = None, base: str = None, palette: dict = None, symmetry: str = None) -> dict:
    "Paint a Blender image in layers, bottom first: fills (color, palette name, image) through masks of faces, a vertex group, a UV box, brush strokes or an image; MIX, MULTIPLY, ADD, ERASE_ALPHA; symmetry. Replies coverage per layer."
    return _blender("paint_texture", layers=layers, image=image, material=material, input=input, object=object_name,
                    uvMap=uv_map, base=base, palette=palette, symmetry=symmetry)


@mcp.tool()
def bl_project_image_decal(object_name: str, image: str, origin: list, direction: list, size: list,
                           rotation: float = None, faces: list = None, limit: int = None, max_depth: float = None,
                           wipe_back_faces: bool = None, uv_map: str = None, mask_attribute: str = None,
                           material: str = None, wire: bool = None) -> dict:
    "Project a Blender image from a plane (origin, direction, size, rotation) onto the faces it hits - a logo on a garment: a UV layer and a mask written, back faces wiped, the image mixed over Base Color. Replies faces covered."
    return _blender("project_image_decal", object=object_name, image=image, origin=origin, direction=direction,
                    size=size, rotation=rotation, faces=faces, limit=limit, maxDepth=max_depth,
                    wipeBackFaces=wipe_back_faces, uvMap=uv_map, maskAttribute=mask_attribute, material=material,
                    wire=wire)


@mcp.tool()
def bl_spray_text_art(text: str, font: str = None, name: str = None, size: list = None, margin: int = None,
                      outline_width: int = None, highlight_width: int = None, softness: float = None,
                      overspray: float = None, overspray_spread: int = None, seed: int = None, colors: dict = None,
                      passes: dict = None, stroke_width: int = None, nozzle_samples: int = None,
                      decal_size: list = None, min_readable_area: float = None, reveal_keys: list = None,
                      color_path: str = None, reveal_path: str = None, overwrite: bool = None,
                      drips: dict = None, run_path: str = None, shadow: dict = None, nozzle: dict = None) -> dict:
    "Text in a font -> a spray-paint colour texture and a reveal map (outline, fill and highlight passes trailing a nozzle, a time per pixel) for a decal, and the first reveal at which the paint covers min_readable_area on a decal of decal_size. nozzle {path, radius} paints each pixel when a given nozzle path first passes within the spray radius of it."
    return _blender("spray_text_art", text=text, font=font, name=name, size=size, margin=margin,
                    outlineWidth=outline_width, highlightWidth=highlight_width, softness=softness,
                    overspray=overspray, overspraySpread=overspray_spread, seed=seed, colors=colors, passes=passes,
                    strokeWidth=stroke_width, nozzleSamples=nozzle_samples, decalSize=decal_size,
                    minReadableArea=min_readable_area, revealKeys=reveal_keys, colorPath=color_path,
                    revealPath=reveal_path, overwrite=overwrite, drips=drips, runPath=run_path, shadow=shadow,
                    nozzle=nozzle)


@mcp.tool()
def bl_list_node_groups(type: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "Every Blender node group in the file, by name, with its tree type, node count, user count and fake-user flag; type filters to GeometryNodeTree, ShaderNodeTree or CompositorNodeTree. bl_add_group_interface, bl_add_group_node, bl_assign_node_group, bl_link_group_nodes and bl_list_group_nodes all take a group NAME, bl_create_node_group makes one, and until 2026-09-08 nothing listed them: a caller had to remember what it had created or read the names out of a deliberate REFUSAL, since the not-found message prints them. Neither asymmetry audit caught it and both were right to pass - node_groups has a creator and is reached by a caller-supplied key, so the shape that was missing is a third one they do not ask about, CREATED AND CONSUMED AND NOT LISTABLE. users and fakeUser are reported because a group with neither is one the next save deletes, which is the same fact bl_file_info gives for orphans, said where somebody choosing a group will actually see it. Pass type to filter by tree type (GeometryNodeTree, ShaderNodeTree, CompositorNodeTree); omit it for all of them."
    return _blender("list_node_groups", type=type, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_describe_op(op: str) -> dict:
    "What a Blender addon op accepts, and the RECORDED REASON for keys it deliberately does not take - the discovery half of the notes work. reject_unknown has been able to answer a wrongly-guessed key since 2026-09-07 (bl_add_modifier says apply is absent because adding and applying are separate on purpose, and points at bl_apply_modifier), but only to a caller who had already been refused. The UE side serves the same table through describe_endpoint and can be read BEFORE the call; this is the addon's equivalent. It reads the accept-list and the notes dict out of the RUNNING SOURCE with ast rather than from a generated table, so unlike the UE side it cannot drift from the guard it describes - accept-lists that are module-level constants are resolved by asking the imported module, which is exact. An op with no reject_unknown guard comes back with acceptedKeys null and a reason, never as an empty list, because reporting no keys for an op that simply has no guard is the same class of quiet lie the notes exist to prevent. `refusals` lists the op's VALUE RULES as written in its source - every refusal message, placeholders like %s left unfilled - so ranges, allowed words and required combinations can be read before the call instead of met one refusal at a time; helpers in the same module are followed one level (refusalsFromHelpers), and a message built at runtime is counted in refusalsUnreadable rather than dropped."
    return _blender("describe_op", op=op)


@mcp.tool()
def bl_add_modifier(object_name: str = None, type: str = None, modifier: str = None,
                    settings: dict = None, index: int = None, copy_from: dict = None, to: list = None,
                    hollow: bool = None) -> dict:
    "Add a modifier to a Blender object's stack - the write half of bl_list_modifiers. The response's facesFold is true when this modifier made faces fold over themselves (triangleOverlapAfter above 1.02: a render-only defect every count misses, typically a bevel wider than half a thin wall after a boolean - put a TRIANGULATE modifier before it). A mesh takes every type; a curve or text takes 28 (ARRAY, BEVEL, SOLIDIFY, SCREW, SUBSURF, WELD, SIMPLE_DEFORM, CAST and more, not the ones that need mesh data); a lattice only deforming ones; a Grease Pencil object the GREASE_PENCIL_* types and LINEART (GP_* on Blender 3.6 and 4.2). A type the object cannot take is refused naming its type. `settings` takes the curated camelCase names or any writable RNA property of the modifier by its identifier (foam_layer_name, limit_method, use_axis) - the path bl_set_modifier uses - and a setting that refuses removes the new modifier again. A SOLIDIFY that would make a second inward shell inside a closed mesh is refused, naming bl_transform_selection's SHRINK_FATTEN to grow it; hollow true keeps the cavity, for a wall to cut through or a shell to print."
    if copy_from is not None or to is not None:
        # COPY MODE takes no object, type, modifier or settings of its own; the add-on refuses any it is sent
        return _blender("add_modifier", copyFrom=copy_from, to=to, index=index, object=object_name, type=type,
                        modifier=modifier, settings=settings, hollow=hollow)
    return _blender("add_modifier", object=object_name, type=type, modifier=modifier,
                    settings=settings, index=index, hollow=hollow)


@mcp.tool()
def bl_apply_modifier(object_name: str = None, modifier: str = None, dry_run: bool = False, as_shape_key: bool = None,
                      keep_modifier: bool = None, key_name: str = None, preview_id: str = None,
                      preview_ttl_seconds: int = None, keep_shape_keys: bool = None) -> dict:
    "Bake a modifier into a Blender mesh's data. Destructive, and it reports what that cost. Meshes only: on a curve or text Blender will not apply a constructive modifier without converting to a mesh, and this refuses saying so. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since. On a mesh with shape keys keep_shape_keys=True bakes a deforming modifier into the basis and each key, keeping them."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, modifier=modifier, dry_run=dry_run,
                                        as_shape_key=as_shape_key, keep_modifier=keep_modifier, key_name=key_name,
                                        preview_ttl_seconds=preview_ttl_seconds, keep_shape_keys=keep_shape_keys)
                or _blender("apply_modifier", previewId=preview_id))
    return _blender("apply_modifier", object=object_name, modifier=modifier, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds,
                    asShapeKey=as_shape_key, keepModifier=keep_modifier, keyName=key_name,
                    keepShapeKeys=keep_shape_keys)


@mcp.tool()
def bl_set_modifier(object_name: str, modifier: str, settings: dict = None, index: int = None,
                    move: str = None, adaptive: dict = None) -> dict:
    "Change the settings of a modifier ALREADY on a Blender object's stack, or move it (index, or move UP, DOWN, TOP or BOTTOM) - the half bl_add_modifier left write-once. `settings` takes either the curated camelCase names bl_add_modifier and bl_list_modifiers use (foamCoverage, vertexGroup, offsetObject - validated exactly as bl_add_modifier validates them) or ANY writable RNA property by its identifier (foam_layer_name, spectrum, limit_method, show_render, use_axis), read off bl_rna at call time the way bl_set_constraint is: enums validated against their own items, finite numbers only, strict booleans, a datablock pointer by NAME with Blender's poll read back, a vertex group or UV map name checked against the object. Naming one attribute twice (foamCoverage and foam_coverage) is refused. ALL OR NOTHING: every attribute is snapshotted and restored if any key refuses. The effect is MEASURED: evaluatedChanged compares the evaluated mesh before and after, and evaluatedNote says when a setting moved no vertex - right for a render toggle or a layer name. Why it exists: an Ocean's foam is a colour attribute a material reads BY NAME, and its name (foam_layer_name) was settable by no op; re-tuning choppiness or a bevel's width meant removing and re-adding the modifier. A geometry-nodes modifier's group and inputs stay with bl_assign_node_group. A FLUID MODIFIER'S SETTINGS ARE ON ITS ROLE: its own property is only fluid_type, so a key it lacks is written on the live role's struct - domain_settings, flow_settings or effector_settings - e.g. use_dissolve_smoke, dissolve_speed, use_collision_border_top, vorticity, flame_vorticity, burning_rate, use_noise, time_scale on a domain; fuel_amount, temperature, surface_distance (in grid CELLS, not metres) on a flow. A key another role owns is refused naming that role; bl_list_modifiers(rna=True) reads the role back under its identifier. A cloth's, collision's or soft body's settings are on its structs - settings, collision_settings, point_cache - sent nested ({\"collision_settings\": {\"distance_min\": 0.01}}) or flat when one struct has the key, and read back as collision_settings.distance_min. The stack evaluates top to bottom, so order is behaviour: index or move places the modifier, read back in stackBefore and stackAfter; a move Blender declines, or one that changes nothing, is refused, and with settings the move is undone if a setting refuses."
    return _blender("set_modifier", object=object_name, modifier=modifier, settings=settings, index=index,
                    move=move, adaptive=adaptive)


@mcp.tool()
def bl_remove_modifier(object_name: str, modifier: str) -> dict:
    "Remove a modifier from a Blender object's stack (mesh, curve, text, surface or lattice) WITHOUT applying it - the mesh data is untouched, and meshUnchanged in the response is measured rather than asserted (null on a non-mesh, which has no mesh data to compare)."
    return _blender("remove_modifier", object=object_name, modifier=modifier)


@mcp.tool()
def bl_set_uv_layer(object: str, layer: str, active: bool = None, rename: str = None,
                    remove: bool = None) -> dict:
    "Choose the ACTIVE UV layer on a Blender mesh, rename one, or remove one. THE ACTIVE LAYER IS THE ONE EVERYTHING WRITES TO - every UV operator and every bake - and bl_uv_info reported it while the only way to CHANGE it was bl_uv_unwrap, which sets it as a side effect of unwrapping. So selecting a layer meant re-unwrapping it and destroying the UVs it held. REMOVING A LAYER SHIFTS EVERY LATER LAYER'S INDEX: measured on 3.6.23 and 5.0.1, with UVMap, A, B, C, removing A moves B from 2 to 1 and C from 3 to 2. Unreal's lightmap coordinate is an INDEX, not a name, so removing an earlier layer silently repoints what the engine reads on a mesh it has already imported - the layers that moved are named in indicesShifted. Removing the LAST layer is refused: it leaves the mesh unwrappable, and texturing and lightmap baking both fail until it is unwrapped again. A mesh is limited to 8 UV layers, which is why removing junk ones matters at all."
    return _blender("set_uv_layer", object=object, layer=layer, active=active,
                    rename=rename, remove=remove)


@mcp.tool()
def bl_uv_info(object_name: str, layer: str = None, max_reported_islands: int = 64, islands: bool = None,
               stretch: str = None) -> dict:
    "Read a Blender mesh's UVs per layer - the verification half of bl_uv_unwrap. bl_uv_unwrap can CREATE a channel and bl_object_info reports only that channels EXIST, so 'did the unwrap produce something Unreal can bake a lightmap into' had"
    return _blender("uv_info", object=object_name, layer=layer,
                    maxReportedIslands=max_reported_islands, islands=islands, stretch=stretch)


@mcp.tool()
def bl_extrude_mesh(object: str, indices: list, elements: str = None, mode: str = None, offset: list = None,
                    distance: float = None, scale=None, rotate: dict = None, steps: int = None, space: str = None,
                    flip_normals: bool = None) -> dict:
    "Extrude Blender mesh faces, edges or vertices (indices) along the normal (distance) or a vector (offset), with scale and rotate {angle, axis} about each step's median, repeated steps times; mode REGION or INDIVIDUAL. Returns each step's new indices."
    return _blender("extrude_mesh", object=object, indices=indices, elements=elements, mode=mode, offset=offset,
                    distance=distance, scale=scale, rotate=rotate, steps=steps, space=space, flipNormals=flip_normals)


@mcp.tool()
def bl_loop_cut(object: str, edge: int = None, edge_near: list = None, count: int = None, slide: float = None,
                even: bool = None, smoothness: float = None) -> dict:
    "Cut edge loops across the ring of quads through a seed edge (index, or edge_near [x,y,z] world) on a Blender mesh: count loops, slid by slide (-1..1). Returns each loop's vertex and edge indices."
    return _blender("loop_cut", object=object, edge=edge, edgeNear=edge_near, count=count, slide=slide, even=even,
                    smoothness=smoothness)


@mcp.tool()
def bl_slide_edges(object: str, edges: list, factor: float, clamp: bool = None) -> dict:
    "Slide a chain or loop of Blender mesh edges along the edges beside it by factor (-1..1, the sign picks the side); clamp=false slides past the rail's end."
    return _blender("slide_edges", object=object, edges=edges, factor=factor, clamp=clamp)


@mcp.tool()
def bl_inset_faces(object: str, faces: list, thickness: float, depth: float = None, individual: bool = None,
                   boundary: bool = None, even: bool = None, outset: bool = None, relative_offset: bool = None,
                   space: str = None) -> dict:
    "Inset Blender mesh faces: a ring thickness wide inside the region (or each face with individual), the inner face moved depth along the normal. Returns the inner and ring face indices."
    return _blender("inset_faces", object=object, faces=faces, thickness=thickness, depth=depth, individual=individual,
                    boundary=boundary, even=even, outset=outset, relativeOffset=relative_offset, space=space)


@mcp.tool()
def bl_knife_cut(object: str, points: list = None, vertex_pairs: list = None, vertex_path: list = None,
                 space: str = None, tolerance: float = None) -> dict:
    "Cut across Blender mesh faces between points (a polyline; each lands on the nearest vertex or edge), or join vertices by vertex_pairs or vertex_path - across the faces between them. Returns the cut's edges and vertices."
    return _blender("knife_cut", object=object, points=points, vertexPairs=vertex_pairs, vertexPath=vertex_path,
                    space=space, tolerance=tolerance)


@mcp.tool()
def bl_join_boundary_loops(object: str, loop_a: dict = None, loop_b: dict = None, mode: str = None, twist: int = None,
                           match: str = None, cuts: int = None, interpolation: str = None, run_a: dict = None,
                           run_b: dict = None, runs_b: list = None, report_only: bool = None,
                           tolerance: float = None) -> dict:
    "Join two open boundary loops of one Blender mesh: BRIDGE (a band of faces), FUSE (weld into one seam) or SEW (loose edges for cloth sewing). Each loop is {near: [x,y,z]}, {vertex}, {edge} or {index}."
    return _blender("join_boundary_loops", object=object, loopA=loop_a, loopB=loop_b, mode=mode, twist=twist,
                    match=match, cuts=cuts, interpolation=interpolation, runA=run_a, runB=run_b, runsB=runs_b,
                    reportOnly=report_only, tolerance=tolerance)


@mcp.tool()
def bl_merge_vertices(object: str, vertices: list = None, pairs: list = None, at: str = None,
                      distance: float = None) -> dict:
    "Merge chosen Blender mesh vertices: all of them into one, each of pairs [[a, b], ...], or those closer than distance among the vertices given. at CENTER (default), FIRST or LAST says where a merged vertex lands."
    return _blender("merge_vertices", object=object, vertices=vertices, pairs=pairs, at=at, distance=distance)


@mcp.tool()
def bl_offset_edges(object: str, edges: list, distance: float, side_face: int = None, space: str = None,
                    clamp_overlap: bool = None) -> dict:
    "Carve a buffer loop beside a Blender mesh edge loop, distance away across each face it touches - the room a bevel needs round a junction a Boolean left in n-gons. edges is one closed loop or open chain; the loop stays in place. side_face (a face touching a closed loop) keeps the buffer on that side only. Replies the rings' edges and measured distance, the buffer faces and maxOffSurface."
    return _blender("offset_edges", object=object, edges=edges, distance=distance, sideFace=side_face, space=space,
                    clampOverlap=clamp_overlap)


@mcp.tool()
def bl_mesh_from_outline(outline: list, name: str = None, holes: list = None, height: float = None, plane: str = None,
                         edge_length: float = None, materials: dict = None, uv_scale: float = None,
                         boundary_group: str = None, location: list = None, rotation: list = None, scale: list = None,
                         collection: str = None, take_over_name: bool = None) -> dict:
    "A Blender mesh from a 2D outline [[x, y], ...] (and holes): a flat panel, or with height a closed slab with top, side and bottom slots and world-density UVs. edge_length adds an inner lattice (cloth). plane XY, XZ or YZ."
    return _blender("mesh_from_outline", outline=outline, name=name, holes=holes, height=height, plane=plane,
                    edgeLength=edge_length, materials=materials, uvScale=uv_scale, boundaryGroup=boundary_group,
                    location=location, rotation=rotation, scale=scale, collection=collection,
                    takeOverName=take_over_name)


@mcp.tool()
def bl_mesh_sections(object: str, heights: list = None, step: float = None, axis: str = None, space: str = None,
                     evaluated: bool = None, limit: int = None) -> dict:
    "Slice a Blender mesh across an axis (default Z) at heights or every step and read each cross-section: loops, closed or open, length, area, min/max/center and size in the other two axes. WORLD and evaluated by default. Read-only."
    return _blender("mesh_sections", object=object, heights=heights, step=step, axis=axis, space=space,
                    evaluated=evaluated, limit=limit)


@mcp.tool()
def bl_set_scene_units(system: str = None, scale_length: float = None, length_unit: str = None, mass_unit: str = None,
                       time_unit: str = None, temperature_unit: str = None, rotation_unit: str = None,
                       separate: bool = None) -> dict:
    "Set the Blender scene's unit system (METRIC, IMPERIAL, NONE), unit scale (1.0 is what the Unreal hand-off assumes) and display units; checked before any is written, before and after read back."
    return _blender("set_scene_units", system=system, scaleLength=scale_length, lengthUnit=length_unit,
                    massUnit=mass_unit, timeUnit=time_unit, temperatureUnit=temperature_unit,
                    rotationUnit=rotation_unit, separate=separate)


@mcp.tool()
def bl_build_vehicle_body(bottom: list, belt: list, half_width: list, roof: list = None, exponents: dict = None,
                          roof_width_ratio: float = None, stations: int = None, section_points: int = None,
                          arches: list = None, glass: list = None, materials: dict = None, name: str = None,
                          location: list = None, rotation: list = None, scale: list = None, collection: str = None,
                          take_over_name: bool = None, lids: list = None, seams: list = None, bumpers: list = None,
                          pockets: list = None, mirrors: list = None, arch_lips: dict = None,
                          wheels: dict = None) -> dict:
    "A parked car's Blender body lofted from side-profile curves [[x, z], ...] (bottom, belt, roof) and a plan (half_width), each applying only in its own x range; lids, seams, bumpers, lamp/plate pockets, mirrors, lips, wheels."
    return _blender("build_vehicle_body", bottom=bottom, belt=belt, halfWidth=half_width, roof=roof,
                    exponents=exponents, roofWidthRatio=roof_width_ratio, stations=stations,
                    sectionPoints=section_points, arches=arches, glass=glass, materials=materials, name=name,
                    location=location, rotation=rotation, scale=scale, collection=collection,
                    takeOverName=take_over_name, lids=lids, seams=seams, bumpers=bumpers, pockets=pockets,
                    mirrors=mirrors, archLips=arch_lips, wheels=wheels)


@mcp.tool()
def bl_build_lathe(profile: list, name: str = None, segments: int = None, axis: str = None, caps: bool = None,
                   smooth: bool = None, uv_scale: float = None, material: str = None, location: list = None,
                   rotation: list = None, scale: list = None, collection: str = None,
                   take_over_name: bool = None) -> dict:
    "Revolve a profile [[r, h], ...] round an axis into a Blender solid of revolution (a can, a bottle, a post) whose every face points out, checked by signed volume. r 0 is a pole; caps close open ends."
    return _blender("build_lathe", profile=profile, name=name, segments=segments, axis=axis, caps=caps, smooth=smooth,
                    uvScale=uv_scale, material=material, location=location, rotation=rotation, scale=scale,
                    collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_tube(radius: float, points: list = None, start: list = None, end: list = None,
                  sag: float = None, samples: int = None, sides: int = None, closed: bool = None, caps: bool = None,
                  smooth: bool = None, uv_scale: float = None, material: str = None, name: str = None,
                  location: list = None, rotation: list = None, scale: list = None, collection: str = None,
                  take_over_name: bool = None, surface: str = None, lift: float = None, step: float = None,
                  hang_ends: list = None, end_caps: str | list = None, stoppers: list = None) -> dict:
    "A round Blender tube along points (closed for a neon loop), laid on a surface mesh lift off it, or hanging between start and end with sag (a catenary): a cable, a pipe, a drawcord. Capped into a closed solid that faces out."
    return _blender("build_tube", radius=radius, points=points, start=start, end=end, sag=sag,
                    samples=samples, sides=sides, closed=closed, caps=caps, smooth=smooth, uvScale=uv_scale,
                    material=material, name=name, location=location, rotation=rotation, scale=scale,
                    collection=collection, takeOverName=take_over_name, surface=surface, lift=lift, step=step,
                    hangEnds=hang_ends, endCaps=end_caps, stoppers=stoppers)


@mcp.tool()
def bl_cut_eyelets(object: str, at: list, shape: str = None, inner_radius: float = None, size: list = None,
                   axis: list = None, ring_width: float = None, slot: str = None, min_boundary: int = None,
                   layers: str = None, lining: str | list = None, layer_depth: float = None, mirror: str = None) -> dict:
    "Cut eyelets through a Blender sheet at world points: round (inner_radius) or oval (size [w, h] along axis) holes at the mesh's density, each with a ring of faces ring_width wide in a material slot."
    return _blender("cut_eyelets", object=object, at=at, shape=shape, innerRadius=inner_radius, size=size, axis=axis,
                    ringWidth=ring_width, slot=slot, minBoundary=min_boundary, layers=layers, lining=lining,
                    layerDepth=layer_depth, mirror=mirror)


@mcp.tool()
def bl_rip_edges(object: str, edges: list = None, selector: dict = None, fill: bool = None) -> dict:
    "Rip (split) a Blender mesh along chosen edges: edges [i...] or selector {bl_select_edges' grammar}. A chain boundary to boundary, or a loop, cuts it into parts. fill bridges each rip with a face. Replies new vertex indices and parts before and after."
    return _blender("rip_edges", object=object, edges=edges, selector=selector, fill=fill)


@mcp.tool()
def bl_fracture_mesh(object: str = None, objects: list = None, source: str = None, source_limit=None,
                     points: list = None, noise: float = None, seed: int = None, cell_scale: list = None,
                     margin: float = None, recursion: int = None, recursion_source_limit: int = None,
                     recursion_clamp: int = None, recursion_chance: float = None, material_index: int = None,
                     collection: str = None, hide_original: bool = None, rigid_body: dict = None) -> dict:
    "Fracture closed Blender meshes into Voronoi cells, one object per cell, in a collection. source RANDOM|VERTS|PARTICLES|POINTS; source_limit per object; recursion; rigid_body {type, collisionShape...}. Replies cell counts, smallest cell, volumes."
    return _blender("fracture_mesh", object=object, objects=objects, source=source, sourceLimit=source_limit,
                    points=points, noise=noise, seed=seed, cellScale=cell_scale, margin=margin, recursion=recursion,
                    recursionSourceLimit=recursion_source_limit, recursionClamp=recursion_clamp,
                    recursionChance=recursion_chance, materialIndex=material_index, collection=collection,
                    hideOriginal=hide_original, rigidBody=rigid_body)


@mcp.tool()
def bl_copy_animation(source: str, targets: list, copy: bool = None, keys: list = None,
                      frame_offset: float = None) -> dict:
    "Copy a Blender object's action onto targets: copy true (default) gives each its own action to edit alone, false shares it. keys keeps curves whose data path starts with one; frame_offset shifts the copied keys."
    return _blender("copy_animation", source=source, targets=targets, copy=copy, keys=keys, frameOffset=frame_offset)


@mcp.tool()
def bl_paint_weights(object: str, group: str, path: list, radius: float, weight: float = None, strength: float = None,
                     falloff: str = None, falloff_shape: str = None, mode: str = None, spacing: float = None,
                     accumulate: bool = None, front_faces_only: bool = None, view_direction: list = None,
                     mask: dict = None, auto_normalize: bool = None, create_group: bool = None) -> dict:
    "Paint a Blender vertex group along a stroke of world points: radius, falloff, strength, mode MIX|ADD|SUBTRACT|REPLACE|LIGHTEN|DARKEN, accumulate, front faces only, a mask, auto normalize. No viewport. Replies vertices touched and weight range."
    return _blender("paint_weights", object=object, group=group, path=path, radius=radius, weight=weight,
                    strength=strength, falloff=falloff, falloffShape=falloff_shape, mode=mode, spacing=spacing,
                    accumulate=accumulate, frontFacesOnly=front_faces_only, viewDirection=view_direction, mask=mask,
                    autoNormalize=auto_normalize, createGroup=create_group)


@mcp.tool()
def bl_circularize(object: str, faces: list = None, loop: list = None, flatten: bool = None, radius: float = None,
                   regularize: bool = None, space: str = None) -> dict:
    "Make a Blender mesh loop (loop [vertex i...]) or a face region's rim (faces) a circle in its best-fit plane. flatten false keeps the host's curvature; regularize spaces vertices evenly; radius defaults to the mean. Replies deviation from a circle before and after."
    return _blender("circularize", object=object, faces=faces, loop=loop, flatten=flatten, radius=radius,
                    regularize=regularize, space=space)


@mcp.tool()
def bl_build_body(armature: str = None, bones: dict = None, preset: str = None, height: float = None,
                  torso: list = None, arms: list = None, legs: list = None, finger_curl: float = None,
                  subdivisions: int = None, game_ready=None, create_armature: bool = None, bind: bool = None,
                  material: str = None, name: str = None, collection: str = None, take_over_name: bool = None) -> dict:
    "A Blender body with clean joins on a skeleton (armature) or a preset (adult, slim, heavy, child): a quad cage lofted from width and depth tables, limbs joined through quad holes, hands and feet, fitted and subdivided."
    return _blender("build_body", armature=armature, bones=bones, preset=preset, height=height, torso=torso, arms=arms,
                    legs=legs, fingerCurl=finger_curl, subdivisions=subdivisions, gameReady=game_ready,
                    createArmature=create_armature, bind=bind, material=material, name=name, collection=collection,
                    takeOverName=take_over_name)


@mcp.tool()
def bl_measure_neckline(body: str, center: list = None, neck_z: float = None, armature: str = None,
                        neck_bone: str = None, angles: int = None, z_step: float = None, search_down: float = None,
                        tolerance: float = None, clip: float = None, smooth_passes: int = None, reach: float = None,
                        evaluated: bool = None, trim: dict = None) -> dict:
    "Where a Blender body's neck column meets chest and shoulders, per angle (rays out from the neck's axis), as baseZ and a ring; trim {garment, distance} deletes a garment's faces above it near the neck."
    return _blender("measure_neckline", body=body, center=center, neckZ=neck_z, armature=armature, neckBone=neck_bone,
                    angles=angles, zStep=z_step, searchDown=search_down, tolerance=tolerance, clip=clip,
                    smoothPasses=smooth_passes, reach=reach, evaluated=evaluated, trim=trim)


@mcp.tool()
def bl_build_collar(neckline, center: list = None, segments: int = None, stand_height: float = None, ease: float = None,
                    stand_rows: int = None, fold_gap: float = None, fall_width: float = None, fall_angle: float = None,
                    fall_rows: int = None, front_angle: float = None, front_gap: float = None,
                    point_drop: float = None, point_angle: float = None, point_span: float = None, shirt: str = None,
                    lay_offset: float = None, lay_reach: float = None, material: str = None, smooth: bool = None,
                    uv_scale: float = None, evaluated: bool = None, name: str = None, collection: str = None,
                    take_over_name: bool = None) -> dict:
    "A Blender shirt collar - stand, fold and fall, collar points and a front gap - as one ring at even angles on a neckline (bl_measure_neckline's ring, or its request), the fall laid onto shirt."
    return _blender("build_collar", neckline=neckline, center=center, segments=segments, standHeight=stand_height,
                    ease=ease, standRows=stand_rows, foldGap=fold_gap, fallWidth=fall_width, fallAngle=fall_angle,
                    fallRows=fall_rows, frontAngle=front_angle, frontGap=front_gap, pointDrop=point_drop,
                    pointAngle=point_angle, pointSpan=point_span, shirt=shirt, layOffset=lay_offset,
                    layReach=lay_reach, material=material, smooth=smooth, uvScale=uv_scale, evaluated=evaluated,
                    name=name, collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_applique(target: str, origin: list, direction: list, outline: list, up: list = None, spin: float = None,
                      resolution: float = None, lift: float = None, thickness: float = None, bottom: bool = None,
                      reach: float = None, material: str = None, side_material: str = None, smooth: bool = None,
                      evaluated: bool = None, name: str = None, collection: str = None,
                      take_over_name: bool = None) -> dict:
    "Lay a patch on a Blender surface where a ray meets it - a pocket flap, badge, name plate: a 2D outline (metres) cast on as a radial fan, lifted and walled into a closed solid. Resolution is the fan's spacing in metres, on the rim and between the rings alike: bigger is coarser and makes fewer faces - a 12 x 14 cm pocket is 286 faces at 2 cm, 950 at 1 cm and 3,626 at 5 mm. Left out, it is the coarsest of 4, 2 and 1 cm that follows the surface, and 1 cm on cloth no step follows, never finer; resolutionFrom (given, surface, floor) and resolutionTried (each step's spanDeviation) say which."
    return _blender("build_applique", target=target, origin=origin, direction=direction, outline=outline, up=up,
                    spin=spin, resolution=resolution, lift=lift, thickness=thickness, bottom=bottom, reach=reach,
                    material=material, sideMaterial=side_material, smooth=smooth, evaluated=evaluated, name=name,
                    collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_band(around: str, z0: float, z1: float, center: list = None, segments: int = None, samples: int = None,
                  offset: float = None, thickness: float = None, stiff: bool = None, smooth_passes: int = None,
                  items: list = None, material: str = None, item_material: str = None, smooth: bool = None,
                  evaluated: bool = None, name: str = None, collection: str = None,
                  take_over_name: bool = None) -> dict:
    "A stiff Blender band round a body from z0 to z1 - a belt, a waistband - by rays onto the garment: one radius per angle (the widest), closed top and bottom, with boxes (items) on it by angle and height."
    return _blender("build_band", around=around, z0=z0, z1=z1, center=center, segments=segments, samples=samples,
                    offset=offset, thickness=thickness, stiff=stiff, smoothPasses=smooth_passes, items=items,
                    material=material, itemMaterial=item_material, smooth=smooth, evaluated=evaluated, name=name,
                    collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_trash_bag(name: str = None, height: float = None, width: float = None, contact: float = None,
                       seed: int = None, knuckles: int = None, knuckle_depth: float = None, pleats: int = None,
                       twist: float = None, ears: bool = None, segments: int = None, rings: int = None,
                       color: list = None, material: str = None, location: list = None, rotation: list = None,
                       collection: str = None, take_over_name: bool = None) -> dict:
    "A tied rubbish bag in Blender standing on the floor: a pear on a flat contact patch, lumped by its contents, pleats twisting into the neck, a knot and two ears - one mesh, every part a closed solid facing out."
    return _blender("build_trash_bag", name=name, height=height, width=width, contact=contact, seed=seed,
                    knuckles=knuckles, knuckleDepth=knuckle_depth, pleats=pleats, twist=twist, ears=ears,
                    segments=segments, rings=rings, color=color, material=material, location=location,
                    rotation=rotation, collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_downspout(name: str = None, height: float = None, offset: float = None, section: str = None,
                       width: float = None, depth: float = None, standoff: float = None, straps: bool = None,
                       strap_spacing: float = None, shoe: bool = None, color: list = None, location: list = None,
                       rotation: list = None, collection: str = None, take_over_name: bool = None) -> dict:
    "Build a downspout down a wall in Blender: a hollow rect or round pipe from the gutter outlet over the eave (a gooseneck), down on screwed straps, out at a kick-out shoe. The wall is y 0, the pipe in +y."
    return _blender("build_downspout", name=name, height=height, offset=offset, section=section, width=width,
                    depth=depth, standoff=standoff, straps=straps, strapSpacing=strap_spacing, shoe=shoe,
                    color=color, location=location, rotation=rotation, collection=collection,
                    takeOverName=take_over_name)


@mcp.tool()
def bl_build_wall_fixture(name: str = None, kind: str = None, scale: float = None, arm_length: float = None,
                          conduit_length: float = None, conduit: str = None, color: list = None,
                          lamp_color: list = None, lamp_strength: float = None, location: list = None,
                          rotation: list = None, collection: str = None, take_over_name: bool = None) -> dict:
    "Build a wall-mounted fixture in Blender - a wall pack light, a gooseneck barn light, or a junction box with its conduit run - every part a closed solid, the lamp its own emissive slot. The wall is y 0."
    return _blender("build_wall_fixture", name=name, kind=kind, scale=scale, armLength=arm_length,
                    conduitLength=conduit_length, conduit=conduit, color=color, lampColor=lamp_color,
                    lampStrength=lamp_strength, location=location, rotation=rotation, collection=collection,
                    takeOverName=take_over_name)


@mcp.tool()
def bl_build_dumpster(name: str = None, width: float = None, depth: float = None, height: float = None,
                      front_slope: float = None, corner_radius: float = None, ribs: int = None, dents: float = None,
                      seed: int = None, lid_angles: list = None, casters: bool = None, fill: float = None,
                      carton: bool = None, color: list = None, lid_color: list = None, location: list = None,
                      rotation: list = None, collection: str = None, take_over_name: bool = None) -> dict:
    "Build a front-load dumpster in Blender: a lofted steel body with a sloped lower front, pressed ribs and dents, a rolled rim, fork pockets, skids or casters, lids opened by angle, and an optional fill checked against the walls."
    return _blender("build_dumpster", name=name, width=width, depth=depth, height=height, frontSlope=front_slope,
                    cornerRadius=corner_radius, ribs=ribs, dents=dents, seed=seed, lidAngles=lid_angles,
                    casters=casters, fill=fill, carton=carton, color=color, lidColor=lid_color,
                    location=location, rotation=rotation, collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_build_prop(kind: str = None, name: str = None, seed: int = None, width: float = None, depth: float = None,
                  height: float = None, wear: float = None, detail: str = None, color: list = None, color2: list = None,
                  metal_color: list = None, lid: bool = None, collision: bool = None, location: list = None,
                  rotation: list = None, collection: str = None, take_over_name: bool = None) -> dict:
    "Build a wooden prop in Blender board by board: crate, pallet, barrel, table, bench, chair, shelf, with eased edges, worn corners and grain along each."
    return _blender("build_prop", kind=kind, name=name, seed=seed, width=width, depth=depth, height=height, wear=wear,
                    detail=detail, color=color, color2=color2, metalColor=metal_color, lid=lid, collision=collision,
                    location=location, rotation=rotation, collection=collection, takeOverName=take_over_name)


@mcp.tool()
def bl_subdivide_mesh(object: str, faces: list = None, edges: list = None, all: bool = None, cuts: int = None,
                      smoothness: float = None, quad_corner: str = None, fractal: float = None, seed: int = None) -> dict:
    "Cut a Blender mesh's chosen faces or edges, or all of it, into cuts + 1 pieces per edge - Blender's Subdivide (a plane cut 25 times is 26 x 26). smoothness 0-1 rounds the new vertices out; quad_corner STRAIGHT_CUT, INNER_VERT, PATH or FAN. Counts before and after, regionFaces."
    return _blender("subdivide_mesh", object=object, faces=faces, edges=edges, all=all, cuts=cuts, smoothness=smoothness,
                    quadCorner=quad_corner, fractal=fractal, seed=seed)


@mcp.tool()
def bl_sculpt_stroke(object: str, brush: str, path: list, radius: float, strength: float = None, falloff: str = None,
                     invert: bool = None, symmetry: list = None, mask: dict = None, shape_key: str = None,
                     spacing: float = None, stabilize: float = None, boundary_deform: str = None,
                     dyntopo: dict = None, multires_level: int = None, cloth_deform: str = None,
                     deform_target: str = None, cloth_area: float = None) -> dict:
    "Sculpt a Blender mesh along a stroke of world points: brush DRAW, CLAY, INFLATE, SMOOTH, FLATTEN, CREASE, GRAB, SNAKE_HOOK, BOUNDARY, CLOTH, CLAY_STRIPS, SCRAPE, FILL, PINCH, LAYER, BLOB, DRAW_SHARP, NUDGE, ELASTIC or POSE, falloff, symmetry, masks, a shape key, dyntopo detail or a Multires level. Mesh space."
    return _blender("sculpt_stroke", object=object, brush=brush, path=path, radius=radius, strength=strength,
                    falloff=falloff, invert=invert, symmetry=symmetry, mask=mask, shapeKey=shape_key, spacing=spacing,
                    stabilize=stabilize, boundaryDeform=boundary_deform, dyntopo=dyntopo,
                    multiresLevel=multires_level, clothDeform=cloth_deform, deformTarget=deform_target,
                    clothArea=cloth_area)


@mcp.tool()
def bl_cloth_filter(object: str = None, filter_type: str = None, strength: float = None, steps: int = None,
                    direction: list = None, center: list = None, mask: dict = None, colliders: list = None,
                    collision_offset: float = None, friction: float = None, dry_run: bool = None,
                    preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Run a cloth filter over a Blender mesh: GRAVITY, INFLATE, EXPAND, PINCH or SCALE with its edge lengths held as cloth, masked vertices pinned, colliders pushed out of. Native mesh-space cloth, not Blender's filter. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, filter_type=filter_type, strength=strength,
                                        steps=steps, direction=direction, center=center, mask=mask,
                                        colliders=colliders, collision_offset=collision_offset, friction=friction,
                                        dry_run=dry_run, preview_ttl_seconds=preview_ttl_seconds)
                or _blender("cloth_filter", previewId=preview_id))
    return _blender("cloth_filter", object=object, filterType=filter_type, strength=strength, steps=steps,
                    direction=direction, center=center, mask=mask, colliders=colliders,
                    collisionOffset=collision_offset, friction=friction, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_groom_attach(object: str, mode: str = None, surface: str = None, uv_map: str = None, attach_mode: str = None,
                    tolerance: float = None, limit: int = None) -> dict:
    "Attach a Blender hair groom (Curves) to its surface - set the surface and UV map, snap the roots onto it (NEAREST, or DEFORM by the existing binding), or report the roots that float off it. See mif_help."
    return _blender("groom_attach", object=object, mode=mode, surface=surface, uvMap=uv_map, attachMode=attach_mode,
                    tolerance=tolerance, limit=limit)


@mcp.tool()
def bl_uv_stitch(object: str, mode: str = None, islands: list = None, faces: list = None, threshold: float = None,
                 uv_layer: str = None) -> dict:
    "Stitch two Blender UV islands along their shared edges (the second fitted onto the first), weld UVs, or merge them by distance; split edges and islands before and after. See mif_help."
    return _blender("uv_stitch", object=object, mode=mode, islands=islands, faces=faces, threshold=threshold,
                    uvLayer=uv_layer)


@mcp.tool()
def bl_sculpt_mask(object: str, mode: str, combine: str = None, invert: bool = None, value: float = None,
                   group: str = None, vertex: int = None, near: list = None, by: str = None, radius: float = None,
                   falloff: str = None, filter: str = None, iterations: int = None, steps: int = None,
                   boundary: str = None, cavity: dict = None, min_weight: float = None) -> dict:
    "Make, change or read a Blender mesh's sculpt mask by rule - from cavity, the boundary, a vertex group (or into one), an expand from a vertex, a filter (smooth, grow, shrink, contrast), invert, set, clear; masked count and stats before and after. See mif_help."
    return _blender("sculpt_mask", object=object, mode=mode, combine=combine, invert=invert, value=value, group=group,
                    vertex=vertex, near=near, by=by, radius=radius, falloff=falloff, filter=filter,
                    iterations=iterations, steps=steps, boundary=boundary, cavity=cavity, minWeight=min_weight)


@mcp.tool()
def bl_mask_extract(object: str, mode: str = None, threshold: float = None, add_boundary_loop: bool = None,
                    smooth_iterations: int = None, solidify: bool = None, thickness: float = None,
                    shrinkwrap: bool = None, fill_holes: bool = None, new_object: bool = None,
                    name: str = None) -> dict:
    "Extract a Blender mesh's masked region into a new object (thickness, smoothed border) or slice it out - Blender's own mask extract and slice under either name (mesh.* to 4.4, sculpt.* from 5.0). See mif_help."
    return _blender("mask_extract", object=object, mode=mode, threshold=threshold, addBoundaryLoop=add_boundary_loop,
                    smoothIterations=smooth_iterations, solidify=solidify, thickness=thickness, shrinkwrap=shrinkwrap,
                    fillHoles=fill_holes, newObject=new_object, name=name)


@mcp.tool()
def bl_mesh_filter(object: str, filter: str, strength: float = None, iterations: int = None, use_mask: bool = None,
                   vertex_group: str = None, center: list = None, seed: int = None, shape_preservation: float = None,
                   current_vertex: float = None) -> dict:
    "A whole-mesh sculpt filter on a Blender mesh in mesh space - smooth, surface smooth, inflate, scale, sphere, random, relax, relax face sets, sharpen, enhance details - the sculpt mask protecting what it covers. See mif_help."
    return _blender("mesh_filter", object=object, filter=filter, strength=strength, iterations=iterations,
                    useMask=use_mask, vertexGroup=vertex_group, center=center, seed=seed,
                    shapePreservation=shape_preservation, currentVertex=current_vertex)


@mcp.tool()
def bl_symmetrize_mesh(object: str, mode: str = None, axis: str = None, direction: str = None,
                       threshold: float = None, factor: float = None, limit: int = None) -> dict:
    "Report where a Blender mesh is not symmetric (vertices with no mirror partner), snap near pairs to exact symmetry, or symmetrize one side onto the other. See mif_help."
    return _blender("symmetrize_mesh", object=object, mode=mode, axis=axis, direction=direction, threshold=threshold,
                    factor=factor, limit=limit)


@mcp.tool()
def bl_smooth_vertices(object: str, vertices: list = None, vertex_group: str = None, min_weight: float = None,
                       iterations: int = None, factor: float = None, axes: str = None, keep_boundary: bool = None,
                       laplacian: bool = None, preserve_volume: bool = None) -> dict:
    "Smooth a region of a Blender mesh - vertices, a vertex group or all - by iterations and factor along chosen axes, boundary kept, optionally Laplacian; displacement read back. See mif_help."
    return _blender("smooth_vertices", object=object, vertices=vertices, vertexGroup=vertex_group,
                    minWeight=min_weight, iterations=iterations, factor=factor, axes=axes, keepBoundary=keep_boundary,
                    laplacian=laplacian, preserveVolume=preserve_volume)


@mcp.tool()
def bl_snap_to_surface(object: str, target: str, vertices: list = None, vertex_group: str = None, mode: str = None,
                       axis: str = None, offset: float = None, outside_only: bool = None,
                       max_distance: float = None) -> dict:
    "Snap a Blender mesh's vertices onto another mesh's surface: NEAREST, PROJECT_NORMAL or PROJECT_AXIS, an offset along the surface normal, outside_only to push out only. Reports the vertices that found no surface."
    return _blender("snap_to_surface", object=object, target=target, vertices=vertices, vertexGroup=vertex_group,
                    mode=mode, axis=axis, offset=offset, outsideOnly=outside_only, maxDistance=max_distance)


@mcp.tool()
def bl_add_quads_on_surface(object: str, target: str, from_edges: list = None, quads: list = None,
                            distance: float = None, direction: list = None, steps: int = None, snap: str = None,
                            offset: float = None, mirror: str = None, weld_distance: float = None) -> dict:
    "Draw quads onto a Blender body: rows grown from boundary edges (from_edges, distance, steps) or quads by world corners, each vertex snapped (FACE, FACE_NEAREST), faces turned out, mirror adds a clipping Mirror."
    return _blender("add_quads_on_surface", object=object, target=target, fromEdges=from_edges, quads=quads,
                    distance=distance, direction=direction, steps=steps, snap=snap, offset=offset, mirror=mirror,
                    weldDistance=weld_distance)


@mcp.tool()
def bl_subdivide_multires(object: str, mode: str = None, levels: int = None, modifier: str = None,
                          sculpt_levels: int = None, viewport_levels: int = None, render_levels: int = None) -> dict:
    "Run a Blender Multires modifier's Subdivide (CATMULL_CLARK, SIMPLE, LINEAR), UNSUBDIVIDE or REBUILD, and set its sculpt, viewport and render levels. Total levels before and after, base and top-level vertex counts."
    return _blender("subdivide_multires", object=object, mode=mode, levels=levels, modifier=modifier,
                    sculptLevels=sculpt_levels, viewportLevels=viewport_levels, renderLevels=render_levels)


@mcp.tool()
def bl_remesh_mesh(object: str, mode: str, target_faces: int = None, target_ratio: float = None,
                   target_edge_length: float = None, symmetry: list = None, preserve_sharp: bool = None,
                   preserve_boundary: bool = None, preserve_attributes: bool = None, smooth_normals: bool = None,
                   seed: int = None, voxel_size: float = None, adaptivity: float = None, fix_poles: bool = None,
                   preserve_volume: bool = None, preserve_vertex_colors: bool = None, preserve_face_sets: bool = None,
                   preserve_mask: bool = None) -> dict:
    "Remesh a closed Blender mesh in place: mode QUADRIFLOW to a face count, ratio or edge length (symmetry welded), or VOXEL with fix poles, preserve volume and the keep flags. Faces, quads and poles before and after."
    return _blender("remesh_mesh", object=object, mode=mode, targetFaces=target_faces, targetRatio=target_ratio,
                    targetEdgeLength=target_edge_length, symmetry=symmetry, preserveSharp=preserve_sharp,
                    preserveBoundary=preserve_boundary, preserveAttributes=preserve_attributes,
                    smoothNormals=smooth_normals, seed=seed, voxelSize=voxel_size, adaptivity=adaptivity,
                    fixPoles=fix_poles, preserveVolume=preserve_volume, preserveVertexColors=preserve_vertex_colors,
                    preserveFaceSets=preserve_face_sets, preserveMask=preserve_mask)


@mcp.tool()
def bl_set_face_sets(object: str, mode: str, faces: list = None, box: dict = None, set_id: int = None,
                     iterations: int = None, strength: float = None) -> dict:
    "Make, extend or clear a Blender mesh's sculpt face sets - FROM_SEAMS, FROM_MATERIAL, FROM_LOOSE_PARTS, FROM_VISIBLE, FROM_FACES, FROM_BOX - or RELAX the borders between them. Set ids and face counts."
    return _blender("set_face_sets", object=object, mode=mode, faces=faces, box=box, setId=set_id,
                    iterations=iterations, strength=strength)


@mcp.tool()
def bl_relax_loop(object: str, vertices: list, mode: str = None, iterations: int = None, interpolation: str = None,
                  on_surface: bool = None) -> dict:
    "Even out a Blender mesh run of vertices (one path or one loop along edges): mode SPACE puts them at equal distances along it (interpolation LINEAR or CUBIC), RELAX averages each with its neighbours; on_surface keeps them on the mesh. Edge lengths before and after."
    return _blender("relax_loop", object=object, vertices=vertices, mode=mode, iterations=iterations,
                    interpolation=interpolation, onSurface=on_surface)


@mcp.tool()
def bl_dissolve_mesh_elements(object: str, indices: list = None, elements: str = None, use_face_split: bool = None,
                              use_boundary_tear: bool = None, also_verts: bool = None, mode: str = None,
                              angle: float = None, delimit: list = None, dissolve_boundaries: bool = None,
                              keep_ngons: bool = None) -> dict:
    "Dissolve Blender mesh vertices, edges or faces (indices) and keep the surface - the faces around them merge into one instead of leaving a hole. mode LIMITED is Limited Dissolve by angle (degrees, default 5) over the whole mesh or the indices given, with delimit and dissolve_boundaries; it splits the n-gons it merges unless keep_ngons."
    return _blender("dissolve_mesh_elements", object=object, indices=indices, elements=elements,
                    useFaceSplit=use_face_split, useBoundaryTear=use_boundary_tear, alsoVerts=also_verts, mode=mode,
                    angle=angle, delimit=delimit, dissolveBoundaries=dissolve_boundaries, keepNgons=keep_ngons)


@mcp.tool()
def bl_fill_mesh_loop(object: str, vertices: list = None, edges: list = None, loop_near: list = None,
                      mode: str = None, span: int = None, offset: int = None) -> dict:
    "Make a face from Blender mesh vertices, or fill an open loop (edges, or loop_near [x,y,z]) as one NGON, TRIANGLES or a GRID of quads (span, offset). New faces take their neighbours' material and UVs."
    return _blender("fill_mesh_loop", object=object, vertices=vertices, edges=edges, loopNear=loop_near, mode=mode,
                    span=span, offset=offset)


@mcp.tool()
def bl_mark_seams(object: str = None, clear: bool = None, by_material_boundary: bool = None,
                  by_vertex_group_boundary: str = None, min_weight: float = None, boundary_loop_near: list = None,
                  path: dict = None, selector: dict = None, dry_run: bool = None, by_bones=None,
                  armature: str = None, mark: str = None, value: bool | float = None, preview_id: str = None,
                  preview_ttl_seconds: int = None, from_islands: bool | str = None) -> dict:
    "Mark edges of a Blender mesh by rule without unwrapping: mark SEAM (default), SHARP, CREASE or BEVEL_WEIGHT, value true/false or 0-1. Rules: material or vertex-group boundaries, bone panels from skin weights (by_bones), the boundary loop nearest a point, the cheapest edge path {from, to, nearPlane {point, normal}} that hugs a plane, UV island borders (from_islands), or a bl_select_edges selector. Rules join; clear removes the mark first. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, clear=clear,
                                        by_material_boundary=by_material_boundary,
                                        by_vertex_group_boundary=by_vertex_group_boundary, min_weight=min_weight,
                                        boundary_loop_near=boundary_loop_near, path=path, selector=selector,
                                        dry_run=dry_run, by_bones=by_bones, armature=armature, mark=mark, value=value,
                                        preview_ttl_seconds=preview_ttl_seconds, from_islands=from_islands)
                or _blender("mark_seams", previewId=preview_id))
    return _blender("mark_seams", object=object, clear=clear, byMaterialBoundary=by_material_boundary,
                    byVertexGroupBoundary=by_vertex_group_boundary, minWeight=min_weight,
                    boundaryLoopNear=boundary_loop_near, path=path, selector=selector, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds,
                    byBones=by_bones, armature=armature, mark=mark, value=value, fromIslands=from_islands)


@mcp.tool()
def bl_mesh_distance(object: str, target: str, bins: dict = None, percentiles: list = None, region: dict = None,
                     evaluated: bool = None, max_distance: float = None, limit: int = None) -> dict:
    "How far one Blender mesh stands from another - a garment from its body, a cuff from its sleeve: each vertex's signed gap to the nearest point of target (negative inside), as stats overall and in bins {zStep, sectors, center} by height and angle round an axis, with radius (cross-section) and edge-length stats per bin."
    return _blender("mesh_distance", object=object, target=target, bins=bins, percentiles=percentiles, region=region,
                    evaluated=evaluated, maxDistance=max_distance, limit=limit)


@mcp.tool()
def bl_deformation_stretch(object: str, frames: list, regions: dict = None, action: str = None,
                           threshold: float = None, percentiles: list = None, worst: int = None,
                           allow_simulation: bool = None) -> dict:
    "How much a skinned Blender mesh's edges stretch over animation frames, by region: posed over rest edge length per frame as p05/p50/p95/max and the share above threshold, per region {name: box | sphere | vertexGroup} in rest space, with the worst edges and their weights."
    return _blender("deformation_stretch", object=object, frames=frames, regions=regions, action=action,
                    threshold=threshold, percentiles=percentiles, worst=worst, allowSimulation=allow_simulation)


@mcp.tool()
def bl_build_rib(object: str, length: float, loop: dict = None, variant: str = None, axis: list = None, bone: dict = None,
                 rings: int = None, radius: float = None, under: str | list = None, ease: float = None,
                 blend: float = None, slot: str = None, slot_length: float = None, tiers: int = None,
                 scale: float = None, fullness: float = None, waves: int = None,
                 separate_objects: bool = None) -> dict:
    "Grow a band off a Blender mesh's open boundary loop along a bone or axis: a RIB (cuff, hem band) to a radius or to rays at under meshes plus ease, or a FRILL in tiers with a fullness reached by folds."
    return _blender("build_rib", object=object, loop=loop, variant=variant, axis=axis, bone=bone, length=length,
                    rings=rings, radius=radius, under=under, ease=ease, blend=blend, slot=slot, slotLength=slot_length,
                    tiers=tiers, scale=scale, fullness=fullness, waves=waves, separateObjects=separate_objects)


@mcp.tool()
def bl_gather_loop(object: str, strength: float, loop: dict = None, group: str = None, reach: float = None,
                   apply: bool = None, smooth_iterations: int = None, set_cloth: bool = None) -> dict:
    "Gather a Blender mesh's open boundary loop like elastic: a shrink weight group falling off from the rim, the cloth's shrink set to strength, and optionally the rim pulled in now and smoothed."
    return _blender("gather_loop", object=object, loop=loop, strength=strength, group=group, reach=reach, apply=apply,
                    smoothIterations=smooth_iterations, setCloth=set_cloth)


@mcp.tool()
def bl_trim_to_outline(object: str, radii, center: list = None, bone: dict = None, axis: list = None,
                       front: list = None, keep: str = None, relax: int = None) -> dict:
    "Trim a Blender sheet to a smooth outline round an axis or bone - radii {front, side, back} or [[angleDeg, r], ...] - cutting along the curve, its new edge snapped onto it and relaxed on the original surface."
    return _blender("trim_to_outline", object=object, radii=radii, center=center, bone=bone, axis=axis, front=front,
                    keep=keep, relax=relax)


@mcp.tool()
def bl_add_inner_layer(object: str, part_near: list = None, near_openings: float = None, openings: list = None,
                       offset: float = None, group: str = None, slot: str = None, faces=None, side: str = None,
                       join: str = None, flip_normals: bool = None, internal_lines: int = None,
                       lines_group: str = None, uv_offset: list = None, name: str = None) -> dict:
    "Make a Blender garment two-sided or lined: flipped, inset copies of chosen faces (part_near, near_openings, or faces), kept loose, joined at the rims into one closed shell, or a new object; evenly spaced lines in a band."
    return _blender("add_inner_layer", object=object, partNear=part_near, nearOpenings=near_openings,
                    openings=openings, offset=offset, group=group, slot=slot, faces=faces, side=side, join=join,
                    flipNormals=flip_normals, internalLines=internal_lines, linesGroup=lines_group,
                    uvOffset=uv_offset, name=name)


@mcp.tool()
def bl_quilt_surface(object: str, cell_height: float, mode: str = None, axis: str = None, across: str = None,
                     cell_width: float = None, start: float = None, start_across: float = None, pinch: float = None,
                     inflation: float = None, group: str = None, puff_group: str = None, pressure: float = None,
                     pin_seams: bool = None) -> dict:
    "Divide a Blender garment into quilted cells: seam loops cut every cell_height (BANDS) or as a GRID, pinched in, cells puffed out; seams and puff become vertex groups, and the cloth's pin and pressure groups when pressure is given."
    return _blender("quilt_surface", object=object, cellHeight=cell_height, mode=mode, axis=axis, across=across,
                    cellWidth=cell_width, start=start, startAcross=start_across, pinch=pinch, inflation=inflation,
                    group=group, puffGroup=puff_group, pressure=pressure, pinSeams=pin_seams)


@mcp.tool()
def bl_add_accessory_physics(object: str, body: str, cage_density: float = None, margin: float = None,
                             pin_near: list = None, pin_radius: float = None, pin_within: float = None,
                             collision_with: list = None, cloth: dict = None, name: str = None,
                             group: str = None) -> dict:
    "Simulate a Blender chain, necklace or lace through a low-poly cage: built round it, pinned near points or where it rests on the body, run as cloth with the body colliding; the accessory follows by a bound Surface Deform."
    return _blender("add_accessory_physics", object=object, body=body, cageDensity=cage_density, margin=margin,
                    pinNear=pin_near, pinRadius=pin_radius, pinWithin=pin_within, collisionWith=collision_with,
                    cloth=cloth, name=name, group=group)


@mcp.tool()
def bl_wrap_to_target(object: str = None, target: str = None, wrap_method: str = None, offset: float = None,
                      pins: dict = None, landmarks: list = None, landmark_radius: float = None,
                      smooth_iterations: int = None, reach: float = None, dry_run: bool = None, preview_id: str = None,
                      preview_ttl_seconds: int = None) -> dict:
    "Fit a Blender mesh onto a target surface keeping its topology (a template head onto a scan): landmark pairs moved first, pinned vertices held, relaxed and re-projected; the distance to the target before and after. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, target=target, wrap_method=wrap_method,
                                        offset=offset, pins=pins, landmarks=landmarks,
                                        landmark_radius=landmark_radius, smooth_iterations=smooth_iterations,
                                        reach=reach, dry_run=dry_run, preview_ttl_seconds=preview_ttl_seconds)
                or _blender("wrap_to_target", previewId=preview_id))
    return _blender("wrap_to_target", object=object, target=target, wrapMethod=wrap_method, offset=offset, pins=pins,
                    landmarks=landmarks, landmarkRadius=landmark_radius, smoothIterations=smooth_iterations,
                    reach=reach, dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_retopo_panels(panels: list, edge_length: float = None, seams: list = None, rings: list = None,
                     uv_map: str = None, name: str = None, collection: str = None) -> dict:
    "Quad-grid retopo of flat Blender pattern panels: each outline split at four corners and filled by a Coons patch at edge_length, the panel's UVs sampled on, and seams [[{panel, near}, {panel, near}]] given equal vertex counts. A hole is ringed by closed quad loops; rings [{panel, near, loops}] sets how many, or lays rows along a curved side."
    return _blender("retopo_panels", panels=panels, edgeLength=edge_length, seams=seams, rings=rings, uvMap=uv_map,
                    name=name, collection=collection)


@mcp.tool()
def bl_transfer_by_uv(object: str, source: str, uv_map: str = None, source_uv_map: str = None,
                      weld_distance: float = None, levels: int = None, evaluated: bool = None) -> dict:
    "Move a Blender mesh's vertices to the point of a source surface at the same UV (a flat retopo onto its drape), weld seam pairs at their centre, and make an upward LOD chain (levels: subdivide, transfer again)."
    return _blender("transfer_by_uv", object=object, source=source, uvMap=uv_map, sourceUvMap=source_uv_map,
                    weldDistance=weld_distance, levels=levels, evaluated=evaluated)

@mcp.tool()
def bl_loft_garment(body: str, armature: str, rings: list, segments: int = None, ease: float = None,
                    offset: float = None, edge_length: float = None, evaluated: bool = None, name: str = None,
                    collection: str = None, garment: str = None, ease_table: dict = None) -> dict:
    "Loft a Blender garment tube round a body through rings [{bone, at, ease, offset}] at its bones: each ring the body's own cross-section square to the bone, eased out; rows at edge_length, UVs down the tube."
    return _blender("loft_garment", body=body, armature=armature, rings=rings, segments=segments, ease=ease,
                    offset=offset, edgeLength=edge_length, evaluated=evaluated, name=name, collection=collection,
                    garment=garment, easeTable=ease_table)


@mcp.tool()
def bl_measure_garment_ease(body: str, garment: str, armature: str, rings: list, segments: int = None,
                            evaluated: bool = None) -> dict:
    "Measure a Blender garment's ease off the body it was fitted to, ring by ring at its bones (rings [{bone, at, region}]): body and garment perimeters, ease, gaps; table {region: ease} feeds bl_loft_garment's ease_table."
    return _blender("measure_garment_ease", body=body, garment=garment, armature=armature, rings=rings,
                    segments=segments, evaluated=evaluated)


@mcp.tool()
def bl_clear_envelope(object: str = None, envelope: str = None, center: list = None, margin: float = None,
                      cone_angle: float = None, mask=None, protect=None, max_push: float = None, wall: float = None,
                      dry_run: bool = None, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Push a lined Blender garment out from a centre until it clears an envelope (a head, body, hair hull, or hair Curves) by margin, shell and lining moved alike so the wall keeps its thickness; mask fades it, protect holds a region. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, envelope=envelope, center=center, margin=margin,
                                        cone_angle=cone_angle, mask=mask, protect=protect, max_push=max_push,
                                        wall=wall, dry_run=dry_run, preview_ttl_seconds=preview_ttl_seconds)
                or _blender("clear_envelope", previewId=preview_id))
    return _blender("clear_envelope", object=object, envelope=envelope, center=center, margin=margin,
                    coneAngle=cone_angle, mask=mask, protect=protect, maxPush=max_push, wall=wall,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_points_inside(object: str, points: list = None, source: str = None, center: list = None, regions: dict = None,
                     wall: float = None, limit: int = None) -> dict:
    "Count points (a list, a hair Curves object, a point cloud or a mesh's vertices) outside, inside or in the wall of a Blender garment mesh, by region; strands with a point outside."
    return _blender("points_inside", object=object, points=points, source=source, center=center, regions=regions,
                    wall=wall, limit=limit)


@mcp.tool()
def bl_fit_mesh(object: str = None, target: str = None, mode: str = None, iterations: int = None,
                samples: int = None, reject: float = None, init: str = None, landmarks: list = None,
                region: dict = None, carry: list = None, tolerance: float = None, dry_run: bool = None,
                preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Fit one Blender mesh onto another (RIGID or UNIFORM scale) by iterative closest points, optionally from landmark pairs; moves it and carry by the transform and reports the fit error before and after. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, target=target, mode=mode, iterations=iterations,
                                        samples=samples, reject=reject, init=init, landmarks=landmarks,
                                        region=region, carry=carry, tolerance=tolerance, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("fit_mesh", previewId=preview_id))
    return _blender("fit_mesh", object=object, target=target, mode=mode, iterations=iterations, samples=samples,
                    reject=reject, init=init, landmarks=landmarks, region=region, carry=carry, tolerance=tolerance,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)

@mcp.tool()
def bl_build_sole(body: str, armature: str, bone: str, thickness: float = None, margin: float = None,
                  heel_thickness: float = None, segments: int = None, ground: float = None,
                  sample_height: float = None, evaluated: bool = None, name: str = None,
                  collection: str = None, last: str = None, footprint_top: float = None,
                  heel_length: float = None) -> dict:
    "A Blender shoe sole round a foot's footprint: the foot under a foot bone (heel to toe) seen from above, from sample_height up to footprint_top (default the bone's higher end, the ankle), eased out by margin, a closed solid from the ground up to thickness at the toe and heel_thickness at the heel. With heel_length the heel is a block: the top flat at heel_thickness from the footprint's back to heel_length along the bone and flat at thickness ahead of it, with a vertical breast between them."
    return _blender("build_sole", body=body, armature=armature, bone=bone, thickness=thickness, margin=margin,
                    heelThickness=heel_thickness, segments=segments, ground=ground, sampleHeight=sample_height,
                    evaluated=evaluated, name=name, collection=collection, last=last, footprintTop=footprint_top,
                    heelLength=heel_length)

@mcp.tool()
def bl_attach_hood(object: str, body: str, armature: str, rings: list, ease: float = None, offset: float = None,
                   edge_length: float = None, neck: dict = None, front: list = None, opening: dict = None,
                   crown_lift: float = None, evaluated: bool = None) -> dict:
    "Loft a Blender hood up from a garment's neckline through rings [{bone, at, ease, offset}] round the head, close it over the crown with a centre seam and cut the face opening {halfAngle, bottom, top}: every new face a quad."
    return _blender("attach_hood", object=object, body=body, armature=armature, rings=rings, ease=ease, offset=offset,
                    edgeLength=edge_length, neck=neck, front=front, opening=opening, crownLift=crown_lift,
                    evaluated=evaluated)

@mcp.tool()
def bl_attach_sleeve(object: str, body: str, armature: str, rings: list, ease: float = None, offset: float = None,
                     edge_length: float = None, armhole_radius: float = None, min_facing: float = None,
                     evaluated: bool = None, garment: str = None, ease_table: dict = None) -> dict:
    "Cut a round armhole in a Blender garment where an arm bone's line meets it and loft the sleeve out of it through rings [{bone, at, ease, offset}] (as bl_loft_garment's): one garment, every new face a quad."
    return _blender("attach_sleeve", object=object, body=body, armature=armature, rings=rings, ease=ease,
                    offset=offset, edgeLength=edge_length, armholeRadius=armhole_radius, minFacing=min_facing,
                    evaluated=evaluated, garment=garment, easeTable=ease_table)


@mcp.tool()
def bl_displace_by_rule(object: str, rules: list, name: str = None) -> dict:
    "A Blender high poly's detail by rule: SEAM grooves and STITCH dashes along a world polyline or the mesh's BOUNDARY, UV_SEAMS or SHARP edges, each vertex moved along its normal - the detail a high-to-low bake carries."
    return _blender("displace_by_rule", object=object, rules=rules, name=name)

@mcp.tool()
def bl_blend_weights(object: str = None, source: str = None, mask=None, smooth: int = None,
                     smooth_factor: float = None, max_influences: int = None, normalize: bool = None,
                     groups: list = None, dry_run: bool = None, preview_id: str = None,
                     preview_ttl_seconds: int = None) -> dict:
    "Blend a source mesh's skin weights into a Blender garment under a soft mask - {box | sphere | band {zFrom, zTo}} with a falloff, or a vertex group's weights - then smooth inside the mask, cap the influences and normalize. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, source=source, mask=mask, smooth=smooth,
                                        smooth_factor=smooth_factor, max_influences=max_influences,
                                        normalize=normalize, groups=groups, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("blend_weights", previewId=preview_id))
    return _blender("blend_weights", object=object, source=source, mask=mask, smooth=smooth,
                    smoothFactor=smooth_factor, maxInfluences=max_influences, normalize=normalize, groups=groups,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_pose_coverage(garment: str, body: str, frames: list, compare_to: str = None, region: dict = None,
                     reach: float = None, limit: int = None, allow_simulation: bool = None) -> dict:
    "Where a Blender body shows through a garment per frame: a ray from each body vertex along its posed normal must meet the garment (inside vs hole). compare_to, an earlier garment version, leaves only what the change broke."
    return _blender("pose_coverage", garment=garment, body=body, frames=frames, compareTo=compare_to, region=region,
                    reach=reach, limit=limit, allowSimulation=allow_simulation)


@mcp.tool()
def bl_recipe_fit_outfit(garment: str, body: str, source_body: str = None, armature: str = None, output: str = None,
                         method: str = None, align: str = None, ease: float = None, landmarks: list = None,
                         falloff: float = None, clearance: float = None, smooth: int = None, poses: list = None,
                         margin: float = None, spread: float = None, max_push: float = None, passes: int = None,
                         cover_distance: float = None, edge_band: float = None, tolerance: int = None,
                         min_patch: int = None,
                         straps: dict = None, hide_covered: bool = None, keep_on_refusal: bool = None, keep_poses: bool = None,
                         allow_simulation: bool = None) -> dict:
    "Fit a bought garment onto a body in one call, check it in test poses, and refuse - naming the pose and the place - where skin shows through or a strap leaves the body. Call mif_help(\"bl_recipe_fit_outfit\") first."
    return _blender("recipe_fit_outfit", garment=garment, body=body, sourceBody=source_body, armature=armature,
                    output=output, method=method, align=align, ease=ease, landmarks=landmarks, falloff=falloff,
                    clearance=clearance, smooth=smooth, poses=poses, margin=margin, spread=spread, maxPush=max_push,
                    passes=passes, coverDistance=cover_distance, edgeBand=edge_band, tolerance=tolerance,
                    minPatch=min_patch,
                    straps=straps, hideCovered=hide_covered, keepOnRefusal=keep_on_refusal, keepPoses=keep_poses,
                    allowSimulation=allow_simulation)


@mcp.tool()
def bl_push_out_for_poses(garment: str = None, body: str = None, frames: list = None, margin: float = None,
                          spread: float = None, region: dict = None, reach: float = None, max_push: float = None,
                          passes: int = None, dry_run: bool = None, allow_simulation: bool = None,
                          preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Push a Blender garment's rest vertices out along their normals where the body shows through in any frame given, enough for the worst pose plus margin, faded over spread; re-measured until no frame fails. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, garment=garment, body=body, frames=frames, margin=margin,
                                        spread=spread, region=region, reach=reach, max_push=max_push, passes=passes,
                                        dry_run=dry_run, allow_simulation=allow_simulation,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("push_out_for_poses", previewId=preview_id))
    return _blender("push_out_for_poses", garment=garment, body=body, frames=frames, margin=margin, spread=spread,
                    region=region, reach=reach, maxPush=max_push, passes=passes, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds, allowSimulation=allow_simulation)


@mcp.tool()
def bl_garment_reshape(object: str = None, body: str = None, mask=None, mode: str = None, axis: dict = None,
                       target: float = None, keep: float = None, smooth: int = None, reach: float = None,
                       dry_run: bool = None, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Pull a masked region of a Blender garment toward the nearest skin of a body, or squeeze it toward an axis against the body, keeping keep x the gap above target; a clearance floor, smoothing and a fold count. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, body=body, mask=mask, mode=mode, axis=axis,
                                        target=target, keep=keep, smooth=smooth, reach=reach, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("garment_reshape", previewId=preview_id))
    return _blender("garment_reshape", object=object, body=body, mask=mask, mode=mode, axis=axis, target=target,
                    keep=keep, smooth=smooth, reach=reach, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_set_skin_vertices(object: str, vertices: list = None, all: bool = None, group: str = None, radius=None,
                         scale: float = None, root: bool = None, loose: bool = None, read: bool = None,
                         limit: int = None) -> dict:
    "Write a Blender Skin modifier's per-vertex data, which lives on the mesh: radius (a number or [x, y]) or scale (multiplies it), root (one per connected piece; the piece's old root is cleared) and loose (branches leave the vertex freely). Pick the vertices with vertices (indices), all or group. Replies the radii written, the roots and the loose count."
    return _blender("set_skin_vertices", object=object, vertices=vertices, all=all, group=group, radius=radius,
                    scale=scale, root=root, loose=loose, read=read, limit=limit)


@mcp.tool()
def bl_uv_pack(object: str, uv_layer: str = None, margin: float = None, rotate: bool = None,
               equalize_by_area: bool = None, texture_size=None, merge_overlap: bool = None,
               target_texel_density: float = None, unit: str = None, region: list = None) -> dict:
    "Pack a Blender mesh's existing UV islands into the 0-1 tile WITHOUT re-unwrapping - optionally equalising them by area first, so every island gets one texel density. Reports the density spread before and after (1.0 = equal), the layout's area and bounds. texture_size packs for that texture: on [w, h] V is stretched by w / h so texels stay square and rotation is refused (the pack treats the tile as square). target_texel_density (needs texture_size, unit PX_PER_CM default) scales the layout to that density instead of the largest that fits. merge_overlap keeps overlapping islands stacked."
    return _blender("uv_pack", object=object, uvLayer=uv_layer, margin=margin, rotate=rotate,
                    equalizeByArea=equalize_by_area, textureSize=texture_size, mergeOverlap=merge_overlap,
                    targetTexelDensity=target_texel_density, unit=unit, region=region)


@mcp.tool()
def bl_uv_set_texel_density(object: str, texture_size, target: float = None, unit: str = None,
                            from_screen: dict = None, uv_layer: str = None, faces: list = None,
                            islands: list = None, mode: str = None, ladder: list = None, pack: bool = None,
                            margin: float = None) -> dict:
    "Scale a Blender mesh's UV islands to a texel density you choose on a texture size, and say which texture size holds the layout. target with unit PX_PER_CM (default) or PX_PER_M - or from_screen {sizeCm, pixels}, the density an object that size needs to fill that many pixels of the frame at its closest; there is no default target. texture_size is a number or [w, h]. mode PER_ISLAND brings each island to the target, AVERAGE the selection as one with relative sizes kept; faces or islands pick which. Replies the density per island before and after (min, median, max, spread), uvFillPercent, each ladder size's fill and whether a pack at the target fits 0-1, smallestFittingSize, texelShape (1 = square texels) and, with pack, the islands left outside."
    return _blender("uv_set_texel_density", object=object, textureSize=texture_size, target=target, unit=unit,
                    fromScreen=from_screen, uvLayer=uv_layer, faces=faces, islands=islands, mode=mode,
                    ladder=ladder, pack=pack, margin=margin)


@mcp.tool()
def bl_uv_transform_islands(object: str, islands: list = None, faces: list = None, translate: list = None,
                            rotate: float = None, scale=None, pivot=None, match_texel_density: dict = None,
                            texture_size=None, uv_layer: str = None, align_rotation: bool = None,
                            randomize: dict = None, arrange=None) -> dict:
    "Move, rotate (degrees) and scale chosen UV islands of a Blender mesh one by one, each about its own pivot, or bring them to a texel density (match_texel_density {target, textureSize} or {toIsland}). Replies per-island UV area, density and box before and after. Also align_rotation, randomize and arrange."
    return _blender("uv_transform_islands", object=object, islands=islands, faces=faces, translate=translate,
                    rotate=rotate, scale=scale, pivot=pivot, matchTexelDensity=match_texel_density,
                    textureSize=texture_size, uvLayer=uv_layer, alignRotation=align_rotation, randomize=randomize,
                    arrange=arrange)


@mcp.tool()
def bl_uv_straighten_islands(object: str, islands: list = None, faces: list = None, mode: str = None,
                             active_face: int = None, texture_size=None, uv_layer: str = None) -> dict:
    "Straighten quad UV islands of a Blender mesh into a grid: GRID makes every quad a square, RECTIFY rectangles in world proportion; area and centre kept. Replies each island's worst corner off 90 degrees and edge spread before and after."
    return _blender("uv_straighten_islands", object=object, islands=islands, faces=faces, mode=mode,
                    activeFace=active_face, textureSize=texture_size, uvLayer=uv_layer)


@mcp.tool()
def bl_uv_move_vertices(object: str, select: dict, translate: list = None, scale=None, rotate: float = None,
                        pivot=None, falloff: dict = None, pin: list = None, stretch: str = None,
                        uv_layer: str = None) -> dict:
    "Move, scale or rotate chosen UV vertices of a Blender mesh, with an optional proportional falloff {type, radius} onto their neighbours and pinned vertices held. Replies the selection's box and the area or angle stretch before and after."
    return _blender("uv_move_vertices", object=object, select=select, translate=translate, scale=scale, rotate=rotate,
                    pivot=pivot, falloff=falloff, pin=pin, stretch=stretch, uvLayer=uv_layer)


@mcp.tool()
def bl_uv_relax(object: str, select: dict = None, iterations: int = None, pin_boundary: bool = None,
                pin: list = None, stretch: str = None, uv_layer: str = None) -> dict:
    "Relax a Blender mesh's UVs with Blender's Minimize Stretch over chosen faces (default all), island borders or named vertices held. Replies faces relaxed, the largest move and the stretch before and after."
    return _blender("uv_relax", object=object, select=select, iterations=iterations, pinBoundary=pin_boundary,
                    pin=pin, stretch=stretch, uvLayer=uv_layer)


@mcp.tool()
def bl_bevel_edges(object_name: str = None, selector: dict = None, offset_uu: float = None,
                   segments: int = None, profile: float = None, preserve_x: bool = False,
                   clamp_overlap: bool = None, loop_slide: bool = None,
                   harden_normals: bool = None, miter_outer: str = None,
                   miter_inner: str = None, spread: float = None,
                   dry_run: bool = None, seam_band: float = None,
                   offset: float = None, preview_id: str = None, preview_ttl_seconds: int = None,
                   finish: bool | dict = None) -> dict:
    "Round or chamfer the selected edges with bmesh.ops.bevel (NOT bpy.ops.mesh.bevel, which needs an EDIT_MESH context and a real VIEW_3D area and therefore cannot run under blender -b). THE ADDON'S DEFAULTS, NOT A ROAD TILE'S: `selector` is required (the same keys as bl_select_edges - preview with it first), and exactly one of `offset` (Blender units) or `offset_uu` (Unreal units); segments defaults to 1, a flat chamfer (>1 rounds), profile to 0.5, and no axis is guarded unless preserve_x=true. Until 2026-09-21 this tool defaulted to a sidewalk tile's settings - 15 uu, 3 segments, the Y boundary edges, preserve_x on - and preserve_x snaps every vertex within the offset of the X extremes back onto them: measured on 5.0.1, bevelling all edges of a plain cube with it left all 96 vertices on the two X planes, the bevel beside those faces flattened, under ok:true. Pass preserve_x=true for a tile that must butt against the next one (mif_mesh_roundtrip keeps the tile defaults). clamp_overlap, loop_slide, harden_normals, miter_outer, miter_inner and spread are bmesh.ops.bevel's own options - left unset they keep the addon's defaults. THE RESULT CAN PASS THROUGH ITSELF while every guard holds: selfIntersectingFacePairs counts faces that cross and share no vertex (with the count before the bevel when there are any), and selfIntersectionNote says when the bevel made them - measured, a 0.7 BU bevel on a radius-0.5 tube's rim crossed 424 pairs unclamped on every build and 386 with clampOverlap on from Blender 4.4. shapeBefore/shapeAfter fingerprint every vertex position, because profile and offset move vertices without changing a count - compare shapeAfter to tell two bevels apart. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since. finish (true or {minAngleDeg, triangleBudget}) is the hard-surface finish: the sharp edges when no selector is given, hardened and weighted normals, flat faces shading flat, triangles held to the budget."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, selector=selector, offset_uu=offset_uu,
                                        segments=segments, profile=profile, preserve_x=preserve_x,
                                        clamp_overlap=clamp_overlap, loop_slide=loop_slide,
                                        harden_normals=harden_normals, miter_outer=miter_outer,
                                        miter_inner=miter_inner, spread=spread, dry_run=dry_run, seam_band=seam_band,
                                        offset=offset, preview_ttl_seconds=preview_ttl_seconds, finish=finish)
                or _blender("bevel_edges", previewId=preview_id))
    # THE SIX OPTIONS ABOVE WERE UNREACHABLE. The addon has always accepted them and passed them
    # straight to bmesh.ops.bevel - clamp_overlap, loop_slide and harden_normals are read at
    # ops_mesh.py:936-938 - and this tool sent none of them, so a caller could only ever get the
    # addon's defaults. Same shape as the cone/torus radii found the same night, and found the same
    # way: diff each addon op's reject_unknown set against the keys any _blender call site sends.
    #
    # Defaulted to None rather than to the addon's values on purpose: _blender drops unset params, so
    # None means "the addon decides" and this tool does not have to track defaults that live in
    # ops_mesh.py. Duplicating them here is how the two halves drift.
    #
    # AND SINCE 2026-09-21 THE REST FOLLOW THE SAME RULE. offset_uu, segments and profile carried a
    # sidewalk tile's values (15, 3, 0.5), the selector fell back to _MIF_DEFAULT_EDGE_SELECTOR and
    # preserve_x defaulted on - none of it the addon's, which asks for an offset, defaults to one
    # segment, guards no axis and refuses to guess edges. The tile defaults hurt a general mesh: the
    # preserve snap flattened every bevel beside a cube's X faces under ok:true (scratchpad measurement,
    # 2026-09-21, 5.0.1). `offset` joins because the addon's own refusal names it and this tool could
    # not send it. mif_mesh_roundtrip, the tile pipeline, keeps its tile defaults.
    # finish picks the sharp edges itself when no selector is given (the addon's finish.minAngleDeg, default 30).
    if selector is None and not finish:
        return {"ok": False, "error": "bl_bevel_edges needs a selector - which edges. Examples: "
                                      "{\"allEdges\": true}; {\"minAngleDeg\": 30} for the sharp ones; "
                                      "{\"boundaryOnly\": true, \"axis\": \"Y\", \"side\": \"both\"} for a "
                                      "road tile's two long edges. bl_select_edges previews a selector "
                                      "without changing anything. NOTHING was changed."}
    try:
        sel = _bl_selector(selector) if selector is not None else dict.fromkeys(
            ("boundaryOnly", "axis", "side", "tolerance", "minAngleDeg", "maxAngleDeg", "edgeIndices", "allEdges"))
    except _MifToolError as exc:
        return {"ok": False, "error": str(exc)}
    pres = _bl_preserve_axes(preserve_x)
    # dryRun reports WHICH edges would be touched and changes nothing. Both of these ops are
    # destructive and select edges by tolerance, so "did my selector catch what I meant" had no cheap
    # answer - you ran it and looked at the result. seamBand is how wide the near-but-off band around
    # the seam is; the addon defaults it to the larger of the snap and on tolerances, and a band
    # NARROWER than the snap tolerance is the blind spot its own comment warns about.
    #
    # THIS COMMENT LIVES ABOVE THE CALL, NOT INSIDE IT, and that is not a style preference. Its first
    # draft sat between the keywords and mentioned "ops_mesh.py" in brackets. param_reach captures a
    # call's keywords with a non-greedy match that stops at the first close paren, so the paren in
    # that citation TRUNCATED the capture and nine already-working parameters - every bevel option
    # added this morning, plus direction and flipNormals - were reported as newly unreachable. The
    # same [^)]* trap that hid move_tree_widget from audit_promise_flags earlier today.
    return _blender("bevel_edges", object=object_name,
                    boundaryOnly=sel["boundaryOnly"], axis=sel["axis"], side=sel["side"],
                    tolerance=sel["tolerance"], minAngleDeg=sel["minAngleDeg"],
                    maxAngleDeg=sel["maxAngleDeg"], edgeIndices=sel["edgeIndices"],
                    allEdges=sel["allEdges"],
                    offset=offset, offsetUU=offset_uu, segments=segments, profile=profile,
                    preserveAxes=pres, assertAxes=pres,
                    clampOverlap=clamp_overlap, loopSlide=loop_slide,
                    hardenNormals=harden_normals, miterOuter=miter_outer,
                    miterInner=miter_inner, spread=spread, finish=finish,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds, seamBand=seam_band)


@mcp.tool()
def bl_extrude_skirt(object_name: str = None, selector: dict = None, depth_uu: float = 15.0,
                     preserve_x: bool = True, direction: str = "down",
                     flip_normals: bool = False, dry_run: bool = None,
                     seam_band: float = None, allow_non_boundary: bool = None, preview_id: str = None,
                     preview_ttl_seconds: int = None) -> dict:
    "Extrude the selected boundary edge loops straight DOWN by depth_uu, forming a skirt - the fix for a flat-edged tile that hovers where the terrain falls away. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object_name=object_name, selector=selector,
                                        flip_normals=flip_normals, dry_run=dry_run, seam_band=seam_band,
                                        allow_non_boundary=allow_non_boundary,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("extrude_skirt", previewId=preview_id))
    try:
        sel = _bl_selector(selector)
    except _MifToolError as exc:
        return {"ok": False, "error": str(exc)}
    pres = _bl_preserve_axes(preserve_x)
    # dryRun reports WHICH edges would be touched and changes nothing. Both of these ops are
    # destructive and select edges by tolerance, so "did my selector catch what I meant" had no cheap
    # answer - you ran it and looked at the result. seamBand is how wide the near-but-off band around
    # the seam is; the addon defaults it to the larger of the snap and on tolerances, and a band
    # NARROWER than the snap tolerance is the blind spot its own comment warns about.
    #
    # THIS COMMENT LIVES ABOVE THE CALL, NOT INSIDE IT, and that is not a style preference. Its first
    # draft sat between the keywords and mentioned "ops_mesh.py" in brackets. param_reach captures a
    # call's keywords with a non-greedy match that stops at the first close paren, so the paren in
    # that citation TRUNCATED the capture and nine already-working parameters - every bevel option
    # added this morning, plus direction and flipNormals - were reported as newly unreachable. The
    # same [^)]* trap that hid move_tree_widget from audit_promise_flags earlier today.
    return _blender("extrude_skirt", object=object_name,
                    boundaryOnly=sel["boundaryOnly"], axis=sel["axis"], side=sel["side"],
                    tolerance=sel["tolerance"], minAngleDeg=sel["minAngleDeg"],
                    maxAngleDeg=sel["maxAngleDeg"], edgeIndices=sel["edgeIndices"],
                    allEdges=sel["allEdges"],
                    depthUU=depth_uu, direction=direction, flipNormals=flip_normals,
                    preserveAxes=pres, assertAxes=pres,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds, seamBand=seam_band,
                    allowNonBoundary=allow_non_boundary)


@mcp.tool()
def bl_set_material_slots(object: str = None, slots: list = None, allow_resize: bool = False, links: list = None,
                          dry_run: bool = False, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Set a Blender object's material slot NAMES, in ORDER. Slot order is what decides which Unreal material lands on which face, so a reordered list renders the wrong material on an otherwise perfect mesh. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, slots=slots, allow_resize=allow_resize, links=links,
                                        dry_run=dry_run, preview_ttl_seconds=preview_ttl_seconds)
                or _blender("set_material_slots", previewId=preview_id))
    return _blender("set_material_slots", object=object, slots=slots, allowResize=allow_resize, links=links,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_create_primitive(kind: str, name: str = "", size: float = None, radius: float = None,
                        location: list = None, rotation: list = None, segments: int = None,
                        ring_count: int = None, subdivisions: int = None, vertices: int = None,
                        depth: float = None, x_subdivisions: int = None,
                        y_subdivisions: int = None, radius1: float = None, radius2: float = None,
                        major_radius: float = None, minor_radius: float = None,
                        align: str = None, fill_type: str = None,
                        major_segments: int = None, minor_segments: int = None) -> dict:
    "Create a Blender primitive mesh - kind cube (size = full edge), sphere, icosphere, cylinder, cone, torus, plane, grid, circle or monkey; align WORLD/VIEW/CURSOR; fill_type NGON/TRIFAN/NOTHING. A sphere (alias uvsphere) takes radius. A CONE takes radius1/radius2 and a TORUS takes major_radius/minor_radius - neither accepts size or radius, and passing the wrong one is refused rather than reinterpreted. align is WORLD/VIEW/CURSOR; fill_type (NGON/TRIFAN/NOTHING) applies to a circle, a cylinder and a cone, whose ends are fans by default, and is refused by name elsewhere."
    # radius1/radius2 and majorRadius/minorRadius were accepted by the addon and sent by nothing, so
    # a cone or a torus could only be created at its DEFAULT dimensions over MCP. The op's own
    # docstring is explicit - "A cone takes radius1/radius2 and a torus majorRadius/minorRadius;
    # neither takes size or radius" - and it REFUSES size/radius for those kinds rather than
    # reinterpreting them, so there was no workaround either. Found 2026-08-31 by diffing each addon
    # op's reject_unknown set against the keys any _blender call site sends, which is param_reach's
    # question asked of the Blender half.
    # align and fillType were the last two create_primitive keys nothing could send, found by the
    # same diff a day later once param_reach stopped counting alias spellings as lost capability.
    # align is WORLD/VIEW/CURSOR and the addon validates it BEFORE creating anything, so a bad value
    # costs an error rather than an orphaned object. fillType applies to a circle (NGON/TRIFAN/NOTHING)
    # and the addon refuses it by name on kinds that have no such extra - a cube says so rather than
    # ignoring it.
    return _blender("create_primitive", kind=kind, name=name or None, size=size, radius=radius,
                    location=location, rotation=rotation, segments=segments,
                    ringCount=ring_count, subdivisions=subdivisions, vertices=vertices,
                    depth=depth, xSubdivisions=x_subdivisions, ySubdivisions=y_subdivisions,
                    radius1=radius1, radius2=radius2,
                    majorRadius=major_radius, minorRadius=minor_radius,
                    align=align, fillType=fill_type, majorSegments=major_segments, minorSegments=minor_segments)


@mcp.tool()
def bl_create_light(type: str = "POINT", name: str = "", location: list = None,
                    rotation: list = None, energy: float = None, color: list = None,
                    radius: float = None, size: float = None, size_y: float = None,
                    shape: str = None, spot_angle: float = None, spot_blend: float = None,
                    angle: float = None, shadow: bool = None, diffuse_factor: float = None,
                    specular_factor: float = None, temperature: float = None,
                    use_temperature: bool = None) -> dict:
    "Create a Blender light: type POINT, SUN, SPOT or AREA; energy in watts (a SUN's is irradiance), color [r,g,b], rotation in RADIANS. Type-specific settings are REFUSED on the wrong type rather than ignored - spot_angle on a POINT light is an error, not a no-op. rotation and spot_angle/angle are RADIANS. The response reports what the light IS, read back off the datablock. COLOUR TEMPERATURE IN KELVIN: temperature is how lighting is actually specified - 3200 tungsten, 6500 daylight - rather than as linear RGB, and it MULTIPLIES the colour rather than replacing it, so colour becomes a tint. Refused by name on a Blender without bpy.types.Light.use_temperature rather than silently ignored."
    # EVERY accepted key is sent. param_reach asks exactly this question of the Blender half, and a
    # capability the addon accepts but no tool forwards is one an MCP caller cannot reach at all -
    # which is how cone and torus dimensions went unreachable until 2026-08-31.
    return _blender("create_light", type=type, name=name or None, location=location,
                    rotation=rotation, energy=energy, color=color, radius=radius,
                    size=size, sizeY=size_y, shape=shape, spotAngle=spot_angle,
                    spotBlend=spot_blend, angle=angle, shadow=shadow,
                    diffuseFactor=diffuse_factor, specularFactor=specular_factor,
                    temperature=temperature, useTemperature=use_temperature)


@mcp.tool()
def bl_create_light_probe(name: str = "LightProbe", type: str = "SPHERE", location: dict | list = None,
                          rotation: dict | list = None, influence_distance: float = None,
                          influence_type: str = None, falloff: float = None,
                          clip_start: float = None, clip_end: float = None, collection: str = None) -> dict:
    "Create an EEVEE LIGHT PROBE - the objects that give a scene reflections and bounced light. The addon could make every other lighting object and not this one: bl_create_light makes lamps and the bl_set_light_* ops tune them, while a probe is what makes that light appear in a reflection or fill a room indirectly. A PROBE EMITS NOTHING - it captures its surroundings, so one placed where nothing reflective sits is the usual reason it looks like it does nothing. BLENDER RENAMED ALL THREE TYPES AT 4.2 and they are aliased here so one script works on 3.6 and 5.0 without branching: SPHERE/CUBE/CUBEMAP are one probe, PLANE/PLANAR another, VOLUME/GRID the third. On 3.6 the name you pass and the name Blender reports DIFFER - you pass CUBE and it calls the result CUBEMAP - so the response carries requestedType, passedType and type separately. FROM 4.2 EACH TYPE IS ITS OWN CLASS with different fields: only the SPHERE one has falloff, clipEnd and influence_type, while PLANE and VOLUME carry influence_distance and clip_start and nothing else. A setting the chosen type does not have is REFUSED naming that, rather than dropped, and `settingsOnThisType` in the response lists what this probe actually took. On 3.6 there is one class and all five apply to every type."
    return _blender("create_light_probe", name=name, type=type, location=location,
                    rotation=rotation, influenceDistance=influence_distance,
                    influenceType=influence_type, falloff=falloff, clipStart=clip_start,
                    clipEnd=clip_end, collection=collection)


@mcp.tool()
def bl_set_light_probe(object: str, settings: dict) -> dict:
    "Write an existing Blender light probe's own settings by RNA identifier - bake settings such as surfel_density, surface_bias, normal_bias, view_bias, use_data_display - each checked against this build's probe and read back; all or nothing."
    return _blender("set_light_probe", object=object, settings=settings)


@mcp.tool()
def bl_bake_light_probe(object: str = None, all: bool = None, type: str = None) -> dict:
    "Bake EEVEE's light probe cache - a volume probe's bounce light - for one probe (4.2+) or all; the right operator per build. Replies points, bake time, the cache state where readable and meshes outside the volume."
    return _blender("bake_light_probe", object=object, all=all, type=type)


@mcp.tool()
def bl_set_light_influence(object: str, cutoff_distance: float = None,
                           use_custom_distance: bool = None, volume_factor: float = None,
                           transmission_factor: float = None, spread: float = None,
                           soft_falloff: bool = None, square: bool = None,
                           show_cone: bool = None, cascade_count: int = None,
                           cascade_max_distance: float = None,
                           cascade_exponent: float = None,
                           cascade_fade: float = None) -> dict:
    "How far a Blender light REACHES and what it reaches into - the half neither bl_set_light nor bl_set_light_shadow covered. bl_set_light writes energy, colour, cone and size; bl_set_light_shadow writes the shadow settings; nothing wrote a light's INFLUENCE RADIUS, its volumetric contribution, an area light's spread, or a sun's shadow cascades - and those are most of what separates a lighting rig that renders well from one that merely exists. cutoff_distance is the performance knob: the distance past which the light is not evaluated at all, the direct equivalent of an attenuation radius in a game engine. A CUTOFF DOES NOTHING UNTIL use_custom_distance IS ON - Blender stores the distance either way, so setting it alone changes a number and not the render while every field reads back exactly as asked; passing cutoff_distance turns the toggle on unless you say otherwise, and the response says so. TWO AXES OF AVAILABILITY, and the refusal says WHICH failed: a property can be missing because this BUILD lacks it (transmission_factor and soft_falloff are 4.2 and later) or because this light TYPE lacks it (spread is AREA only, square and show_cone are SPOT only, the cascades are SUN only). Every row was read off bl_rna on 3.6.23, 4.2.17, 4.4.0 and 5.0.1 for all four light types. square and showCone (viewport display) change no pixel in EEVEE or Cycles on any build, and the cascade keys none under Cycles or under EEVEE from 4.2 (measured on 3.6-5.2); the reply names them under measuredNoEffect."
    return _blender("set_light_influence", object=object, cutoffDistance=cutoff_distance,
                    useCustomDistance=use_custom_distance, volumeFactor=volume_factor,
                    transmissionFactor=transmission_factor, spread=spread,
                    softFalloff=soft_falloff, square=square, showCone=show_cone,
                    cascadeCount=cascade_count, cascadeMaxDistance=cascade_max_distance,
                    cascadeExponent=cascade_exponent, cascadeFade=cascade_fade)


@mcp.tool()
def bl_set_light_shadow(object: str, enabled: bool = None, soft_size: float = None,
                        buffer_clip_start: float = None, color: list = None,
                        filter_radius: float = None, jitter: bool = None,
                        jitter_overblur: float = None, max_resolution: int = None,
                        contact_shadow: bool = None, contact_distance: float = None,
                        contact_bias: float = None, contact_thickness: float = None,
                        cycles_cast_shadow: bool = None, cycles_max_bounces: int = None,
                        cycles_mis: bool = None, is_portal: bool = None,
                        is_caustics_light: bool = None) -> dict:
    "The engine-specific shadow settings that live on a Blender LIGHT - the general half (hide_render, per-ray visibility, holdout, shadow catcher) is bl_set_object_visibility, because those live on the OBJECT. THIS AREA MOVES MORE THAN ANY OTHER, read off bl_rna on 3.6.23, 4.2.17, 4.4.0 and 5.0.1: light.cycles.cast_shadow exists on 3.6 ONLY and was removed at 4.2; CONTACT SHADOWS exist on 3.6 and 4.2 and EEVEE Next dropped them at 4.4; shadow_buffer_samples/_size are 3.6 only; and jitter, filter radius and max resolution arrived at 4.2. ANYTHING THIS BUILD LACKS IS REFUSED, naming which builds have it - accepting a key and writing it only where it exists is how a caller asks for shadows off, gets shadows on, and is told nothing. Values are read back per key because several are silently CLAMPED rather than refused, and `clamped` reports where the stored value differs from the request. Settings the ACTIVE ENGINE will not read (cycles.* under EEVEE, contact shadows under Cycles) are still written - a scene is often set up under one renderer and rendered with another - but reported in inertUnderThisEngine so the difference is visible."
    return _blender("set_light_shadow", object=object, enabled=enabled, softSize=soft_size,
                    bufferClipStart=buffer_clip_start, color=color, filterRadius=filter_radius,
                    jitter=jitter, jitterOverblur=jitter_overblur, maxResolution=max_resolution,
                    contactShadow=contact_shadow, contactDistance=contact_distance,
                    contactBias=contact_bias, contactThickness=contact_thickness,
                    cyclesCastShadow=cycles_cast_shadow, cyclesMaxBounces=cycles_max_bounces,
                    cyclesMIS=cycles_mis, isPortal=is_portal, isCausticsLight=is_caustics_light)


@mcp.tool()
def bl_set_light_linking(object: str, receiver_collection: str = None,
                         blocker_collection: str = None, clear_receivers: bool = None,
                         clear_blockers: bool = None, receiver_members: dict = None,
                         blocker_members: dict = None) -> dict:
    "Control WHICH objects a Blender light affects. 'This key light hits the product and not the backdrop' is routine in product and archviz work and cannot be faked - moving the light changes the look, flagging it with geometry changes the reflections, and turning it down changes everything. Light linking is the only correct answer and nothing here could reach it. receiver_collection limits what the light illuminates; blocker_collection limits what casts its shadows; both are created if they do not exist. Requires Blender 4.0+, and an older build is REFUSED BY NAME with the version rather than accepting the keys and doing nothing. litsNothing is reported because an EMPTY receiver collection illuminates nothing at all - a legitimate state mid-setup and a catastrophic one to render from, and identical to a correct link from every other field. Call mif_help(\"bl_set_light_linking\") first."
    return _blender("set_light_linking", object=object, receiverCollection=receiver_collection,
                    blockerCollection=blocker_collection, clearReceivers=clear_receivers,
                    clearBlockers=clear_blockers, receiverMembers=receiver_members,
                    blockerMembers=blocker_members)


@mcp.tool()
def bl_set_light_ies(object: str, filepath: str = None, text: str = None,
                     strength: float = None, clear: bool = None) -> dict:
    "Give a Blender light a real-world IES photometric profile, or clear it. An IES file is a MEASURED distribution from a fixture manufacturer - the shape of light a real luminaire throws - and it is how archviz and product lighting stop looking like computer graphics. No amount of energy, radius or spot-angle adjustment substitutes for one. A LIGHT'S DISTRIBUTION IS NOT A PROPERTY, IT IS A NODE TREE: Blender wires an IES Texture node into the Strength of an Emission shader, which is why this is its own op rather than a key on bl_set_light, and the addon had never touched a light's node tree at all. Pass filepath for an external .ies, or text for the data inline; both together are refused, and a filepath that does not exist is refused BEFORE the tree is built, because a half-built tree on a light that then renders black is worse than no change. linkedToEmission is the postcondition - an IES node sitting unconnected changes nothing and looks entirely correct in the node list. Call mif_help(\"bl_set_light_ies\") first."
    return _blender("set_light_ies", object=object, filepath=filepath, text=text,
                    strength=strength, clear=clear)


@mcp.tool()
def bl_move_keyframes(object: str = None, data_path: str = None, index: int = None,
                      offset: float = None, scale: float = None, pivot: float = None,
                      frame_start: float = None, frame_end: float = None, target: str = None,
                      datablock: str = None) -> dict:
    "Retime Blender keyframes - shift them by an offset, or scale the timing about a pivot. Retiming is a core animation operation and nothing here could do it: bl_delete_keyframe and bl_set_keyframe together can only rebuild an animation from scratch, losing every handle, interpolation and easing on the way, so 'make this 20% slower' meant re-authoring it. THE HANDLES MOVE WITH THE KEYS - a bezier handle is stored in ABSOLUTE frame coordinates, so moving only the key leaves its handles behind and silently reshapes the curve while the interpolation still reads BEZIER. Keys are moved in the direction of travel (later-first when shifting forwards) because Blender keeps them sorted and a key crossing one that has not moved yet makes the walk skip or revisit it. offset and scale are refused together as an ambiguous order. framesChanged reports whether the frame list actually differs, since an offset of 0 or a scale of 1 does nothing whatever the call said. A NODE TREE: target world, material or node_group with datablock (its name) works on that tree's own curves instead of an object's - data_path is the path on the tree, e.g. 'nodes[\"Mapping\"].inputs[2].default_value'. Call mif_help(\"bl_move_keyframes\") first."
    return _blender("move_keyframes", object=object, dataPath=data_path, index=index,
                    offset=offset, scale=scale, pivot=pivot, frameStart=frame_start,
                    frameEnd=frame_end, target=target, datablock=datablock)


@mcp.tool()
def bl_copy_keyframes(object: str, data_paths: list = None, bones: list = None, all: bool = None, index: int = None,
                      from_range: dict = None, to: dict = None, to_data_path: str = None, to_object: str = None,
                      repeat: int = None, period: float = None, merge: str = None, value_scale: float = None,
                      value_offset: float = None, pivot=None, flip: bool = None, name_rule=None) -> dict:
    "Copy Blender keys with their handles to another frame, channel or object - repeated, value-scaled about a pivot, or flipped onto the other side; reports the evaluated difference."
    return _blender("copy_keyframes", object=object, dataPaths=data_paths, bones=bones, all=all, index=index,
                    fromRange=from_range, to=to, toDataPath=to_data_path, toObject=to_object, repeat=repeat,
                    period=period, merge=merge, valueScale=value_scale, valueOffset=value_offset, pivot=pivot,
                    flip=flip, nameRule=name_rule)


@mcp.tool()
def bl_clean_keyframes(object: str, data_paths: list = None, bones: list = None, all: bool = None, index: int = None,
                       threshold: float = None, channels: bool = None, euler_filter=None, frame_start: float = None,
                       frame_end: float = None) -> dict:
    "Remove Blender keys whose removal moves the curve less than threshold, and/or make euler curves continuous (euler_filter true or 'ONLY'); reports keys before and after and the largest deviation."
    return _blender("clean_keyframes", object=object, dataPaths=data_paths, bones=bones, all=all, index=index,
                    threshold=threshold, channels=channels, eulerFilter=euler_filter, frameStart=frame_start,
                    frameEnd=frame_end)


@mcp.tool()
def bl_set_camera_panorama(object: str, panorama_type: str = None, fisheye_fov: float = None,
                           fisheye_lens: float = None, latitude_min: float = None,
                           latitude_max: float = None, longitude_min: float = None,
                           longitude_max: float = None) -> dict:
    "Configure a panoramic Blender camera - the settings bl_create_camera could accept a PANO type for and never reach. That made PANO a declared-and-unreachable: the type validated, the camera was created, and nothing in the addon could set a single panorama property, which is worse than not offering the type. A camera that is not PANO is refused rather than having the settings stored where they will never be used. Panorama settings moved between versions - camera.cycles in 3.x, the camera data in 4.x+ - so both are tried and storedOn reports which one this Blender uses. PANORAMIC RENDERING IS A CYCLES FEATURE: engineHonoursPanorama says whether the current engine will actually use any of this, rather than leaving it to be discovered when the render comes back rectilinear. Call mif_help(\"bl_set_camera_panorama\") first."
    return _blender("set_camera_panorama", object=object, panoramaType=panorama_type,
                    fisheyeFov=fisheye_fov, fisheyeLens=fisheye_lens, latitudeMin=latitude_min,
                    latitudeMax=latitude_max, longitudeMin=longitude_min,
                    longitudeMax=longitude_max)


@mcp.tool()
def bl_set_nla_track(object: str, track: str, mute: bool = None, solo: bool = None,
                     lock: bool = None, rename: str = None, remove: bool = None) -> dict:
    "Mute, solo, lock, rename or remove an NLA track (\"*\": all)."
    return _blender("set_nla_track", object=object, track=track, mute=mute, solo=solo,
                    lock=lock, rename=rename, remove=remove)


@mcp.tool()
def bl_add_nla_strip(object: str, action: str, track: str = None, start: int = None,
                     strip_name: str = None, blend_type: str = None, influence: float = None,
                     push_down_active: bool = None) -> dict:
    "Put an action on an NLA track as a strip - how several clips live on one Blender object. An object holds ONE active action at a time; the NLA is how a walk, an idle and a wave coexist on the same rig, how they blend, and what glTF reads to export multiple clips. THE TRAP IS THAT THE ACTIVE ACTION SHADOWS THE WHOLE STACK: Blender evaluates animation_data.action ON TOP of the NLA, so an object with an active action set plays that and every strip below contributes nothing while the stack reads as perfectly correct. activeActionShadowsNla reports it, and push_down_active moves the active action onto its own track first - what the UI's Push Down button does. Overlapping strips on one track are refused by Blender and the message says so. Read the result back with bl_list_animation_data. Call mif_help(\"bl_add_nla_strip\") first. INFLUENCE IS ANIMATED OR IGNORED: Blender plays a strip at full strength unless its animated influence is on, so passing influence turns it on and keys the value (measured on every supported build: 0.5 plays at half; without the flag every value played the full action). The response and bl_list_animation_data report animatedInfluence and effectiveInfluence - what the strip plays at. With nothing beneath the strip, a partial influence blends every channel toward 0, not toward the object's own value - influenceNote says so when the strip lands on the bottom track."
    return _blender("add_nla_strip", object=object, action=action, track=track, start=start,
                    stripName=strip_name, blendType=blend_type, influence=influence,
                    pushDownActive=push_down_active)


@mcp.tool()
def bl_list_custom_properties(object: str, bone: str = None, limit: int = None, data: bool = None) -> dict:
    "Custom properties on a Blender object or pose bone, with their UI range. These are how a rig exposes controls, and glTF writes them into the engine as `extras` - so they are metadata that TRAVELS, not just annotations. The UI min/max is a separate store from the value and is reported alongside it, because a slider without a range is not a control. Blender's own internal keys (cycles settings, _RNA_UI) share the namespace and are named under skippedInternalKeys rather than silently folded into the count. `limit` bounds the rows - measured at ~132 bytes a property, so 200 on one object is 26 KB - while `count` stays the true total. Call mif_help(\"bl_list_custom_properties\") first."
    # `limit` added 2026-09-20: measured at ~132 bytes a property, and 200 on one cube answered
    # 26,454 bytes. Metadata that travels is metadata somebody generates.
    return _blender("list_custom_properties", object=object, bone=bone, limit=limit, data=data)


@mcp.tool()
def bl_set_custom_property(object: str, key: str, value=None, bone: str = None,
                           min: float = None, max: float = None, description: str = None,
                           delete: bool = None, data: bool = None, library_overridable: bool = None) -> dict:
    "Set a custom property, and its UI range, on a Blender object or pose bone. THE TYPE IS REPORTED BACK because Blender coerces silently: an int written where a float was meant stays an int, and a driver or an exporter reading it later gets a different type than the caller thinks they stored - typeChanged names that. min/max set the UI range, without which a rig slider is not a control; if this Blender will not take the range the op says the value WAS written and the bounds were not, rather than reporting a clean success. Pass delete:true to remove one. Writing an internal key (cycles, _RNA_UI) is refused rather than allowed to collide with Blender's own storage. Call mif_help(\"bl_set_custom_property\") first."
    return _blender("set_custom_property", object=object, key=key, value=value, bone=bone,
                    min=min, max=max, description=description, delete=delete, data=data,
                    libraryOverridable=library_overridable)


@mcp.tool()
def bl_add_driver(object: str, data_path: str, index: int = None, expression: str = None,
                  variables: list = None, driver_type: str = None, curve: dict = None, map_range: dict = None) -> dict:
    "Drive a Blender property (data_path, index for one element) with an expression over variables [{name, object, dataPath}], and prove the driver actually EVALUATES. Drivers are the one animation feature that fails completely silently: a broken expression, or a variable pointing at an object that no longer exists, stays in place and evaluates to ZERO - nothing errors, nothing warns, and every field a caller can read looks perfectly correct. Blender shows it as a coloured field in the UI and reports it nowhere else. So this refuses a data_path that does not resolve (Blender would create the driver permanently invalid), refuses a variable whose target object does not exist, refuses a second driver on a path that already has one, and reports isValid plus the driven property read back through the depsgraph. Read them back with bl_list_animation_data. Call mif_help(\"bl_add_driver\") first."
    return _blender("add_driver", object=object, dataPath=data_path, index=index,
                    expression=expression, variables=variables, driverType=driver_type, curve=curve,
                    mapRange=map_range)


@mcp.tool()
def bl_remove_driver(object: str, data_path: str, index: int = None) -> dict:
    "Remove a Blender driver and report what the property fell back to. A property with its driver removed returns to whatever it was last set to, which is NOT necessarily what it was displaying while driven - so the value is read back rather than assumed. A path with no driver is refused with a pointer to bl_list_animation_data rather than reported as a successful removal of nothing, and the removal is re-checked afterwards. Call mif_help(\"bl_remove_driver\") first."
    return _blender("remove_driver", object=object, dataPath=data_path, index=index)


@mcp.tool()
def bl_add_constraint(object: str, type: str, bone: str = None, target: str = None,
                      subtarget: str = None, influence: float = None,
                      constraint_name: str = None) -> dict:
    "Add an object or bone constraint in Blender, and MEASURE that it moves the thing. Constraints are how camera and light rigs are actually built - a Track To on an empty stays correct as the target moves, which a one-shot bl_aim_object cannot - and nothing here could create one. THE MEASUREMENT IS THE DESIGN: a constraint does NOT touch obj.matrix_world, it is applied by the depsgraph at evaluation, so reading the object's own transform reports no change for every constraint that works perfectly. This samples the EVALUATED world matrix before and after and reports movedDistance and turnedRadians. hadEffect false is not automatically wrong - a Copy Location onto something already in place moves nothing - so read it beside isValid: valid and inert means nothing to do, INVALID and inert means a target that does not resolve. Call mif_help(\"bl_add_constraint\") first."
    return _blender("add_constraint", object=object, type=type, bone=bone, target=target,
                    subtarget=subtarget, influence=influence, constraintName=constraint_name)


@mcp.tool()
def bl_set_constraint(object: str, constraint: str, settings: dict, bone: str = None) -> dict:
    "Write the TYPE-SPECIFIC settings of an existing Blender constraint - the half nothing could reach. bl_add_constraint sets target, subtarget and influence, which every constraint shares; everything that makes a constraint do its particular job was unreachable. Measured on 4.4: 28 constraint types carry 253 type-specific writable properties between them and NOT ONE type has zero, so no constraint could ever be fully configured. What that cost concretely: a LIMIT_ROTATION or LIMIT_LOCATION is INERT until its limits are set, so the addon could add a limit constraint that limits nothing - present in the stack, reported by bl_list_constraints, doing precisely nothing; a TRACK_TO tracked along Blender's default axis, wrong for most cameras and turrets; an IK took the default chain length. GENERIC, NOT CURATED, deliberately the opposite choice from bl_add_modifier: properties are read off bl_rna at call time, so this accepts exactly what the running Blender accepts and refuses the rest BY NAME with the settable list in the refusal - there is no list here to go stale. Enums are validated against their own items, object pointers resolve by name (pass null to clear one, which is how a pole target is removed), and every key is checked BEFORE the first write so a bad key cannot leave a half-configured constraint. target/subtarget/influence are refused here on purpose: two ways to set one value is a parallel system. Values are read BACK off the constraint rather than echoed from the request."
    return _blender("set_constraint", object=object, constraint=constraint, settings=settings,
                    bone=bone)


@mcp.tool()
def bl_list_constraints(object: str, bone: str = None, limit: int = None, rna: bool = None) -> dict:
    "Every constraint on a Blender object or one of its pose bones, with each one's validity. invalidCount is the field to read: a constraint whose target has been DELETED stays in the stack, contributes nothing, and is indistinguishable from a working one everywhere except is_valid - Blender shows it red in the UI and reports it nowhere else an API caller can reach. Also reports influence, mute, target and subtarget per constraint, and how many are muted. `limit` bounds the rows at ~190 bytes each; invalidCount, invalid and mutedCount are computed from the whole stack whatever the limit, so a bound cannot hide a broken constraint. Call mif_help(\"bl_list_constraints\") first."
    # `limit` added 2026-09-20: ~190 bytes a constraint, and invalidCount / invalid / mutedCount
    # are still computed from the FULL stack, so a bound cannot turn the diagnostic into a sample.
    return _blender("list_constraints", object=object, bone=bone, limit=limit, rna=rna)


@mcp.tool()
def bl_copy_constraints(object: str, from_bone: str, to: list, constraints: list = None,
                        rename_subtarget=None) -> dict:
    "Copy a Blender pose bone's constraints, every setting, onto other bones; each subtarget follows its bone by rename_subtarget ({'DF_': 'FK_'} or [[old, new], ...]) and must resolve before anything is made."
    return _blender("copy_constraints", object=object, fromBone=from_bone, to=to, constraints=constraints,
                    renameSubtarget=rename_subtarget)


@mcp.tool()
def bl_remove_constraint(object: str, constraint_name: str, bone: str = None) -> dict:
    "Remove a Blender constraint by name, and report where the object went when it came off. Removing a constraint moves the object back to its own transform, and THAT movement is the proof the constraint was doing something - measured through the evaluated depsgraph for the same reason adding one is. The response counts constraints before and after with countsAgree, because constraints.remove() returns None either way. Call mif_help(\"bl_remove_constraint\") first."
    return _blender("remove_constraint", object=object, constraintName=constraint_name, bone=bone)


@mcp.tool()
def bl_list_markers(limit: int = None, name_excludes: list = None) -> dict:
    "Every timeline marker in the Blender scene, and which CAMERA each one cuts to. Camera binding is the reason markers matter beyond being labels: a marker with a camera bound makes the scene switch to it at that frame, which is how a multi-camera edit is done in Blender and is invisible from everywhere else in this addon - bl_list_cameras reports which camera is active NOW, this reports which one each part of the timeline uses. sceneCutsBetweenCameras says outright whether the scene changes camera mid-render, in which case scene.camera only describes the frames before the first binding. Bounded by limit/name_excludes because timeline_markers is scene-wide; count and the camera summary stay the MATCHED totals. Call mif_help(\"bl_list_markers\") first."
    return _blender("list_markers", limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_set_marker(name: str, frame: int = None, camera: str = None, unbind_camera: bool = None,
                  rename: str = None, delete: bool = None) -> dict:
    "Create, move, rename, camera-bind or delete a Blender timeline marker. Binding a camera makes the scene CUT to it at that frame. Markers are matched BY NAME and Blender permits duplicates, so the response reports how many matched - acting on several silently is how a caller ends up moving the wrong one. Binding a non-camera object is refused by type rather than accepted and ignored, and passing both a camera and unbind_camera is refused as two answers to one question. Call mif_help(\"bl_set_marker\") first."
    return _blender("set_marker", name=name, frame=frame, camera=camera,
                    unbindCamera=unbind_camera, rename=rename, delete=delete)


@mcp.tool()
def bl_bake_to_keyframes(object: str, frame_start: int = None, frame_end: int = None,
                         step: int = None, visual_keying: bool = None,
                         clear_constraints: bool = None, clear_parents: bool = None,
                         remove_rigid_body: bool = None, allow_simulation: bool = None) -> dict:
    "Bake an object's evaluated motion (constraints, drivers, a simulation) into REAL keyframes an exporter will carry; frame_start/frame_end default to the scene range, step to 1. This matters more than it sounds: bl_bake_physics bakes POINT CACHES and no exporter reads them, so a rigid-body simulation authored through this bridge could be rendered here and handed to NOTHING. Constraints and drivers have the same problem one step removed - they evaluate correctly in Blender and export as a static object, because an exporter writes keyframes and a constraint is not one. THE POSTCONDITION IS THE MOTION, NOT THE KEY COUNT: the evaluated world matrix is sampled across the range before the bake and again after, and maxPositionError / maxRotationErrorRadians / motionPreserved report whether the movement survived. Producing the right NUMBER of keys while losing the motion is the normal failure when visual keying is off, and a key count cannot see it. Sources left in place keep evaluating ON TOP of the new keys - clear_constraints and remove_rigid_body exist for that, and hadConstraints/hadRigidBody say what was there. Call mif_help(\"bl_bake_to_keyframes\") first."
    return _blender("bake_to_keyframes", object=object, frameStart=frame_start,
                    frameEnd=frame_end, step=step, visualKeying=visual_keying,
                    clearConstraints=clear_constraints, clearParents=clear_parents,
                    removeRigidBody=remove_rigid_body, allowSimulation=allow_simulation)


@mcp.tool()
def bl_set_bone_pose(object: str, bone: str, location: list = None, rotation: list = None,
                     quaternion: list = None, scale: list = None, rotation_mode: str = None, mirror: dict = None,
                     turn: dict = None, ik: dict = None) -> dict:
    "Pose one bone of a Blender armature: location [x,y,z] in the bone's own space, rotation [x,y,z] euler radians or quaternion [w,x,y,z], scale [x,y,z], with rotation_mode applied first. Character animation had zero coverage here - bones could be listed and renamed and nothing else - and this was unreachable through bl_set_keyframe until the same day, because its dotted-path walk stripped subscripts so pose.bones[\"x\"].location resolved to the bone COLLECTION. rotation (euler) and quaternion are refused against the bone's actual rotation_mode rather than silently ignored, and refused together. THE READ-BACK IS EVALUATED through the depsgraph: a bone with an IK chain, a Copy Rotation or a Limit does not end up where you put it, and pose_bone.matrix is the raw value. The response reports `written` and `evaluated*` separately - if they differ, that is the constraint working, not a fault. Call mif_help(\"bl_set_bone_pose\") first. rotation_mode (QUATERNION, XYZ ... AXIS_ANGLE) is applied BEFORE the values: a new bone is QUATERNION and ignores a euler, so pass rotation_mode XYZ with rotation in one call. Changing the mode keeps the bone's current pose (measured on 3.6-5.2); rotationModeBefore reports what it was."
    return _blender("set_bone_pose", object=object, bone=bone, location=location,
                    rotation=rotation, quaternion=quaternion, scale=scale, rotationMode=rotation_mode, mirror=mirror,
                    turn=turn, ik=ik)


@mcp.tool()
def bl_set_armature_properties(object: str, pose_mode: str = None, display_type: str = None, show_names: bool = None,
                               show_axes: bool = None, show_bone_shapes: bool = None, show_bone_colors: bool = None,
                               show_in_front: bool = None, x_mirror_edit: bool = None, x_mirror_pose: bool = None,
                               auto_ik: bool = None) -> dict:
    "Set a Blender armature's display and pose options - rest or pose position, display type, names, axes, X-mirror, auto IK - each read back with the previous values."
    return _blender("set_armature_properties", object=object, poseMode=pose_mode, displayType=display_type,
                    showNames=show_names, showAxes=show_axes, showBoneShapes=show_bone_shapes,
                    showBoneColors=show_bone_colors, showInFront=show_in_front, xMirrorEdit=x_mirror_edit,
                    xMirrorPose=x_mirror_pose, autoIK=auto_ik)


@mcp.tool()
def bl_key_pose(object: str, frame: float, bones: list = None, name_prefix: str = None, all: bool = None,
                channels: list = None, custom_properties=None, interpolation: str = None, reset: bool = None) -> dict:
    "Key Blender pose bones at a frame as they stand (bones, name_prefix or all): location, rotation in each bone's own mode, scale, and custom_properties (true or names); reset keys the rest pose."
    return _blender("key_pose", object=object, frame=frame, bones=bones, namePrefix=name_prefix, all=all,
                    channels=channels, customProperties=custom_properties, interpolation=interpolation, reset=reset)


@mcp.tool()
def bl_copy_pose(object: str, from_frame: float, to_frame: float, bones: list = None, name_prefix: str = None,
                 all: bool = None, flip: bool = None, name_rule=None, key: bool = None, interpolation: str = None,
                 allow_simulation: bool = None) -> dict:
    "Copy a Blender armature's pose from one frame to another, optionally flipped by name_rule (FLIP_NAME, a suffix pair or a table), keyed by default; reports the mirror error."
    return _blender("copy_pose", object=object, fromFrame=from_frame, toFrame=to_frame, bones=bones,
                    namePrefix=name_prefix, all=all, flip=flip, nameRule=name_rule, key=key, interpolation=interpolation,
                    allowSimulation=allow_simulation)


@mcp.tool()
def bl_create_pose_asset(object: str, asset: str, bones: list = None, name_prefix: str = None, all: bool = None,
                         library: str = None, catalog: str = None, description: str = None,
                         thumbnail: str = None) -> dict:
    "Store a Blender armature's current pose (all bones, or bones / name_prefix) as a pose asset in this file or in an asset library by name, with a catalog path and a thumbnail; the rig's own action is kept."
    return _blender("create_pose_asset", object=object, asset=asset, bones=bones, namePrefix=name_prefix, all=all,
                    library=library, catalog=catalog, description=description, thumbnail=thumbnail)


@mcp.tool()
def bl_apply_pose_asset(object: str, asset: str, library: str = None, blend: float = None, flipped: bool = None,
                        name_rule=None, bones: list = None) -> dict:
    "Apply a pose asset to a Blender armature at a blend of 0 to 1, optionally flipped by name_rule, onto all its bones or bones; reports the bones moved and the largest change."
    return _blender("apply_pose_asset", object=object, asset=asset, library=library, blend=blend, flipped=flipped,
                    nameRule=name_rule, bones=bones)


@mcp.tool()
def bl_mark_asset(datablock: str, kind: str, catalog: str = None, description: str = None, tags: list = None,
                  preview: bool = None, clear: bool = None) -> dict:
    "Mark a Blender datablock (ACTION, COLLECTION, MATERIAL, NODE_TREE, OBJECT, WORLD) as an asset of this file with a catalog, description and tags, or clear the mark."
    return _blender("mark_asset", datablock=datablock, kind=kind, catalog=catalog, description=description, tags=tags,
                    preview=preview, clear=clear)


@mcp.tool()
def bl_list_asset_libraries() -> dict:
    "List the asset libraries in Blender's preferences - name, folder, import method, relative path, .blend count - with the build's import methods and whether the preferences are saved. Read-only."
    return _blender("list_asset_libraries")


@mcp.tool()
def bl_set_asset_library(name: str, action: str = None, path: str = None, import_method: str = None,
                         relative_path: bool = None, enabled: bool = None, new_name: str = None,
                         save: bool = None) -> dict:
    "Register a folder as a Blender asset library (action ADD), change one (EDIT: path, import_method, relative_path, new_name, enabled on 5.2) or REMOVE it; save=true also writes the preferences."
    return _blender("set_asset_library", name=name, action=action, path=path, importMethod=import_method,
                    relativePath=relative_path, enabled=enabled, newName=new_name, save=save)


@mcp.tool()
def bl_list_libraries(library: str = None, limit: int = None) -> dict:
    "List the .blend libraries this Blender file links - path, missing or not, what each links - and every library override with its state (system or editable) and the edits it stores. Read-only."
    return _blender("list_libraries", library=library, limit=limit)


@mcp.tool()
def bl_set_library(library: str, filepath: str = None, reload: bool = None) -> dict:
    "Relocate a linked Blender library to the .blend it now lives at (filepath), or reload it from disk (reload=true), without reopening this file; reports the overrides a reload brought in."
    return _blender("set_library", library=library, filepath=filepath, reload=reload)


@mcp.tool()
def bl_library_override(target: str, action: str = None, kind: str = None, scope: str = None,
                        editable: list = None, fully_editable: bool = None, data: list = None,
                        hierarchy: bool = None) -> dict:
    "Make a library override of a linked Blender collection or object (MAKE), mark members EDITABLE or SYSTEM, RESYNC, RESET, CLEAR it back to linked data, or MAKE_LOCAL. Call mif_help(\"bl_library_override\") first."
    return _blender("library_override", target=target, action=action, kind=kind, scope=scope, editable=editable,
                    fullyEditable=fully_editable, data=data, hierarchy=hierarchy)


@mcp.tool()
def bl_mirror_shape_key(object: str, key: str = None, keys: list = None, prefix: str = None, axis: str = None,
                        new_name: str = None, pairs: list = None, tolerance: float = None,
                        overwrite: bool = None) -> dict:
    "Mirror Blender shape keys across an axis onto new keys named for the other side (smile_L -> smile_R, or new_name): each offset goes to the mirrored vertex. Reports the vertex pairs matched and the unmatched vertices; drivers are not carried."
    return _blender("mirror_shape_key", object=object, key=key, keys=keys, prefix=prefix, axis=axis,
                    newName=new_name, pairs=pairs, tolerance=tolerance, overwrite=overwrite)


@mcp.tool()
def bl_set_shape_key(object: str, key: str, value: float = None, slider_min: float = None,
                     slider_max: float = None, mute: bool = None, vertex_group: str = None,
                     relative_key: str = None, remove: bool = None, move_to: int = None, make_basis: bool = None,
                     apply_to_basis: bool = None) -> dict:
    "Set a Blender shape key's influence, and optionally its slider range. It is the write half of bl_list_shape_keys, which could only read. The range is applied BEFORE the value, because setting a value outside the OLD range would be clamped to it and then look wrong even though the new range allows it. Blender CLAMPS SILENTLY: asking for 2.0 on a 0..1 key leaves 1.0 and reports nothing, so the response carries requestedValue, the actual value, and a `clamped` flag naming the difference rather than leaving it to be discovered in a render. remove true deletes the key instead (keys relative to it are re-pointed at the basis; the basis is refused while other keys remain). Call mif_help(\"bl_set_shape_key\") first."
    return _blender("set_shape_key", object=object, key=key, value=value, sliderMin=slider_min,
                    sliderMax=slider_max, mute=mute, vertexGroup=vertex_group, relativeKey=relative_key,
                    remove=remove, moveTo=move_to, makeBasis=make_basis, applyToBasis=apply_to_basis)


@mcp.tool()
def bl_apply_shape_keys(object: str, mix: bool = None, keys: list = None, keep_basis: bool = None) -> dict:
    "Bake a Blender mesh's shape keys into its base mesh: the whole current mix (default) or keys [names] at their values, then remove them. keep_basis keeps a Basis holding the new shape. Replies keys before/after, maxDisplacement and mixDeviation."
    return _blender("apply_shape_keys", object=object, mix=mix, keys=keys, keepBasis=keep_basis)


@mcp.tool()
def bl_set_point_positions(object: str, points: list = None, shape_key: str = None, tilt: list = None,
                           handle_type: dict = None, radius: list = None, handles: list = None) -> dict:
    "Move a Blender lattice's points, a shape key's vertices or a curve's control points by index. It is the edit that makes a lattice or a key deform. points is [[index, x, y, z], ...] in the object's LOCAL space; a lattice's index is u + v*pointsU + w*pointsU*pointsV, so a 3x3x3's centre is 13. On an object that has shape keys, shape_key is REQUIRED: the Basis key holds the rest shape and base points moved underneath it change nothing (measured on 4.4 for meshes and lattices). Every row is validated before any is written, positions are read back, and each object the points deform is fingerprinted before and after - evaluatedChanged names what actually moved, and note says when nothing could (a lattice nothing is bound to, a key at value 0). A curve takes radius too, and with nothing to write is READ. Call mif_help(\"bl_set_point_positions\") first."
    return _blender("set_point_positions", object=object, points=points, shapeKey=shape_key,
                    tilt=tilt, handleType=handle_type, radius=radius, handles=handles)


@mcp.tool()
def bl_list_actions(name_contains: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "Every action in the Blender file, who uses it, and whether it will SURVIVE A SAVE. An action with no users and no fake user is deleted the next time the file is saved - silently, by the save succeeding - so willBeDeletedOnSave names them before that happens. Each row carries curve and keyframe counts, frame range, and usedBy built by walking objects, because an action knows its user COUNT and not their names while 'which object is this clip on' is the actual question. Action names are also the CLIP NAMES glTF and FBX write into an engine, so an auto-generated 'Action.003' becomes a name somebody downstream has to live with. Call mif_help(\"bl_list_actions\") first."
    return _blender("list_actions", nameContains=name_contains, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_set_action(action: str, fake_user: bool = None, rename: str = None, frame_range: list = None,
                  use_frame_range: bool = None, cyclic: bool = None) -> dict:
    "Protect a Blender action from being destroyed by the next save, or rename it. THE ADDON ALREADY TOLD YOU THIS WAS COMING AND COULD NOT HELP: bl_list_actions reports survivesSave for every action, and bl_assign_action reports previousSurvivesSave for the action it just displaced - which is exactly the moment one drops to zero users. An action with no users and no fake user is DELETED the next time the file is saved, silently, BY THE SAVE SUCCEEDING; it is the same purge bl_save_file reports as purgedOrphans, seen from the side that can prevent it. use_fake_user was writable only by bl_create_action, at birth, so an action that arrived with an imported mesh or was displaced by an assignment could not be protected at all. A fake user is a deliberate reference meaning \"keep this even though nothing points at it\". Renaming reports the name you actually got, because Blender suffixes a clash rather than refusing and anything looking the action up by the name it asked for would then miss."
    return _blender("set_action", action=action, fakeUser=fake_user, rename=rename, frameRange=frame_range,
                    useFrameRange=use_frame_range, cyclic=cyclic)


@mcp.tool()
def bl_create_action(name: str, object: str = None, fake_user: bool = None) -> dict:
    "Create a NAMED Blender action, optionally assigning it. Naming is the entire point: an object gets whatever Blender auto-named its action, and that string is what glTF and FBX write into the engine as the clip name - so every clip exported through this bridge previously arrived downstream named after nothing. fake_user defaults TRUE, because a freshly created unassigned action has zero users by definition and would be deleted on the next save; the safe default is the one that does not lose work. The response reports nameWasSuffixed, since Blender uniquifies silently and a caller looking up the name they asked for would find a different action or none. Call mif_help(\"bl_create_action\") first."
    return _blender("create_action", name=name, object=object, fakeUser=fake_user)


@mcp.tool()
def bl_assign_action(object: str, action: str = None, clear: bool = None, slot: str = None,
                     rest_pose: bool = None) -> dict:
    "Put an existing action on a Blender object, or clear it. An object held ONE action forever - bl_set_keyframe creates one on first use and nothing could swap it, so a second clip on the same rig was impossible. Pass clear:true to unlink instead, which is the dangerous direction and says so: an unlinked action with no fake user drops to zero users and is deleted on the next save, so the response reports whether the action you just unlinked will survive. On Blender 4.4+ an object plays an action only through a SLOT, so the op binds the slot named for the object or the action's only object slot and reports slot and slotUsers; slot picks one by identifier or name when there are several. The assignment is re-read afterwards and refuses to claim success if it did not take. Call mif_help(\"bl_assign_action\") first."
    return _blender("assign_action", object=object, action=action, clear=clear, slot=slot, restPose=rest_pose)


@mcp.tool()
def bl_edit_fcurve(object: str = None, data_path: str = None, index: int = None, frame: int = None,
                   interpolation: str = None, easing: str = None, handle_type: str = None,
                   extrapolation: str = None, amplitude: float = None, period: float = None,
                   back: float = None, mute: bool = None, lock: bool = None, hide: bool = None, target: str = None, datablock: str = None,
                   handle_left: list = None, handle_right: list = None, driver: bool = None,
                   value: float = None) -> dict:
    "Change how a Blender curve moves BETWEEN its keys. bl_set_keyframe can set an interpolation at INSERT time and reaches three of Blender's thirteen; nothing could change one afterwards, and nothing could touch EASING at all - which is most of the craft in motion graphics, where BACK with ease-out versus LINEAR is the entire look. Retiming or re-feeling an existing animation meant deleting and re-keying it. Omit frame to apply to every key on the curve, omit index for every array element. All four enums are validated against this Blender's own RNA rather than a remembered list. amplitude and period shape an ELASTIC key (period is in FRAMES; the default spring is 0.3 of the segment and on a long one is over almost at once) and back sets a BACK key's overshoot - each is read back into keyParamsAfter, and a key whose interpolation ignores it is named in warnings. A NODE TREE: target world, material or node_group with datablock (its name) works on that tree's own curves instead of an object's - data_path is the path on the tree, e.g. 'nodes[\"Mapping\"].inputs[2].default_value'. Call mif_help(\"bl_edit_fcurve\") first."
    return _blender("edit_fcurve", object=object, dataPath=data_path, index=index, frame=frame,
                    interpolation=interpolation, easing=easing, handleType=handle_type,
                    extrapolation=extrapolation, amplitude=amplitude, period=period, back=back, mute=mute,
                    lock=lock, hide=hide, handleLeft=handle_left, handleRight=handle_right, target=target, datablock=datablock,
                    driver=driver, value=value)


@mcp.tool()
def bl_add_fcurve_modifier(object: str, data_path: str = None, type: str = None, index: int = None,
                           mode_before: str = None, mode_after: str = None,
                           strength: float = None, scale: float = None, settings: dict = None,
                           control_points: list = None, bones: list = None, all: bool = None,
                           action: str = None, driver: bool = None, remove: str = None,
                           stack_index: int = None) -> dict:
    "Put a modifier on a Blender curve - most usefully CYCLES, which is how an animation LOOPS. There was no way to loop anything: every turntable, idle, cycling fan and blinking light had to be keyed out to full length by hand, and a two-key rotation could not be made to repeat at all. A CYCLES modifier on a curve with fewer than two keyframes is REFUSED, because there is no cycle to repeat and Blender adds the modifier anyway and does nothing with it, which looks like success. Every other type is configured through settings - any writable property of that type by Blender's own name (LIMITS use_max_y + max_y, STEPPED frame_step, GENERATOR poly_order + coefficients, FNGENERATOR function_type + amplitude, SMOOTH sigma + filter_width on 5.2), checked against this Blender before anything is added; an out-of-range value is refused rather than clamped - and an ENVELOPE takes control_points [{frame, min, max}]. The response counts modifiers off the curves before and after, reads each setting back off the modifier, and samples the curve before and after: noEffect is true, with a note saying why, when the modifier changed nothing - a LIMITS with no bound enabled and an ENVELOPE with no points both do exactly that. To prove a loop is live, sample past the last key with bl_evaluate_at_frame. Call mif_help(\"bl_add_fcurve_modifier\") first."
    return _blender("add_fcurve_modifier", object=object, dataPath=data_path, type=type,
                    index=index, modeBefore=mode_before, modeAfter=mode_after,
                    strength=strength, scale=scale, settings=settings, controlPoints=control_points, bones=bones,
                    all=all, action=action, driver=driver, remove=remove, stackIndex=stack_index)


@mcp.tool()
def bl_evaluate_at_frame(object: str, frames: list, data_paths: list = None, allow_simulation: bool = None,
                         bones: list = None, point: str = None) -> dict:
    "What a Blender object ACTUALLY is at a LIST of frames, read through the evaluated depsgraph: each sample's world location/rotationEuler/scale, plus any data_paths you name. Every other read in this addon reads the RAW property off the datablock, which is not what the scene evaluates to whenever a constraint, driver, NLA stack, parent or simulation cache is involved - a constraint does not touch obj.matrix_world at all, so reading the base object reports every constraint as having done nothing. This is the substrate for verifying anything procedural: reading back the value you wrote is a proxy that cannot fail. The scene frame is restored and the restoration is ASSERTED, because leaving somebody's scene on frame 47 as a side effect of a READ is exactly the quiet damage this bridge refuses. movedAcrossFrames answers the question behind most uses - a list of identical matrices is what a dead driver, a muted NLA track and a bake that lost its motion all look like. WHERE THE OBJECT IS is each sample's top-level location/rotationEuler/scale, the world matrix: a data_paths entry like 'location' is the object's OWN channel, which keys and drivers move and a parent, a constraint or a rigid body never does - a baked body's values.location stays at its start while it falls. Asking for a transform channel reports transformChannelsDiffer, and channelsNote names what moves the object. Call mif_help(\"bl_evaluate_at_frame\") first."
    return _blender("evaluate_at_frame", object=object, frames=frames, dataPaths=data_paths,
                    allowSimulation=allow_simulation, bones=bones, point=point)


@mcp.tool()
def bl_motion_paths(object: str, bones: list = None, action: str = None, type: str = None, range: str = None,
                    frame_start: int = None, frame_end: int = None, frame_before: int = None, frame_after: int = None,
                    frame_step: int = None, bake_location: str = None, display: dict = None, max_points: int = None,
                    allow_simulation: bool = None) -> dict:
    "Calculate, update or clear a Blender motion path of an object or pose bones and return its points (frame, world position, gap to the previous), length and spacing. Call mif_help(\"bl_motion_paths\") first."
    return _blender("motion_paths", object=object, bones=bones, action=action, type=type, range=range,
                    frameStart=frame_start, frameEnd=frame_end, frameBefore=frame_before, frameAfter=frame_after,
                    frameStep=frame_step, bakeLocation=bake_location, display=display, maxPoints=max_points,
                    allowSimulation=allow_simulation)


@mcp.tool()
def bl_spread_pose(object: str, frame: int = None, bones: list = None, name_prefix: str = None, all: bool = None,
                   before: int = None, after: int = None, falloff: str = None, influence: float = None,
                   keys: str = None, channels: list = None, mode: str = None, end_frame: int = None,
                   slider: str = None, factor: float = None) -> dict:
    "Spread the change made to Blender pose bones at a frame over the frames before and after it with a falloff, or run Blender's pose propagate (mode). With slider BREAKDOWN, PUSH or RELAX and a factor it keys the pose sliders' in-between at the frame instead. Call mif_help(\"bl_spread_pose\") first."
    return _blender("spread_pose", object=object, frame=frame, bones=bones, namePrefix=name_prefix, all=all,
                    before=before, after=after, falloff=falloff, influence=influence, keys=keys, channels=channels,
                    mode=mode, endFrame=end_frame, slider=slider, factor=factor)


@mcp.tool()
def bl_filter_keys(object: str, filter: str, data_paths: list = None, bones: list = None, all: bool = None,
                   frame_start: float = None, frame_end: float = None, settings: dict = None) -> dict:
    "Smooth, Gaussian-smooth, Butterworth-filter, decimate or blend a Blender object's F-curve keys over a frame range, replying each curve's keys before and after and its largest change. The Graph Editor sliders too - ease, blend offset, blend to ease or default, match slope, scale average or from neighbor, push-pull, shear, time offset. Call mif_help(\"bl_filter_keys\") first."
    return _blender("filter_keys", object=object, filter=filter, dataPaths=data_paths, bones=bones, all=all,
                    frameStart=frame_start, frameEnd=frame_end, settings=settings)


@mcp.tool()
def bl_edit_action_slots(action: str, op: str, slot: str = None, name: str = None, id_type: str = None,
                         object: str = None, target: str = None, new_slot: bool = None, from_actions: list = None,
                         slots: list = None, new_action: str = None) -> dict:
    "Make, list, rename, assign, merge, separate or move a Blender action's slots (4.4+), or replace an action on everything that plays it. Call mif_help(\"bl_edit_action_slots\") first."
    return _blender("edit_action_slots", action=action, op=op, slot=slot, name=name, idType=id_type, object=object,
                    target=target, newSlot=new_slot, slots=slots, newAction=new_action, fromActions=from_actions)


@mcp.tool()
def bl_set_nla_strip(object: str, track: str, strip: str = None, rename: str = None, frame_start: float = None,
                     frame_end: float = None, action_start: float = None, action_end: float = None,
                     repeat: float = None, scale: float = None, blend_in: float = None, blend_out: float = None,
                     auto_blend: bool = None, mute: bool = None, reverse: bool = None, sync_length: bool = None,
                     slot: str = None) -> dict:
    "Edit an existing Blender NLA strip - frames, action range, repeat, scale, blends, mute, reverse, name, its slot (4.4+) and sync length - replying it before and after. Call mif_help(\"bl_set_nla_strip\") first."
    return _blender("set_nla_strip", object=object, track=track, strip=strip, rename=rename, frameStart=frame_start,
                    frameEnd=frame_end, actionStart=action_start, actionEnd=action_end, repeat=repeat, scale=scale,
                    blendIn=blend_in, blendOut=blend_out, autoBlend=auto_blend, mute=mute, reverse=reverse,
                    syncLength=sync_length, slot=slot)


@mcp.tool()
def bl_match_world_transform(object: str, bone: str = None, target: str = None, target_bone: str = None,
                             frames: list = None, frame_start: int = None, frame_end: int = None, key: bool = None,
                             channels: list = None, relative_to: str = None, relative_to_bone: str = None,
                             allow_simulation: bool = None) -> dict:
    "Match a Blender object's or pose bone's world transform to another's at frames, keyed, or carry it along with a moving object, replying the position and rotation error after. Call mif_help(\"bl_match_world_transform\") first."
    return _blender("match_world_transform", object=object, bone=bone, target=target, targetBone=target_bone,
                    frames=frames, frameStart=frame_start, frameEnd=frame_end, key=key, channels=channels,
                    relativeTo=relative_to, relativeToBone=relative_to_bone, allowSimulation=allow_simulation)


@mcp.tool()
def bl_set_object_visibility(object: str, hide_viewport: bool = None, hide_render: bool = None,
                             visible_camera: bool = None, visible_diffuse: bool = None,
                             visible_glossy: bool = None, visible_transmission: bool = None,
                             visible_volume_scatter: bool = None, visible_shadow: bool = None,
                             holdout: bool = None, shadow_catcher: bool = None,
                             hidden_in_view_layer: bool = None, display_as: str = None, show_wire: bool = None,
                             show_in_front: bool = None, show_bounds: bool = None, bounds_type: str = None,
                             hide_select: bool = None) -> dict:
    "Control what a Blender object is visible TO - the viewport, the render, and each ray type separately. visible_glossy=false is how you stop a softbox appearing as a white rectangle in every reflection, which is the single most common adjustment in product and archviz lighting and was unreachable here. It also answers 'why is my object missing from the render', because hide_render and visible_camera are exactly where that answer lives - the response reports appearsInRender directly. Works on EVERY object type, because ray visibility is an object property and a mesh acting as a reflector needs it as much as a lamp does. Ray visibility moved off the Cycles addon onto the object in Blender 3.0, so both spellings are tried; a flag this build does not expose is REFUSED by name rather than silently ignored, and every flag is resolved before any is written. Call mif_help(\"bl_set_object_visibility\") first."
    return _blender("set_object_visibility", object=object, hideViewport=hide_viewport,
                    hideRender=hide_render, visibleCamera=visible_camera,
                    visibleDiffuse=visible_diffuse, visibleGlossy=visible_glossy,
                    visibleTransmission=visible_transmission,
                    visibleVolumeScatter=visible_volume_scatter, visibleShadow=visible_shadow,
                    holdout=holdout, shadowCatcher=shadow_catcher, hiddenInViewLayer=hidden_in_view_layer,
                    displayAs=display_as, showWire=show_wire, showInFront=show_in_front, showBounds=show_bounds,
                    boundsType=bounds_type, hideSelect=hide_select)


@mcp.tool()
def bl_list_animation_data(object: str, target: str = None) -> dict:
    "Every route by which a Blender object is animated - action fcurves, DRIVERS and NLA strips. bl_list_keyframes reads animation_data.action only, which is one of three places animation lives, so an object driven entirely by drivers came back with curveCount 0 from an op whose purpose is verification - a wrong answer, not a missing one. This reports animatedBy (action/drivers/nla), the action's name and whether it has a fake user (an action WITHOUT one is deleted on save), each driver's expression, validity and variable targets, and every NLA track and strip. invalidDrivers counts drivers whose variables point at something that no longer exists - the silent failure where a driver evaluates to zero and reports nothing. Call mif_help(\"bl_list_animation_data\") first."
    return _blender("list_animation_data", object=object, target=target)


@mcp.tool()
def bl_delete_keyframe(object: str = None, data_path: str = None, frame: int = None,
                       index: int = None, target: str = None, datablock: str = None) -> dict:
    "Remove keyframes from a Blender channel - the correction path bl_set_keyframe never had, which forced delete-and-recreate for any mistake. Omit frame to clear every key on the path; omit index to clear every array element. Counted before and after off the fcurves, because keyframe removal reports a bool that is False both for 'there was nothing there' and for 'it refused', and those are different answers. The response carries keyframesBefore, keyframesAfter, removed and countsAgree - the last is the op checking its own arithmetic against the curve rather than asking you to trust it. A NODE TREE: target world, material or node_group with datablock (its name) works on that tree's own curves instead of an object's - data_path is the path on the tree, e.g. 'nodes[\"Mapping\"].inputs[2].default_value'. Call mif_help(\"bl_delete_keyframe\") first."
    return _blender("delete_keyframe", object=object, dataPath=data_path, frame=frame, index=index, target=target,
                    datablock=datablock)


@mcp.tool()
def bl_set_camera(object: str, type: str = None, lens: float = None, sensor_width: float = None,
                  sensor_height: float = None, sensor_fit: str = None, ortho_scale: float = None,
                  clip_start: float = None, clip_end: float = None, shift_x: float = None,
                  shift_y: float = None, f_stop: float = None, dof_distance: float = None,
                  look_at: list = None, location: list = None, rotation: list = None,
                  make_active: bool = None, field_of_view: float = None,
                  focus_object: str = None) -> dict:
    "Change a Blender camera that already exists, including SWITCHING which camera the scene renders through - make_active was previously available only at creation, so choosing between two existing cameras was impossible without bl_run_python. sensor_fit is here because without it sensor_width is a half-answer and no real lens can be matched. Type-gated settings are validated against the type the camera will BE after the call, so retyping to ORTHO and setting ortho_scale together is legal. Every refusal fires before any write. The response carries before, after and changedFields, with FOV derived at the current render resolution. rotation and look_at are mutually exclusive. Call mif_help(\"bl_set_camera\") first."
    return _blender("set_camera", object=object, type=type, lens=lens, sensorWidth=sensor_width,
                    sensorHeight=sensor_height, sensorFit=sensor_fit, orthoScale=ortho_scale,
                    clipStart=clip_start, clipEnd=clip_end, shiftX=shift_x, shiftY=shift_y,
                    fStop=f_stop, dofDistance=dof_distance, lookAt=look_at, location=location,
                    rotation=rotation, makeActive=make_active,
                    fieldOfView=field_of_view, focusObject=focus_object)


@mcp.tool()
def bl_set_camera_display(object: str, passepartout: bool = None, passepartout_alpha: float = None,
                          guides: dict = None, safe_areas: bool = None, safe_center: bool = None,
                          safe_area_margins: dict = None, show_limits: bool = None, show_mist: bool = None,
                          show_name: bool = None, show_sensor: bool = None, display_size: float = None,
                          show_background_images: bool = None, background_images: list = None,
                          clear_background_images: bool = None) -> dict:
    "What a Blender camera's VIEW shows - passepartout, composition guides, safe areas and background images (a reference plate behind a matchmove camera). None of it changes a render. guides {thirds: true, ...} by this Blender's names; safe_area_margins {title, action, titleCenter, actionCenter: [x, y]} is a scene setting; background_images [{image | imageName | movie | movieName, alpha, depth, frameMethod, offset, rotation, scale, flipX, flipY, show}] are appended (clear_background_images empties the list first). Every value is read back in display; only object is the read. Call mif_help(\"bl_set_camera_display\") first."
    return _blender("set_camera_display", object=object, passepartout=passepartout,
                    passepartoutAlpha=passepartout_alpha, guides=guides, safeAreas=safe_areas,
                    safeCenter=safe_center, safeAreaMargins=safe_area_margins, showLimits=show_limits,
                    showMist=show_mist, showName=show_name, showSensor=show_sensor, displaySize=display_size,
                    showBackgroundImages=show_background_images, backgroundImages=background_images,
                    clearBackgroundImages=clear_background_images)


@mcp.tool()
def bl_list_cameras(name_contains: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "List every camera in the Blender file and, crucially, WHICH ONE the scene renders through. sceneCamera was unobtainable anywhere in the addon before this: bl_scene_info omits it, bl_set_render_settings reports a bare boolean, and bl_render_still names it only by blocking for a whole render. Each row carries the full optical set plus fovDegrees derived at the current render resolution - the number that actually determines framing, and which depends on sensor fit as well as focal length. Call mif_help(\"bl_list_cameras\") first."
    return _blender("list_cameras", nameContains=name_contains, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_aim_object(object: str, target: str = None, look_at: list = None) -> dict:
    "Point a Blender object's local -Z axis - the way a light or camera faces - at an object (target) or a point (look_at [x,y,z]). Nothing could aim anything after creation - bl_create_camera took look_at at birth and that was the only user of the aiming maths, so a spot light could not be aimed at all. Pass exactly one of target (an object, aimed at its world origin) or look_at (a point). THE POSTCONDITION IS MEASURED, not assumed: it reports the ANGLE between the object's world-space local -Z and the direction to the target after the write, and refuses above ~1e-3 rad. That matters because the aiming derivation was once wrong by exactly pi - aiming 166 degrees off while returning a perfectly plausible euler - and reading back the euler you just wrote is a proxy that cannot catch it."
    return _blender("aim_object", object=object, target=target, lookAt=look_at)


@mcp.tool()
def bl_render_info(rna: bool = None) -> dict:
    "Everything that decides what a Blender render will look like - engine, samples and WHICH property they live on, effective resolution after the percentage multiplier, output format, colour management (view transform, look, exposure - the usual cause of 'washed out'), frame range, scene camera, world, and how many lights actually contribute. The read half of bl_set_render_settings, which reports only the five fields it can write. It also answers the black-render question directly: `blockers` is a measured list of reasons this render will produce nothing useful - no scene camera, no world datablock (a scene with no world contributes NO ambient light, so interiors go black and the lights get blamed), every light hidden or at zero energy - and `wouldRenderSomething` is the one-field answer. outOfProcessSimulations names the simulations bl_render_animation's saved-file render would not animate: an unbaked rigid body or soft body stays frozen, an unbaked cloth restarts when the render starts after its cache, an unbaked dynamic paint surface renders flat. No parameters. Call mif_help(\"bl_render_info\") first. rna=True adds `eevee`, `cycles` and `render`: every readable setting of scene.eevee, scene.cycles and scene.render by identifier - the read half of bl_set_render_settings' eevee / cycles / render."
    return _blender("render_info", rna=rna)


@mcp.tool()
def bl_file_info() -> dict:
    "What .blend this Blender session is, whether it has unsaved work, what it holds, and - the part Blender never volunteers - which datablocks a SAVE WOULD DELETE. Anything with no users and no fake user is purged on write, permanently and silently: an unlinked action, an unused image, a node group nobody instanced. Call this before bl_save_file if the session has been edited for a while. Cheap, no parameters. Call mif_help(\"bl_file_info\") first."
    return _blender("file_info")


@mcp.tool()
def bl_save_file(filepath: str, overwrite: bool = None, repoint_session: bool = None,
                 compress: bool = None) -> dict:
    "Write the Blender session to a .blend on disk. Until this existed NOTHING the addon authored survived the process - every light, camera, keyframe, world and node group died when Blender closed, and the only artefacts were a mesh FBX, a baked texture and one rendered frame. SAVE-A-COPY BY DEFAULT: the session keeps its own filepath, so a later save still goes where it did before; pass repoint_session to make this the working file. An existing file is REFUSED unless overwrite is passed. The response reports purgedOrphans - what the save destroyed - which is data loss caused by the successful operation and invisible otherwise. Call mif_help(\"bl_save_file\") first."
    return _blender("save_file", filepath=filepath, overwrite=overwrite,
                    repointSession=repoint_session, compress=compress)


@mcp.tool()
def bl_open_file(filepath: str, discard_unsaved: bool = None) -> dict:
    "Open a .blend, DISCARDING everything currently in memory. There is no undo across a file load, so an open over a DIRTY session is refused unless discard_unsaved is passed - this is the one Blender op here that can lose work which was never on disk. Save first with bl_save_file if you want it. The response re-reads the session path and refuses to claim success if the session did not actually become the file you asked for. Call mif_help(\"bl_open_file\") first."
    return _blender("open_file", filepath=filepath, discardUnsaved=discard_unsaved)


@mcp.tool()
def bl_set_light(object: str, type: str = None, energy: float = None, color: list = None,
                 radius: float = None, size: float = None, size_y: float = None,
                 shape: str = None, spot_angle: float = None, spot_blend: float = None,
                 angle: float = None, shadow: bool = None, diffuse_factor: float = None,
                 specular_factor: float = None, location: list = None,
                 rotation: list = None, temperature: float = None,
                 use_temperature: bool = None, energy_scale: float = None, stops: float = None) -> dict:
    "Change a Blender light that already exists - energy, colour, cone, area size, shadow, or its TYPE. Until this existed a light could be created and never adjusted, which is most of lighting work. Type-specific settings are validated against the type the light will BE after the call, so retyping to SPOT and setting spot_angle together is legal while spot_angle on a POINT light is refused. Every refusal fires before any write. The response carries before, after and changedFields, so you can see what actually moved - including a property Blender DISCARDS on a retype. rotation and spot_angle/angle are RADIANS. Call mif_help(\"bl_set_light\") first. COLOUR TEMPERATURE IN KELVIN: temperature is how lighting is actually specified - 3200 tungsten, 6500 daylight - rather than as linear RGB, and it MULTIPLIES the colour rather than replacing it, so colour becomes a tint. Refused by name on a Blender without bpy.types.Light.use_temperature rather than silently ignored."
    # EVERY accepted key is forwarded, and every optional one defaults to None so _blender drops it.
    # A concrete default here would be sent on every call, and set_light refuses a per-type key on
    # the wrong type - which is exactly how six UE tools were made uncallable on 2026-09-03.
    return _blender("set_light", object=object, type=type, energy=energy, color=color,
                    radius=radius, size=size, sizeY=size_y, shape=shape, spotAngle=spot_angle,
                    spotBlend=spot_blend, angle=angle, shadow=shadow,
                    diffuseFactor=diffuse_factor, specularFactor=specular_factor,
                    location=location, rotation=rotation,
                    temperature=temperature, useTemperature=use_temperature, energyScale=energy_scale,
                    stops=stops)


@mcp.tool()
def bl_list_lights(name_contains: str = None, type: str = None, limit: int = None,
                   name_excludes: list = None) -> dict:
    "List every light in the Blender file with its full state - type, energy, colour, cone, area size, shadow, transform - plus hideViewport/hideRender, because a perfectly configured light that is hidden from the render is the usual reason a scene comes back black. bl_object_info reports the same record under its `light` key; on Blender 4.0+ both carry lightLinking, the collections bl_set_light_linking limited the light to. sceneHasAnyLight answers the first question worth asking about a dark render. Call mif_help(\"bl_list_lights\") first."
    return _blender("list_lights", nameContains=name_contains, type=type, limit=limit,
                    nameExcludes=name_excludes)


@mcp.tool()
def bl_create_camera(name: str = "", location: list = None, rotation: list = None,
                     look_at: list = None, lens: float = None, sensor_width: float = None,
                     type: str = None, ortho_scale: float = None, clip_start: float = None,
                     clip_end: float = None, f_stop: float = None, dof_distance: float = None,
                     shift_x: float = None, shift_y: float = None,
                     make_active: bool = None, field_of_view: float = None,
                     focus_object: str = None) -> dict:
    "Create a Blender camera, optionally aimed with look_at instead of rotation - passing both is refused, since they are two answers to the same question. A Blender camera faces its local -Z, which is what hand-written aiming gets wrong, so look_at derives the euler for you. f_stop or dof_distance enables depth of field; neither is turned on by default. Call mif_help(\"bl_create_camera\") first. FIELD OF VIEW AND FOCUS: field_of_view is the SAME property as lens in different units - setting 90 degrees moves lens from 50mm to 18mm, measured on all four builds - so passing both is refused rather than resolved by argument order, and it is PERSP only. focus_object racks focus onto an object instead of a fixed distance, and it OVERRIDES dof_distance silently: Blender uses the object's distance and ignores the number, while both fields keep reading back exactly as written. Both were REPORTED by bl_object_info and writable by nothing until 2026-09-04, so a camera could be described in more detail than it could be built."
    return _blender("create_camera", name=name or None, location=location, rotation=rotation,
                    lookAt=look_at, lens=lens, sensorWidth=sensor_width, type=type,
                    orthoScale=ortho_scale, clipStart=clip_start, clipEnd=clip_end,
                    fStop=f_stop, dofDistance=dof_distance, shiftX=shift_x, shiftY=shift_y,
                    makeActive=make_active,
                    fieldOfView=field_of_view, focusObject=focus_object)


@mcp.tool()
def bl_set_keyframe(object: str = None, frame: float = None, location: list = None, rotation: list = None,
                    scale: list = None, data_path: str = None, value=None, index: int = None,
                    target: str = None, interpolation: str = None, datablock: str = None,
                    modifier_input: dict = None) -> dict:
    "Key a Blender object at a frame: location, rotation (radians) or scale [x,y,z] key its transform; data_path + value key any other property - a list for an array, or index for one element. A data_path is routed to the object or its data automatically - a light's energy lives on the data datablock, not the object. keyframe_insert stores the CURRENT value, so the value is WRITTEN first and the object is left holding it. A list sent as text ('[0,1,0]') is refused: send the list itself. interpolation CONSTANT/LINEAR/BEZIER: a flicker needs CONSTANT or it eases and stops reading as a flicker. rotationModeNote says when rotation_euler keys land on an object or bone in QUATERNION or AXIS_ANGLE mode, where they move nothing - pose bones start in QUATERNION. A NODE TREE: target world, material or node_group with datablock (its name) works on that tree's own curves instead of an object's - data_path is the path on the tree, e.g. 'nodes[\"Mapping\"].inputs[2].default_value'. Call mif_help(\"bl_set_keyframe\") first."
    return _blender("set_keyframe", object=object, frame=frame, location=location,
                    rotation=rotation, scale=scale, dataPath=data_path, value=value,
                    index=index, target=target, interpolation=interpolation, datablock=datablock,
                    modifierInput=modifier_input)


@mcp.tool()
def bl_set_frame_range(start: float = None, end: float = None, fps: float = None,
                       current: float = None, fps_base: float = None, frame_step: float = None,
                       preview_start: float = None, preview_end: float = None,
                       use_preview_range: bool = None, sync_mode: str = None,
                       allow_simulation: bool = None, fit_to_strips: str = None) -> dict:
    "Set the Blender scene's frame range, frame rate, step, current frame and PREVIEW RANGE. An end before start is REFUSED and the previous range restored - Blender accepts it and then renders nothing. fps_base is the other half of the frame rate and was previously unreachable: Blender stores 29.97 as fps 30 with fps_base 1.001, and 23.976 as 24 with 1.001, so every broadcast rate was impossible to set through fps alone. durationSeconds now divides by the TRUE rate (fps/fps_base) - it divided by fps alone until 2026-09-03, reporting every NTSC duration 0.1% short, which is a frame and a half over an hour. A preview range is for playback and viewport renders only: a render - bl_render_animation included - uses frame_start..frame_end (or bl_render_animation's own frame_start/frame_end) whatever the preview range, and the response's rendersFrames is that scene range. Call mif_help(\"bl_set_frame_range\") first."
    return _blender("set_frame_range", start=start, end=end, fps=fps, current=current,
                    fpsBase=fps_base, frameStep=frame_step, previewStart=preview_start,
                    previewEnd=preview_end, usePreviewRange=use_preview_range, syncMode=sync_mode,
                    allowSimulation=allow_simulation, fitToStrips=fit_to_strips)


@mcp.tool()
def bl_list_keyframes(object: str = None, target: str = None, limit: int = None,
                      max_keyframes: int = None, name_excludes: list = None, datablock: str = None) -> dict:
    "Read the animation curves on a Blender object and/or its data (target object, data or both), with the frames and values actually stored. The read half of bl_set_keyframe - a write is not verified by the writer. target: object | data | both (default both). BOUND IT ON ANYTHING BAKED: max_keyframes caps the points inside each curve, which is the direction that explodes - measured at 15 bytes a keyframe, so a 60-channel rig over 3,000 frames is about 686,000 tokens in one reply. limit caps the curves, name_excludes drops curves by dataPath (e.g. pose.bones). curveCount and keyframeTotal stay the true totals whatever you cap. A NODE TREE: target world, material or node_group with datablock (its name) works on that tree's own curves instead of an object's - data_path is the path on the tree, e.g. 'nodes[\"Mapping\"].inputs[2].default_value'."
    return _blender("list_keyframes", object=object, target=target, limit=limit,
                    maxKeyframes=max_keyframes, nameExcludes=name_excludes, datablock=datablock)


# --------------------------------------------------------------------------
# Blender: rendering, world, physics, particles and geometry-node authoring.
# Added 2026-09-01 to close the last five capability families that had no typed
# op at all and could only be reached through run_python.
# --------------------------------------------------------------------------

@mcp.tool()
def bl_set_render_settings(engine: str = None, resolution_x: int = None, resolution_y: int = None,
                           percentage: int = None, samples: int = None, file_path: str = None,
                           file_format: str = None, film_transparent: bool = None,
                           color_mode: str = None, use_denoising: bool = None,
                           exposure: float = None, color_depth: int = None,
                           eevee_raytracing: bool = None, eevee: dict = None, cycles: dict = None,
                           render: dict = None, ffmpeg: dict = None, image: dict = None,
                           cycles_curves: dict = None, texture_cache: str = None,
                           texture_cache_memory: bool = None) -> dict:
    "Configure the Blender render: engine (EEVEE/CYCLES aliases resolve to whatever THIS build calls them), resolution, samples, output path and format. The sample count lives in a different property per engine and writing the wrong one is a silent no-op, so this routes it and says which property it used. Warns when there is no scene camera, since a render would then fail. COLOUR DEPTH: bl_render_info has always reported colorDepth and nothing could set it, so an 8-bit normal map or HDR pass was quantised, existed, and looked roughly right. Two things make it awkward and both are handled. The enum LIES - bl_rna.properties['color_depth'].enum_items reports 8, 10, 12, 16 and 32 for EVERY format including JPEG, measured on 3.6 and 5.0 - while Blender validates on ASSIGNMENT and raises naming the real set, so the attempt is the validation and its own message is passed through. And order matters: the format is applied FIRST, so file_format='OPEN_EXR' with color_depth=32 works in one call. If the depth is then refused, the format AND the depth are put back, because changing format clamps depth and restoring one without the other leaves half the state. eevee_raytracing turns EEVEE's SCENE raytracing on or off - eevee.use_raytracing from 4.2, use_ssr + use_ssr_refraction on 3.6 - which glass and water need to refract the scene rather than the world (measured: a glass ball's render moves ~25/255 when it is toggled); the material's refraction flags come from bl_set_material_settings. The response reports it before and after and names the property written. EEVEE AND CYCLES' OWN SETTINGS: eevee={identifier: value} and cycles={identifier: value} write any writable property of scene.eevee / scene.cycles by its RNA identifier, each checked against its own type (enums against their items, finite numbers, strict booleans) - e.g. eevee={'volumetric_tile_size': '2', 'volumetric_samples': 128} (EEVEE draws volumes on an 8-pixel grid by default, which turns smoke and fire into soft blobs) or cycles={'volume_step_rate': 0.5}. They are written last and inside the call's snapshot, so any refusal puts the whole call back; a setting this tool also names (samples, eevee_raytracing, use_denoising) is refused if given both ways. render={identifier: value} does the same for scene.render - motion blur (use_motion_blur, motion_blur_shutter) lives there from 4.2 - and refuses fps and the compositing switches, naming bl_set_frame_range / bl_set_compositing, which own them. bl_render_info(rna=True) reads them back. IMAGE AND NESTED SETTINGS: image={identifier: value} writes scene.render.image_settings - exr_codec, use_preview, color_management - and a dotted identifier reaches a nested settings struct in any group: image={'color_management': 'OVERRIDE', 'linear_colorspace_settings.name': 'ACEScg'}, eevee={'ray_tracing_options.resolution_scale': '1'} (4.2 and later); the format, depth and mode stay file_format, color_depth and color_mode."
    return _blender("set_render_settings", engine=engine, resolutionX=resolution_x,
                    resolutionY=resolution_y, percentage=percentage, samples=samples,
                    filePath=file_path, fileFormat=file_format, filmTransparent=film_transparent,
                    colorMode=color_mode, useDenoising=use_denoising, exposure=exposure, colorDepth=color_depth,
                    eeveeRaytracing=eevee_raytracing, eevee=eevee, cycles=cycles, render=render, ffmpeg=ffmpeg,
                    image=image, cyclesCurves=cycles_curves, textureCache=texture_cache,
                    textureCacheMemory=texture_cache_memory)


@mcp.tool()
def bl_preview_garment(object: str, output_dir: str, textures: dict = None, hdri: str = None,
                       hdri_strength: float = None, views: list = None, elevation: float = None,
                       resolution: int = None, samples: int = None, prefix: str = None,
                       backface_culling: bool = None, film_transparent: bool = None) -> dict:
    "Render a garment alone as an asset viewer lights it: EEVEE, an HDRI world, backface culling, base color x ORM AO and ORM blue to metallic; per view. Nothing in the file changes. Replies files, coverage, luminance."
    return _blender("preview_garment", object=object, outputDir=output_dir, textures=textures, hdri=hdri,
                    hdriStrength=hdri_strength, views=views, elevation=elevation, resolution=resolution,
                    samples=samples, prefix=prefix, backfaceCulling=backface_culling,
                    filmTransparent=film_transparent)


@mcp.tool()
def bl_render_still(file_path: str = None, frame: float = None, samples: int = None,
                    resolution_x: int = None, resolution_y: int = None, percentage: int = None,
                    write_still: bool = None, return_image: bool = None,
                    preview_max_px: int = None, preview_max_bytes: int = None,
                    allow_simulation: bool = None, compute_backend: str = None, region: list = None,
                    region_px: list = None, crop: bool = None) -> dict:
    "Render the current frame to a file. BLOCKS Blender's main thread for the whole render, so start small - a heavy frame exceeds the addon's job timeout and reads as a hung bridge. render() returns FINISHED whether or not a file appeared, so the file is stat'd afterwards and wroteFile/fileBytes are measurements rather than the operator's opinion. PASS return_image=True TO SEE THE RENDER: it comes back as an image block, scaled to preview_max_px on its longest edge (default 1024, at most 1568) and to preview_max_bytes of PNG (default 512 KB), smaller rather than refused; imageBlocks describes it and filePath stays the full file. With _image 'data' it comes back the old way, as `image`, a base64 PNG in the text at preview_max_px (default 128), omitted if its PNG exceeds preview_max_bytes (default 30000); imageOmittedReason then directs the caller to reduce preview_max_px or inspect file_path. renderedWidth/renderedHeight always describe the full artifact. A frame that would step a live simulation is refused unless allow_simulation. COMPUTE BACKEND (2026-10-05): compute_backend NONE (the CPU), OPTIX, CUDA, HIP, METAL or ONEAPI chooses Cycles' device for this in-process still - the choice that bl_render_animation's gpu option makes in its own process. It writes compute_device_type, a USER PREFERENCE and not a scene setting: it stays for every later Cycles render in this Blender session (and is kept if the preferences are saved), and the reply's computeBackend says so with preferenceWas / preferenceNow, the devices ticked and the scene's device. A backend that lists no device here is REFUSED with the backends that do, the preference put back and nothing rendered - measured headless on 5.2.2 with no GPU: compute_device_type takes CUDA though CUDA lists nothing, and Cycles then renders on the CPU in silence. Refused too on a scene that does not render with Cycles. Measured: NONE on the CPU on 4.2.23, 4.4.0, 5.0.1 and 5.2.2; a GPU backend's success path was not run (no GPU on that host)."
    # THE ADD-ON'S OWN DEFAULTS (128 px, 30,000 bytes) ARE FOR BASE64 IN TEXT. A block is fitted here (_with_pictures), so the
    # add-on hands over a big preview rather than withholding one: 10 of the 25 previews asked for in a 22-run measurement
    # came back imageOmitted, and a model said so ("the preview was too large to come back to me").
    block = bool(return_image) and _IMAGE_MODE.get() is True
    res = _blender("render_still", filePath=file_path, frame=frame, samples=samples,
                   resolutionX=resolution_x, resolutionY=resolution_y, percentage=percentage,
                   writeStill=write_still, returnImage=return_image,
                   previewMaxPx=(min(preview_max_px or _IMAGE_DEFAULT_PX, _IMAGE_HARD_PX) if block else preview_max_px),
                   previewMaxBytes=(_IMAGE_WIRE_BYTES if block else preview_max_bytes), allowSimulation=allow_simulation,
                   _timeout=600.0, computeBackend=compute_backend, region=region, regionPx=region_px, crop=crop)
    if block and isinstance(res, dict):
        res = dict(res, _imageLimits=[preview_max_px, preview_max_bytes])
    return res


@mcp.tool()
def bl_compare_to_reference(image: str, reference: str, compare_px: int = None,
                            threshold: float = None) -> dict:
    "TURN 'match this reference' INTO A NUMBER. Returns silhouetteIoU (intersection over union of the two subject masks - ignores colour and lighting entirely and asks only whether the subject occupies the same space in frame, which is the number for matching a blockout) and meanAbsDiff (per-pixel luminance difference - sensitive to lighting and materials, so useful for 'is this the same shot' and useless for 'is this the same shape'). Reported separately rather than blended, because one score would hide which of the two moved. RENDER WITH A TRANSPARENT FILM FIRST (bl_set_render_settings film_transparent) - with an opaque background every pixel counts as subject, the mask becomes the whole frame, and IoU is 1.0 by definition; the op detects that and returns null with degenerateMask explaining, rather than a meaningless number. This is PIXEL similarity, not semantic similarity: a render matching a blockout's silhouette exactly with entirely wrong materials scores well and should."
    return _blender("compare_to_reference", image=image, reference=reference,
                    comparePx=compare_px, threshold=threshold)


@mcp.tool()
def bl_set_stereo(enabled: bool = None, views_format: str = None, output: str = None, display_mode: str = None,
                  camera: str = None, convergence_mode: str = None, interocular_distance: float = None,
                  convergence_distance: float = None, pivot: str = None, views: dict = None) -> dict:
    "Stereoscopic and multiview rendering in Blender: on or off, stereo or multiview, how the views are written, and the camera's convergence, interocular distance and pivot - read back. See mif_help."
    return _blender("set_stereo", enabled=enabled, viewsFormat=views_format, output=output, displayMode=display_mode,
                    camera=camera, convergenceMode=convergence_mode, interocularDistance=interocular_distance,
                    convergenceDistance=convergence_distance, pivot=pivot, views=views)


@mcp.tool()
def bl_set_color_management(view_transform: str = None, look: str = None, exposure: float = None,
                            gamma: float = None, display_device: str = None,
                            sequencer_colorspace: str = None,
                            use_curve_mapping: bool = None, working_space: str = None,
                            convert_colors: bool = None) -> dict:
    "View transform, look, exposure and gamma - the settings that silently change every pixel of every render and saved image, and the usual answer to 'why does this look washed out'. EVERY ENUM IS VALIDATED AGAINST THE OCIO CONFIG ACTUALLY LOADED, read from the instance rather than bpy.types, because unlike light or camera type these sets are populated at runtime: a studio config renames all of them, and the stock default moved from Filmic to AgX in 4.0. A hard-coded list would refuse the only values that work. The LOOK IS NAMESPACED BY THE VIEW TRANSFORM (\"AgX - Punchy\"), so the transform is applied first and the look is validated against what the NEW transform offers - and because Blender silently resets look when the transform changes, every requested write is read back from the scene rather than echoed."
    return _blender("set_color_management", viewTransform=view_transform, look=look,
                    exposure=exposure, gamma=gamma, displayDevice=display_device,
                    sequencerColorspace=sequencer_colorspace, useCurveMapping=use_curve_mapping,
                    workingSpace=working_space, convertColors=convert_colors)


@mcp.tool()
def bl_render_animation(frame_start: int = None, frame_end: int = None,
                        frame_step: int = None, scene: str = None, gpu: bool = None,
                        gpu_backend: str = None, memory_limit_gb: float = None, region: list = None,
                        region_px: list = None, crop: bool = None) -> dict:
    "Render a frame RANGE out of process and return immediately with a jobId to poll - the only shape that works, because every addon op runs on Blender's main thread under a 150s job timeout, so an in-process animation render freezes the bridge and the MCP gives up while the render carries on. It renders the SAVED .blend, not this session, so it refuses on an unsaved or dirty file rather than silently rendering the wrong scene; there is deliberately no output override, because -o would desynchronise the frame paths from the ones progress is measured against. Every expected frame path is stat'd BEFORE the render so a leftover file from an earlier run cannot be counted as progress. Poll with bl_render_status. gpu=true switches Cycles to the GPU with the OptiX denoiser for this job only, by injecting a script that runs after the file loads and before the first frame. Its warnings name every simulation that will not animate in that render (bl_render_info outOfProcessSimulations): bake_physics, then save, then render. gpu:true tries OPTIX, CUDA, HIP, METAL, ONEAPI in turn and keeps the first that lists a device; gpu_backend (with gpu:true) uses only that one - pass CUDA where OptiX lists the card but its kernel fails to load and no frame renders. On the GPU the job also turns on persistent data and, off OptiX, the denoiser on the GPU where the build has it - measured 18.5 s to 4.1 s for 12 frames on an RTX 4080 SUPER with CUDA, frames unchanged; the job log names what was applied. One heavy job at a time machine-wide (MIF_MAX_CHILD_JOBS, default 1): refused while another holds the slot; memory_limit_gb ends the render's process tree past that (default half of RAM, 0 none)."
    return _blender("render_animation", frameStart=frame_start, frameEnd=frame_end,
                    frameStep=frame_step, scene=scene, gpu=gpu, gpuBackend=gpu_backend, memoryLimitGB=memory_limit_gb,
                    region=region, regionPx=region_px, crop=crop)


@mcp.tool()
def bl_render_status(job_id: str = None, log_lines: int = None) -> dict:
    "How far an out-of-process render has actually got, measured on disk rather than asked of the process. framesRendered counts only files whose mtime is at or after the job start. Keeps three answers distinct that must never be collapsed: unknownJob (never started here, or Blender restarted and the table went with it - NOT 'unfinished', which is how a caller waits forever), running, and exited-with-a-code, where a non-zero exit still reports the real frames already on disk. For a movie container it reports framesVerifiable:false instead of a frame count it never checked. Omit job_id to list the jobs this Blender knows about. gpuNote names gpu_backend CUDA when the log shows the OptiX kernel failing to load."
    return _blender("render_status", jobId=job_id, logLines=log_lines)

@mcp.tool()
def bl_compose_atlas(frames: list, columns: int, rows: int, output: str, overwrite: bool = None,
                     alpha_mode: str = None, alpha_gain: float = None, alpha_floor: float = None,
                     loop_blend: int = None, normal_output: str = None, normal_strength: float = None,
                     alpha_frames: list = None, motion_output: str = None, motion_scale: float = None) -> dict:
    "Tile rendered frames into one flipbook atlas PNG (left-to-right, top-to-bottom, the order Unreal's SubUV reads). Call mif_help(\"bl_compose_atlas\") first."
    return _blender("compose_atlas", frames=frames, columns=columns, rows=rows, output=output, overwrite=overwrite,
                    alphaMode=alpha_mode, alphaGain=alpha_gain, alphaFloor=alpha_floor, loopBlend=loop_blend,
                    normalOutput=normal_output, normalStrength=normal_strength, alphaFrames=alpha_frames,
                    motionOutput=motion_output, motionScale=motion_scale)


@mcp.tool()
def bl_set_scene_audio(distance_model: str = None, doppler_speed: float = None, doppler_factor: float = None,
                       volume: float = None) -> dict:
    "Set the Blender scene's 3D audio: distance model, Doppler speed and factor, master volume. Call mif_help(\"bl_set_scene_audio\") first."
    return _blender("set_scene_audio", distanceModel=distance_model, dopplerSpeed=doppler_speed,
                    dopplerFactor=doppler_factor, volume=volume)


@mcp.tool()
def bl_add_sound(file: str, mode: str, frame: int = None, name: str = None, volume: float = None,
                 pitch: float = None, location: list = None, distance_reference: float = None,
                 distance_max: float = None, attenuation: float = None, cone_angle_inner: float = None,
                 cone_angle_outer: float = None, cone_volume_outer: float = None, channel: int = None,
                 volume_keys: list = None, scene: str = None) -> dict:
    "Put a sound file in the Blender scene at a frame, as a 3D speaker or a sequencer strip. Call mif_help(\"bl_add_sound\") first."
    return _blender("add_sound", file=file, mode=mode, frame=frame, name=name, volume=volume, pitch=pitch,
                    location=location, distanceReference=distance_reference, distanceMax=distance_max,
                    attenuation=attenuation, coneAngleInner=cone_angle_inner, coneAngleOuter=cone_angle_outer,
                    coneVolumeOuter=cone_volume_outer, channel=channel, volumeKeys=volume_keys, scene=scene)


@mcp.tool()
def bl_mixdown_audio(filepath: str, container: str = None, sample_rate: int = None, bit_format: str = None,
                     frame_start: int = None, frame_end: int = None, channels: str = None, scene: str = None) -> dict:
    "Render the Blender scene's audio to a WAV or FLAC file, measured off the file. Call mif_help(\"bl_mixdown_audio\") first."
    # 600 s like bl_render_still: Blender steps every frame of the range before it mixes, so a heavy scene is slow.
    return _blender("mixdown_audio", filepath=filepath, container=container, sampleRate=sample_rate,
                    bitFormat=bit_format, frameStart=frame_start, frameEnd=frame_end, channels=channels,
                    scene=scene, _timeout=600.0)


@mcp.tool()
def bl_add_strip(type: str, scene: str = None, name: str = None, channel: int = None, frame: int = None,
                 length: int = None, file: str = None, files: list = None, fit_method: str = None,
                 with_sound: bool = None, source_scene: str = None, scene_input: str = None, scene_camera: str = None,
                 inputs: list = None, text: str = None, font_size: float = None, color: list = None,
                 location: list = None, align_x: str = None, strips: list = None, settings: dict = None) -> dict:
    "Add a strip to Blender's sequencer: MOVIE (with its sound), IMAGE, SCENE, META, or an effect - COLOR, TEXT, ADJUSTMENT, a transition (CROSS, WIPE...) or a filter over inputs. An occupied channel is refused. Call mif_help(\"bl_add_strip\") first."
    return _blender("add_strip", type=type, scene=scene, name=name, channel=channel, frame=frame, length=length,
                    file=file, files=files, fitMethod=fit_method, withSound=with_sound, sourceScene=source_scene,
                    sceneInput=scene_input, sceneCamera=scene_camera, inputs=inputs, text=text, fontSize=font_size,
                    color=color, location=location, alignX=align_x, strips=strips, settings=settings)


@mcp.tool()
def bl_edit_strip(strip: str, scene: str = None, channel: int = None, frame: int = None, trim_start: int = None,
                  trim_end: int = None, slip: int = None, split: int = None, split_method: str = None,
                  delete: bool = None, blend_type: str = None, opacity: float = None, crop: dict = None,
                  transform: dict = None, saturation: float = None, multiply: float = None, mute: bool = None,
                  lock: bool = None, volume: float = None, fade_in: int = None, fade_out: int = None,
                  settings: dict = None, modifiers: list = None, swap_inputs: bool = None, rename: str = None,
                  duplicate: dict = None, connect: list = None, disconnect: bool = None, proxy: dict = None) -> dict:
    "Edit a Blender sequencer strip: move, trim, slip, split, delete; blend, opacity, crop, transform; mute, lock, fades; RNA settings; its modifier stack (add, set, move, remove). Call mif_help(\"bl_edit_strip\") first."
    return _blender("edit_strip", strip=strip, scene=scene, channel=channel, frame=frame, trimStart=trim_start,
                    trimEnd=trim_end, slip=slip, split=split, splitMethod=split_method, delete=delete,
                    blendType=blend_type, opacity=opacity, crop=crop, transform=transform, saturation=saturation,
                    multiply=multiply, mute=mute, lock=lock, volume=volume, fadeIn=fade_in, fadeOut=fade_out,
                    settings=settings, modifiers=modifiers, swapInputs=swap_inputs, rename=rename,
                    duplicate=duplicate, connect=connect, disconnect=disconnect, proxy=proxy)


@mcp.tool()
def bl_list_strips(scene: str = None, limit: int = None) -> dict:
    "List every strip in a Blender scene's sequencer - range, channel, source, inputs, modifiers - with named channels and, on 5.0+, the workspace's sequencer scene. Read-only."
    return _blender("list_strips", scene=scene, limit=limit)


@mcp.tool()
def bl_set_sequencer(scene: str = None, channels: dict = None, sequencer_scene: str = None,
                     scene_time_sync: bool = None) -> dict:
    "Name, mute and lock Blender sequencer channels ({\"3\": {name, mute, lock}}); on 5.0+ set the workspace's sequencer scene and Sync Scene Time."
    return _blender("set_sequencer", scene=scene, channels=channels, sequencerScene=sequencer_scene,
                    sceneTimeSync=scene_time_sync)


@mcp.tool()
def bl_load_clip(filepath: str, name: str = None, colorspace: str = None, active: bool = None,
                 set_scene_range: bool = None) -> dict:
    "Load a movie, or an image sequence from its first frame, as a Blender movie clip for tracking; reports size, frames, source, fps against the scene's and the color space."
    return _blender("load_clip", filepath=filepath, name=name, colorspace=colorspace, active=active,
                    setSceneRange=set_scene_range)


@mcp.tool()
def bl_list_clip(clip: str = None, limit: int = None) -> dict:
    "Read a Blender movie clip's tracking: each tracking object's tracks (frame span, bundle, error) and solve (average and per-frame error), and the lens, tracker and stabilization settings. Read-only."
    return _blender("list_clip", clip=clip, limit=limit)


@mcp.tool()
def bl_set_clip(clip: str = None, camera: dict = None, settings: dict = None, stabilization: dict = None,
                tracks: dict = None, copy_to_scene_camera: bool = None, proxy: dict = None,
                stabilization_tracks: dict = None) -> dict:
    "Set a Blender movie clip's lens (camera), tracker defaults (settings), 2D stabilization and per-track settings by their RNA names; all checked before any write. Call mif_help(\"bl_set_clip\") first."
    return _blender("set_clip", clip=clip, camera=camera, settings=settings, stabilization=stabilization,
                    tracks=tracks, copyToSceneCamera=copy_to_scene_camera, proxy=proxy,
                    stabilizationTracks=stabilization_tracks)


@mcp.tool()
def bl_edit_clip_tracks(action: str, tracks: list = None, clip: str = None, object: str = None, frame: int = None,
                        clear_mode: str = None, backwards: bool = None, keep_original: bool = None, source: str = None,
                        threshold: float = None, name: str = None, image: str = None, opacity: float = None,
                        auto_keying: bool = None) -> dict:
    "Edit a Blender movie clip's tracks: CLEAR_PATH, REFINE, LOCK, UNLOCK, JOIN, AVERAGE, COPY_SETTINGS, FILTER_SPIKES, PLANE (a plane track from 4), TO_MESH, TO_EMPTIES or DELETE; read back."
    return _blender("edit_clip_tracks", action=action, tracks=tracks, clip=clip, object=object, frame=frame,
                    clearMode=clear_mode, backwards=backwards, keepOriginal=keep_original, source=source,
                    threshold=threshold, name=name, image=image, opacity=opacity, autoKeying=auto_keying)


@mcp.tool()
def bl_build_tracking_scene(clip: str = None, skip: list = None, background: bool = None) -> dict:
    "Build the compositing scene for a solved Blender camera track: camera solver, foreground/background layers, shadow-catcher ground, light, cube, compositor; skip parts; the clip behind the camera."
    return _blender("build_tracking_scene", clip=clip, skip=skip, background=background)


@mcp.tool()
def bl_create_grease_pencil(name: str = None, layer: str = None, material: str = None, color: list = None,
                            location: list = None, collection: str = None) -> dict:
    "Create a Blender Grease Pencil object with a first layer and a stroke material, on this build's API generation (legacy GPENCIL on 3.6/4.2, GREASEPENCIL on 4.3+), named in the reply."
    return _blender("create_grease_pencil", name=name, layer=layer, material=material, color=color, location=location,
                    collection=collection)


@mcp.tool()
def bl_draw_grease_pencil_strokes(object: str, strokes: list, layer: str = None, frame: int = None, space: str = None,
                                  surface: dict = None, replace: bool = None) -> dict:
    "Draw strokes on a Blender Grease Pencil layer at a frame: points, radius (4.3+) or lineWidth (legacy), strength, material, cyclic; world or local space, or laid on a mesh surface at an offset. Each stroke also takes caps {start, end}, fill {color, opacity}, curveType (4.3+) and hideStroke (5.2+)."
    return _blender("draw_grease_pencil_strokes", object=object, strokes=strokes, layer=layer, frame=frame,
                    space=space, surface=surface, replace=replace)


@mcp.tool()
def bl_set_grease_pencil_layers(object: str, action: str = None, layer: str = None, new_name: str = None,
                                settings: dict = None, direction: str = None, group: str = None,
                                mask: str = None) -> dict:
    "List, add, remove, rename a Blender Grease Pencil object's layers or set one's properties by RNA name (opacity, blend_mode, use_lights, hide, lock, tint); frames and stroke counts read back. MOVE with direction UP, DOWN, TOP or BOTTOM changes the drawing order. DATA writes the Grease Pencil data's onion skin and auto-lock settings by RNA name. GROUP and UNGROUP put a layer in a layer group (4.3+); MASK and UNMASK add or remove a masking layer (not on 4.3-5.0)."
    return _blender("set_grease_pencil_layers", object=object, action=action, layer=layer, newName=new_name,
                    settings=settings, direction=direction, group=group, mask=mask)


@mcp.tool()
def bl_set_grease_pencil_frames(object: str, layer: str = None, action: str = None, frames: list = None, to: int = None,
                                offset: int = None, factor: float = None, pivot: int = None, step: int = None,
                                keyframe_type: str = None, detail: bool = None, space: str = None,
                                limit: int = None) -> dict:
    "List and edit the keyframes on a Blender Grease Pencil layer: insert blank ones, delete, move, duplicate, scale their timing, interpolate in-betweens, or set their keyframe type - on either API generation."
    return _blender("set_grease_pencil_frames", object=object, layer=layer, action=action, frames=frames, to=to,
                    offset=offset, factor=factor, pivot=pivot, step=step, keyframeType=keyframe_type, detail=detail,
                    space=space, limit=limit)


@mcp.tool()
def bl_edit_grease_pencil_strokes(object: str, layer: str, action: str, frame: int = None, strokes: list = None,
                                  settings: dict = None) -> dict:
    "Clean up Blender Grease Pencil strokes on a keyframe - simplify, smooth, subdivide, merge by distance, thickness, opacity, caps, cyclic, direction, join, reproject onto a surface, separate (4.3+), outline (5.0+); point counts before and after. See mif_help."
    return _blender("edit_grease_pencil_strokes", object=object, layer=layer, action=action, frame=frame,
                    strokes=strokes, settings=settings)


@mcp.tool()
def bl_trace_image_to_strokes(image: str = None, file: str = None, name: str = None, threshold: float = None,
                              turn_policy: str = None, thickness: int = None, resolution: int = None, frame: int = None,
                              convert_to: str = None) -> dict:
    "Trace a black-and-white image - a logo, a sketch, a scan - into a new Blender Grease Pencil object's strokes with Blender's own trace, optionally converted to a mesh. See mif_help."
    return _blender("trace_image_to_strokes", image=image, file=file, name=name, threshold=threshold,
                    turnPolicy=turn_policy, thickness=thickness, resolution=resolution, frame=frame,
                    convertTo=convert_to)


@mcp.tool()
def bl_bake_grease_pencil(objects: list, frame_start: int = None, frame_end: int = None, step: int = None,
                          frame_target: int = None, project: str = None, name: str = None) -> dict:
    "Bake Blender Grease Pencil objects' modifiers and animation - Line Art included - into a new object of plain strokes, a keyframe per baked frame. See mif_help."
    return _blender("bake_grease_pencil", objects=objects, frameStart=frame_start, frameEnd=frame_end, step=step,
                    frameTarget=frame_target, project=project, name=name)


@mcp.tool()
def bl_transform_grease_pencil_points(object: str, layer: str, frame: int = None, strokes: list = None,
                                      translate: list = None, rotate: list = None, scale=None, pivot=None) -> dict:
    "Translate, rotate and scale the points of a Blender Grease Pencil layer's strokes at one keyframe - written to the stroke data, no edit mode - for a head turn on its own layer, a drawing nudged or resized in place."
    return _blender("transform_grease_pencil_points", object=object, layer=layer, frame=frame, strokes=strokes,
                    translate=translate, rotate=rotate, scale=scale, pivot=pivot)


@mcp.tool()
def bl_set_grease_pencil_material(material: str, create: bool = None, object: str = None, stroke: dict = None,
                                  fill: dict = None, settings: dict = None) -> dict:
    "Make or edit a Blender Grease Pencil material: stroke {show, color, style, mode}, fill {show, color, style}, any other setting by RNA name; optionally add it to a Grease Pencil object."
    return _blender("set_grease_pencil_material", material=material, create=create, object=object, stroke=stroke,
                    fill=fill, settings=settings)


@mcp.tool()
def bl_set_grease_pencil_effects(object: str, action: str = None, effect: str = None, type: str = None,
                                 settings: dict = None) -> dict:
    "List, add, remove or set a Blender Grease Pencil object's visual effects - Blur, Colorize, Flip, Glow, Pixelate, Rim, Shadow, Swirl, Wave - with every setting read back by RNA name."
    return _blender("set_grease_pencil_effects", object=object, action=action, effect=effect, type=type,
                    settings=settings)


@mcp.tool()
def bl_track_clip(clip: str = None, object: str = None, markers: list = None, detect: dict = None,
                  track: dict = None) -> dict:
    "Place markers, detect features at chosen frames and track them both ways through a Blender movie clip; reports each track's frame span and how many reach each end. Call mif_help(\"bl_track_clip\") first."
    return _blender("track_clip", clip=clip, object=object, markers=markers, detect=detect, track=track,
                    _timeout=600.0)


@mcp.tool()
def bl_solve_clip(clip: str = None, object: str = None, keyframes=None, tripod: bool = None, refine: list = None,
                  clean: dict = None, rounds: dict = None) -> dict:
    "Solve a Blender movie clip's camera or object motion, then delete the worst tracks and re-solve in rounds to a target error; reports the error per round and per frame. Call mif_help(\"bl_solve_clip\") first."
    return _blender("solve_clip", clip=clip, object=object, keyframes=keyframes, tripod=tripod, refine=refine,
                    clean=clean, rounds=rounds, _timeout=600.0)


@mcp.tool()
def bl_orient_clip(clip: str = None, object: str = None, floor: list = None, wall: list = None, origin=None,
                   use_median: bool = None, x_axis: str = None, y_axis: str = None, scale: dict = None) -> dict:
    "Orient a solved Blender movie clip: floor or wall from three tracks, origin at a track, an axis through one, and a real distance between two; moves the scene camera."
    return _blender("orient_clip", clip=clip, object=object, floor=floor, wall=wall, origin=origin,
                    useMedian=use_median, xAxis=x_axis, yAxis=y_axis, scale=scale)


@mcp.tool()
def bl_extract_audio(filepath: str, out_path: str, gain_db: float = None, bit_format: str = None) -> dict:
    "Write a file's audio - a rendered video's soundtrack or any sound file - to a WAV, with an optional gain. Call mif_help(\"bl_extract_audio\") first."
    return _blender("extract_audio", filepath=filepath, outPath=out_path, gainDb=gain_db, bitFormat=bit_format,
                    _timeout=600.0)


@mcp.tool()
def bl_normalize_audio(target="ebu_r128", filepath: str = None, out_path: str = None, set_scene_volume: bool = None,
                       true_peak: float = None) -> dict:
    "Normalize the Blender scene's audio or a file to EBU R 128 or ASWG-R001 loudness without passing the true-peak ceiling: a normalized WAV, and the scene volume a video render reads. Call mif_help(\"bl_normalize_audio\") first."
    # THE TRANSPORT ITSELF, as render_flipbook_atlas hands its module _blender: normalize.py names its three ops
    # (mixdown_audio, extract_audio, set_scene_audio) as literals, and test_blender_audio_mux BM102 sends them to a real
    # add-on. A lambda calling _blender(op, ...) here left parity_check a call site it cannot read.
    return _audio("normalize").run(functools.partial(_blender, _timeout=600.0), target=target,
                                   filepath=filepath, out_path=out_path, set_scene_volume=set_scene_volume,
                                   true_peak=true_peak)


@mcp.tool()
def bl_bake_sound_keys(file: str, object: str, data_path: str, index: int = None, frame_start: int = None,
                       min_value: float = None, max_value: float = None, attack_ms: float = None,
                       release_ms: float = None, mode: str = None) -> dict:
    "Key a Blender property from a sound's loudness, one key per frame. Call mif_help(\"bl_bake_sound_keys\") first."
    return _blender("bake_sound_keys", file=file, object=object, dataPath=data_path, index=index,
                    frameStart=frame_start, minValue=min_value, maxValue=max_value, attackMs=attack_ms,
                    releaseMs=release_ms, mode=mode)

@mcp.tool()
def bl_create_empty(name: str = None, location: list = None, rotation: list = None,
                    display_type: str = None, display_size: float = None,
                    collection: str = None,
                    image: dict = None, images: list = None, opacity: float = None, depth: str = None,
                    side: str = None, show_perspective: bool = None, show_orthographic: bool = None,
                    offset: list = None, selectable: bool = None) -> dict:
    "Create a Blender Empty - the most-used object in Blender that this addon could not make. An Empty is what a Track To or Copy Location constraint points AT, what a rig is controlled by, what a camera is aimed at, and what objects are parented to for one shared pivot. bl_add_constraint and bl_aim_object both take a target and neither could create the object people overwhelmingly use as one, so a constraint could only be set up against something that already existed. It has no geometry and renders nothing; display_size is viewport only. IMAGE BACKDROPS (2026-10-05): image {file | name} makes an IMAGE empty holding it, with opacity (0-1), depth and side (Blender's image-empty choices; a wrong one is refused with the list), show_perspective, show_orthographic, offset [x, y] and selectable; images [{view FRONT | BACK | LEFT | RIGHT | TOP | BOTTOM, file | name, scaleToCube, location} - keys as written] places one per entry, rotated to face that numpad view, the shared settings on all - front, side, top and back drawings in one call; scaleToCube sizes the longer side to 2 (the default cube). bl_object_info's empty.image reads every setting back. A reference image on an empty, compared in viewport captures - not the camera's background images (declined 2026-09-03)."
    return _blender("create_empty", name=name, location=location, rotation=rotation,
                    displayType=display_type, displaySize=display_size, collection=collection,
                    image=image, images=images, opacity=opacity, depth=depth, side=side,
                    showPerspective=show_perspective, showOrthographic=show_orthographic, offset=offset,
                    selectable=selectable)


@mcp.tool()
def bl_create_mesh(vertices: list, faces: list = None, edges: list = None, material_indices: list = None,
                   materials: list = None, uvs: dict = None, vertex_groups: dict = None, smooth: bool | list = None,
                   name: str = None, location: list = None, rotation: list = None, scale: list = None,
                   collection: str = None, take_over_name: bool = None) -> dict:
    "Build a Blender mesh object from vertex, face and loose-edge lists, with optional per-face material slots, UV layers, vertex groups and smooth flags; bad input is refused before anything exists. Call mif_help(\"bl_create_mesh\") first."
    return _blender("create_mesh", vertices=vertices, faces=faces, edges=edges, materialIndices=material_indices,
                    materials=materials, uvs=uvs, vertexGroups=vertex_groups, smooth=smooth, name=name,
                    location=location, rotation=rotation, scale=scale, collection=collection,
                    takeOverName=take_over_name)


@mcp.tool()
def bl_create_force_field(name: str = "Field", type: str = "WIND", location: list = None,
                          rotation: list = None, collection: str = None, strength: float = None,
                          flow: float = None, noise: float = None, seed: int = None,
                          falloff_power: float = None, max_distance: float = None,
                          shape: str = None) -> dict:
    "Create a Blender FORCE FIELD that simulations respond to - type WIND, TURBULENCE, VORTEX, FORCE, DRAG, MAGNET, HARMONIC, CHARGE, LENNARDJ, TEXTURE, GUIDE, BOID or FLUID_FLOW. bl_add_cloth, bl_add_soft_body and bl_set_particles all read and write effectorWeights, which scales how much the fields in a scene affect that sim; with no field present the only effector is gravity, so those were dials on something that could not exist. type: WIND | TURBULENCE | VORTEX | FORCE | DRAG | MAGNET | HARMONIC | CHARGE | LENNARDJ | TEXTURE | GUIDE | BOID | FLUID_FLOW - refused by name if the running Blender does not offer it, and the list in the refusal is read off that Blender rather than quoted, so it cannot go stale. NEGATIVE STRENGTH PULLS instead of pushing, which is how you make a drain or an attractor out of a WIND or FORCE field. `max_distance` sets the flag AND the number together: Blender ignores distance_max unless use_max_distance is on, so setting the number alone reads back as the value you asked for and changes nothing. The object is created with bpy.data and a name you choose, then made a field - the obvious route, bpy.ops.object.effector_add, works but names the object after its type and drops it at the 3D cursor, so your name and location would only be suggestions. `shape`: POINT | LINE | PLANE. WIND and VORTEX DEFAULT TO PLANE, and a PLANE field reaches everything on either side of its own plane - max_distance is measured perpendicular to it, so it never stops at the sides and fields placed in a row overlap completely; the response carries a shapeNote saying so. Pass shape POINT for a field that ends in every direction. SURFACE and POINTS are refused: they take their shape from a mesh, and this makes an empty. Verified on Blender 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1: all four attach the field, take every setting, and offer the same 14 types. A field affects a simulation only as far as that simulation's effector weights allow. Rigid bodies listen too, through the rigid body world's weights - but a simulation baked BEFORE the field existed ignores it, so bake with bl_bake_physics rebake=True after adding one."
    return _blender("create_force_field", name=name, type=type, location=location,
                    rotation=rotation, collection=collection, strength=strength, flow=flow,
                    noise=noise, seed=seed, falloffPower=falloff_power, maxDistance=max_distance,
                    shape=shape)


@mcp.tool()
def bl_create_curve(points: list, name: str = None, spline_type: str = None,
                    cyclic: bool = None, bevel_depth: float = None,
                    bevel_resolution: int = None, extrude: float = None, resolution: int = None,
                    use_path: bool = None, location: list = None, rotation: list = None,
                    dimensions: str = None, collection: str = None, radii: list = None, bevel_object: str = None,
                    taper_object: str = None) -> dict:
    "Create a Blender curve - a path to follow, a profile to bevel, or a cable. bl_add_constraint accepts FOLLOW_PATH and nothing could make the one object it requires. THE SPLINE TYPE DECIDES WHAT THE POINTS MEAN: POLY and NURBS points live in spline.points with a 4th weight component while BEZIER points live in spline.bezier_points with handles - different collections with different lengths, so building into the wrong one produces a curve with no points and no error, and the point count is checked afterwards. use_path defaults ON because a Follow Path constraint evaluates to NOTHING without it while the constraint and the curve both read back perfectly. radii is one radius per point, scaling the bevel there: a stem, branch, horn or cable that TAPERS is radii [1.0, ..., 0.1] (read back as `radii`; bl_mesh_stats measures the tube it builds). Every coordinate and radius must be a finite number."
    return _blender("create_curve", points=points, name=name, splineType=spline_type,
                    cyclic=cyclic, bevelDepth=bevel_depth, bevelResolution=bevel_resolution,
                    extrude=extrude, resolution=resolution, usePath=use_path, location=location,
                    rotation=rotation, dimensions=dimensions, collection=collection, radii=radii,
                    bevelObject=bevel_object, taperObject=taper_object)


@mcp.tool()
def bl_set_curve(object: str = None, objects: list = None, collection: str = None, bevel_mode: str = None,
                 bevel_object: str = None, taper_object: str = None, taper_radius_mode: str = None,
                 fill_mode: str = None, resolution_u: int = None, bevel_resolution: int = None,
                 bevel_depth: float = None, extrude: float = None, offset: float = None,
                 triangles: bool = None) -> dict:
    "Write Blender curve data settings - bevel_mode, bevel_object, taper_object, fill_mode, resolution_u, bevel_resolution, bevel_depth, extrude, offset - on one curve, a list or a collection at once. Values before and after; triangles adds the evaluated triangle count."
    return _blender("set_curve", object=object, objects=objects, collection=collection, bevelMode=bevel_mode,
                    bevelObject=bevel_object, taperObject=taper_object, taperRadiusMode=taper_radius_mode,
                    fillMode=fill_mode, resolutionU=resolution_u, bevelResolution=bevel_resolution,
                    bevelDepth=bevel_depth, extrude=extrude, offset=offset, triangles=triangles)


@mcp.tool()
def bl_create_text(body: str, name: str = None, size: float = None, extrude: float = None,
                   bevel_depth: float = None, align: str = None, align_y: str = None,
                   location: list = None, rotation: list = None, collection: str = None, font: str = None) -> dict:
    "Create a Blender text object - titles, labels, mograph, anything with words in the render. An empty body is refused because such an object renders nothing while existing perfectly. align/align_y are validated against the enum this Blender offers. Says so when extrude is 0, since flat text is right for a 2D title and wrong for anything meant to catch a light. A TYPEFACE, NEW 2026-09-07: `font` takes a PATH to a .ttf or .otf. Left out, the text is in Blender's built-in face - which is what every text object this addon made before then was stuck in, because nothing could set one. There is no font library to name into, so it is a path; the same file passed twice reuses one datablock rather than stacking copies. The response says which face the text is actually in and whether it is the built-in."
    return _blender("create_text", body=body, name=name, size=size, extrude=extrude,
                    bevelDepth=bevel_depth, align=align, alignY=align_y, location=location,
                    rotation=rotation, collection=collection, font=font)


@mcp.tool()
def bl_add_bones(object: str, bones: list = None, replace_existing: bool = False, sample: dict = None,
                 enable_rigify: bool = None, rename_from: dict = None, remove: list = None,
                 merge_weights: bool = None) -> dict:
    "Add bones to an armature that ALREADY EXISTS. bl_create_armature builds bones and every other rigging tool edits what is there - bl_list_bones, bl_rename_bones, bl_set_bone_pose, the weight tools - so edit_bones appeared in exactly one place in the whole addon and a skeleton that arrived through bl_import_mesh could be renamed, posed and weighted but never given another bone. Adding a socket bone, or an ik_hand_gun beside the hand, is ordinary skeletal work for a game engine and had no typed path. A parent may be a bone that was already on the rig OR one created earlier in the same call. `bones` is [{name, head:[x,y,z], tail:[x,y,z], parent, connect, roll}]. TWO REFUSALS WORTH KNOWING: a bone whose head equals its tail has ZERO length and Blender DELETES it when edit mode is left - silently, with no error - so it is refused up front rather than surfacing later as a confusing count mismatch. And a name already on the rig is refused, because Blender would add '.001' beside it and everything looking that name up afterwards, including the vertex groups that skin it, would find the wrong bone. Restoring OBJECT mode is a postcondition rather than tidy-up: left in edit mode, every call after this one fails on an editor nobody asked for, so the mode is asserted afterwards and reported. REMOVE: remove [names] deletes bones (children reparented; merge_weights moves their weights)."
    return _blender("add_bones", object=object, bones=bones, replaceExisting=replace_existing, sample=sample,
                    enableRigify=enable_rigify, renameFrom=rename_from, remove=remove, mergeWeights=merge_weights)


@mcp.tool()
def bl_set_bone_collections(object: str, action: str = None, collection=None, bones: list = None,
                            name_prefix: str = None, parent: str = None, new_name: str = None, index: int = None,
                            visible: bool = None, solo: bool = None, ui_row: int = None, ui_title: str = None,
                            selection_set: bool = None, color_set=None, enable_rigify: bool = None) -> dict:
    "List, create, rename, remove, move, assign, unassign, show, solo and nest a Blender armature's bone collections and set their Rigify UI rows (SET_UI); collection is a name, or a layer index 0-31 on 3.6."
    return _blender("set_bone_collections", object=object, action=action, collection=collection, bones=bones,
                    namePrefix=name_prefix, parent=parent, newName=new_name, index=index, visible=visible, solo=solo,
                    uiRow=ui_row, uiTitle=ui_title, selectionSet=selection_set, colorSet=color_set,
                    enableRigify=enable_rigify)


@mcp.tool()
def bl_set_rigify_type(object: str, bones: list = None, chain_from: str = None, type: str = None,
                       parameters: dict = None, copy_to: dict = None, enable_rigify: bool = None) -> dict:
    "Write Rigify rig types and their parameters on metarig bones (bones or chain_from), copy_to {bones | chainChildren | mirror}; a parameter the type does not define is refused."
    return _blender("set_rigify_type", object=object, bones=bones, chainFrom=chain_from, type=type,
                    parameters=parameters, copyTo=copy_to, enableRigify=enable_rigify)


@mcp.tool()
def bl_generate_rigify_rig(metarig: str = None, name: str = None, target_rig: str = None, collection: str = None,
                           dry_run: bool = None, backup: bool = None, allow_straight_chains: bool = None,
                           enable_rigify: bool = None, preview_id: str = None, preview_ttl_seconds: int = None) -> dict:
    "Generate or regenerate a Rigify rig from a metarig after checks the generator skips (scale, UI rows, leftover face bone, straight limbs); reports bones by collection, IK/FK switches, erased bones and the backup. With dry_run=True it previews a new rig by the name it will get and returns a preview_id; preview_id alone commits it, refused if anything changed since."
    if preview_id:
        return (_preview_commit_refusal(preview_id, metarig=metarig, name=name, target_rig=target_rig,
                                        collection=collection, dry_run=dry_run, backup=backup,
                                        allow_straight_chains=allow_straight_chains, enable_rigify=enable_rigify,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("generate_rigify_rig", previewId=preview_id, _timeout=600))
    return _blender("generate_rigify_rig", metarig=metarig, name=name, targetRig=target_rig, collection=collection,
                    dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds, backup=backup,
                    allowStraightChains=allow_straight_chains, enableRigify=enable_rigify, _timeout=600)


@mcp.tool()
def bl_rigify_snap(object: str, limb: str = None, direction: str = None, frame: int = None, key: bool = None,
                   switch: bool = None, bake: dict = None, allow_simulation: bool = None) -> dict:
    "Snap a generated Rigify rig's limb IK_TO_FK or FK_TO_IK with its own UI operators (optionally keyed, switched or baked), or LIST its snap buttons and which IK_FK value means IK."
    return _blender("rigify_snap", object=object, limb=limb, direction=direction, frame=frame, key=key,
                    switch=switch, bake=bake, allowSimulation=allow_simulation)


@mcp.tool()
def bl_create_lattice(name: str = None, resolution: list = None, points_u: int = None,
                      points_v: int = None, points_w: int = None,
                      interpolation: str = None,
                      use_outside: bool = None, location: list = None, rotation: list = None,
                      scale: list = None, collection: str = None, fit_to: list = None, margin: float = None) -> dict:
    "Create a Blender LATTICE object - the thing a LATTICE modifier deforms with. Found by asking what the addon can consume and cannot produce, the sixth time that question has paid: bl_add_modifier learned to point a LATTICE modifier at an object and bpy.data.lattices appeared nowhere else, so the modifier could be aimed and there was nothing to aim it at. TWO THINGS THAT MAKE A CORRECT-LOOKING LATTICE DO NOTHING, both reported rather than left to be discovered. A default lattice is 2x2x2, and eight corner points describe an AFFINE transform and nothing else - it can scale, shear and translate but cannot bend, however far its points are moved; canDeformNonLinearly says which you have. And a lattice only influences geometry INSIDE its own volume, so one created at the origin will not touch a mesh standing elsewhere while the modifier reports itself perfectly configured. Interpolation is validated against this build's own enum. FIT TO OBJECTS (2026-10-05) - Blender's Lattice Deform Selected in one call: fit_to [object names] places and scales the lattice over the objects' joint WORLD bounding box (margin world units added on every side, default 0; the lattice's rest grid is -0.5..0.5, so the box's size is its scale) and puts a LATTICE modifier aimed at it on each object - read the bounds, create, then one bl_add_modifier per object were three steps. Measured: two 2 m cubes at [0, 0, 0] and [4, 0, 1] with margin 0.1 give a lattice at [2, 0, 0.5] scaled [6.2, 2.2, 3.2]. fittedTo in the reply names the objects, the box and the modifier added to each. Refused before anything is made: fit_to with location, rotation or scale; an object that does not exist; one a Lattice modifier cannot deform (MESH, CURVE, SURFACE, FONT, LATTICE only - a light is refused); a negative margin; a box with no thickness on an axis and margin 0."
    return _blender("create_lattice", name=name, resolution=resolution, pointsU=points_u,
                    pointsV=points_v, pointsW=points_w,
                    interpolation=interpolation, useOutside=use_outside, location=location,
                    rotation=rotation, scale=scale, collection=collection, fitTo=fit_to, margin=margin)


@mcp.tool()
def bl_create_curves(name: str = None, location: list = None, rotation: list = None,
                     scale: list = None, collection: str = None, surface: str = None, uv_map: str = None,
                     count: int = None, length: float = None, points_per_curve: int = None, direction: str = None,
                     custom_direction: list = None, density_group: str = None, seed: int = None,
                     surface_deform: bool = None, preset: str = None, density: str = None, radius: float = None,
                     view_percentage: float = None, apply_hair_guides: bool = None, use_noise: bool = None,
                     use_frizz: bool = None) -> dict:
    "Create a Blender CURVES object (the hair/curves type a geometry node group fills; not bl_create_curve's legacy CURVE). With surface and count it grows real guide curves on a mesh. With surface_deform it is Blender's own Empty Hair, and preset 'fur' is its Quick Fur. Call mif_help(\"bl_create_curves\") first."
    return _blender("create_curves", name=name, location=location, rotation=rotation,
                    scale=scale, collection=collection, surface=surface, uvMap=uv_map, count=count, length=length,
                    pointsPerCurve=points_per_curve, direction=direction, customDirection=custom_direction,
                    densityGroup=density_group, seed=seed, surfaceDeform=surface_deform, preset=preset,
                    density=density, radius=radius, viewPercentage=view_percentage,
                    applyHairGuides=apply_hair_guides, useNoise=use_noise, useFrizz=use_frizz)


@mcp.tool()
def bl_create_point_cloud(name: str = None, location: list = None, rotation: list = None,
                          scale: list = None, collection: str = None) -> dict:
    "Create a Blender POINTCLOUD object - points with attributes, driven by geometry nodes. Same argument and same measurement as bl_create_curves: on 3.6.23, 4.4.0 and 5.0.1 an empty pointcloud accepts a NODES modifier and a generating group evaluates to real points. IT ARRIVES EMPTY AND BLENDER HAS NO UI FOR PLACING POINTS IN ONE BY HAND EITHER - the type exists to be generated. The response says pointCount 0 and isEmpty true rather than staying quiet about it. Fill it with bl_create_node_group plus bl_assign_node_group, and the group's OUTPUT must be POINT geometry (a Mesh Grid into Mesh to Points, for instance) - a mesh-outputting group evaluates without error and lands nothing. bl_object_info reports the stored and evaluated counts separately, which is the only way to see that it worked."
    return _blender("create_point_cloud", name=name, location=location, rotation=rotation,
                    scale=scale, collection=collection)


@mcp.tool()
def bl_create_volume(name: str = None, location: list = None, rotation: list = None,
                     scale: list = None, collection: str = None, filepath: str = None, sequence=None) -> dict:
    "Create a Blender VOLUME object - the container for OpenVDB grids, filled by geometry nodes or loaded from a .vdb file. The third object type on the same measured argument as bl_create_curves and bl_create_point_cloud, and checked for THIS type rather than assumed from the other two: with a volume-outputting group (Mesh Grid into Mesh to Volume) an empty volume evaluates to a grid on 3.6.23, 4.4.0 and 5.0.1 alike. OPENVDB: filepath loads a .vdb (sequence=true or {frameStart, frameOffset, mode}: every numbered file beside it), and the reply lists its grids - name, data type, channels, voxel size - and the world box; a file Blender cannot read is refused and the object removed. A VOLUME WITH GRIDS AND NO VOLUME SHADER RENDERS NOTHING, which is how this most often looks broken - a fresh volume object has no material at all (reported as materialSlots, a number rather than a warning to be believed) and Blender's default material is a SURFACE shader, which a volume does not use. No material is created for you: choosing somebody's shader is not this op's job. bl_object_info reports the stored and evaluated grid counts separately, which is how you tell an empty volume from a generated one."
    return _blender("create_volume", name=name, location=location, rotation=rotation,
                    scale=scale, collection=collection, filepath=filepath, sequence=sequence)


@mcp.tool()
def bl_create_nurbs_surface(name: str = None, kind: str = None, radius: float = None, location: list = None,
                            rotation: list = None, scale: list = None, collection: str = None,
                            control_points: list = None, order_u: int = None, order_v: int = None,
                            resolution_u: int = None, resolution_v: int = None, endpoint_u: bool = None,
                            endpoint_v: bool = None, cyclic_u: bool = None, cyclic_v: bool = None) -> dict:
    "Create a Blender SURFACE object - a NURBS surface with real control points (a PATCH, CYLINDER, SPHERE or TORUS), its grid, orders and resolutions read back with the faces it evaluates to. See mif_help."
    return _blender("create_nurbs_surface", name=name, kind=kind, radius=radius, location=location,
                    rotation=rotation, scale=scale, collection=collection, controlPoints=control_points,
                    orderU=order_u, orderV=order_v, resolutionU=resolution_u, resolutionV=resolution_v,
                    endpointU=endpoint_u, endpointV=endpoint_v, cyclicU=cyclic_u, cyclicV=cyclic_v)


@mcp.tool()
def bl_create_metaball(name: str = None, location: list = None, rotation: list = None, scale: list = None,
                       collection: str = None, elements: list = None, resolution: float = None,
                       render_resolution: float = None, threshold: float = None, update_method: str = None) -> dict:
    "Create a Blender META object - a metaball with its elements (ball, capsule, plane, ellipsoid, cube; radius, stiffness, negative), never an empty one; the family it joins and the faces it evaluates to read back. See mif_help."
    return _blender("create_metaball", name=name, location=location, rotation=rotation, scale=scale,
                    collection=collection, elements=elements, resolution=resolution,
                    renderResolution=render_resolution, threshold=threshold, updateMethod=update_method)


@mcp.tool()
def bl_append_from_file(filepath: str, objects: list = None, collections: list = None,
                        materials: list = None, node_groups: list = None, actions: list = None,
                        worlds: list = None, fonts: list = None, link: bool | str = None,
                        active_collection: str = None, list_only: bool = None,
                        instance_empty: bool = None) -> dict:
    "Bring datablocks IN from another .blend - the half of the Blender file ops that did not exist. bl_save_file wrote and bl_open_file REPLACES the session, so a scene this addon built could be saved and then never reused; every asset-library workflow is this call. APPEND COPIES, LINK REFERENCES, and the difference decides whether you can edit what you just brought in: link=false (the default) copies the datablocks into this file - editable, and they survive the source being moved; link=true leaves them in the source file and points at it, which means NOT editable (a transform on a linked object is refused by Blender, and reads as this addon being broken if you did not know you linked) and the source must exist forever. The response says which happened and what it costs. list_only=true READS THE FILE AND IMPORTS NOTHING, so you can see what is in it before choosing names. A name the file does not contain is REFUSED naming what it does have - Blender's own append silently imports nothing for a missing name and reports no error, which is a call that succeeds and does nothing. Objects and collections are linked into the scene's active collection so they are actually VISIBLE: bpy.data.libraries.load puts them in bpy.data and nothing else, and an object in no collection is invisible, unrendered, and saved anyway. Colliding names get Blender's .001 suffix and the response reports every one under `renamed`, so you know what you actually have. FONTS ARE IN THE LIST because a typeface is the one thing you cannot re-derive from a file you did not build: bl_create_text's `font` takes a PATH, so text set in somebody else's face has no name you can reach - list_only:true now shows you."
    return _blender("append_from_file", filepath=filepath, objects=objects,
                    collections=collections, materials=materials, nodeGroups=node_groups,
                    actions=actions, worlds=worlds, fonts=fonts, link=link,
                    activeCollection=active_collection, listOnly=list_only, instanceEmpty=instance_empty)


@mcp.tool()
def bl_set_text(object: str, body: str = None, size: float = None, extrude: float = None,
                bevel_depth: float = None, align: str = None, align_y: str = None,
                font: str = None) -> dict:
    "Change an existing Blender text object - the half that did not exist, so text was WRITE-ONCE. bl_create_text could set a body, a size and an alignment and nothing could change any of them afterwards, while bl_object_info REPORTS `body` for a FONT object - the string was readable and unwritable. Re-labelling a sign meant deleting the object and rebuilding it, which loses its transform, its material and anything parented to it. EVERY FIELD IS OPTIONAL and only what you name is touched, so it is safe to call with one key. The response carries `before`, `after` and `changed`, and `changedAnything` is false when the values asked for were the ones already there - so a caller can tell a no-op from a write, which a bare ok:true cannot. `font` is a PATH to a .ttf/.otf, as on bl_create_text."
    return _blender("set_text", object=object, body=body, size=size, extrude=extrude,
                    bevelDepth=bevel_depth, align=align, alignY=align_y, font=font)


@mcp.tool()
def bl_batch(ops: list = None, continue_on_error: bool = None, budget_seconds: float = None,
             stop_on_budget: bool = None, results: str = None, preview_id: str = None, dry_run: bool = None,
             preview_ttl_seconds: int = None) -> dict:
    "Run several Blender ops as ONE job ({op, ...params} each, in its bl_ tool's names), with an honest partial-failure contract; results='errors' returns only the ops that failed. The addon is single-threaded by design - one worker draining one queue - so this is not concurrency; it is one round trip instead of N, which matters because every op here is a socket call. IT IS NOT ATOMIC AND THE RESPONSE SAYS SO. A batch failing at op 7 of 10 leaves six applied - but the whole batch is ONE undo step, so a single bl_undo takes all six back, and atomicNote says whether this batch's step was recorded. The response carries `atomic: false`, `applied`, `failed` and `skipped` by name, plus a per-op `results` list in order with each op's own result and timing - so what happened is readable rather than a single verdict. THE WHOLE BATCH IS VALIDATED BEFORE ANY OF IT RUNS: an unknown op name in the last entry refuses the batch without applying the first, and an empty ops[] is a hard failure rather than ok:true with zero rows, because the response IS the audit trail. Each op's own parameters are checked by the handler it dispatches to, which already refuses unknown keys by name. IT STOPS AT THE FIRST FAILURE by default, because a batch is usually a build sequence where a later op depends on an earlier one and running on turns one diagnosable error into several; continue_on_error=True runs the rest when the ops are independent. budget_seconds is a TIME budget, not an op cap, and that is deliberate: measured op costs span four orders of magnitude (bl_list_objects 0.05ms, bl_add_modifier SUBSURF 205ms, a first bl_render_still 610ms), so 1000 ordinary ops is about a second while 250 renders is 150 seconds. The budget is checked BETWEEN ops - nothing can interrupt a running handler - so the guarantee is that it will not START an op it cannot afford, and it names the one it skipped. Hitting the budget is not a failure: the ops that ran succeeded. With dry_run=True it previews every op's changes with their values before and returns a preview_id, refused when an op takes no preview or two ops change one target; preview_id alone commits them all, rolled back together on a mismatch."
    refusal = _batch_results_refusal(results)
    if refusal:
        return refusal
    # THE CHANGE CAP (docs/15 section 6): a batch over the open file's cap answers a preview with a previewId, and
    # preview_id alone commits exactly the ops it listed.
    # THE BATCH PREVIEW (docs/15 section 5): dry_run plans every op and writes nothing; its preview_id commits them all.
    if preview_id:
        if ops is not None or dry_run or preview_ttl_seconds:
            return {"ok": False, "endpoint": "batch", "nothingWritten": True,
                    "error": "a commit takes preview_id alone - the ops were fixed when they were previewed. NOTHING was run."}
        reply = _blender("batch", previewId=preview_id)
        return _fit_batch_reply(_errors_only(reply) if results == "errors" else reply)
    ops, refused = _bl_batch_ops(ops)
    if refused:
        return {"ok": False, "endpoint": "batch", "error": refused}
    reply = _blender("batch", ops=ops, continueOnError=continue_on_error, budgetSeconds=budget_seconds,
                     stopOnBudget=stop_on_budget, dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)
    return _fit_batch_reply(_errors_only(reply) if results == "errors" else reply)


def _bl_batch_ops(ops):
    """bl_batch's ops as the add-on reads them, [{op, params}], and a refusal or None.

    THE FLAT FORM IS ACCEPTED because the add-on read only `params` and IGNORED every other key of an entry, so
    {op: 'create_primitive', kind: 'cube'} - the shape Unreal's batch takes - ran create_primitive with NO parameters and
    was refused "'kind' is required" beside the kind it had been given. the benchmark's bracket and grid runs sent exactly
    that nine times (docs/TOKEN_OPTIMIZATION.md). An entry carrying BOTH params and top-level keys is refused here
    rather than guessed at: the add-on would drop the top-level ones in silence.

    AN OP NAMED BY ITS TOOL GOES AS THE OP THAT TOOL POSTS. No add-on op starts with bl_, so {op: 'bl_create_primitive'}
    could only be refused - and 11baf1c7's bracket runs wrote exactly that twice, reading this tool's "in its bl_ tool's
    names" as covering the op too. The op is read off the tool's own `_blender` call, never derived from its name:
    bl_status posts ping."""
    if not isinstance(ops, list):
        return ops, None
    out = []
    for i, entry in enumerate(ops):
        if not isinstance(entry, dict):
            out.append(entry)            # the add-on refuses it, naming the index
            continue
        key = "op" if "op" in entry else "endpoint"
        named = entry.get(key)
        posted = _bl_tool_op(named) if isinstance(named, str) and named.startswith("bl_") else None
        if posted:
            entry = dict(entry, **{key: posted})
        extra = {k: v for k, v in entry.items() if k not in ("op", "endpoint", "params")}
        params = entry.get("params")
        if extra and params is not None:
            return None, ("ops[%d] has both 'params' and top-level keys (%s). Give its parameters one way - all "
                          "under params, or all beside op. NOTHING was run." % (i, ", ".join(sorted(extra))))
        if params is None:
            params = extra
        if isinstance(params, dict):
            params = _backend_op_params("blender", entry.get("op") or entry.get("endpoint"), params)
        out.append(dict({k: entry[k] for k in ("op", "endpoint") if k in entry}, params=params))
    return out, None


@mcp.tool()
def bl_create_texture(name: str = None, type: str = None, image: str = None,
                      intensity: float = None, contrast: float = None, file: str = None,
                      colorspace: str = None) -> dict:
    "Create a legacy Blender TEXTURE datablock - what a DISPLACE modifier reads. Same producer/consumer gap as bl_create_lattice and found in the same pass: bl_add_modifier can point DISPLACE at a texture and bpy.data.textures appeared nowhere else in the addon. THESE ARE NOT SHADER NODES. bpy.data.textures is Blender's older texture system and is what the MODIFIER STACK reads - a DISPLACE modifier cannot take a ShaderNodeTexNoise. Both systems exist, they are not interchangeable, and that is the confusion this is most likely to be met with. The type is validated against this build's own enum (BLEND, CLOUDS, DISTORTED_NOISE, IMAGE, MAGIC, MARBLE, MUSGRAVE, NOISE, STUCCI, VORONOI, WOOD - identical on 3.6, 4.2, 4.4 and 5.0, read off the enum rather than assumed), and `image` is refused for any type but IMAGE rather than accepted and ignored. `file` loads an image from disk as the datablock (type IMAGE implied, colorspace default Non-Color, because a height map is data)."
    return _blender("create_texture", name=name, type=type, image=image,
                    intensity=intensity, contrast=contrast, file=file, colorspace=colorspace)


@mcp.tool()
def bl_create_armature(name: str = None, bones: list = None, display_type: str = None,
                       show_in_front: bool = None, location: list = None, rotation: list = None,
                       collection: str = None, metarig: str = None, height: float = None, scale: float = None,
                       enable_rigify: bool = None) -> dict:
    "Create a Blender armature with its bones - without which the whole rigging family could only EDIT. ops_rig has twelve ops and not one creates an armature, so nothing could be rigged from scratch. BONES ONLY EXIST IN EDIT MODE (armature.edit_bones is absent outside it), so this switches mode, builds, and switches back - and restoring the mode is a POSTCONDITION, not a courtesy, because being left in edit mode strands every op that follows. Every bone is validated in full BEFORE any mode change, so a bad entry cannot leave a half-built rig with Blender stuck in an editor. Bones are counted from data.bones after leaving edit mode, not from the edit_bones the op made, since those only exist inside it. Parents must be listed before their children. Each bone takes name, head, tail, parent, connect and rotationMode (the pose bone's, default QUATERNION); any other key in a bone is refused. rotationModes reads every bone's mode back."
    return _blender("create_armature", name=name, bones=bones, displayType=display_type,
                    showInFront=show_in_front, location=location, rotation=rotation,
                    collection=collection, metarig=metarig, height=height, scale=scale,
                    enableRigify=enable_rigify)


@mcp.tool()
def bl_create_collection(name: str, parent: str = None, objects: list = None,
                         link: bool = None, color_tag: str = None) -> dict:
    "Create a Blender collection and LINK it into the scene, because bpy.data.collections.new() alone makes one that belongs to no scene - its objects are outside the view layer, outside the depsgraph and outside the render, while every field on it reads perfectly. Before this the only collection creation anywhere in the addon was a private helper inside bl_set_light_linking that made an EMPTY one, so light linking could reach only its broken state (litsNothing) and nothing could fix it without bl_run_python. Pass objects to fill it at birth. The postcondition is REACHABILITY from the scene collection, not existence."
    return _blender("create_collection", name=name, parent=parent, objects=objects,
                    link=link, colorTag=color_tag)


@mcp.tool()
def bl_list_collections(view_layer: str = None, with_objects: bool = None, limit: int = None,
                        name_excludes: list = None) -> dict:
    "The Blender collection tree with the four different things 'hidden' can mean, kept apart: hideViewport and hideRender are GLOBAL, while exclude and hideInViewLayer (the eye icon, the one people actually click) are PER VIEW LAYER and live on a LayerCollection rather than the collection. A null in a per-layer field means the collection is not in that view layer at all, which is why excluding an orphan does nothing. Also reports orphanCollections and objectsInNoCollection - both invisible everywhere while every field still reads correctly."
    return _blender("list_collections", viewLayer=view_layer, withObjects=with_objects,
                    limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_link_objects(collection: str = None, scene: str = None, objects: list = None,
                    object: str = None, move: bool = None) -> dict:
    "Put Blender objects into a collection OR into a scene. An object can be in MANY collections at once, so this ADDS rather than moves - pass move:true to unlink it from every other collection first, including the scene root, which a version that only walked bpy.data.collections would miss. Every name is resolved before anything is linked, so a typo in the fourth name does not leave a half-populated collection. Warns when the target collection is not itself in the scene, since filling an orphan changes nothing anybody can see. Pass `scene` to target a scene's own master collection: that datablock never appears in bpy.data.collections, so it cannot be reached by ANY name through `collection` - which meant bl_create_scene could make a scene and nothing could put an object into it. Exactly one of collection or scene."
    return _blender("link_objects", collection=collection, scene=scene, objects=objects,
                    object=object, move=move)


@mcp.tool()
def bl_unlink_objects(collection: str = None, scene: str = None, objects: list = None,
                      object: str = None, allow_orphans: bool = None) -> dict:
    "Take Blender objects out of a collection, or out of a scene's own master collection, REFUSING by default to leave one in no collection at all. An object in zero collections still exists in bpy.data and is in no scene: invisible in the viewport, absent from the render, gone from the outliner, nothing warns, and it survives the save. The check runs across every named object before anything is unlinked, so the refusal cannot fire partway through. Pass allow_orphans to mean it. `scene` is the counterpart of bl_link_objects' - a scene's master collection is not in bpy.data.collections and has no name `collection` can resolve. Exactly one of collection or scene."
    return _blender("unlink_objects", collection=collection, scene=scene, objects=objects,
                    object=object, allowOrphans=allow_orphans)


@mcp.tool()
def bl_set_collection_visibility(collection: str, view_layer: str = None,
                                 hide_viewport: bool = None, hide_render: bool = None,
                                 exclude: bool = None, hide_in_view_layer: bool = None,
                                 indirect_only: bool = None, holdout: bool = None) -> dict:
    "The four meanings of 'hide this Blender collection', taken by their real names because writing the wrong one is a silent no-op that reads back as success on the property that WAS written. hide_viewport/hide_render are global; exclude, hide_in_view_layer (the eye), indirect_only and holdout are per view layer and are refused outright if the collection is not in that view layer, since there is no LayerCollection to write to. EXCLUDE IS NOT HIDING: it removes the collection from the depsgraph entirely, so constraints, drivers and modifiers depending on those objects change behaviour too. Every requested write is verified individually afterwards. indirect_only is a CYCLES feature - under EEVEE it changes nothing (measured 0.0/255 on every supported build), so the response reports inertUnderThisEngine when it is written outside Cycles."
    return _blender("set_collection_visibility", collection=collection, viewLayer=view_layer,
                    hideViewport=hide_viewport, hideRender=hide_render, exclude=exclude,
                    hideInViewLayer=hide_in_view_layer, indirectOnly=indirect_only,
                    holdout=holdout)


@mcp.tool()
def bl_delete_collection(collection: str, delete_objects: bool = None,
                         reparent_to: str = None) -> dict:
    "Delete a Blender collection and say what became of its objects. bpy.data.collections.remove() deletes the collection and leaves the objects alone, which sounds safe and is exactly how objects end up in no collection at all - still in bpy.data, in no scene, invisible, surviving the save. So this decides it explicitly: objects that would be stranded move to the scene root (or reparent_to), or are deleted outright with delete_objects. Objects that are ALSO in another collection are left where they are rather than silently reorganised. Child collections are relinked rather than orphaned. Only objects the deletion would strand are rehomed: an object also linked in another collection or the scene root stays there, and objectsKeptElsewhere maps each such object to the homes it keeps (with a keptNote when reparent_to was given). bl_link_objects with move:true moves them. With delete_objects, rigid bodies leave the physics world first, as bl_delete_object's do: a joint holding one of the collection's bodies is deleted with it even when it sits elsewhere (jointsRemovedWithBodies), and a rigid body bake the delete leaves stale is freed (freedRigidBodyBake, rigidBodyBakeNote)."
    return _blender("delete_collection", collection=collection, deleteObjects=delete_objects,
                    reparentTo=reparent_to)


@mcp.tool()
def bl_duplicate_collection(collection: str, new_name: str = None, linked: bool = None, parent: str = None) -> dict:
    "Duplicate a Blender collection with its objects and child collections under new_name: each object copied once (linked shares its mesh), parents remapped to the copies."
    return _blender("duplicate_collection", collection=collection, newName=new_name, linked=linked, parent=parent)


@mcp.tool()
def bl_add_collection_instance(collection: str = None, mode: str = None, transforms: list = None, points: list = None,
                               name: str = None, target: str = None, instance_offset: list = None, objects: list = None,
                               limit: int = None) -> dict:
    "Instance a Blender collection (an Empty drawing it) at transforms or points without copying geometry; LIST them with each one's world box, or REALIZE them into real objects. A target inside the collection itself is refused. See mif_help."
    return _blender("add_collection_instance", collection=collection, mode=mode, transforms=transforms, points=points,
                    name=name, target=target, instanceOffset=instance_offset, objects=objects, limit=limit)

@mcp.tool()
def bl_create_world(name: str, use_as_scene_world: bool = None, color: list = None,
                    strength: float = None) -> dict:
    "Create a Blender World datablock. The addon could NAME a world - bl_world_info and bl_set_world both take one - and nothing could make one, which two independent audits reported on the same day from opposite ends. A World that is not the SCENE's contributes no light at all: it is a datablock sitting in the file, and the response says isSceneWorld plus a note rather than leaving a caller to wonder why the render did not change. Pass use_as_scene_world to make it the active one. A name that already exists is refused rather than silently reused, so `created` never lies; edit an existing world with bl_set_world instead."
    return _blender("create_world", name=name, useAsSceneWorld=use_as_scene_world,
                    color=color, strength=strength)


@mcp.tool()
def bl_set_world(color: list = None, strength: float = None, hdri: str = None,
                 rotation: float = None, mist_use: bool = None, mist_start: float = None,
                 mist_depth: float = None, use_as_light: bool = None, name: str = None, sun: dict = None,
                 probe_resolution: str = None) -> dict:
    "Set the Blender world background - a flat colour or an HDRI, plus strength and mist. A scene with no world contributes no ambient light at all, so an interior renders pure black outside its own fixtures and the lights get blamed. strength multiplies emission: 1.0 with mid grey is roughly overcast and washes out a dark interior, which usually wants 0.02-0.1. Passing both an hdri and a colour is refused - the texture would silently override the colour. useAsLight writes the world's Cycles ray visibility, which EEVEE does not read (measured no change on 3.6-5.2); under EEVEE the reply says so with inertUnderThisEngine and an engineNote."
    return _blender("set_world", color=color, strength=strength, hdri=hdri, rotation=rotation,
                    mistUse=mist_use, mistStart=mist_start, mistDepth=mist_depth,
                    useAsLight=use_as_light, name=name, sun=sun, probeResolution=probe_resolution)


@mcp.tool()
def bl_world_info(name: str = None) -> dict:
    "What the Blender world actually IS - the read half of bl_set_world, which had none. The family was write-only: bl_scene_info omits the world entirely and bl_render_info reports only its NAME, so 'what is my world set to' was unanswerable on the one datablock that decides whether an interior renders black. Every answer is taken from the LINK rather than the node, because that is where a shader tree's effect lives: backgroundConnected (a Background node not wired to the world output accepts every write and contributes nothing), and environmentTextureDriving, found by walking backwards from the Colour socket so a Mapping in between still counts while a texture left unlinked correctly does not. Also reports useNodes, because a world with it off IGNORES the whole tree and renders its flat colour while every node reads perfectly. blockers is the diagnosis, not the inputs to it. A scene with no world returns a blocker rather than an error - that is the commonest black-render cause, not a failure. A LINKED COLOUR RENDERS ITS LINK: colorLinked says whether anything drives the Background's Colour, colorDrivenBy names it, skyTextureDriving names a Sky Texture as environmentTextureDriving names an HDRI, and `color` is then the socket's unused value (colorNote says so). Until 2026-09-20 a sky-lit world whose socket value was black read contributesLight:false."
    return _blender("world_info", name=name)


@mcp.tool()
def bl_add_rigid_body(object: str = None, type: str = None, mass: float = None, friction: float = None,
                      bounciness: float = None, collision_shape: str = None,
                      kinematic: bool = None, margin: float = None, linear_damping: float = None,
                      angular_damping: float = None, objects: list = None, collection: str = None,
                      copy_from: str = None, update: bool = None, density: float = None, material: str = None,
                      use_deactivation: bool = None, start_deactivated: bool = None,
                      deactivate_linear_velocity: float = None, deactivate_angular_velocity: float = None,
                      enabled: bool = None) -> dict:
    "Make Blender objects (object, objects or a collection) ACTIVE (falls) or PASSIVE (is landed on) rigid bodies, or update existing ones: deactivation, the Dynamic flag (enabled), copy_from and a mass from density or material. The simulation lives in a scene-level rigid body world that Blender creates on demand, so this goes through the operator with a context override - assigning obj.rigid_body directly is not possible. A rigid body is stepped forward from the start frame, so jumping to a late frame shows it at REST: call bl_bake_physics before rendering that frame. A body added to a world that is ALREADY BAKED is not in those frames and does not move at all - the response carries bakedWorldNote, and bl_bake_physics rebake=true is the fix. BOUNCINESS NEEDS BOTH SURFACES: measured on every supported build, a sphere at 0.99 rebounds 0.00 m off a floor at 0 and 2.29 m off a floor at 0.99, so bouncinessNote names the bodies in the world that have none."
    return _blender("add_rigid_body", object=object, type=type, mass=mass, friction=friction,
                    bounciness=bounciness, collisionShape=collision_shape, kinematic=kinematic,
                    margin=margin, linearDamping=linear_damping, angularDamping=angular_damping, objects=objects,
                    collection=collection, copyFrom=copy_from, update=update, density=density, material=material,
                    useDeactivation=use_deactivation, startDeactivated=start_deactivated,
                    deactivateLinearVelocity=deactivate_linear_velocity,
                    deactivateAngularVelocity=deactivate_angular_velocity, enabled=enabled)


@mcp.tool()
def bl_add_cloth(object: str, quality: int = None, mass: float = None, stiffness: float = None,
                 damping: float = None, gravity: float = None, use_pressure: bool = None,
                 pressure: float = None, collision_quality: int = None,
                 self_collision: bool = None, pin_group: str = None,
                 pin_stiffness: float = None, preset: str = None, bending: float = None,
                 bending_damping: float = None, bending_model: str = None, bending_group: str = None,
                 bending_max: float = None, pressure_group: str = None, pressure_factor: float = None) -> dict:
    "Add a cloth simulation to a Blender mesh (refused under 16 vertices). preset COTTON/DENIM/LEATHER/RUBBER/SILK from the build's own files, or FLEECE/CANVAS; bending springs; pin_group pins; pressure_group scales pressure by region."
    return _blender("add_cloth", object=object, quality=quality, mass=mass, stiffness=stiffness,
                    damping=damping, gravity=gravity, usePressure=use_pressure, pressure=pressure,
                    collisionQuality=collision_quality, selfCollision=self_collision,
                    pinGroup=pin_group, pinStiffness=pin_stiffness, preset=preset, bending=bending,
                    bendingDamping=bending_damping, bendingModel=bending_model, bendingGroup=bending_group,
                    bendingMax=bending_max, pressureGroup=pressure_group, pressureFactor=pressure_factor)


@mcp.tool()
def bl_add_soft_body(object: str, mass: float = None, friction: float = None, speed: float = None,
                     use_goal: bool = None, goal_default: float = None, goal_spring: float = None,
                     pull: float = None, push: float = None, bend: float = None,
                     use_edges: bool = None, self_collision: bool = None,
                     gravity: float = None, goal_vertex_group: str = None, goal_min: float = None,
                     goal_max: float = None, goal_friction: float = None) -> dict:
    "Add a SOFT BODY simulation to a Blender mesh - the deformable one, between cloth and a rigid body: jelly, a beanbag, a sagging cable, flesh over a rig. GOAL IS ON BY DEFAULT AND IT IS WHY NOTHING MOVES: Blender ships use_goal=True with goal_default=0.7, which pins every vertex toward its ORIGINAL position, so a soft body added with defaults sags a little and stops - and reads as a simulation that does not work. That default is NOT overridden, because silently inverting what the engine chose is worse than reporting it; the response says useGoal and carries a goalNote explaining it. Pass use_goal=false for a free-falling body, or lower goal_default to loosen the pin. TO PIN PART OF THE MESH - a jelly's base on its plate, a tail's root - pass goal_vertex_group (an existing vertex group, made with bl_set_vertex_weights): each vertex's goal becomes its weight there TIMES goal_default, mapped into goal_min..goal_max, so weight-1 vertices hold at goal_default's strength and weight-0 ones move; goal_friction damps them. goal_default=0 beside a group is refused, since every weight would multiply to zero and pin nothing. Without it every vertex shares one goal and the body sways as one lump. An unknown group, a group beside use_goal=false, or goal_min above goal_max is refused with nothing added. Settings live on the OBJECT (object.soft_body), not on the modifier as cloth's do, which is the trap that makes a value look like it landed when it did not - this writes and reads back the same place. Refused on a mesh under 16 vertices, since a soft body deforms the geometry it is given and a quad has nothing to bend, and refused on an object that already has one, because a second solver does not make a stronger simulation, it makes two fighting over the same vertices. `gravity` is a WEIGHT on the scene's gravity, not an acceleration - 1.0 normal, 0.0 floats. Every attribute this writes was verified present on Blender 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1, so there is nothing version-gated here. Like any simulation it is stepped forward: bake_physics before jumping to a late frame."
    return _blender("add_soft_body", object=object, mass=mass, friction=friction, speed=speed,
                    useGoal=use_goal, goalDefault=goal_default, goalSpring=goal_spring,
                    pull=pull, push=push, bend=bend, useEdges=use_edges,
                    selfCollision=self_collision, gravity=gravity, goalVertexGroup=goal_vertex_group,
                    goalMin=goal_min, goalMax=goal_max, goalFriction=goal_friction)


@mcp.tool()
def bl_add_fluid(object: str, type: str, name: str = "Fluid", domain_type: str = None,
                 resolution_max: int = None, cache_start: int = None, cache_end: int = None,
                 use_adaptive_domain: bool = None, flow_type: str = None,
                 flow_behavior: str = None, use_inflow: bool = None,
                 surface_distance: float = None, subframes: int = None,
                 effector_type: str = None, use_effector: bool = None, use_mesh: bool = None,
                 time_scale: float = None, cache_type: str = None, cache_directory: str = None,
                 cache_resumable: bool = None) -> dict:
    "Make a Blender mesh a fluid DOMAIN, FLOW or EFFECTOR - smoke, fire and liquid. FLUID has been in the addon's _SIM_MODIFIERS since that list was written, so bake_physics and physics_info always handled one and nothing could create one: the same consumer-with-no-creator shape as soft body. bl_add_modifier can add the FLUID modifier but not configure it - the modifier carries exactly ONE writable property, fluid_type, and everything that matters lives on domain_settings/flow_settings/effector_settings. THIS DOES NOT BAKE, deliberately: a fluid bake is minutes to hours and blocks Blender's main thread for all of it, so call bl_bake_physics when the setup is right and expect it to be slow. TWO DEFAULTS MAKE A CORRECT SETUP LOOK BROKEN and are reported rather than overridden - resolution_max is 32, Blender's default and low enough that smoke is visibly blocky; and flow_behavior defaults to GEOMETRY, which turns the MESH into fluid ONCE, a single puff, where the continuous emitter people mean by 'add smoke' is INFLOW. Neither is changed for you, because picking a simulation resolution or guessing which behaviour you meant is not this op's business, but the response says so. A SETTING SENT TO THE WRONG ROLE IS REFUSED naming the role that owns it: resolution_max with type=FLOW is a caller who believes they set a resolution, and dropping it quietly would leave them believing it. One object is one role - a domain that is also a flow is not something Mantaflow solves. Verified on 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1. A LIQUID domain gets mesh generation ON (use_mesh) on every build: 3.6 starts it off, which simulates a liquid with no surface - useMeshNote says when this op turned it on; use_mesh:false keeps it off. A GAS DOMAIN WITH NO VOLUME SHADER renders as a solid box in EEVEE and as nothing in Cycles (measured on every build), so the response's volumeNote says when one is missing and names the calls: bl_create_material, bl_set_material_volume with surface='disconnect', then bl_set_material_slots. EVERY OTHER SETTING of the role - smoke dissolve, the domain's border collisions, flame and vorticity, noise upres, time scale, a flow's fuel and temperature - is written afterwards with bl_set_modifier(object, modifier='Fluid', settings={identifier: value}). surface_distance is in grid CELLS (Blender's default 1.5): 0.08, read as metres, emits almost nothing from a thin burner and the domain stays empty."
    return _blender("add_fluid", object=object, type=type, name=name, domainType=domain_type,
                    resolutionMax=resolution_max, cacheStart=cache_start, cacheEnd=cache_end,
                    useAdaptiveDomain=use_adaptive_domain, flowType=flow_type,
                    flowBehavior=flow_behavior, useInflow=use_inflow,
                    surfaceDistance=surface_distance, subframes=subframes,
                    effectorType=effector_type, useEffector=use_effector, useMesh=use_mesh, timeScale=time_scale,
                    cacheType=cache_type, cacheDirectory=cache_directory, cacheResumable=cache_resumable)


@mcp.tool()
def bl_add_dynamic_paint(object: str, type: str, name: str = "Dynamic Paint",
                         surface_type: str = None, surface_format: str = None,
                         frame_start: int = None, frame_end: int = None,
                         use_dissolve: bool = None, dissolve_speed: int = None,
                         paint_color: list = None, paint_alpha: float = None,
                         paint_wetness: float = None, paint_source: str = None,
                         paint_distance: float = None, use_absolute_alpha: bool = None) -> dict:
    "Make a Blender mesh a dynamic paint CANVAS or BRUSH - wetmaps, footprints in snow, weathering, anything where one object marks another. The last of the modifier types this addon could add and not configure: like FLUID it sits in _SIM_MODIFIERS so bake_physics and physics_info always handled one, and its modifier carries exactly ONE writable property, ui_type, with everything real on canvas_settings or brush_settings. ui_type ALONE DOES NOTHING - setting it leaves those settings None on 3.6, 4.4 and 5.0 alike, so a modifier that looks configured in the outliner can have no settings at all; the settings are created by an operator, which this runs for you. A CANVAS ARRIVES WITH ONE SURFACE and this configures that one - several surfaces on a canvas is a real setup and is not built here, so the count does not depend on how often you called this. surface_type IS A DYNAMIC ENUM whose members depend on the surface FORMAT, so it is validated against Blender at the moment of writing and a refusal names what is available NOW - which is the only correct answer to 'why can I not set WAVE'. Format is applied before type for the same reason. A setting sent to the wrong role is refused naming the role that owns it: a canvas receives paint and a brush applies it. Nothing is baked."
    return _blender("add_dynamic_paint", object=object, type=type, name=name,
                    surfaceType=surface_type, surfaceFormat=surface_format,
                    frameStart=frame_start, frameEnd=frame_end, useDissolve=use_dissolve,
                    dissolveSpeed=dissolve_speed, paintColor=paint_color, paintAlpha=paint_alpha,
                    paintWetness=paint_wetness, paintSource=paint_source,
                    paintDistance=paint_distance, useAbsoluteAlpha=use_absolute_alpha)


@mcp.tool()
def bl_add_rigid_body_constraint(object1: str = None, object2: str = None, name: str = "Joint",
                                 type: str = "FIXED", location: list = None,
                                 rotation: list = None,
                                 disable_collisions: bool = None,
                                 breaking_threshold: float = None,
                                 enabled: bool = None, objects: list = None, active: str = None,
                                 pattern: str = None, pivot: str = None, collection: str = None,
                                 solver_iterations: int = None, use_breaking: bool = None,
                                 update: bool = None, names: list = None) -> dict:
    "Join Blender RIGID BODIES with physics joints - type FIXED, POINT (a chain), HINGE (a door), SLIDER, PISTON, GENERIC, GENERIC_SPRING or MOTOR - two (object1, object2) or many (objects, pattern, pivot, collection), or edit joints (update, names). GENERIC limits make a ragdoll. bl_add_rigid_body could make things fall and bl_set_physics_world could tune the solver, and nothing could join two bodies. NOT the same as bl_add_constraint: those are OBJECT constraints (Track To, Copy Location) evaluated in the depsgraph, while these are solved by the rigid body world. type: FIXED | POINT | HINGE | SLIDER | PISTON | GENERIC | GENERIC_SPRING | MOTOR, refused by name if the running Blender lacks it. BOTH objects must ALREADY be rigid bodies - call bl_add_rigid_body on each first, or this refuses naming the one that is not. THE JOINT IS AN EMPTY AND ITS LOCATION IS THE PIVOT: a HINGE swings about the constraint object's origin, so put `location` on the hinge line or the door swings about the wrong axis - nothing here can guess where that is. THE JOINT'S ROTATION IS THE AXIS: a HINGE turns about the joint's local Z and SLIDER, PISTON and MOTOR work along its local X (measured on all five Blender builds), so an unrotated hinge is VERTICAL and a plank on it swings like a door and never falls. `rotation` is the joint's Euler XYZ in RADIANS ([1.5708, 0, 0] lays a hinge axis along world -Y), and the response reports axisLocal and axisWorld read off the joint. `breaking_threshold` sets use_breaking AND the number together, because Blender ignores the threshold unless the flag is on and the number alone reads back as asked while never breaking. A constraint is COLLECTION MEMBERSHIP rather than a modifier: the object is linked into scene.rigidbody_world.constraints and Blender creates the settings, which is why this needs no operator. Verified on Blender 3.6.23, 4.4.0 and 5.0.1. Rigid bodies are stepped forward - bake_physics before jumping to a late frame."
    return _blender("add_rigid_body_constraint", name=name, type=type, object1=object1,
                    object2=object2, location=location, rotation=rotation, disableCollisions=disable_collisions,
                    breakingThreshold=breaking_threshold, enabled=enabled, objects=objects, active=active,
                    pattern=pattern, pivot=pivot, collection=collection, solverIterations=solver_iterations,
                    useBreaking=use_breaking, update=update, names=names)


@mcp.tool()
def bl_add_collision(object: str, damping: float = None, friction: float = None,
                     thickness: float = None, remove: bool = None) -> dict:
    "Give a Blender object a Collision modifier so cloth, softbody and particles collide with it. RIGID BODIES DO NOT USE THIS - they collide through the rigid body world, so a floor for a falling crate needs bl_add_rigid_body type=PASSIVE instead. Giving the floor a Collision modifier and expecting a bounce is a common silent mistake."
    return _blender("add_collision", object=object, damping=damping, friction=friction,
                    thickness=thickness, remove=remove)


@mcp.tool()
def bl_set_physics_world(gravity: list = None, use_gravity: bool = None,
                         substeps: int = None, solver_iterations: int = None,
                         time_scale: float = None, split_impulse: bool = None,
                         enabled: bool = None, cache_start: int = None,
                         cache_end: int = None, cache_step: int = None) -> dict:
    "The SCENE-level Blender simulation settings. bl_physics_info has always reported substeps and solverIterations and nothing could write either, so the addon could add rigid bodies, bake them, and report exactly how the solver was configured while being unable to change any of it. Those two are what you raise when bodies jitter or tunnel through a floor, and scene gravity drives every rigid body in the file. A GRAVITY VECTOR IS INERT WHILE use_gravity IS OFF - Blender stores it either way, so setting it alone changes three numbers and not the simulation, and passing gravity turns the toggle on unless you say otherwise. CHANGING A SOLVER SETTING DOES NOT RE-SIMULATE: an existing bake stays on disk and stays marked valid, so every frame you look at was computed with the OLD settings - cacheIsBaked and cacheIsOutdated are reported, and a bake made stale by the change is named. The rigid-body-world keys are REFUSED when the scene has no world rather than conjuring one, because a scene-wide simulation is not a side effect a settings call should have; add a rigid body first."
    return _blender("set_physics_world", gravity=gravity, useGravity=use_gravity,
                    substeps=substeps, solverIterations=solver_iterations,
                    timeScale=time_scale, splitImpulse=split_impulse, enabled=enabled,
                    cacheStart=cache_start, cacheEnd=cache_end, cacheStep=cache_step)


@mcp.tool()
def bl_physics_info(object: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "What the Blender physics setup IS - the read half of a family that could only write. bl_add_rigid_body, bl_add_cloth, bl_add_collision and bl_bake_physics all set and nothing reported what they had set; bl_scene_info carries no physics at all, and a rigid body is NOT a modifier (it lives on obj.rigid_body) so bl_list_modifiers cannot see it either. Catches the inert state that settings cannot reveal: an object can carry a fully configured rigid body - mass, friction, shape, all reading back perfectly - and never simulate, because the sim only acts on objects in the RigidBodyWorld's COLLECTION, so it hangs in the air. Reported as inSimulation. Also flags unbaked caches (a late frame shows the REST state and a render of it is simply wrong) and BAKED-BUT-SHORT ones, where a bake made before the frame range was extended stays valid while the frames past its end silently fall back."
    return _blender("physics_info", object=object, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_bake_physics(start: int = None, end: int = None, clear: bool = None,
                    rebake: bool = None, object: str = None, modifier: str = None, disk_cache: bool = None,
                    to_frame: int = None, free: bool = None, stages: list = None, resume: bool = None,
                    free_stages: list = None, bake_node: Any = None, bake_mode: str = None) -> dict:
    "Bake Blender's physics point caches so a given frame shows the simulated state rather than the rest state - or one fluid domain stage by stage (stages GUIDES, DATA, NOISE, MESH, PARTICLES; resume; free_stages). BLOCKS for the length of the bake. Reports which caches actually hold frames, because bake_all returns success even when nothing in the scene has a cache to bake. An EXISTING bake is kept (keptExistingBake, staleBakeNote) - pass rebake=true to free and bake again. Before any bake that starts from nothing it rewrites a force-field setting (fieldsRefreshed) and a rigid body world setting (rigidBodyWorldRefreshed): measured, a body added after a bake stayed frozen through a rebake that only freed, and moved once a world setting was written. allBaked says whether every cache ended up baked; `baked` only says bake_all ran. Each Mantaflow domain gets a caches row of kind fluid - cacheType, cacheDirectory, framesOnDisk per subfolder, framesStored, isBaked - read off its cache folder: bake_all simulates a REPLAY cache by stepping frames and bakes nothing for MODULAR or ALL, and fluidNote names a domain with no frames. A liquid's FLIP display system is not listed as a point cache. start/end reach a Mantaflow domain too (its cache_frame_start/end, written only when they differ), and a domain is listed in rangeAppliedTo or rangeNotHeld by the frames its cache folder holds. With clear=true it also empties caches that hold frames WITHOUT a bake - a cloth stepped frame by frame into a disk cache, which the next run replayed - and leaves the scene at their start frame, because going back re-fills them (framesBefore, framesAfter, unbakedFramesFreed, frame, frameBefore)."
    return _blender("bake_physics", start=start, end=end, clear=clear, rebake=rebake, object=object,
                    modifier=modifier, diskCache=disk_cache, toFrame=to_frame, free=free, stages=stages,
                    resume=resume, freeStages=free_stages, bakeNode=bake_node, bakeMode=bake_mode, _timeout=600.0)


@mcp.tool()
def bl_step_simulation(object: str, end: int, start: int = None, max_speed: float = None,
                       max_memory_gb: float = None, max_seconds: float = None, fastest: int = None) -> dict:
    "Step a Blender simulation frame by frame and stop BEFORE the next frame once it runs away: each frame reports the object's max and p99 vertex speed (m/s), its box, the process memory and the time; it stops at end, past max_speed, past max_memory_gb (default 12) or after max_seconds (default 60), says which in stoppedBecause, and names the fastest vertices."
    return _blender("step_simulation", object=object, start=start, end=end, maxSpeed=max_speed,
                    maxMemoryGB=max_memory_gb, maxSeconds=max_seconds, fastest=fastest, _timeout=900.0)


@mcp.tool()
def bl_add_particles(object: str, type: str = None, count: int = None, seed: int = None,
                     frame_start: float = None, frame_end: float = None, lifetime: float = None,
                     lifetime_random: float = None, emit_from: str = None,
                     distribution: str = None, physics_type: str = None,
                     normal_factor: float = None, random_factor: float = None,
                     gravity_factor: float = None, damping_factor: float = None,
                     size: float = None, size_random: float = None, render_type: str = None,
                     instance_object: str = None, instance_collection: str = None,
                     hair_length: float = None, child_count: int = None,
                     show_emitter: bool = None, system_name: str = None,
                     use_modifier_stack: bool = None, rotation_mode: str = None,
                     use_rotations: bool = None, children: dict = None, material: str | int = None,
                     collection: dict = None, rotation: dict = None, groups: dict = None,
                     vertex_group_density: str = None, object_align_factor: list = None,
                     object_factor: float = None, hair_dynamics: dict = None, strand_shape: dict = None) -> dict:
    "Add a Blender particle system: EMITTER throws particles over time (steam, sparks) and HAIR instances geometry across a surface without moving (rubble, grass). Type-specific settings are refused on the wrong type. render_type OBJECT without instance_object renders NOTHING and Blender reports no error, so the two are validated together. AN EMITTER DEFAULTS TO render_type HALO, WHICH ALSO RENDERS NOTHING in EEVEE, Cycles and Workbench - pass instance_object or instance_collection to see the particles; the response's renderNote says whenever a system will draw nothing. HAIR takes NONE, PATH, OBJECT or COLLECTION and defaults to PATH. ON A HAIR SYSTEM normal_factor IS THE STRAND LENGTH TOO: hair_length = 4 x normal_factor, one stored value, so normal_factor 0 means no hair - send one of the two; a pair that disagrees is refused."
    return _blender("add_particles", object=object, type=type, count=count, seed=seed,
                    frameStart=frame_start, frameEnd=frame_end, lifetime=lifetime,
                    lifetimeRandom=lifetime_random, emitFrom=emit_from, distribution=distribution,
                    physicsType=physics_type, normalFactor=normal_factor,
                    randomFactor=random_factor, gravityFactor=gravity_factor,
                    dampingFactor=damping_factor, size=size, sizeRandom=size_random,
                    renderType=render_type, instanceObject=instance_object,
                    instanceCollection=instance_collection, hairLength=hair_length,
                    childCount=child_count, showEmitter=show_emitter, systemName=system_name,
                    useModifierStack=use_modifier_stack, rotationMode=rotation_mode,
                    useRotations=use_rotations, children=children, material=material,
                    collection=collection, rotation=rotation, groups=groups, vertexGroupDensity=vertex_group_density,
                    objectAlignFactor=object_align_factor, objectFactor=object_factor, hairDynamics=hair_dynamics,
                    strandShape=strand_shape)


@mcp.tool()
def bl_set_particles(object: str, system: str = None, type: str = None, count: int = None, seed: int = None,
                     frame_start: float = None, frame_end: float = None, lifetime: float = None,
                     lifetime_random: float = None, emit_from: str = None,
                     distribution: str = None, physics_type: str = None,
                     normal_factor: float = None, random_factor: float = None,
                     gravity_factor: float = None, damping_factor: float = None,
                     size: float = None, size_random: float = None, render_type: str = None,
                     instance_object: str = None, instance_collection: str = None,
                     hair_length: float = None, child_count: int = None,
                     show_emitter: bool = None,
                     use_modifier_stack: bool = None, rotation_mode: str = None,
                     use_rotations: bool = None, children: dict = None, material: str | int = None,
                     collection: dict = None, rotation: dict = None, groups: dict = None,
                     edit: dict = None, vertex_group_density: str = None, object_align_factor: list = None,
                     object_factor: float = None, hair_dynamics: dict = None, strand_shape: dict = None) -> dict:
    "Change a Blender particle system that ALREADY EXISTS. bl_add_particles STACKS - three calls give three systems on one object, all emitting - which is right for particles, where dust and sparks on one mesh is ordinary, and it meant the only way to change a count was to add a fourth system beside the one you meant to edit. THE SYSTEM IS NAMED, OR THERE MUST BE EXACTLY ONE: an object with two systems and no `system` argument is refused rather than guessed at, because editing the wrong emitter looks exactly like the edit doing nothing. Retyping between EMITTER and HAIR is allowed, and the per-type key checks run against the type the system will BE after the call: hair_length or child_count on an emitter, and frame_start, frame_end or lifetime on a hair system, are refused with nothing changed, as bl_add_particles refuses them. Every setting goes through the same writer bl_add_particles uses, so the two cannot drift. The response carries before/changedFields alongside the current state."
    return _blender("set_particles", object=object, system=system, type=type, count=count, seed=seed,
                    frameStart=frame_start, frameEnd=frame_end, lifetime=lifetime,
                    lifetimeRandom=lifetime_random, emitFrom=emit_from, distribution=distribution,
                    physicsType=physics_type, normalFactor=normal_factor,
                    randomFactor=random_factor, gravityFactor=gravity_factor,
                    dampingFactor=damping_factor, size=size, sizeRandom=size_random,
                    renderType=render_type, instanceObject=instance_object,
                    instanceCollection=instance_collection, hairLength=hair_length,
                    childCount=child_count, showEmitter=show_emitter, useModifierStack=use_modifier_stack, rotationMode=rotation_mode,
                    useRotations=use_rotations, children=children, material=material,
                    collection=collection, rotation=rotation, groups=groups, edit=edit,
                    vertexGroupDensity=vertex_group_density, objectAlignFactor=object_align_factor,
                    objectFactor=object_factor, hairDynamics=hair_dynamics, strandShape=strand_shape)


@mcp.tool()
def bl_list_particles(object: str, limit: int = None) -> dict:
    "Read every particle system on a Blender object off its datablocks - the verification half of bl_add_particles. rendersNothing and renderNote flag a system that draws nothing: render_type HALO (an emitter's default) or NONE, or OBJECT / COLLECTION with nothing to instance. Each row carries the whole particle record, childType and renderedChildCount beside childCount - about 800 bytes, the fattest of any listing here - and bl_add_particles stacks by design, so `limit` bounds the rows while `count` stays the true total."
    # `limit` added 2026-09-20: the fattest row of any listing here at ~800 bytes a system, and
    # bl_add_particles stacks by design, so the count is bounded by nothing.
    return _blender("list_particles", object=object, limit=limit)


@mcp.tool()
def bl_list_view_layers(with_passes: bool = None, limit: int = None) -> dict:
    "Every Blender view layer, what it outputs, and whether it renders at all. `renders` is view_layer.use, and it is the one that decides whether any of the rest happens: with it off, every pass and collection assignment on the layer reads back perfectly and no pixel of it is ever produced, with nothing to warn you. This is the biggest response in the addon: with_passes adds a ~60-entry map to EVERY row, and 61 layers measured 52,880 bytes, so `limit` bounds the rows. notRendering is computed from every layer whatever the limit, because that warning is about the scene rather than the page."
    # `limit` added 2026-09-20 - this was the biggest unbounded response in the addon. Measured on
    # 5.0.1: 61 view layers answered 8,960 bytes plain and 52,880 with passes. notRendering is
    # still computed from every layer, because that warning is about the scene, not the page.
    return _blender("list_view_layers", withPasses=with_passes, limit=limit)


@mcp.tool()
def bl_set_view_layer(name: str = None, use: bool = None, enable_passes: list = None,
                      disable_passes: list = None, passes: dict = None,
                      samples: int = None, cycles: dict = None, eevee: dict = None,
                      cryptomatte_depth: int = None, lightgroups: list = None, aovs: list = None,
                      material_override: str = None) -> dict:
    "Turn Blender render passes on or off - what decides WHAT THE COMPOSITOR CAN SEE, since a Render Layers node only offers sockets for passes the layer actually outputs. Ask for a Z-depth composite with the Z pass off and there is nothing to connect. Pass names drop the use_pass_ prefix (\"z\", \"normal\", \"mist\", \"cryptomatte_object\") and are validated against THIS layer rather than a list in the addon, because the set depends on the Blender version and render engine - a hard-coded one would refuse passes that exist. Every requested pass is read back individually: several use_pass_* properties are read-only under a given engine, so the write is accepted and the value does not move. cycles / eevee write view_layer.cycles / .eevee by identifier (denoising_store_passes gives a Denoise node its Albedo and Normal), cryptomatte_depth sets the Cryptomatte levels, lightgroups and aovs ([{name, type}]) are made when missing, and material_override renders the layer with one material (null clears it)."
    return _blender("set_view_layer", name=name, use=use, enablePasses=enable_passes,
                    disablePasses=disable_passes, passes=passes, samples=samples, cycles=cycles, eevee=eevee,
                    cryptomatteDepth=cryptomatte_depth, lightgroups=lightgroups, aovs=aovs,
                    materialOverride=material_override)


@mcp.tool()
def bl_create_view_layer(name: str, copy_from: str = None, use: bool = None) -> dict:
    "Add a Blender view layer - a second pass over the same scene with its own collection visibility and its own outputs. copy_from carries the enabled passes across, because Blender's new layers start from defaults, which is rarely what somebody splitting a shot into layers wants."
    return _blender("create_view_layer", name=name, copyFrom=copy_from, use=use)


@mcp.tool()
def bl_delete_view_layer(name: str) -> dict:
    "Remove a Blender view layer, refusing to remove the last one - a scene with no view layer cannot be rendered at all, and the API will happily let you get there."
    return _blender("delete_view_layer", name=name)


@mcp.tool()
def bl_knife_project(object: str, cutter: str, direction: list = None, cut_through: bool = None,
                     clean_weld: bool = None, weld_distance: float = None) -> dict:
    "Cut a cutter object's outline (a mesh, curve or text) into a Blender mesh along a direction, through the whole mesh or only the faces seen first, without a 3D View - the crossed faces divided, nothing removed, the seam welded clean. See mif_help."
    return _blender("knife_project", object=object, cutter=cutter, direction=direction, cutThrough=cut_through,
                    cleanWeld=clean_weld, weldDistance=weld_distance)


@mcp.tool()
def bl_groom_stroke(object: str, mode: str, center: list = None, radius: float = None, falloff: str = None,
                    strength: float = None, vector: list = None, factor: float = None, iterations: int = None,
                    length: float = None, count: int = None, seed: int = None) -> dict:
    "Comb, lengthen or shorten, smooth, cut, puff, pinch or slide a Blender hair Curves object's strands, or add and remove strands - natively on its points, in a region or everywhere; lengths, moves and counts read back. See mif_help."
    return _blender("groom_stroke", object=object, mode=mode, center=center, radius=radius, falloff=falloff,
                    strength=strength, vector=vector, factor=factor, iterations=iterations, length=length,
                    count=count, seed=seed)


@mcp.tool()
def bl_create_mask(name: str, layers: list, compositor: bool = None, strip: str = None, scene: str = None) -> dict:
    "Make a Blender mask - layers of splines (point lists, or CIRCLE / SQUARE primitives) with feather, keyed at frames and parented to a motion-tracking track - and wire it into a compositor Mask node and / or a sequencer strip's MASK modifier. See mif_help."
    return _blender("create_mask", name=name, layers=layers, compositor=compositor, strip=strip, scene=scene)


@mcp.tool()
def bl_set_compositing(enabled: bool = None, use_compositing: bool = None,
                       use_sequencer: bool = None, with_default_nodes: bool = None) -> dict:
    "Turn the Blender scene compositor on, and the SECOND switch that also has to be on. TWO INDEPENDENT FLAGS decide whether compositing happens: scene.use_nodes (a tree exists and is edited) and scene.render.use_compositing (the render PIPELINE runs it). With the first on and the second off, the whole tree reads perfectly, the compositor backdrop updates, and the rendered file is completely unprocessed - nothing reports it. Both are set and both read back. Wires a default Render Layers -> Composite pair when the tree is empty, because an empty compositor writes nothing at all. Afterwards, address the tree from bl_add_group_node / bl_link_group_nodes / bl_list_group_nodes by passing tree:'scene:compositor'. The same resolver reaches 'scene:world', 'material:<name>' and 'world:<name>'. enabled:false turns the pipeline switch off on every build. From Blender 5.0, scene.use_nodes stays on whatever is written (measured on 5.0.1 and 5.2.1), so there the reply carries useNodesFixed:true and a useNodesNote instead of a refusal - render.use_compositing is the switch the render reads."
    return _blender("set_compositing", enabled=enabled, useCompositing=use_compositing,
                    useSequencer=use_sequencer, withDefaultNodes=with_default_nodes)


@mcp.tool()
def bl_compositor_info(view_layer: str = None, limit: int = None, name_excludes: list = None) -> dict:
    "What the Blender compositor IS, and every way it can be on and doing nothing. The whole subsystem was unreachable before: bl_create_node_group can make a CompositorNodeTree, but that is a node GROUP in bpy.data.node_groups, while the scene's compositor is scene.node_tree - a different tree nothing could address - so glare, grading, denoise, cryptomatte and file output were outside the typed path. Reports four distinct blockers because the fix differs for each: use_nodes off; use_compositing off (the classic - backdrop updates, file untouched); no Composite node linked (a Viewer is NOT a substitute, it feeds the backdrop only, which is why it looks right in the compositor and the saved file is wrong); and no Render Layers feeding it. Also muted nodes, VSE strips that replace the compositor's output wholesale, and which view-layer passes are actually enabled."
    return _blender("compositor_info", viewLayer=view_layer, limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_create_node_group(name: str = None, type: str = None, with_group_io: bool = None, tool: Any = None) -> dict:
    "Create a Blender node group (geometry by default) whose Group Input and Group Output are NOT linked: until something reaches Group Output, an object it drives evaluates to nothing. Both sit on a Geometry socket pair; a geometry group with no geometry sockets cannot drive a modifier at all. Group sockets moved to tree.interface in Blender 4.0 and the old tree.inputs/outputs are GONE rather than deprecated; both are handled. tool: true (OBJECT and EDIT, MESH) or {modes: [OBJECT|EDIT|SCULPT], types: [MESH|CURVES|POINTCLOUD]} marks a geometry group as a node tool (Blender 4.0+), run by bl_run_node_tool."
    return _blender("create_node_group", name=name, type=type, withGroupIO=with_group_io, tool=tool)


@mcp.tool()
def bl_add_group_node(group: str, type: str = None, name: str = None, location: list = None,
                      inputs: dict = None, label: str = None, operation: str = None,
                      data_type: str = None, domain: str = None, mode: str = None,
                      node_group: str = None, properties: dict = None, parent: str = None,
                      zone: dict = None) -> dict:
    "Add a node to a Blender node group. `inputs` is {socketName: value} and a name that does not match a real socket is refused with the sockets the node actually has - a value written to a socket that is not there vanishes without a word. A vector or colour value needs ALL its components ([r,g,b] into a colour keeps its alpha) and only finite numbers: until 2026-09-20 a NaN was stored, a long list died as a raw IndexError and a short one was written in part and echoed as applied. bl_set_node_property writes the same inputs on a node that already exists. The `group` argument also addresses trees that are OWNED by something rather than living in bpy.data.node_groups: 'scene:compositor', 'scene:world', 'material:<name>' and 'world:<name>'. That is how a MATERIAL's shader graph is authored - bl_describe_material could read one in full while bl_set_material_properties could write only the Principled BSDF's own sockets, so the addon could describe a graph in detail and not add a node to it. NESTING: pass node_group to make a Group node CONTAIN another tree - without it a Group node points at nothing, has no sockets at all and does nothing, while reporting success. Set it in the same call as `inputs`, because the node has no sockets to write until it holds a tree. A RECURSIVE nest is refused: Blender rejects one SILENTLY, leaving node_tree empty and raising nothing, so the assignment is read back and the node removed rather than left inert. PROPERTIES: operation/data_type/domain/mode are friendly spellings of four common knobs; `properties` writes ANY node property by its real RNA name, dispatched on the property's own declared type - an enum is validated against its own items and reported as the identifier Blender stored, not the string you sent. That is what reaches a Math node's use_clamp, a Noise texture's noise_dimensions or a Map Range's interpolation_type, none of which a curated list of four was ever going to cover."
    return _blender("add_group_node", group=group, type=type, name=name, location=location,
                    inputs=inputs, label=label, operation=operation, dataType=data_type,
                    domain=domain, mode=mode, nodeGroup=node_group, properties=properties, parent=parent,
                    zone=zone)


@mcp.tool()
def bl_link_group_nodes(group: str, to_node: str, from_node: str = None, from_socket: str = None,
                        to_socket: str = None, remove: bool = None) -> dict:
    "Wire one node's output into another's input inside a Blender node group - or, with remove=true, take links OUT. links.new returns a link object even when Blender immediately drops it as invalid (mismatched socket types), so the link is read back and `linked` is a measurement rather than the call's return. A LINK INTO AN INPUT THAT ALREADY HAS ONE REPLACES IT and Blender says nothing: the displaced link comes back in replacedLinks (with replacedNote), in full, so it can be restored; a multi-input socket such as Join Geometry keeps both. remove=true needs to_node plus to_socket and/or from_node (from_socket narrows further), removes every link matching that scope and lists each one in removedLinks - a scope matching nothing is refused rather than reported as a success. Before this a link could only be undone by deleting a node. The `group` argument also addresses trees that are OWNED by something rather than living in bpy.data.node_groups: 'scene:compositor', 'scene:world', 'material:<name>' and 'world:<name>'. That is how a MATERIAL's shader graph is authored."
    return _blender("link_group_nodes", group=group, fromNode=from_node, toNode=to_node,
                    fromSocket=from_socket, toSocket=to_socket, remove=remove)


@mcp.tool()
def bl_add_group_interface(group: str, name: str = None, socket_type: str = None, in_out: str = None,
                           default: float = None, min: float = None, max: float = None, rename: dict = None,
                           move: dict = None, remove: list = None, confirm: bool = None,
                           default_input: str = None) -> dict:
    "Expose a value as a group input or output - what turns a Blender node tree into a modifier with sliders on it. socket_type is a NodeSocket id such as NodeSocketFloat."
    return _blender("add_group_interface", group=group, name=name, socketType=socket_type,
                    inOut=in_out, default=default, min=min, max=max, rename=rename, move=move, remove=remove,
                    confirm=confirm, defaultInput=default_input)


@mcp.tool()
def bl_build_pass_composite(view_layer: str = None, passes: list = None, group: bool = None, denoise: str = None,
                            extras: dict = None, output: str = None, file_path: str = None,
                            file_format: str = None) -> dict:
    "Rebuild a Blender render from its light passes in the scene compositor: (Direct + Indirect) x Color per pass, summed, then Emission and Environment, with optional denoise, AO, glare and mist."
    return _blender("build_pass_composite", viewLayer=view_layer, passes=passes, group=group, denoise=denoise,
                    extras=extras, output=output, filePath=file_path, fileFormat=file_format)


@mcp.tool()
def bl_list_group_nodes(group: str, values: bool = False, limit: int = None, name_excludes: list = None, max_links: int = None) -> dict:
    "Every node and link in a Blender node group, plus whether the Group Output is actually reachable. That last line is the point: an unlinked Group Output is NOT an error, and it does NOT pass the geometry through - the modifier evaluates to EMPTY and the object disappears from the viewport and every export (measured on 3.6, 4.2, 4.4 and 5.0). The TERMINAL depends on the tree type - a Composite node for the compositor, a Material/World Output for a shader tree, a Group Output for a geometry group - so the answer is right for each rather than reporting a correctly wired compositor as inert. The `group` argument also addresses trees that are OWNED by something rather than living in bpy.data.node_groups: 'scene:compositor', 'scene:world', 'material:<name>' and 'world:<name>'. That is how a MATERIAL's shader graph is authored - bl_describe_material could read one in full while bl_set_material_properties could write only the Principled BSDF's own sockets, so the addon could describe a graph in detail and not add a node to it."
    # values=values, never `values or None`: that dropped an explicit False, which is safe only while every reader of a
    # `values` key defaults false - describe_conversation's does not (BL107).
    return _blender("list_group_nodes", group=group, values=values,
                    limit=limit, nameExcludes=name_excludes, maxLinks=max_links)


@mcp.tool()
def bl_inspect_node_output(object: str, node: str, modifier: str = None, socket: Any = None,
                           attributes: list = None) -> dict:
    "Evaluate a Blender object's Geometry Nodes tree AT ONE NODE - its geometry output routed to the Group Output, read (counts per component, attributes, world box), then every link put back and the restore checked. Finds which node empties the geometry. See mif_help."
    return _blender("inspect_node_output", object=object, node=node, modifier=modifier, socket=socket,
                    attributes=attributes)


@mcp.tool()
def bl_set_node_property(group: str, node: str, properties: dict = None, inputs: dict = None,
                         label: str = None, location: list = None, items: dict = None,
                         base_path: str = None, hide_unused_sockets: bool = None, curve: dict | list = None) -> dict:
    "Change a Blender node that ALREADY EXISTS - its PROPERTIES (enums, toggles, vector fields: a Math node's operation, a Noise Texture's dimensions) and its INPUT SOCKETS (a Noise Texture's Scale, a Bump's Strength and Distance, a Mix Shader's Factor). Both were write-once until they came here: properties until 2026-09-07, input sockets until 2026-09-20, when a plaster variant could not re-tune its PeelBump for want of this call. ADDRESSED LIKE EVERY OTHER NODE OP: `group` takes a node group by name, or 'scene:compositor', 'scene:world', 'material:<name>', 'world:<name>'. `inputs` is {socket: value} keyed by the socket's name, its identifier (Value_001 - a Math node's three inputs are all called Value) or its index, converted exactly as bl_add_group_node's `inputs` are: finite numbers only, a vector or colour needs ALL its components ([r,g,b] into a colour keeps its alpha), a datablock socket takes a NAME. It is applied AFTER `properties`, because a data_type or operation decides which of several same-named sockets is live, and a LINKED input is refused - Blender reads the link and never the value (remove the link with bl_link_group_nodes remove=true first). inputsApplied is read BACK off each socket. ALL OR NOTHING: every property and input is snapshotted first and put back if any key refuses, so a refusal's 'The node was NOT changed.' is true - it was not until 2026-09-20. Unknown or read-only property names are refused BY NAME with that node type's settable list attached, enums validate against their own items, and values are read back rather than echoed. A property that holds a DATABLOCK - a Texture Coordinate's object, an Image node's image - takes the datablock's NAME; null clears it. A group node's node_tree stays with bl_add_group_node's node_group, which checks the kind of tree. ITEMS: items={collection, add [{name, type, format}], rename {old: new}, remove [names]} edits a node's item collection - a simulation or repeat zone's state items, capture, bake, menu and index-switch items, a File Output node's slots (with a per-slot image format) - each type from the build's own list, and base_path sets a File Output's folder on any build."
    return _blender("set_node_property", group=group, node=node, properties=properties,
                    inputs=inputs, label=label, location=location, items=items, basePath=base_path,
                    hideUnusedSockets=hide_unused_sockets, curve=curve)


@mcp.tool()
def bl_group_nodes(group: str, nodes: list, mode: str, label: str = None, name: str = None) -> dict:
    "Wrap existing Blender nodes in a labelled frame (mode FRAME), or move them into a new node group (mode GROUP) whose sockets are the links that crossed the boundary (each copy compared with its node before the originals go; refused and undone if one differs)."
    return _blender("group_nodes", group=group, nodes=nodes, mode=mode, label=label, name=name)


@mcp.tool()
def bl_remove_node(group: str, node: str, allow_structural: bool = False, dissolve: bool = None) -> dict:
    "Delete a Blender node and report exactly which links died with it. bl_add_group_node and bl_link_group_nodes had no counterpart, which is what made the bunker showcase's material stage one-way - its own docstring says 'no typed op REMOVES a node ... it cannot take a palette off, and it cannot re-tune one in place'. bl_set_node_property answered the re-tuning half; this is the other. REMOVAL SEVERS LINKS IN SILENCE: measured on 4.4, deleting one middle node took a three-link tree to one and Blender said nothing, so a caller who removed a colour ramp would be looking at an unlinked Base Color with no explanation. Every link touching the node is captured BEFORE the delete and returned as fromNode/fromSocket/toNode/toSocket - enough to rebuild with bl_link_group_nodes. Structural nodes (Material Output, Group Output, Composite and friends) are REFUSED by default: a material with no output renders black and a group with no Group Output evaluates to EMPTY, which makes the object vanish - pass allow_structural=true if that is genuinely what you want. `group` addresses the tree as everywhere else: a node group by name, or 'scene:compositor', 'scene:world', 'material:<name>', 'world:<name>'. dissolve=true keeps the link through it (2026-10-06): the node's first linked output is matched to its first linked input of the same socket type, and that input's source is linked to everything the output fed - keptLinks, read back off the tree; refused with nothing removed when there is no linked input, no linked output, or no input of the output's type."
    return _blender("remove_node", group=group, node=node, allowStructural=allow_structural, dissolve=dissolve)


@mcp.tool()
def bl_assign_node_group(object: str, group: str, modifier_name: str = None,
                         inputs: dict = None, reuse: bool = True, output_attributes: dict = None,
                         library: str = None) -> dict:
    "Attach a Blender geometry node group to an object as a Nodes modifier and set its exposed inputs. The modifier addresses inputs by IDENTIFIER (Socket_2), not by name, which is the most confusing thing about driving geometry nodes from script - this resolves names to identifiers for you and reports anything it could not place. A DATABLOCK INPUT TAKES A NAME: an Object, Collection, Material or Image socket is resolved from the name you pass, because assigning the raw string is ACCEPTED by Blender, stored, and then ignored - the modifier reads an empty socket and the effect silently does nothing. REUSE IS THE DEFAULT: calling this twice for the same object and group used to stack a second modifier, so an assign-look-adjust loop tripled the stack invisibly. It now updates the modifier already holding that group and reports reused:true; pass reuse:false to force a second instance deliberately. A DIFFERENT group always gets its own modifier."
    return _blender("assign_node_group", object=object, group=group, modifierName=modifier_name,
                    inputs=inputs, reuse=reuse, outputAttributes=output_attributes, library=library)


@mcp.tool()
def bl_run_node_tool(object: str, tool: str, inputs: dict = None, mode: str = None, library: str = None) -> dict:
    "Run a geometry-nodes node TOOL (a group marked with bl_create_node_group tool) once on a mesh, curves or points object, with its inputs by name or identifier; mode EDIT runs on the edit-mode selection. Replies counts before and after."
    return _blender("run_node_tool", object=object, tool=tool, inputs=inputs, mode=mode, library=library)


@mcp.tool()
def bl_set_viewport_shading(shading: str = None, use_scene_lights: bool = None,
                            use_scene_world: bool = None, studio_light: str = None,
                            show_overlays: bool = None, show_gizmos: bool = None,
                            color_type: str = None, show_xray: bool = None, show_cavity: bool = None,
                            show_grid: bool = None) -> dict:
    "Set the Blender 3D viewport shading: WIREFRAME, SOLID, MATERIAL or RENDERED. This is what makes lighting work VISIBLE - SOLID ignores materials and lamps entirely, so a correctly lit scene looks grey. MATERIAL preview uses a studio light and does NOT show your scene's lamps; only RENDERED runs the render engine, so it is the only mode in which a flickering light actually flickers. The reply reads the shading back, colorType and studioLight included, and a call with no arguments changes nothing and reports the current shading."
    return _blender("set_viewport_shading", shading=shading, useSceneLights=use_scene_lights,
                    useSceneWorld=use_scene_world, studioLight=studio_light,
                    showOverlays=show_overlays, showGizmos=show_gizmos, colorType=color_type, showXray=show_xray,
                    showCavity=show_cavity, showGrid=show_grid)


@mcp.tool()
def bl_frame_viewport(object: str = None, camera: bool = None) -> dict:
    "Point the Blender viewport at one object, at everything, or through the scene camera. Framing matters when a person is watching a build happen - work that happens off-screen is work nobody can see."
    return _blender("frame_viewport", object=object, camera=camera)


@mcp.tool()
def bl_set_viewport_view(focus: list = None, distance: float = None, azimuth: float = None,
                         elevation: float = None, look_from: list = None,
                         perspective: str = None, lens: float = None) -> dict:
    "Place the Blender viewport's own view - the orbit pivot (focus), how far back, and the angle. The viewport is an ORBIT, not a camera: there is no eye position to set directly, so pass look_from and it derives the pivot, distance and rotation for you. azimuth/elevation are RADIANS; azimuth 0 looks along +Y and positive elevation looks DOWN at the focus. Use this to drive a walkthrough without the human touching the viewport."
    return _blender("set_viewport_view", focus=focus, distance=distance, azimuth=azimuth,
                    elevation=elevation, lookFrom=look_from, perspective=perspective, lens=lens)







@mcp.tool()
def bl_transform_object(object: str, location: list = None, rotation: list = None,
                        scale: list = None, relative: bool = False) -> dict:
    "Move, rotate or scale a Blender object WITHOUT baking the transform into its mesh data: location [x,y,z], rotation [x,y,z] in RADIANS, scale [x,y,z]; relative=True adds to the current values."
    return _blender("transform_object", object=object, location=location, rotation=rotation,
                    scale=scale, relative=relative)


@mcp.tool()
def bl_duplicate_object(objects: list | str, names: list = None, suffix: str = None, linked: bool = None,
                        collection: str = None, offset: list = None, rotation: list = None, scale: list = None,
                        include_children: bool = None, hide: str = None) -> dict:
    "Duplicate Blender objects under new names (names, or suffix): a full copy with its own mesh, or linked (the same mesh, as Alt+D); offset, rotation (radians) and scale; include_children; hide original or copy."
    return _blender("duplicate_object", objects=objects, names=names, suffix=suffix, linked=linked,
                    collection=collection, offset=offset, rotation=rotation, scale=scale,
                    includeChildren=include_children, hide=hide)


@mcp.tool()
def bl_transform_objects(objects: list = None, pivot: list = None, rotation: list = None,
                         translation: list = None, dry_run: bool = False, preview_id: str = None,
                         preview_ttl_seconds: int = None) -> dict:
    """Rotate/translate independent Blender objects together around a WORLD-space pivot.

    Angles are XYZ radians. The group mapping is translation + pivot + rotation; for example,
    pivot=[2,0,0], rotation=[0,0,-1.5707963267948966], translation=[-2,0,0] turns a group
    around its old center and recenters it. Refuses parent hierarchies and validates the whole
    list before writing. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since.
    Mesh data is never baked.
    """
    # THE BULK PREVIEW (docs/15 section 5) and THE CHANGE CAP (section 6): dry_run previews, and over the file's cap the
    # add-on answers the same preview unasked; preview_id alone commits it.
    if preview_id:
        return (_preview_commit_refusal(preview_id, objects=objects, pivot=pivot, rotation=rotation,
                                        translation=translation, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("transform_objects", previewId=preview_id))
    return _blender("transform_objects", objects=objects, pivot=pivot, rotation=rotation,
                    translation=translation, dryRun=dry_run or None, previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_set_parent(object: str, parent: str = None, clear: bool = False, keep_transform: bool = True,
                  type: str = "OBJECT", bone: str = None, fallback: str = None, max_influences: int = None,
                  bones: list = None, include_parents: bool = None, head_tail: float = None,
                  vertices: list | dict = None) -> dict:
    "Parent a Blender object to another object, or to a bone of an armature (type BONE with bone), or clear its parent with clear:true - the hierarchy a prop on an empty, a camera on a rig or an FBX into Unreal needs. keep_transform (default true) keeps it where it is in the world, and the response MEASURES that: worldBefore / worldAfter and worldMoved. A cycle is refused before anything is written. type VERTEX or VERTEX_3 with vertices [i] / [i, j, k] or {near: [x,y,z]} rides one vertex or a triangle of a mesh parent. Call mif_help(\"bl_set_parent\") first."
    return _blender("set_parent", object=object, parent=parent, clear=clear, keepTransform=keep_transform,
                    type=type, bone=bone, fallback=fallback, maxInfluences=max_influences, bones=bones,
                    includeParents=include_parents, headTail=head_tail, vertices=vertices)


@mcp.tool()
def bl_join_objects(target: str, objects: list, rename_suffix: str = None, keep_sources: bool = None) -> dict:
    "Join Blender mesh objects into one. DESTRUCTIVE AND ASYMMETRIC: the sources are DELETED and everything lands in target. Armatures join too: clashing bones are renamed first (rename_suffix, default .001) and REPORTED, and the skinned meshes follow. With keep_sources=True it joins copies and leaves every source, which the change cap counts as the one object it writes; a consuming join (the default) also counts each source it removes, so 29 parts at a cap of 25 is refused, naming keep_sources."
    return _blender("join_objects", target=target, objects=objects, renameSuffix=rename_suffix,
                    keepSources=keep_sources)


@mcp.tool()
def bl_separate_mesh(object: str, mode: str = "loose", faces=None, vertex_group: str = None, min_weight: float = None,
                     loop=None, duplicate: bool = None, new_name: str = None) -> dict:
    "Split a Blender mesh into separate objects - mode 'loose' (each island), 'material' (one per material in use) or 'selection': faces (indices or a bl_select_faces reply), vertex_group or loop go to one new object; duplicate keeps the original whole."
    return _blender("separate_mesh", object=object, mode=mode, faces=faces, vertexGroup=vertex_group,
                    minWeight=min_weight, loop=loop, duplicate=duplicate, newName=new_name)


@mcp.tool()
def bl_convert_object(object: str, target: str = "MESH", keep_original: bool = False,
                      drop_instances: bool = False, pose: str = None, particle_system: str = None) -> dict:
    "Convert a Blender object to another type - a curve or text to a MESH, a mesh to a CURVE - with its WHOLE modifier stack baked in. Target HAIR_PARTICLES or HAIR_CURVES converts hair between Curves and a particle system. Call mif_help(\"bl_convert_object\") first."
    return _blender("convert_object", object=object, target=target, keepOriginal=keep_original,
                    dropInstances=drop_instances, pose=pose, particleSystem=particle_system)


@mcp.tool()
def bl_rename_bones(object: str = None, renames: dict = None, dry_run: bool = False, preview_id: str = None,
                    preview_ttl_seconds: int = None) -> dict:
    "Rename Blender armature bones through a {old: new} map. Blender already renames the matching vertex groups and updates constraint and driver references by itself - what this adds is refusing the NAME COLLISION where that sync silently fails: renaming a bone onto a name another bone holds gives you 'Hips.001' and leaves the vertex group under its old name, matching no bone, so that part of the mesh stops deforming with nothing to say so. Collisions are refused before anything is written, every rename is read back, swaps are supported, and orphaned vertex groups are reported. With dry_run=True it previews each change with its value before and returns a preview_id; preview_id alone commits it, refused if anything changed since. See mif_help."
    if preview_id:
        return (_preview_commit_refusal(preview_id, object=object, renames=renames, dry_run=dry_run,
                                        preview_ttl_seconds=preview_ttl_seconds)
                or _blender("rename_bones", previewId=preview_id))
    return _blender("rename_bones", object=object, renames=renames, dryRun=dry_run or None,
                    previewTtlSeconds=preview_ttl_seconds)


@mcp.tool()
def bl_bake_texture(object: str, type: str = "AO", width: int = 512, height: int = 512,
                    image_name: str = None, filepath: str = None, uv_layer: str = None,
                    margin: int = 4, samples: int = 16, keep_node: bool = False,
                    device: str = None, ao_distance: float = None, colorspace: str = None, float: bool = None,
                    format: str = None, source: str | list = None, extrusion: float = None,
                    max_ray_distance: float = None, cage: str = None, green_channel: str = None,
                    blob_fix: bool | dict = None, pieces: list = None, explode: float = None,
                    multires: dict = None, include: list = None, target: str = None, color_attribute: str = None,
                    id_by: str = None, height_range: float = None, thickness_distance: float = None) -> dict:
    "Bake AO / NORMAL / DIFFUSE / COMBINED / ROUGHNESS / EMIT / GLOSSY / SHADOW / CURVATURE / METALLIC / ALPHA / MATERIAL_ID / HEIGHT / POSITION / THICKNESS into an image (or target COLOR_ATTRIBUTE) on a Blender mesh - how a high-poly detail becomes a texture an Unreal material can use. Judged by the IMAGE, not the operator: bpy.ops.object.bake returns FINISHED and writes NOTHING when there is no active image-texture node, so the result is checked with is_dirty plus a before/after pixel signature and a blank bake is reported as the failure it is. Needs a UV layer. Render engine, device, samples and selection are all restored afterwards. Pass filepath to write it to disk - without one the image is in memory only. See mif_help."
    # device is CPU or GPU for the Cycles bake (scene.cycles.device, ops_material.py:554). The
    # docstring already promised "Render engine, device, samples and selection are all restored
    # afterwards" - so the tool DESCRIBED a device it gave no way to choose. Unset leaves the
    # addon's "CPU", which is the safe default on a machine with no usable GPU compute device.
    return _blender("bake_texture", object=object, type=type, width=width, height=height,
                    imageName=image_name, filepath=filepath, uvLayer=uv_layer,
                    margin=margin, samples=samples, keepNode=keep_node or None,
                    device=device, aoDistance=ao_distance, colorspace=colorspace, float=float, format=format,
                    source=source, extrusion=extrusion, maxRayDistance=max_ray_distance, cage=cage,
                    greenChannel=green_channel, blobFix=blob_fix, pieces=pieces, explode=explode, multires=multires,
                    include=include, target=target, colorAttribute=color_attribute, idBy=id_by,
                    heightRange=height_range, thicknessDistance=thickness_distance)


@mcp.tool()
def bl_bake_hidden_face_map(body: str, garments: str | list, uv_map: str = None, resolution: int = None,
                            max_distance: float = None, gray_band: int = None, margin: int = None, keep: list = None,
                            output_path: str = None, overwrite: bool = None, max_cull_value: float = None,
                            min_keep_value: float = None) -> dict:
    "Bake a body hidden-face map from garments in the Blender body's UV layout (black covered, gray edge band, white open) - a MetaHuman outfit's body mask - and count what Unreal's geometry removal would cut with it."
    return _blender("bake_hidden_face_map", body=body, garments=garments, uvMap=uv_map, resolution=resolution,
                    maxDistance=max_distance, grayBand=gray_band, margin=margin, keep=keep, outputPath=output_path,
                    overwrite=overwrite, maxCullValue=max_cull_value, minKeepValue=min_keep_value)


@mcp.tool()
def bl_fix_normal_map(image: str = None, filepath: str = None, output_path: str = None, overwrite: bool = None,
                      max_tilt_deg: float = None, min_blob_px: int = None) -> dict:
    "Repair blobs in a Blender normal map - patches tilted past max_tilt_deg (70) where bake rays hit a facing surface - by filling them from the texels round them; thin real lines are kept. texelsFilled, blobs."
    return _blender("fix_normal_map", image=image, filepath=filepath, outputPath=output_path, overwrite=overwrite,
                    maxTiltDeg=max_tilt_deg, minBlobPx=min_blob_px)


@mcp.tool()
def bl_boolean_op(target: str, cutter: str | list, operation: str = "difference",
                  delete_cutter: bool = False, solver: str = None,
                  material_mode: str = None, cleanup: bool | dict = None) -> dict:
    "Cut, merge or intersect mesh `target` with mesh `cutter` - operation 'difference' (default), 'union' or 'intersect'. APPLIES the modifier rather than leaving it stacked, and reports before/after vertex and face counts as the evidence. Says changed:false with the likely cause when the boolean legally did nothing. bl_add_modifier can name a cutter too (settings object) but leaves the modifier unapplied. The cutter is KEPT unless delete_cutter. material_mode 'transfer' gives the faces the cutter adds the cutter's own material (a slot is added to the target if it lacks one), 'index' (Blender's default) the target's slot with the same number - EXACT solver only. cutter may be a list: cutters, and a cutter's shells, are cut one at a time (touching shells cut at once came back empty). cleanup (on unless false; true or {angle, minFaceArea, ngons} to set it) cleans only the faces the cut made or touched - slivers welded, a Limited Dissolve, then n-gons split into quads (default), triangles or kept, needles dissolved and collapsed UVs opened - and counts faces, triangles, quads and n-gons before and after in the answer's cleanup. See mif_help."
    return _blender("boolean_op", target=target, cutter=cutter, operation=operation,
                    deleteCutter=delete_cutter, solver=solver, materialMode=material_mode, cleanup=cleanup)


@mcp.tool()
def bl_create_material(name: str, reuse: bool = False, base_color: list = None,
                       metallic: float = None, roughness: float = None,
                       skip_linked_inputs: bool = False, copy_from: str = None) -> dict:
    "Create a Blender material with a Principled BSDF - base_color [r,g,b] or [r,g,b,a], metallic and roughness can be set inline. Before this the addon could assign material slot NAMES and could not create a material or set a single shading value. `reuse` makes it get-or-create and STILL APPLIES the inline values, which is right when you mean to set them and fatal when you are only fetching the material: a Principled input that some node DRIVES ignores its default value completely, so writing one is refused rather than silently doing nothing. `skip_linked_inputs` is for the get-or-create case - a driven input is left alone and NAMED in skippedLinkedInputs alongside the node driving it, while every unlinked value is still written normally. It is off by default so the refusal still catches a caller who meant it. The case this exists for: a pipeline calls create_material with the same literal values in several stages purely to fetch the material, then a later stage wires a procedural pattern into Base Color, and every one of those earlier calls starts failing - which is exactly what made the bunker showcase unrunnable in its own documented stage order until 2026-09-07."
    return _blender("create_material", name=name, reuse=reuse, baseColor=base_color,
                    metallic=metallic, roughness=roughness,
                    skipLinkedInputs=skip_linked_inputs, copyFrom=copy_from)


@mcp.tool()
def bl_set_material_properties(material: str, base_color: list = None, metallic: float = None,
                               roughness: float = None, specular: float = None,
                               ior: float = None, alpha: float = None, emissive: list = None,
                               emissive_strength: float = None, transmission: float = None,
                               sheen: float = None, clearcoat: float = None,
                               anisotropic: float = None, rename: str = None, viewport_color: list = None,
                               viewport_metallic: float = None, viewport_roughness: float = None,
                               subsurface_weight: float = None, subsurface_scale: float = None,
                               subsurface_radius: list = None, subsurface_ior: float = None,
                               subsurface_anisotropy: float = None, subsurface_method: str = None,
                               weathering: dict | bool = None) -> dict:
    "Write Principled BSDF values on a Blender material, or rename it (rename: refused when another material has the name). THE VERSION SPREAD IS THE WHOLE DIFFICULTY: Blender RENAMED these inputs between 3.6 and 4.0 - 'Specular' became 'Specular IOR Level', 'Emission' became 'Emission Color', 'Transmission'"
    return _blender("set_material_properties", material=material, baseColor=base_color,
                    metallic=metallic, roughness=roughness, specular=specular, ior=ior,
                    alpha=alpha, emissive=emissive, emissiveStrength=emissive_strength,
                    transmission=transmission, sheen=sheen, clearcoat=clearcoat,
                    anisotropic=anisotropic, rename=rename, viewportColor=viewport_color,
                    viewportMetallic=viewport_metallic, viewportRoughness=viewport_roughness,
                    subsurfaceWeight=subsurface_weight, subsurfaceScale=subsurface_scale,
                    subsurfaceRadius=subsurface_radius, subsurfaceIor=subsurface_ior,
                    subsurfaceAnisotropy=subsurface_anisotropy, subsurfaceMethod=subsurface_method,
                    weathering=weathering)


@mcp.tool()
def bl_list_materials(name_contains: str = "", used_only: bool = False, limit: int = None,
                      name_excludes: list = None) -> dict:
    "List every material in the Blender file with its user count. Reports `unused` - materials with zero users - because a material with no users is NOT written to an FBX at all, so one created and never assigned never reaches Unreal."
    return _blender("list_materials", nameContains=name_contains or None, usedOnly=used_only,
                    limit=limit, nameExcludes=name_excludes)


@mcp.tool()
def bl_set_material_settings(material: str, blend_method: str = None,
                             shadow_method: str = None, backface_culling: bool = None,
                             alpha_threshold: float = None, screen_refraction: bool = None,
                             displacement_method: str = None,
                             surface_render_method: str = None,
                             transparent_shadow: bool = None,
                             raytrace_refraction: bool = None) -> dict:
    "The Blender material settings that are NOT Principled inputs - transparency, shadows, culling, refraction. bl_describe_material and bl_create_material have ALWAYS reported blendMethod and nothing anywhere could write it, which is a read/write asymmetry sitting in the middle of the most-used material op. It is also the property that decides whether a transparent material is actually transparent: an alpha of 0.2 on an OPAQUE material renders SOLID, and every field in every response still reads correctly. WHAT MOVES, read off bl_rna on 3.6.23, 4.2.17, 4.4.0 and 5.0.1 rather than from release notes: blend_method is on every build with the same four values; shadow_method is 3.6 and 4.2 ONLY, since EEVEE Next dropped it at 4.4; displacement_method, surface_render_method, use_transparent_shadow and use_raytrace_refraction are 4.2 and later and absent on 3.6. Anything this build lacks is REFUSED naming which builds have it, and every key is checked before any is written so a half-supported request does not leave half of it applied. THESE ARE EEVEE'S SETTINGS - Cycles decides transparency from the shader and ignores blend_method, so the active engine is reported and the keys it will not read are named."
    return _blender("set_material_settings", material=material, blendMethod=blend_method,
                    shadowMethod=shadow_method, backfaceCulling=backface_culling,
                    alphaThreshold=alpha_threshold, screenRefraction=screen_refraction,
                    displacementMethod=displacement_method,
                    surfaceRenderMethod=surface_render_method,
                    transparentShadow=transparent_shadow,
                    raytraceRefraction=raytrace_refraction)


@mcp.tool()
def bl_set_material_volume(material: str, density: float = None, color: list = None, anisotropy: float = None,
                           absorption_color: list = None, emission_strength: float = None,
                           emission_color: list = None, blackbody_intensity: float = None,
                           blackbody_tint: list = None, temperature: float = None,
                           density_attribute: str = None, color_attribute: str = None,
                           temperature_attribute: str = None, surface: str = None,
                           remove: bool = None) -> dict:
    "Give a Blender material a VOLUME shader - smoke, fire, fog - as a Principled Volume on its output's Volume socket. bl_add_fluid could build a smoke or fire simulation and nothing could make it appear, because every other material op writes the Principled BSDF on Surface. MEASURED on 3.6, 4.2, 4.4, 5.0 and 5.2: a GAS domain with no material renders as a SOLID BOX in EEVEE and as NOTHING in Cycles, and a Principled Volume with no surface shader renders the rising smoke in both. SURFACE IS EXPLICIT (keep | disconnect, default keep): a surface shader renders in front of the volume inside the mesh - bl_create_material's Principled BSDF left on Surface hid the smoke completely in both engines - so pass surface='disconnect' for smoke, fire or fog, and keep it only for a surface meant to be seen, like glass around the volume; surfaceNote says whenever a surface is still linked. density_attribute left out is left as it is - a new Principled Volume starts at Blender's 'density', which is what a smoke domain writes; an empty one measured as a filled box in EEVEE and nothing in Cycles, and densityAttributeNote says so when a GAS domain (listed in gasDomains) uses the material. A Principled Volume already on the output is edited in place; any other node there - Volume Scatter, a Mix Shader - is refused by name. Every value validates before anything is created or written. remove=True unlinks the volume shader from the output and deletes the node unless another node reads it (keptBecauseReadBy names them); it takes no other setting. Put the material on the domain with bl_set_material_slots (allow_resize=True when the mesh has no slots yet)."
    return _blender("set_material_volume", material=material, density=density, color=color,
                    anisotropy=anisotropy, absorptionColor=absorption_color,
                    emissionStrength=emission_strength, emissionColor=emission_color,
                    blackbodyIntensity=blackbody_intensity, blackbodyTint=blackbody_tint,
                    temperature=temperature, densityAttribute=density_attribute,
                    colorAttribute=color_attribute, temperatureAttribute=temperature_attribute,
                    surface=surface, remove=remove)


@mcp.tool()
def bl_describe_material(material: str, links: bool = False, nodes: bool = False) -> dict:
    "Read one Blender material in full: its Principled BSDF values, the node tree shape, every image texture with its FILE PATH, and with links=true every node link as from/to 'Node.Socket'."
    return _blender("describe_material", material=material, links=links,
                    nodes=nodes or None)


@mcp.tool()
def bl_assign_material_to_faces(object: str, slot: int, faces: list = None,
                                from_slot: int = None) -> dict:
    "Point a range of polygons at one of a Blender object's material SLOTS. bl_set_material_slots decides which materials a mesh has and in what order; this decides which faces use which. Omit faces to assign every polygon, or pass from_slot to move every face CURRENTLY on that slot - the operation you want after the slot list is reordered or resized. A from_slot no polygon uses is REFUSED rather than reported as changed:0, unlike an empty faces list: asking for nothing is a request, but believing faces live on an empty slot is a wrong assumption about the mesh."
    return _blender("assign_material_to_faces", object=object, slot=slot, faces=faces,
                    fromSlot=from_slot)


@mcp.tool()
def bl_set_material_texture(material: str, input: str = None, file: str = None, image: str = None,
                            colorspace: str = None, uv_map: str = None, strength: float = None,
                            interpolation: str = None, extension: str = None,
                            replace: bool = True, scale=None, offset=None,
                            rotation: float = None, to_node: str = None, to_socket: str = None,
                            from_socket: str = None, node_name: str = None, link: bool = None,
                            invert: bool = None, alpha_mode: str = None, displacement_scale: float = None,
                            midlevel: float = None, normal_convention: str = None) -> dict:
    "Put an IMAGE from disk into a Blender material - the write half of what bl_describe_material has always been able to read. Nothing else in the addon could do it: bpy.data.images.load appears once, in the world/HDRI path, so a material could be given numbers and never a map. `input` takes any bl_set_material_properties name plus 'normal' and 'displacement'. TWO TRAPS ARE HANDLED RATHER THAN LEFT TO THE CALLER. Colour space is chosen per input - sRGB for base_color and emissive, Non-Color for everything else - because Blender decodes from the IMAGE's setting, not from where it is plugged in, so a roughness map loaded as sRGB comes back through the transfer curve and every value the shader reads is wrong while looking plausible. And 'normal' is wired through a Normal Map node, because the Principled Normal socket takes a VECTOR and a tangent-space map is RGB; linking the image straight in is the usual mistake and reads as correct in every field of every response. A MISSING FILE IS REFUSED BEFORE ANYTHING IS CREATED: Blender would make the datablock anyway and render it MAGENTA, which is loud on screen and completely silent to anything reading a response. The link is verified off the socket afterwards, and the nodes are removed again if Blender rejected it. TILING is scale / offset (a number, or [u, v]) and rotation in DEGREES, through a Mapping node - per-texture and non-destructive, where the only alternative was bl_uv_unwrap's uv_transform, which scales the MESH's UVs and is therefore shared by every texture on the object and undone by the next unwrap. The addon has accepted these three since the day the Mapping support landed and no tool sent them, so the capability existed and was unreachable. For a pattern rather than a file, see bl_add_procedural_texture, which shares this vocabulary. A uvMap naming a UV layer the material's meshes lack is accepted and answered with uvMapNote: a UV Map node naming a missing layer renders a fixed fallback, not the active layer (measured on 3.6-5.2). ANY SOCKET OF THE TREE: to_node + to_socket instead of input wires the image into any node's input (a node name, or a node type such as colorRamp, mapRange, mix, math, invert or displacement to create one), link=false leaves the node free, node_name names the Image Texture node or reuses one to swap its image, invert=true puts an Invert node on the way, and from_socket takes Alpha."
    return _blender("set_material_texture", material=material, input=input, file=file,
                    image=image, colorspace=colorspace, uvMap=uv_map, strength=strength,
                    interpolation=interpolation, extension=extension, replace=replace,
                    scale=scale, offset=offset, rotation=rotation, toNode=to_node, toSocket=to_socket,
                    fromSocket=from_socket, nodeName=node_name, link=link, invert=invert, alphaMode=alpha_mode,
                    displacementScale=displacement_scale, midlevel=midlevel, normalConvention=normal_convention)


@mcp.tool()
def bl_import_pbr_texture_set(material: str, folder: str, include: list = None, resolution: str = None,
                              mapping: dict = None, udim: bool = None) -> dict:
    "Wire a folder of PBR maps into one Blender material in one call, each found by its file name and set in the colour space it needs: base colour, normal, roughness, metallic, displacement, AO, opacity, emission, subsurface. Replies what was wired and every file skipped, with why."
    return _blender("import_pbr_texture_set", material=material, folder=folder, include=include, resolution=resolution,
                    mapping=mapping, udim=udim)


@mcp.tool()
def bl_add_procedural_texture(material: str, type: str, node_name: str = None,
                              input: str = None, to_node: str = None, to_socket: str = None,
                              from_socket: str = None, settings: dict = None,
                              properties: dict = None, space: str = None, uv_map: str = None,
                              scale=None, offset=None, rotation: float = None,
                              color_ramp: list = None, relief: str = None,
                              strength: float = None, distance: float = None,
                              replace: bool = True) -> dict:
    "Build a PROCEDURAL pattern into a Blender material and wire it where it is read - type noise, voronoi, wave, gradient, magic, brick, checker, musgrave, whiteNoise or gabor - no texture files. It makes concrete that reads as concrete, worn steel, grain and rust. The write half this addon did not have: bl_create_material sets base colour, metallic and roughness, bl_set_material_texture loads an IMAGE off disk, and everything between was out of reach. type: noise | voronoi | wave | gradient | magic | brick | checker | musgrave | whiteNoise | gabor. MUSGRAVE IS VERSION-GATED and refused by name with the running version where it is absent - it is in 3.6's shader menu and none of 4.2's, 4.4's or 5.0's, having been folded into the Noise texture, and the refusal reads the replacement's real properties off its own RNA rather than quoting a changelog. `input` takes a Principled input by ANY version's spelling - baseColor or 'Base Color', emissive or 'Emission Color' or 'Emission' - because Blender renamed those sockets at 4.0 and a graph should not have to branch on bpy.app.version; `to_node`+`to_socket` targets any node instead, which is how two patterns get combined through a Mix or Math node. THREE TRAPS ARE HANDLED RATHER THAN LEFT TO THE CALLER. A Mapping node is only created when there is something to map and is NEVER left with its Vector unconnected, because an unconnected Mapping Vector is a CONSTANT ZERO and paints the whole surface one flat colour - that is not 'use the default coordinates', which is what an unconnected texture node does, and the two facts sit one node apart. Rotation is DEGREES here and radians on the socket. And a pattern driving a scalar input takes the texture's scalar output (Fac, or Distance on a Voronoi, which has no Fac at all) while one driving a colour takes Color, because Blender converts silently either way and the wrong choice renders plausibly. input:'normal' goes through a BUMP node, not a Normal Map node - the opposite of what bl_set_material_texture does with an image, and deliberately: a procedural pattern is a HEIGHT field, and a Normal Map node would read it as tangent-space RGB and light the surface wrongly while every field still read as correct. Pass relief:'normalMap' for a pattern that genuinely is tangent-space RGB. `color_ramp` inserts a ramp between the pattern and the destination, which is what turns grey mush into a material. EVERY NODE THIS MAKES IS REMOVED AGAIN IF ANY PART OF IT REFUSES, the link is MEASURED off the destination socket afterwards rather than assumed from the call, and a link it replaces is put back if Blender rejects the new one. The response carries a full readback of the texture node so the values can be checked in the same round trip. A bump with no distance gets 1.0 on every build (Blender 5.0 lowered the node default to 0.001, which renders almost no relief); reliefStrength and reliefDistance are read back, and distanceNote says when it filled one in. A uvMap naming a UV layer the material's meshes lack is accepted and answered with uvMapNote: a UV Map node naming a missing layer renders a fixed fallback, not the active layer (measured on 3.6-5.2). scale and offset take a number (X and Y), [u, v] or [x, y, z] - in Object or Generated space pass all three, or the pattern stretches along Z on any face that does not face Z (the reply's scaleNote says so)."
    return _blender("add_procedural_texture", material=material, type=type, nodeName=node_name,
                    input=input, toNode=to_node, toSocket=to_socket, fromSocket=from_socket,
                    settings=settings, properties=properties, space=space, uvMap=uv_map,
                    scale=scale, offset=offset, rotation=rotation, colorRamp=color_ramp,
                    relief=relief, strength=strength, distance=distance, replace=replace)


@mcp.tool()
def bl_set_color_ramp(node: str, material: str = None, tree: str = None, stops: list = None,
                      interpolation: str = None, color_mode: str = None,
                      hue_interpolation: str = None) -> dict:
    "Set the STOPS of a Blender ColorRamp node - the one thing the generic node writer cannot do. bl_add_group_node writes node properties by their RNA name and REFUSES pointers, correctly, because a datablock named as a string is stored and ignored; a colour ramp is a pointer to a ColorRamp holding a COLLECTION of elements, so it falls outside that writer twice over. The result was that a ramp could be added to a material and never given a stop - two black-to-white defaults and no way to change them, which makes the node an expensive no-op. A ramp is how a procedural pattern becomes a material: noise on its own is grey mush, noise through three stops is concrete, rust or moss. Address the tree with `material` (a material name) or `tree` (any tree address - a node group, 'material:<name>', 'scene:world', 'scene:compositor'); they are separate parameters rather than two spellings of one because a material NAME and a tree ADDRESS are different things. stops is [{position: 0..1, color: [r,g,b] or [r,g,b,a]}]. A POSITION OUTSIDE 0..1 IS REFUSED RATHER THAN CLAMPED, because Blender clamps silently and the ramp then reads back agreeing with itself and disagreeing with the request. Every stop is validated before the first is written, since the write clears the ramp down to one element before rebuilding it. Blender keeps elements SORTED BY POSITION, so stops sent out of order come back reordered and the response says when that happened - the ramp is correct and the indices are Blender's."
    return _blender("set_color_ramp", node=node, material=material, tree=tree, stops=stops,
                    interpolation=interpolation, colorMode=color_mode,
                    hueInterpolation=hue_interpolation)


@mcp.tool()
def bl_select_faces(object: str, material: str = None, slot: int = None, axis: str = None,
                    direction: list | str = None, angle: float = None, min_area: float = None,
                    max_area: float = None, inside_box: list | bool = None, box_min: list = None,
                    box_max: list = None, smooth: bool = None, require_all: bool = None,
                    limit: int = None, evaluated: bool = None, toward_point: list = None,
                    non_manifold: bool = None, uv_box: list = None, uv_layer: str = None, trait: str = None) -> dict:
    "Find the Blender faces matching a description - the missing half of bl_assign_material_to_faces, which takes face INDICES that nothing could compute. Working out which indices you wanted was raw Python; bl_select_edges had no counterpart for faces. Criteria: material or slot, a WORLD-space normal direction (axis 'Z'/'-Z'/..., or direction as a vector [x,y,z], with a cone half-angle in DEGREES), area range, a world box (inside_box=[[minX,minY,minZ],[maxX,maxY,maxZ]], or box_min and box_max [x,y,z]; inside_box:true beside them is the same box), smooth/flat, and toward_point [x,y,z]: faces whose normal points at that world point within the angle (a hood's inner shell faces the head - hand the indices to bl_delete_mesh_elements), non_manifold (faces on an edge shared by more than two), trait (NON_MANIFOLD: on an edge not shared by exactly two faces; INTERIOR; LOOSE: sharing no edge; BOUNDARY; DEGENERATE: under 1e-9 m2), and uv_box [[u,v],[u,v]]: faces whose UV centre lies in that box, on uv_layer or the active map (a lining from its shell, by UV island). They AND together by default. Read-only - nothing is written and no selection state is touched. The NORMAL TEST IS WORLD SPACE through the inverse transpose, because a face pointing up in local space on a rotated object does not point up in the world and 'which faces are the floor' is a world question every time. The box test uses the face CENTRE, so a large face straddling the boundary is decided by its middle. matchedPerCriterion is returned so that 'nothing matched' names the culprit instead of being a dead end, and the count is always exact even when the index list is capped (limit 0 returns every index; indicesOmitted counts the rest). An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2)."
    return _blender("select_faces", object=object, material=material, slot=slot, axis=axis,
                    direction=direction, angle=angle, minArea=min_area, maxArea=max_area,
                    insideBox=inside_box, boxMin=box_min, boxMax=box_max, smooth=smooth,
                    requireAll=require_all, limit=limit, evaluated=evaluated, towardPoint=toward_point,
                    nonManifold=non_manifold, uvBox=uv_box, uvLayer=uv_layer, trait=trait)


@mcp.tool()
def bl_bisect_plane(object: str, plane_co: list, axis: str = None, plane_no: list = None,
                    clear_inner: bool = None, clear_outer: bool = None, fill: bool = None,
                    threshold: float = None, faces: list = None, near_bone: dict = None) -> dict:
    "Cut a Blender mesh with a plane - the fundamental 'cut at a plane' that bevel, boolean, separate, decimate and clean are not. A boolean needs a second object authored, positioned and cleaned up for what is one plane. Works through bmesh rather than bpy.ops.mesh.bisect, which needs EDIT mode and a selection - a mode switch from a socket call can strand every op after it. THE PLANE IS WORLD SPACE and converted in, because 'cut at z=2.4' is a world statement every time and handing a local plane to a moved object cuts somewhere else silently. The postcondition is that the mesh actually MOVED: bisect raises nothing when the plane misses the geometry, so it would report success having changed not one vertex - if that happens the refusal quotes the object's world bounding box so you can see where the plane should have been. With clear_inner or clear_outer it also CLOSES the opening with a cap face unless fill=False - pass fill=False to keep it open; a cut that clears nothing leaves no opening to fill. faces [indices] or near_bone {bone, radius, armature} limits the cut and any clear to a region (faces whose centre is within radius of the bone's rest segment), so a cuff cut no longer slices the torso. The reply says what the fill did - capped, cappedFaces, cappedFaceCount and capNote, with fillFrom saying whether fill was sent or took its default - and a garment cut (a hem, a cuff, a neckline) passes fill=False, since the default caps the opening and a later fold finds no open boundary."
    return _blender("bisect_plane", object=object, planeCo=plane_co, axis=axis, planeNo=plane_no,
                    clearInner=clear_inner, clearOuter=clear_outer, fill=fill,
                    threshold=threshold, faces=faces, nearBone=near_bone)


@mcp.tool()
def bl_set_shading(object: str, smooth: bool = None, indices: list = None,
                   auto_smooth_angle: float = None, weighted_normals: bool = None,
                   weighted_normals_mode: str = None, keep_sharp: bool = None) -> dict:
    "Smooth or flat shading, the auto-smooth angle, and weighted normals - none of which the addon addressed at all, so setting flat shading per polygon was a raw Python loop. Hard-surface game assets need it constantly: it is the difference between a bevelled edge reading as a crease and reading as a smear. AUTO-SMOOTH MOVED IN 4.1 AND THE OLD PROPERTY IS GONE - mesh.use_auto_smooth and auto_smooth_angle were REMOVED and replaced by a Smooth by Angle geometry-nodes modifier, so writing the old property on a 4.1+ build raises, and swallowing that would report success having changed nothing on exactly the builds most people run. Both routes are implemented and the response says which was taken, which matters because the 4.1+ one is a MODIFIER: it shows in the modifier list, applies on export, and can be removed. weighted_normals adds the modifier that fixes a bevelled corner shading wrong no matter what the smooth flags say."
    return _blender("set_shading", object=object, smooth=smooth, indices=indices,
                    autoSmoothAngle=auto_smooth_angle, weightedNormals=weighted_normals,
                    weightedNormalsMode=weighted_normals_mode, keepSharp=keep_sharp)


@mcp.tool()
def bl_objects_overlap(a: str, b: str, tolerance: float = None, max_samples: int = None,
                       evaluated: bool = None) -> dict:
    "Do two Blender objects actually intersect - asked of the GEOMETRY rather than of their bounding boxes, which is what this was being approximated with. An AABB test says two objects overlap when they do not: measured on 3.6, 4.4 and 5.0, a 2m cube rotated 45 degrees about Z and a 1m cube at (1.45,1.45,0) have world AABBs that DO intersect and geometry that never touches. Both answers are returned, so aabbWouldHaveLied shows you the disagreement. FIVE VERDICTS, because 'do they overlap' is several questions: DISJOINT, TOUCHING (surfaces meet, no shared volume), INTERSECTING, A_INSIDE_B and B_INSIDE_A. CONTAINMENT IS THE CASE A SURFACE TEST CANNOT SEE - overlap() reports face-to-face intersection only, so a cube entirely inside another returns ZERO pairs and len(overlap()) alone calls it DISJOINT, the exact opposite of the truth. Trees are built in WORLD space from a transformed bmesh, never BVHTree.FromObject, which silently ignores matrix_world - two cubes FIVE UNITS APART report 2 overlap pairs through it - and which segfaults Blender 3.6 outright on a non-mesh object. facePairs is a yes/no signal and NOT a contact count: two identical coincident cubes return 2 pairs, not 36. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2)."
    return _blender("objects_overlap", a=a, b=b, tolerance=tolerance, maxSamples=max_samples,
                    evaluated=evaluated)


@mcp.tool()
def bl_coplanar_overlaps(objects: str | list, plane_tolerance_cm: float = None, min_area_cm2: float = None,
                         seal_distance_cm: float = None, drop: list = None, occluders: list = None, limit: int = None,
                         evaluated: bool = None) -> dict:
    "Find faces that z-fight before a mesh goes to Unreal: overlapping triangles in one plane (within 0.15 cm), facing one way, ranked by the overlap's own area; buried, sealed and identical ones are dropped (drop [] keeps them). See mif_help."
    return _blender("coplanar_overlaps", objects=objects, planeToleranceCm=plane_tolerance_cm, minAreaCm2=min_area_cm2,
                    sealDistanceCm=seal_distance_cm, drop=drop, occluders=occluders, limit=limit,
                    evaluated=evaluated)


@mcp.tool()
def bl_channel_pack_image(channels: dict = None, name: str = None, width: int = None, height: int = None,
                          filepath: str = None, format: str = None, color_depth: int = None, mode: str = None,
                          image: str = None, outputs: dict = None, overwrite: bool = None,
                          fake_user: bool = None) -> dict:
    "Pack channels of several Blender images or constants into one data texture (ORM, RMA, a mask map) - channels {R, G, B, A} each a number or {image, channel, invert} - or SPLIT one into grayscale maps. Non-Color, resampled to one size, read back per channel. See mif_help."
    return _blender("channel_pack_image", channels=channels, name=name, width=width, height=height,
                    filepath=filepath, format=format, colorDepth=color_depth, mode=mode, image=image,
                    outputs=outputs, overwrite=overwrite, fakeUser=fake_user)


@mcp.tool()
def bl_remap_datablock(kind: str, source: Any = None, target: str = None, mode: str = None, names: list = None,
                       remove: bool = None, dry_run: bool = None) -> dict:
    "Replace every use of one Blender datablock with another (materials, meshes, images, node groups, actions...), or mode DEDUPE: merge `.001` copies whose content matches their base, keeping and naming the ones that differ. Users moved per remap. See mif_help."
    return _blender("remap_datablock", kind=kind, source=source, target=target, mode=mode, names=names,
                    remove=remove, dryRun=dry_run)


@mcp.tool()
def bl_batch_rename(kind: str, names: list = None, match: str = None, find: str = None, replace: str = None,
                    regex: bool = None, case_sensitive: bool = None, prefix: str = None, suffix: str = None,
                    remove_prefix: str = None, remove_suffix: str = None, case: str = None,
                    skip_if_prefixed: bool = None, armature: str = None, dry_run: bool = None,
                    limit: int = None) -> dict:
    "Rename many Blender datablocks of one kind (objects, meshes, materials, images, actions, collections, bones...) by find/replace or regex, prefix, suffix, strip and case; a rename onto a name in use is refused with the pair. See mif_help."
    return _blender("batch_rename", kind=kind, names=names, match=match, find=find, replace=replace, regex=regex,
                    caseSensitive=case_sensitive, prefix=prefix, suffix=suffix, removePrefix=remove_prefix,
                    removeSuffix=remove_suffix, case=case, skipIfPrefixed=skip_if_prefixed, armature=armature,
                    dryRun=dry_run, limit=limit)


@mcp.tool()
def bl_manage_external_data(action: str = None, method: str = None, directory: str = None, kinds: list = None,
                            missing_only: bool = None, limit: int = None) -> dict:
    "A .blend's external files: REPORT every path (exists, packed, relative), PACK_ALL, UNPACK, MAKE_RELATIVE, MAKE_ABSOLUTE, FIND_MISSING in a folder - counts before and after, missing files named. See mif_help."
    return _blender("manage_external_data", action=action, method=method, directory=directory, kinds=kinds,
                    missingOnly=missing_only, limit=limit)


@mcp.tool()
def bl_artifact_inventory(roots: list = None, status: list = None, limit: int = None, offset: int = None, hash: bool = None,
                          archive_root: str = None, max_files: int = None) -> dict:
    "Which files MifBlender's ops wrote and what a folder holds, each classed referenced, protected, unknown, candidate or gone, with why. Reads only: it moves and deletes nothing, and never calls a project clean. See mif_help."
    return _blender("artifact_inventory", roots=roots, status=status, limit=limit, offset=offset, hash=hash,
                    archiveRoot=archive_root, maxFiles=max_files)


@mcp.tool()
def bl_object_relations(action: str, objects: Any = None, source: str = None, targets: Any = None,
                        what: Any = None, keep_hierarchy: bool = None, parent: bool = None) -> dict:
    "Blender's Object > Relations: MAKE_SINGLE_USER (data, materials, actions), MAKE_INSTANCES_REAL (collection, particle or node instances to objects), LINK_DATA (a source's data, materials, modifiers or collections onto targets). See mif_help."
    return _blender("object_relations", action=action, objects=objects, source=source, targets=targets, what=what,
                    keepHierarchy=keep_hierarchy, parent=parent)


@mcp.tool()
def bl_list_library_assets(library: str = None, folder: str = None, file: str = None, kinds: list = None,
                           catalog: str = None, tag: str = None, name: str = None, limit: int = None,
                           offset: int = None, catalogs: bool = None) -> dict:
    "Browse a Blender asset library (a library name, a folder or one .blend) without opening its files: assets with kind, catalog, tags, description, author, preview; filtered and paged; the catalog tree. Bring one in with bl_append_from_file. See mif_help."
    return _blender("list_library_assets", library=library, folder=folder, file=file, kinds=kinds, catalog=catalog,
                    tag=tag, name=name, limit=limit, offset=offset, catalogs=catalogs)


@mcp.tool()
def bl_set_collection_exporter(action: str = None, collection: str = None, format: str = None,
                               filepath: str = None, settings: dict = None, index: int = None,
                               all: bool = None) -> dict:
    "Blender 4.2+ collection exporters kept in the .blend: LIST, ADD (FBX, GLTF, USD, OBJ, ALEMBIC, PLY, STL with a path and settings), EDIT, REMOVE, EXPORT one collection's or all - file, size and a newer mtime per exporter. See mif_help."
    return _blender("set_collection_exporter", action=action, collection=collection, format=format,
                    filepath=filepath, settings=settings, index=index, all=all)


@mcp.tool()
def bl_set_keying_set(action: str = None, name: str = None, paths: list = None, active: bool = None,
                      frame: float = None) -> dict:
    "A Blender scene keying set: LIST, CREATE, ADD_PATHS / REMOVE_PATHS ({object, dataPath, index, group}), KEY every path at a frame, DELETE. See mif_help."
    return _blender("set_keying_set", action=action, name=name, paths=paths, active=active, frame=frame)


@mcp.tool()
def bl_set_custom_normals(object: str, source: str, proxy: str = None, center: list = None, direction: list = None,
                          vertex_group: str = None, mix: float = None) -> dict:
    "Write a Blender mesh's custom split normals from a PROXY mesh, a SPHERE center, one DIRECTION, FROM_FACES, SMOOTH, or CLEAR them - masked by a vertex group, mixed; the angle each corner moved read back. Foliage and proxy normals for Unreal. See mif_help."
    return _blender("set_custom_normals", object=object, source=source, proxy=proxy, center=center,
                    direction=direction, vertexGroup=vertex_group, mix=mix)


@mcp.tool()
def bl_transfer_mesh_data(source: str, targets: Any, data: list = None, attributes: list = None,
                          mapping: str = None, vertex_group: str = None, mix: float = None) -> dict:
    "Copy UVs, colors, custom normals, edge and face flags, face sets or named attributes from one Blender mesh onto others - the destination layers made first, so nothing transfers silently into a missing layer. Per target what changed. See mif_help."
    return _blender("transfer_mesh_data", source=source, targets=targets, data=data, attributes=attributes,
                    mapping=mapping, vertexGroup=vertex_group, mix=mix)


@mcp.tool()
def bl_transfer_shape_keys(source: str, target: str, keys: list = None, falloff: float = None,
                           strength: float = None, replace: bool = None) -> dict:
    "Carry shape keys from a Blender mesh onto one of DIFFERENT topology through Surface Deform, key by key - names, sliders and relative keys kept, each key's displacement measured on both. See mif_help."
    return _blender("transfer_shape_keys", source=source, target=target, keys=keys, falloff=falloff,
                    strength=strength, replace=replace)


@mcp.tool()
def bl_set_mesh_attribute(object: str, action: str, name: str, new_name: str = None, domain: str = None,
                          data_type: str = None, value: Any = None, values: list = None, indices: list = None,
                          from_vertex_group: str = None, sample: int = None) -> dict:
    "CREATE, WRITE, RENAME, CONVERT or READ a generic Blender mesh attribute (FLOAT, INT, vectors, colors, FLOAT2, QUATERNION 4.2+) on POINT, EDGE, FACE or CORNER - by value, per element, by index or from a vertex group. See mif_help."
    return _blender("set_mesh_attribute", object=object, action=action, name=name, newName=new_name, domain=domain,
                    dataType=data_type, value=value, values=values, indices=indices,
                    fromVertexGroup=from_vertex_group, sample=sample)


@mcp.tool()
def bl_add_hook(object: str, vertices: list = None, vertex_group: str = None, hook_object: str = None,
                armature: str = None, bone: str = None, name: str = None, strength: float = None,
                falloff: str = None, falloff_radius: float = None, empty_name: str = None,
                empty_size: float = None) -> dict:
    "Hook a Blender mesh's, curve's or lattice's points to an object, a bone or a new empty at their centroid, with the inverse matrix reset so nothing moves when it is added - read back. See mif_help."
    return _blender("add_hook", object=object, vertices=vertices, vertexGroup=vertex_group, hookObject=hook_object,
                    armature=armature, bone=bone, name=name, strength=strength, falloff=falloff,
                    falloffRadius=falloff_radius, emptyName=empty_name, emptySize=empty_size)


@mcp.tool()
def bl_spin_mesh(object: str, edges: list = None, faces: list = None, vertices: list = None, axis: list = None,
                 center: list = None, space: str = None, angle: float = None, steps: int = None,
                 dupli: bool = None, merge_seam: bool = None, flip_normals: bool = None) -> dict:
    "Spin chosen edges, faces or vertices of an existing Blender mesh about an axis - a lip, a flange, a radial copy (dupli) - in place; new faces and the welded seam reported. See mif_help."
    return _blender("spin_mesh", object=object, edges=edges, faces=faces, vertices=vertices, axis=axis,
                    center=center, space=space, angle=angle, steps=steps, dupli=dupli, mergeSeam=merge_seam,
                    flipNormals=flip_normals)


@mcp.tool()
def bl_export_actions(armature: str, folder: str, actions: list = None, pattern: str = None, deform_only: bool = None,
                      simplify: float = None, include_mesh: bool = None, overwrite: bool = None) -> dict:
    "Export each action of a Blender armature to its OWN FBX (deform bones, the action's frame range, named A_{rig}_{action}) - the clips half of a character hand-off; an action keying bones the rig lacks is refused. Per file frames, fps, bones. See mif_help."
    return _blender("export_actions", armature=armature, folder=folder, actions=actions, pattern=pattern,
                    deformOnly=deform_only, simplify=simplify, includeMesh=include_mesh, overwrite=overwrite)


@mcp.tool()
def bl_export_material_textures(folder: str, objects: Any = None, materials: list = None, pattern: str = None,
                                normal_convention: str = None, pack_orm: bool = None, format: str = None,
                                overwrite: bool = None) -> dict:
    "Save every image a Blender material's Principled inputs read, with engine names (T_{material}_BC / _N / _ORM / _E), color spaces, the normal map's green flipped for DIRECTX and roughness + metallic packed as ORM; constants and node chains not carried are listed. See mif_help."
    return _blender("export_material_textures", folder=folder, objects=objects, materials=materials, pattern=pattern,
                    normalConvention=normal_convention, packOrm=pack_orm, format=format, overwrite=overwrite)


@mcp.tool()
def bl_export_camera(camera: str, file: str, frame_range: list = None, samples: int = None,
                     overwrite: bool = None) -> dict:
    "Export an animated Blender camera to FBX in Unreal's axes with its lens animation; focal length, focus distance and f-stop sampled into the reply (FBX has no f-stop channel). The Unreal half reads the file onto a camera binding. See mif_help."
    return _blender("export_camera", camera=camera, file=file, frameRange=frame_range, samples=samples,
                    overwrite=overwrite)


@mcp.tool()
def bl_export_groom(objects: Any, file: str, frame_range: list = None, scale: float = None,
                    overwrite: bool = None) -> dict:
    "Export Blender hair - Curves objects (4.2+) or a mesh's hair particles - as an Alembic Unreal's groom importer reads, each object one group, in centimeters; what the exporter drops is listed. See mif_help."
    return _blender("export_groom", objects=objects, file=file, frameRange=frame_range, scale=scale,
                    overwrite=overwrite)


@mcp.tool()
def bl_material_from_maps(material: str, folder: str, maps: list, overwrite: bool = None) -> dict:
    "Wire a Blender material's Principled inputs from texture files, each map given a role and the channel it is read from - a packed ORM split back to three maps, a DirectX normal's green flipped. See mif_help."
    return _blender("material_from_maps", material=material, folder=folder, maps=maps, overwrite=overwrite)


@mcp.tool()
def bl_mesh_from_heightmap(file: str, cell_size: float, height_scale: float, width: int = None, height: int = None,
                           cell_size_y: float = None, mid_value: float = None, origin: list = None, name: str = None,
                           collection: str = None) -> dict:
    "Build a Blender grid mesh from a 16-bit heightmap (.r16 or 16-bit PNG) at a stated cell size and height scale - an Unreal landscape as a surface. See mif_help."
    return _blender("mesh_from_heightmap", file=file, width=width, height=height, cellSize=cell_size,
                    cellSizeY=cell_size_y, heightScale=height_scale, midValue=mid_value, origin=origin, name=name,
                    collection=collection)


@mcp.tool()
def bl_heightmap_from_mesh(object: str, file: str, width: int, height: int, cell_size: float, height_scale: float,
                           cell_size_y: float = None, mid_value: float = None, origin: list = None,
                           miss_value: int = None, samples: list = None, overwrite: bool = None) -> dict:
    "Ray-cast a Blender terrain mesh straight down on a grid and write a 16-bit .r16 heightmap at the same mapping as bl_mesh_from_heightmap; misses and clamped heights are counted. See mif_help."
    return _blender("heightmap_from_mesh", object=object, file=file, width=width, height=height, cellSize=cell_size,
                    cellSizeY=cell_size_y, heightScale=height_scale, midValue=mid_value, origin=origin,
                    missValue=miss_value, samples=samples, overwrite=overwrite)


@mcp.tool()
def bl_retarget_action(source: str, target: str, action: str = None, map: Any = None, frame_range: list = None,
                       new_action: str = None, root_bone: str = None, source_root: str = None, root_scale: float = None,
                       assign: bool = None, sample_frames: int = None) -> dict:
    "Move a Blender action from one armature onto another with other bone names and rest pose - bone map (AUTO by name, or explicit), rest-pose difference compensated, root translation scaled, baked to a new action; per-bone error in degrees. See mif_help."
    return _blender("retarget_action", source=source, target=target, action=action, map=map, frameRange=frame_range,
                    newAction=new_action, rootBone=root_bone, sourceRoot=source_root, rootScale=root_scale,
                    assign=assign, sampleFrames=sample_frames)


@mcp.tool()
def bl_extract_root_motion(armature: str, action: str = None, hips: str = None, root: str = None, mode: str = None,
                           axes: str = None, yaw: bool = None, feet: list = None, frame_range: list = None,
                           create_root: bool = None) -> dict:
    "Move a Blender clip's hips travel (and yaw) onto a root bone - made when the rig has none - or remove it (IN_PLACE); the feet's world positions are compared before and after. The pair of Unreal's extract_root_motion. See mif_help."
    return _blender("extract_root_motion", armature=armature, action=action, hips=hips, root=root, mode=mode,
                    axes=axes, yaw=yaw, feet=feet, frameRange=frame_range, createRoot=create_root)


@mcp.tool()
def bl_set_freestyle(enabled: bool = None, view_layer: str = None, line_thickness: float = None,
                     thickness_mode: str = None, crease_angle: float = None, line_sets: list = None,
                     remove_line_sets: list = None, edge_marks: Any = None) -> dict:
    "Blender Freestyle line rendering: on or off, line thickness, line sets by edge type and collection with color, thickness and alpha, and Freestyle edge marks on meshes - read back. Cycles and EEVEE draw it. See mif_help."
    return _blender("set_freestyle", enabled=enabled, viewLayer=view_layer, lineThickness=line_thickness,
                    thicknessMode=thickness_mode, creaseAngle=crease_angle, lineSets=line_sets,
                    removeLineSets=remove_line_sets, edgeMarks=edge_marks)


@mcp.tool()
def bl_set_sun_position(latitude: float, longitude: float, object: str = None, date: str = None, time: str = None,
                        utc_offset: float = None, dst: bool = None, north_offset: float = None, sky: Any = None,
                        distance: float = None, strength: float = None, key_frame: int = None) -> dict:
    "Aim a Blender Sun light (and a sky texture's sun) from latitude, longitude, date, local time and UTC offset - azimuth, elevation, sunrise and sunset back; key_frame keys it for a time-lapse. Pairs with Unreal's set_sun_time. See mif_help."
    return _blender("set_sun_position", latitude=latitude, longitude=longitude, object=object, date=date, time=time,
                    utcOffset=utc_offset, dst=dst, northOffset=north_offset, sky=sky, distance=distance,
                    strength=strength, keyFrame=key_frame)


@mcp.tool()
def bl_render_viewport(file_path: str = None, folder: str = None, frame_start: int = None, frame_end: int = None,
                       camera: str = None, shading: str = None, color_type: str = None, resolution_x: int = None,
                       resolution_y: int = None, percentage: int = None, overlays: bool = None) -> dict:
    "A Blender playblast: Viewport Render in a session with a 3D View, a matching Workbench render headless - a still or a frame range in seconds per frame; the reply names the route. See mif_help."
    return _blender("render_viewport", filePath=file_path, folder=folder, frameStart=frame_start, frameEnd=frame_end,
                    camera=camera, shading=shading, colorType=color_type, resolutionX=resolution_x,
                    resolutionY=resolution_y, percentage=percentage, overlays=overlays, _timeout=600.0)


@mcp.tool()
def bl_bake_vertex_animation(object: str, folder: str, frame_range: list = None, name: str = None, format: str = None,
                             normals: bool = None, unreal_axes: bool = None, max_frames: int = None) -> dict:
    "Bake a deforming Blender mesh (cloth, a fixed-topology fluid, shape keys, nodes) over frames into vertex animation textures - offsets and normals, a row per frame, a column per vertex - and a rest mesh with VAT_UV; a changing vertex count is refused. Pair of Unreal's bake_vertex_animation. See mif_help."
    return _blender("bake_vertex_animation", object=object, folder=folder, frameRange=frame_range, name=name,
                    format=format, normals=normals, unrealAxes=unreal_axes, maxFrames=max_frames, _timeout=600.0)


@mcp.tool()
def bl_bake_pivot_painter(folder: str, object: str = None, collection: str = None, name: str = None,
                          format: str = None, uv_layer: str = None, parent_index_encoding: str = None) -> dict:
    "Bake a Blender model's elements (a mesh's loose parts, or a collection's objects) to Pivot Painter 2 data for Unreal's foliage wind functions: pivot + parent index and X axis + extent textures and the element-index UV. See mif_help."
    return _blender("bake_pivot_painter", folder=folder, object=object, collection=collection, name=name,
                    format=format, uvLayer=uv_layer, parentIndexEncoding=parent_index_encoding)


@mcp.tool()
def bl_hair_to_cards(object: str, cards: int = None, segments: int = None, width: float = None, taper: float = None,
                     atlas: list = None, name: str = None, iterations: int = None) -> dict:
    "Turn Blender hair (Curves or hair particles) into game hair cards: strands clumped by root into N cards, a tapered ribbon per card along its clump, each card in its own atlas cell (V root to tip). See mif_help."
    return _blender("hair_to_cards", object=object, cards=cards, segments=segments, width=width, taper=taper,
                    atlas=atlas, name=name, iterations=iterations)


@mcp.tool()
def bl_bake_texture_set(object: str = None, types: list = None, folder: str = None, pattern: str = None,
                        width: int = None, height: int = None, samples: int = None, margin: int = None,
                        uv_layer: str = None, source: Any = None, extrusion: float = None,
                        max_ray_distance: float = None, cage: str = None, green_channel: str = None,
                        pack_orm: bool = None, out_of_process: bool = None, job_id: str = None,
                        memory_limit_gb: float = None, baked_material: str = None,
                        assign_baked: bool = None) -> dict:
    "Bake several maps of one Blender setup (high-to-low or self) in one job, out of process by default - a second Blender bakes a copy of this session - named T_{object}_{suffix}, ORM packed on request; job_id alone polls it. See mif_help."
    return _blender("bake_texture_set", object=object, types=types, folder=folder, pattern=pattern, width=width,
                    height=height, samples=samples, margin=margin, uvLayer=uv_layer, source=source,
                    extrusion=extrusion, maxRayDistance=max_ray_distance, cage=cage, greenChannel=green_channel,
                    packOrm=pack_orm, outOfProcess=out_of_process, jobId=job_id, memoryLimitGB=memory_limit_gb,
                    bakedMaterial=baked_material, assignBaked=assign_baked, _timeout=600.0)


@mcp.tool()
def bl_recipe_detail_bake(object: str = None, detail: str = None, build: dict = None, types: list = None,
                          folder: str = None, pattern: str = None, width: int = None, height: int = None,
                          samples: int = None, margin: int = None, uv_layer: str = None, green_channel: str = None,
                          cage: str = None, cage_distance: float = None, max_ray_distance: float = None,
                          occluders: list = None, stitches: list = None, checks: dict = None,
                          out_of_process: bool = None, job_id: str = None, keep_cage: bool = None,
                          memory_limit_gb: float = None) -> dict:
    "Bake a Blender game mesh's NORMAL, AO and CURVATURE from a detail mesh in one call - the detail given, or built from the game mesh (subdivide, folds, displace rules, sculpt) - with a cage from the measured gap, seams and stitches drawn into the maps in millimetres, and checks before it answers: ray misses, black texels in UV islands, a tilted or flat normal map. See mif_help."
    return _blender("recipe_detail_bake", object=object, detail=detail, build=build, types=types, folder=folder,
                    pattern=pattern, width=width, height=height, samples=samples, margin=margin, uvLayer=uv_layer,
                    greenChannel=green_channel, cage=cage, cageDistance=cage_distance,
                    maxRayDistance=max_ray_distance, occluders=occluders, stitches=stitches, checks=checks,
                    outOfProcess=out_of_process, jobId=job_id, keepCage=keep_cage, memoryLimitGB=memory_limit_gb,
                    _timeout=1800.0, _job_timeout=1770.0)


@mcp.tool()
def bl_audit_placement(bounds: Any = None, floor: float = None, floor_tolerance: float = None,
                       prefix: str = None, collection: str = None, ignore: list = None,
                       check_overlaps: bool = None, max_report: int = None,
                       evaluated: bool = None) -> dict:
    "Sweep a whole Blender scene for objects that stick out of the room, float, sink into the floor, or collide - the level-dressing question, which bl_objects_overlap cannot answer because it asks about ONE pair and dressing asks about every object at once, hundreds of round trips away. IT CAUGHT A REAL DEFECT: two camp cots authored 0.75 x 1.95 were rotated 90 degrees about an object origin still at the world origin, swinging them from x 11.13 to 13.08 - straight through an east wall at x 13.0, which nothing in the render made obvious. EVERY TEST HERE IS AABB, and said so in the response: a cage and the pallet inside it share a bounding box without touching, so treat 'overlaps' as CANDIDATES and settle the ones that matter with bl_objects_overlap, which tests real geometry. A FLOATING OBJECT IS NOT AUTOMATICALLY WRONG - lights, wall fixtures and anything hung read as floating because they are - so 'floor' is opt-in and 'ignore' takes name prefixes, letting the caller say what is meant to rest on the ground. Bounds and floor are both optional and each enables its own test; with neither, this reports only what it audited. check_overlaps is off by default because it is O(n^2). Lists are capped at max_report and every one is returned beside its untruncated *Total, so a cap can never read as a clean scene. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2)."
    return _blender("audit_placement", bounds=bounds, floor=floor,
                    floorTolerance=floor_tolerance, prefix=prefix, collection=collection,
                    ignore=ignore, checkOverlaps=check_overlaps, maxReport=max_report,
                    evaluated=evaluated)


@mcp.tool()
def bl_ray_cast(origin: list = None, direction: list = None, target: list = None, object: str = None,
                distance: float = None, evaluated: bool = None, rays: list = None, all_hits: bool = None,
                objects: list = None) -> dict:
    "Fire a ray into a Blender scene and report what it hits. Chosen by COUNTING escapes, not from a wishlist: 13 of 21 bl_run_python escapes in one day's real work were ray casts - it is what answers 'what is at this point', 'what is under it', 'is anything across this edge', which every layout, level and hard-surface job is built on. WORLD COORDINATES IN AND OUT: obj.ray_cast is LOCAL on both sides, so handing it world coordinates gives a miss or a plausible WRONG hit with nothing to say so, and a moved or rotated object is the normal case - the conversion happens in the addon. Normals come back through the inverse transpose, not the object matrix, which is only correct under uniform scale. With no object it casts against the whole scene through the depsgraph, so it hits what is actually there - subdivided, displaced, mirrored, geometry-nodes output. With an object it uses that object's BASE mesh unless evaluated:true, and says so when the object has modifiers that make those differ. Pass target instead of direction to cast towards a point. rays [{origin, direction | target}, ...] casts a BATCH (at most 10000) in one call instead: one row per ray, in order, with rayCount and hitCount; object, distance and evaluated are shared. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2). all_hits returns EVERY surface the ray crosses within distance, nearest first, as hits [{location, normal, faceIndex, distance, entering}] with crossings - entering where the ray meets a face from its front; a lined garment's lining and shell, not only the first. Objects [names] casts against those meshes only - each its base mesh, or its modifier result with evaluated - the nearest hit winning and naming its object, and hitsByObject counts each one's hits; a scene-wide cast laying straps over a shirt reached the head."
    return _blender("ray_cast", origin=origin, direction=direction, target=target, object=object,
                    distance=distance, evaluated=evaluated, rays=rays, allHits=all_hits, objects=objects)


@mcp.tool()
def bl_closest_point_on_mesh(object: str, point: list, distance: float = None, evaluated: bool = None) -> dict:
    "The nearest point ON a Blender mesh to a point in space - the other half of bl_ray_cast, for when there is no obvious direction to cast in: snapping to a surface, measuring clearance, finding which face something sits over. World in, world out, with the same local-space conversion and the same inverse-transpose normal. Also returns signedOffset - positive means the query point is outside that face, negative behind it - which answers 'is this inside the mesh' without a second cast. Answers on the MODIFIER RESULT by default (what renders); evaluated:false answers on the object's own base mesh instead, and `against` says which was used, beside `modifiers`. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2)."
    return _blender("closest_point_on_mesh", object=object, point=point, distance=distance, evaluated=evaluated)


@mcp.tool()
def bl_face_info(object: str, material: str = None, slot: int = None, indices: list = None,
                 limit: int = None, with_faces: bool = None, evaluated: bool = None) -> dict:
    "Which faces of a Blender mesh carry which material, how many, and where they are. The second most-escaped-to question - 15 of one day's 21 bl_run_python escapes were reading per-face material - and a pure read/write asymmetry: bl_assign_material_to_faces could WRITE that relation and nothing could read it back, so every 'operate on just the glass faces' job left the typed path for bl_run_python. AGGREGATE FIRST: per-slot counts, areas and bounding boxes answer most of it in a fixed-size response, and with_faces asks for rows once you know the slot. The bounding box comes in BOTH spaces, and the world one is built from all EIGHT corners rather than by transforming min and max, which is correct only for an axis-aligned matrix and plausible-but-wrong under any rotation. Empty material slots are reported: a slot holding a material with no faces is what a wrong slot index leaves behind, and it is invisible from the material list. Truncation is never silent. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2). Each face row (with_faces) also gives vertices, the face's vertex indices in winding order - the indices bl_vertex_info reads - so a rule that needs topology no longer keeps its own copy of the face list."
    return _blender("face_info", object=object, material=material, slot=slot, indices=indices,
                    limit=limit, withFaces=with_faces, evaluated=evaluated)


@mcp.tool()
def bl_vertex_info(object: str, indices: list = None, limit: int = None, evaluated: bool = None,
                   group: str = None) -> dict:
    "Where a Blender mesh's vertices ARE, and one vertex group's weight on each - bl_face_info's vertex twin. Nothing could read either: bl_list_vertex_groups gives counts and weight ranges, and bl_export_mesh writes FBX, so a position or a weight written through the bridge could be checked only through a render or a bounding box. Each row has index, co (local) and world. indices picks vertices in the order asked; without it the first `limit` come back (default 200, at most 20000) and verticesTruncated says when that is not all. group adds each row's weight - null for a vertex not in the group, 0.0 for one in it at zero - from the base mesh; evaluated reads the modifier result instead, whose vertex indices are its own, so the two are refused together. A read in edit mode is stale and says so. Call mif_help(\"bl_vertex_info\") first. An object that draws instances - Geometry Nodes instances, particles, instancing - answers with instancesNote: instances are not part of any evaluated mesh, so this does not see them (measured on 3.6-5.2)."
    return _blender("vertex_info", object=object, indices=indices, limit=limit, evaluated=evaluated, group=group)


@mcp.tool()
def bl_export_vertex_positions(object: str, file: str, root: str = None, overwrite: bool = None, shape_key: str = None) -> dict:
    "Write a Blender mesh's vertex positions (or one shape key's, with shape_key) to a .json file in Unreal's mesh frame (cm, Y negated), in its own vertex order - the file set_skeletal_mesh_vertices reads. On an import of export_asset's FBX that order is the export order."
    return _blender("export_vertex_positions", object=object, file=file, root=root, overwrite=overwrite, shapeKey=shape_key)


@mcp.tool()
def bl_import_scene(file: str, collection: str = None, import_materials: bool = None, textures_mode: str = None,
                    textures_dir: str = None, import_all_materials: bool = None, usd_preview: bool = None,
                    material_blend: bool = None, material_name_collision: str = None, image_plane: dict = None,
                    svg: dict = None, bvh: dict = None) -> dict:
    "Read OBJ, USD/USDA/USDC/USDZ, Alembic, STL, PLY, an image (as a textured plane) or SVG (as curves) - the other half of bl_export_scene, which on its own left the addon able to WRITE six formats and read three. FBX and glTF are read by bl_import_mesh and are refused here by name: two ops reading one format is how they drift apart, and that one knows things this does not - that useCustomNormals is an FBX option with no glTF equivalent, and that an axis conversion applied to glTF is applied twice because the spec already fixes +Y up. THE POSTCONDITION IS WHAT ARRIVED, by set difference, because every import operator returns FINISHED and none returns the objects it made: a file that parses and holds nothing importable - an animation-only USD, a camera-only export - reports success and adds nothing, and that is refused rather than returned as ok:true. A named collection is resolved BEFORE the import, so a bad name cannot leave objects already in the scene. An SVG comes in as Grease Pencil strokes with svg={'as': 'GREASE_PENCIL'}."
    return _blender("import_scene", file=file, collection=collection, importMaterials=import_materials,
                    texturesMode=textures_mode, texturesDir=textures_dir, importAllMaterials=import_all_materials,
                    usdPreview=usd_preview, materialBlend=material_blend,
                    materialNameCollision=material_name_collision, imagePlane=image_plane, svg=svg, bvh=bvh)


@mcp.tool()
def bl_export_scene(file: str, objects: list = None, object: str = None,
                    selected_only: bool = None, apply_modifiers: bool = None,
                    frame_start: int = None, frame_end: int = None, animation: bool = None,
                    overwrite: bool = None, include_lights: bool = None, include_cameras: bool = None,
                    exporter_options: dict = None, bvh: dict = None) -> dict:
    "Write glTF/GLB, OBJ, USD/USDA/USDC/USDZ, Alembic, STL or PLY - everything except the FBX bl_export_mesh owns. Measured: bl_import_mesh took .fbx/.gltf/.glb and bl_export_mesh wrote .fbx and NOTHING else, so glTF could come IN and not go OUT and USD was absent in both directions. The format comes from the EXTENSION. Blender moved its OBJ, STL and PLY exporters to wm.*_export at DIFFERENT versions (OBJ and PLY by 3.6, STL at 4.2) and EACH EXPORTER SPELLS ITS 'only the selection' OPTION DIFFERENTLY - four different keyword names across the six operators - so each format carries a list of candidate operators and its own keyword rather than a guess. Carrying one operator's spelling to another either raises or is silently ignored. A missing exporter (they are add-ons and can be disabled) is reported as a sentence naming the format, not an AttributeError. A frame range on a format that carries no time is REFUSED rather than ignored, because 'I exported an animation' and 'I exported frame 1' look identical afterwards. And ok:true is not a file: the mtime is taken before the call, so a leftover from an earlier run cannot pass as this export. Nor is it everything that was there: each format leaves some object types out - glTF writes lights and cameras only when asked, and OBJ, STL and PLY hold mesh geometry only - so the answer's notCarried names every object in scope the file does not carry, with the parameter that would include it where there is one. Lights and cameras stay out of a glTF by default, as in Blender's own exporter, because in a hand-off file they clutter an Unreal import; include_lights puts point, spot and sun lights in and include_cameras cameras. Both are glTF options, refused on every other format, which decides for itself (USD writes both, Alembic writes cameras). An existing file is REPLACED by default and the answer says so: overwrote, and when true replacedBytes and replacedModified (UTC), measured off the old file before the write; overwrite=False refuses the path by name. Grease Pencil strokes go out as .svg or .pdf (svgStrokes counts an SVG's strokes in the file), refused on Blender 4.3-4.5; a 5.x PDF the exporter does not write is refused after the call."
    return _blender("export_scene", file=file, objects=objects, object=object,
                    selectedOnly=selected_only, applyModifiers=apply_modifiers,
                    frameStart=frame_start, frameEnd=frame_end, animation=animation,
                    overwrite=overwrite, includeLights=include_lights, includeCameras=include_cameras,
                    exporterOptions=exporter_options, bvh=bvh)


@mcp.tool()
def bl_export_mesh(object_name: str = None, file: str = None, object_types: list = None,
                   add_leaf_bones: bool = None, armature_deform_only: bool = None,
                   primary_bone_axis: str = None, secondary_bone_axis: str = None,
                   bake_anim: bool = None, mesh_smooth_type: str = None,
                   use_triangles: bool = None, use_tspace: bool = None,
                   use_mesh_modifiers: bool = None, overwrite: bool = None,
                   local_space: bool = None, bake_anim_all_actions: bool = None, bake_anim_nla: bool = None,
                   bake_anim_simplify: float = None, bake_anim_force_keying: bool = None,
                   animations_only: bool = None, objects: list = None, global_scale: float = None,
                   apply_scale_options: str = None, path_mode: str = None, embed_textures: bool = None,
                   facing_body=None, split_shape_corners: bool = None) -> dict:
    "Export Blender objects to FBX for reimport into Unreal - object_name, or objects for one file in which every mesh keeps the scene origin as its pivot. The two axis arguments are the whole ballgame and are set for you: axis_up='Z', axis_forward='Y', which are NOT the operator defaults ('Y' / '-Z', the Maya convention) - the defaults"
    # THE FOUR OVERRIDES BELOW WERE UNREACHABLE, and they are the ones that decide what UNREAL
    # receives rather than what Blender thinks it exported. The addon has always mapped them onto
    # the FBX exporter (_EXPORT_OVERRIDES, ops_mesh.py:227-231) and this tool sent none of them:
    #
    #   mesh_smooth_type    FACE / EDGE / OFF. Unreal reads smoothing groups from this; OFF is why a
    #                       mesh can arrive faceted with no way to ask for anything else.
    #   use_tspace          tangents and binormals in the file. Without them Unreal recomputes, and a
    #                       normal map baked against Blender's tangents will not match. ON BY DEFAULT since
    #                       2026-10-05 (zero-area UV faces warned "nearly zero tangents" in Unreal without them).
    #   use_triangles       triangulate on export rather than letting the importer choose.
    #   use_mesh_modifiers  apply modifiers, or export the base cage.
    #
    # None means UNSET: _blender drops unset params, so the addon's defaults stand and nothing
    # changes for an existing caller. Found 2026-08-31 by param_reach's Blender half.
    # overwrite IS EXPOSED so a caller can say no. This comment used to say the addon refuses an existing
    # path unless told otherwise; it does not, and has not since at least 2026-09-08 - op_export_mesh
    # reads overwrite with default=True, so a second export silently REPLACES the file (found 2026-09-21
    # by the pack-rule review). overwrite=False is the way to refuse: "already exists and overwrite:false".
    # local_space writes the file round the object's own origin and puts the object back (ops_mesh.py
    # _local_space_move) - Unreal bakes each FBX node's world transform into the vertices.
    return _blender("export_mesh", object=object_name, file=file, objectTypes=object_types,
                    overwrite=overwrite, localSpace=local_space,
                    addLeafBones=add_leaf_bones, armatureDeformOnly=armature_deform_only,
                    primaryBoneAxis=primary_bone_axis, secondaryBoneAxis=secondary_bone_axis,
                    bakeAnim=bake_anim, meshSmoothType=mesh_smooth_type,
                    useTriangles=use_triangles, useTspace=use_tspace,
                    useMeshModifiers=use_mesh_modifiers, bakeAnimAllActions=bake_anim_all_actions,
                    bakeAnimNla=bake_anim_nla, bakeAnimSimplify=bake_anim_simplify,
                    bakeAnimForceKeying=bake_anim_force_keying, animationsOnly=animations_only, objects=objects,
                    globalScale=global_scale, applyScaleOptions=apply_scale_options, pathMode=path_mode,
                    embedTextures=embed_textures, facingBody=facing_body, splitShapeCorners=split_shape_corners)


@mcp.tool()
def bl_create_scene(name: str, copy_from: str = None, type: str = None) -> dict:
    "Create a new, EMPTY Blender scene and return the name it actually got. Every op that takes a scene - bl_render_animation among them - takes it BY NAME, and until this existed nothing in the addon could make one, so a second render configuration (different resolution, frame range or output path) meant editing the only scene there was. The new scene does NOT copy the current one unless copy_from / type ask (NEW, EMPTY, LINK_COPY, FULL_COPY), and is NOT activated - the scene you were in stays current, so building one does not move you out of your work; switch with bl_set_current_scene when you want it. Blender appends .001 on a name collision rather than failing, so use the returned `name`, not the string you sent - `renamed` tells you when they differ."
    # ops_scene.py:op_create_scene. bpy.data.scenes.new(), not bpy.ops.scene.new() - the operator
    # needs a window context and does nothing under blender -b, which is the path every suite takes.
    return _blender("create_scene", name=name, copyFrom=copy_from, type=type)


@mcp.tool()
def bl_delete_scene(name: str) -> dict:
    "Remove a Blender scene by name - the counterpart bl_create_scene never had, so a scene made through the bridge lived for the rest of the session and was written into the .blend if the file was saved. REFUSED in three cases, each of which makes a file Blender cannot use: the scene does not exist (named separately so a typo does not read as a policy refusal), it is the LAST scene, or it is the ACTIVE one, which would leave bpy.context.scene dangling - switch away first with bl_set_current_scene. THE OBJECTS DO NOT GO WITH IT: removing a scene unlinks its master collection and its objects stay in bpy.data.objects with zero users, purged on the next save-and-reload rather than now - so `objectsUnlinked` is what was unlinked, NOT what was deleted."
    # ops_scene.py:op_delete_scene. The count has to be read from scene.objects BEFORE the remove:
    # that is a view onto a datablock the call is about to free, so taking it after answers nothing.
    #
    # THE ACTIVE-SCENE REFUSAL USED TO GIVE A FALSE REASON - "re-pointing bpy.context.scene needs a
    # window, which does not exist under `blender -b`". Measured 2026-09-20 on 3.6.23, 4.2.17 LTS,
    # 4.4.0, 5.0.1 and 5.2.1 LTS (tools/probe_blender_background_window.py): every one has a window under -b
    # and window.scene is assignable. The refusal stands on its own merits; the reason is now the
    # true one and it names the op that clears it.
    return _blender("delete_scene", name=name)


@mcp.tool()
def bl_set_current_scene(name: str, all_windows: bool = False) -> dict:
    "Switch Blender's ACTIVE scene. bpy.context.scene is what almost every other op reads when you do not name a scene - render engine and settings, frame range, active view layer and world - so this re-points that read surface (bl_list_objects is NOT scoped by it: it walks every object in the file, every scene), and the response reports the new scene's frame range, engine, world, camera and object count so you can see where you landed. Idempotent: asking for the scene that is already current answers changed:false rather than refusing. A GUI Blender can hold several windows on different scenes deliberately, so by default only the context window moves and `windows` carries a row per window saying where each one is; all_windows moves every one. Under `blender -b` there is exactly one window and the two are the same."
    # ops_scene.py:op_set_current_scene. THIS OP WAS NAMED BY bl_list_scenes' own guidance for six
    # days before it existed - "set_current_scene is what switches to one" - because the addon
    # believed switching was impossible headless. It is not: measured on 3.6.23, 4.2.17 LTS, 4.4.0,
    # 5.0.1 and 5.2.1 LTS, `blender -b` has one window and window.scene assigns cleanly
    # (tools/probe_blender_background_window.py). The write goes to window.scene and the claim is
    # about context.scene, so the op reads context.scene back before reporting success.
    return _blender("set_current_scene", name=name, allWindows=all_windows)


@mcp.tool()
def bl_undo(steps: int = 1) -> dict:
    "Undo the bridge's last call(s) that changed the Blender scene - one step per call, and a bl_batch is one step. Calls that only read (list_*, *_info, evaluate_at_frame, ...) are not steps, so reading between writes costs nothing. Reports the steps undone by name, what the scene gained and lost (objectsAdded, objectsRemoved, datablockCountsChanged) and how many steps remain each way. Stops at the first step the bridge recorded, or at the oldest one Blender still keeps (undo_steps, 32 back by default), and says why (stoppedBecause) rather than walking into what came before it. Files written by an export or render stay written - notReverted names them. Refused outside OBJECT mode, when Blender's Global Undo preference is off, and when the Blender was started with MIF_BLENDER_UNDO=0. In an interactive Blender, edits made in the UI are undo steps too and undo walks back through them in order; the objects added and removed are always measured, so the response says what really changed. While the file holds a Mantaflow fluid domain no step is recorded: an undo step pushed after a flow object changes stops the domain simulating (measured on 4.2-5.2), so bl_undo_history reports recording:false with the reason, and steps recorded before the domain existed stay undoable."
    return _blender("undo", steps=steps)


@mcp.tool()
def bl_redo(steps: int = 1) -> dict:
    "Redo what bl_undo took back, one bridge call per step. Any call that changes the scene after an undo ends the redo history, exactly as in Blender - stoppedBecause says so. Reports the steps redone and what the scene gained and lost."
    return _blender("redo", steps=steps)


@mcp.tool()
def bl_undo_history() -> dict:
    "The undo steps this bridge has recorded since Blender started or the file was opened - each with whether it is currently applied - plus undoAvailable, redoAvailable, Blender's step limit, how many older steps Blender has already let go of (olderStepsDropped), and whether steps are being recorded at all (recording, whyNotRecording). UI edits in an interactive Blender are steps too but are not listed; a file load starts the list again. While the file holds a Mantaflow fluid domain no step is recorded: an undo step pushed after a flow object changes stops the domain simulating (measured on 4.2-5.2), so bl_undo_history reports recording:false with the reason, and steps recorded before the domain existed stay undoable."
    return _blender("undo_history")


@mcp.tool()
def bl_delete_object(object_name: str, purge_orphans: bool = None) -> dict:
    "Delete one object from the Blender scene by name. purge_orphans removes object data with no user - meshes, curves, lights, cameras, armatures, particle settings, in the whole file - which frees those names for a rebuild, named in orphansPurgedNames. Materials, images, node groups and actions the delete leaves with no user are KEPT and named in orphansKeptNames (since 2026-10-05: a purged car took its paint materials and the rebuild's bl_create_mesh refused them); bl_purge_orphans takes them by name, bl_clear_scene sweeps every orphan in the file. It defaults to FALSE here (unlike bl_clear_scene, which defaults it TRUE), so a mesh deleted this way leaves its mesh data behind until something purges it. Use it to clean up a specific import; bl_clear_scene is the whole-scene form. Deleting a RIGID BODY out of a baked rigid body world frees that bake (freedRigidBodyBake, rigidBodyBakeNote): the cache is stored by position in the world, so the bodies left behind would replay the deleted ones' frames. Deleting a rigid body also deletes every RIGID BODY JOINT that holds it (jointsRemovedWithBodies, and in `removed`): removing a body and its joint in the wrong order crashed Blender - 5.2.1 one way round, 4.2.17 the other - so both are taken out of the physics world first and the joint goes with its body. bl_undo takes a delete back, the object's modifiers and materials with it."
    # ops_scene.py:219 - `take_bool(params, "purgeOrphans", "purge", default=False)`. The differing
    # defaults between this and clear_scene are the addon's, not a mistake here, and worth stating in
    # the docstring: deleting one object leaves its data, clearing the scene does not.
    return _blender("delete_object", object=object_name, purgeOrphans=purge_orphans)


@mcp.tool()
def bl_clear_scene(object_type: str = None, purge_orphans: bool = None) -> dict:
    "Delete objects in the Blender scene. object_type limits it to ONE Blender type ('MESH', 'ARMATURE', 'EMPTY', ...) so a rig or the lights and cameras can be kept while the meshes go; omit it to clear everything. purge_orphans (default TRUE in the addon) sweeps the WHOLE FILE for datablocks with zero users, not only the ones this call orphaned - so a material or node group made and not yet assigned is destroyed by it, and object_type='LIGHT' in a scene with no lights answers removedCount 0 while still purging. Measured; orphansPurgedNames and orphanPurgeNote now say what went. Pass purge_orphans=false to keep work in progress. bl_import_mesh already clears by default, so the usual reason to call this directly is to inspect a failed run's leftovers first and then reset. A baked rigid body world whose bodies go is freed and reported (freedRigidBodyBake): it would otherwise replay the old frames onto the next scene's bodies, which made shape, friction and bounciness look inert. Rigid body joints and bodies are taken out of the physics world before they are removed - removing them in either order crashed Blender on some build - and a joint whose body is cleared goes with it (jointsRemovedWithBodies), even when object_type would have kept it. bl_undo takes the whole clear back as one step. It deletes objects, not collections: emptyCollectionsLeft names the collections left empty, since bl_create_collection refuses a name that is still taken. Colour management is scene settings and survives it too: colorManagementKept reports it, with a note naming anything not at Blender's factory values."
    # The addon has always taken both (ops_scene.py:204) and this wrapper sent NEITHER - it called
    # _blender("clear_scene") bare, so "clear the meshes and keep my armature" was unaskable and the
    # only available answer was to delete everything. purge_orphans matters separately: leaving
    # orphaned meshes and materials behind keeps a .blend growing across a long import session, and
    # turning it OFF is what you want when something else still references them.
    return _blender("clear_scene", type=object_type, purgeOrphans=purge_orphans)


@mcp.tool()
def bl_run_python(code: str = "", file: str = "", return_locals: bool = False, fresh: bool = False) -> dict:
    "Run Python on Blender's main thread: code (bpy, bmesh and math imported) or a .py file; stdout, stderr and what the code assigns to `result` come back. Refused while the MifBlender add-on's preferences do not allow it. The escape hatch for everything MifBlender has no first-class op for - prefer a first-class op whenever one exists, because those have checked parameters and this does not. Names persist between calls until a .blend loads; fresh=True runs one call in a clean namespace, and namesFromEarlierCalls lists any earlier call's names the code read."
    reply = _asset_tools_once(_python_hint("blender", _blender("run_python", code=code or None, file=file or None,
                                                               returnLocals=return_locals or None,
                                                               fresh=fresh or None)))
    _py_record("blender", code, file, reply)
    return reply


# THE ASSET TOOLS, NAMED AT THE MOMENT AN AGENT DECIDES HOW TO BUILD - in the reply of the session's first Blender Python
# call that succeeds, once. benchmark round ten (2026-10-01, 69b2e433): 10 of 10 Blender asset runs (a crate, a kit, a
# boulder, a rock family, ground cover, LODs, vertex masks, a tree, two hand-offs to Unreal) built everything with
# run_python and called no bl_build_*, bl_recipe_game_ready or mif_send_to_unreal. What they saw on the lean default
# was three tools: run_python's sent text is its first sentence only (the "prefer a typed tool" sentence is past
# _DESC_BUDGET and never sent), and the steering of 2.16 lives in find_tools answers and Unreal tools' descriptions,
# which two runs reached and none for this. In all ten, request 1 was a read-only look at the scene through run_python
# and the first geometry came in request 2 - so this reply lands between the look and the build.
# WHY HERE AND NOT THE INSTRUCTIONS OR run_python's DESCRIPTION: those ride every request of every session, Unreal-only
# ones included; this rides only the requests after a Blender Python call, of sessions that make one - 71 of round
# ten's 81 requests against 81 and every other session's (docs/TOKEN_OPTIMIZATION.md 2.17). test_mcp_packs P148.
# AN INSTRUCTION WITH THE CALL, NOT A LIST OF NAMES: as a list it reached all 15 shipped A3/A5/A8 runs and 14 went
# straight on to a 100-200 line run_python build; the one that called bl_build_rock was its cell's cheapest (63k
# tokens against a 99k median; 2.18). The example's keys are pinned to bl_build_rock's parameters (P150).
# EACH BUILDER SAYS WHAT IT MAKES, 2026-10-02. Round ten B: the shipped arm called bl_build_rock in 13 of 15 rock runs
# (A3, A4, X1) and the kit builder in 5 of 5, but bl_build_plant and bl_build_tree in none of 10, though every one of those runs got
# this note (and the Python-off arm verified 10 of 10 with them). The note then read "bl_build_rock, bl_build_plant,
# bl_build_tree and bl_build_kit_piece (UV and lightmap layers, UCX_ hulls)" - and A5 asks for ground cover with "no
# collision hulls and no lightmap" and one UV map, A8 for a tree with one UV map: the note said the plant and tree
# builders make what those specs forbid. They make one UV map, leaves on the whole 0-1 square, a Col wind weight; the
# plant no hull, the tree a trunk hull (ops_nature.op_build_plant, ops_tree.op_build_tree). test_mcp_packs P153.
_ASSET_TOOLS_HINT = (
    "For a rock, plant, tree, kit piece or game-ready pass, call the matching tool instead of writing the mesh in "
    "Python - one call replaces the script, e.g. mif_call('bl_build_rock', {'name': 'SM_Rock', 'size': [180, 120, "
    "90], 'triangles': 5000}). The tools, all through mif_call: bl_build_rock and bl_build_kit_piece (UV and "
    "lightmap layers, UCX_ hulls); bl_build_plant (grass, fern, bush) and bl_build_tree, foliage of real blades and "
    "leaves: one UV map, each leaf on the whole 0-1 square, a Col wind weight, no lightmap, a hull on a tree's trunk "
    "only; bl_recipe_game_ready, bl_create_collision_hull, bl_build_lod_chain, bl_set_vertex_color_by_rule; "
    "mif_send_to_unreal imports into Unreal.")
_ASSET_TOOLS_HINTED = []


def _asset_tools_once(reply):
    # A REFUSAL BECAUSE PYTHON IS OFF CARRIES IT TOO: then the typed tools are the only way to build, and that refusal
    # is the first thing such a session reads. benchmark round ten's Python-off arm opened all 20 runs with it, never saw
    # the note (it rode only a success), and went to find_tools('blender mesh', limit 40) - 14,737 characters replayed
    # on every later request - to find the builders (docs/TOKEN_OPTIMIZATION.md 2.18). Any other failure still does not
    # spend it.
    if _ASSET_TOOLS_HINTED or not isinstance(reply, dict):
        return reply
    if reply.get("ok") is not True and "run_python is disabled" not in str(reply.get("error") or ""):
        return reply
    _ASSET_TOOLS_HINTED.append(True)
    return dict(reply, assetTools=_ASSET_TOOLS_HINT)


# --------------------------------------------------------------------------
# GENERATION - local text/image -> 3D through ComfyUI, hosted in the addon.
#
# The chain is Flux.1 (prompt -> reference image) -> Hunyuan3D-2 shape (image ->
# untextured mesh) -> Hunyuan3D-2 paint (mesh -> PBR textures baked from
# multiview renders) -> imported into the open Blender scene. Nothing here calls
# a paid API and nothing leaves the machine; it needs a local ComfyUI, by
# default at 127.0.0.1:8188, with those custom nodes installed.
#
# It lives in the addon rather than a mod's tools/ directory because Blender is
# where the result gets inspected and fixed, so the generator belongs on the same
# side of the socket as the mesh ops that clean it up.
#
# ALWAYS call bl_gen_status first. These are long jobs - the defaults run to 600s
# for an image and 3600s for a full asset - and a missing checkpoint should be
# found before an hour of GPU time, not after.
#
# GENERATIVE: ON BY DEFAULT, AN OFF-SWITCH AT EACH END, AND A MODE EACH PROJECT CHOOSES (2026-10-01). These
# five are the only tools in MifBridge that run a trained model. Earlier the same day they were made opt-in at both
# ends; the reverse was then asked for, with the choice made visible: "that forced generative mode should be a
# toggle that users are aware of, and don't make non-generative the default, make generative the default, and maybe on
# each new project claude asks the user what mode they want to use". So:
#   - THREE OFF-SWITCHES, each enough alone: MIF_GENERATIVE=off for this server (_GENERATIVE), the add-on preference
#     allow_generative, and the PROJECT's mode, 'non-generative', recorded with mif_project_mode - for an Unreal project
#     in <projectDir>/MifBridgeProject.json (beside the .uproject: kept when Saved/ is deleted, can be committed, and
#     never staged into a cooked build), for Blender in the open .blend (ops_gen.project_mode).
#   - A PROJECT WHOSE MODE IS NOT CHOSEN works as generative, and the first reply from it in a session carries ONE
#     note asking the agent to ask the user and record the answer (_project_mode_after_call). Once a mode is set
#     nothing is added to any reply, and the server instructions did not grow.
#   - OFF BY ANY SWITCH: _blender refuses a gen_* op, and a bl_batch carrying one, before the socket opens; the five
#     leave the tool list, the packs and find_tools (_sync_gen_visibility); mif_call answers the refusal. Before a gen
#     op is sent the open .blend's mode is read again, so a file opened since the last read is not missed.
#   - AN UNREAL PROJECT'S MODE HOLDS THE BLENDER TOOLS TOO. They run in Blender, but what they make is made for the
#     project being worked on, so either project being non-generative is enough.
# self_audit and bl_status say which way it is and which switch holds it; the add-on's sidebar and journal say it too.
# --------------------------------------------------------------------------

_GEN_TOOLS = ("bl_gen_status", "bl_gen_image", "bl_gen_mesh", "bl_gen_texture", "bl_gen_asset")
_GEN_HIDDEN = {}      # tool name -> its function, for the gen tools taken out of the registry or _UNLOADED
_GEN_ORIGIN = {}      # tool name -> "registry" | "unloaded": where a tool hidden for a project's mode goes back to
_PROJECT_FILE = "MifBridgeProject.json"
_MODES = ("generative", "non-generative")
# backend -> {"project", "mode", "where"} once read this session; None until then. mode is one of _MODES, "unset" (not
# chosen yet), or "unknown" (the editor answered but could not say - an add-on older than project_mode).
_PM = {"unreal": None, "blender": None}
_PM_TOUCHED = set()   # backends that answered a call while their mode was unread
_PM_NOTED = set()     # (backend, project) pairs this session has already carried the note for
_PM_PROBING = False
_BL_FILE_OPS = ("open_file",)   # ops after which another .blend may be open - its mode is read again
_GEN_OFF_BY = {
    "server": ("for this MCP server (MIF_GENERATIVE=off)",
               "Start the MCP server without MIF_GENERATIVE=off (or MIF_ALLOW_GENERATIVE=0)."),
    "unreal": ("for this Unreal project (its mode is non-generative, in %s)",
               "Only the user changes a project's mode: mif_project_mode('generative') when they ask for it."),
    "blender": ("for the open Blender file (its mode is non-generative)",
                "Only the user changes a project's mode: mif_project_mode('generative') when they ask for it."),
}
# THE MODELS' LICENSE TERMS: THE ONE TABLE. MifBridge ships no model, but two of the three its gen ops drive restrict use, so
# the assistant is told the terms before and whenever it uses them, and can tell the person. THIS IS DISCLOSURE, NOT
# ENFORCEMENT: nothing here looks up a location or asks where anyone is, nothing is collected, and no request is added.
#   - every successful gen op's reply carries `license`, a list with one block per model that call ran, for the Flux
#     variant it ran. A failed call carries none, and neither does a refusal.
#   - gen_status always names all three, even when ComfyUI cannot be reached: it is the documented first call, so this is
#     where the terms can be told before the first generation. Only a generative switch being off leaves it out.
#   - a bl_batch's reply carries it for the gen ops that succeeded in it.
#   - the project-mode question (_PROJECT_MODE_NOTE) names each model's `limit`.
# Built here on the server, not in the add-on, so an older add-on's replies carry it too.
# THIS TABLE OWNS THE TERMS. The five tools' descriptions, their tool_help.json entries and README "Model licenses" restate
# them in words a table cannot generate; test_generative_mode GM141 holds each to it. A pack rule points at the block and
# restates nothing. Each row is what the license text said when read on 2026-11-04 (the links), not advice about any use.
_GEN_LICENSES = {
    "flux-schnell": {
        "model": "FLUX.1 [schnell]",
        "licenseName": "Apache-2.0",
        "licenseUrl": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "limit": "Apache-2.0",
        "restrictions": [],
        "says": "FLUX.1 [schnell] is released under Apache-2.0; its model card says personal, scientific and commercial use."},
    "flux-dev": {
        "model": "FLUX.1 [dev]",
        "licenseName": "FLUX.1 [dev] Non-Commercial License",
        "licenseUrl": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        "limit": "non-commercial",
        "restrictions": [
            "The model may be used only for non-commercial purposes, which exclude revenue-generating activity; other use "
            "needs a license from Black Forest Labs.",
            "Outputs may be used for any purpose, commercial included, except to train, fine-tune or distill a model "
            "competitive with FLUX.1 [dev].",
            "Outputs must carry a disclosure that they were made with AI, to the extent applicable law requires."],
        "says": "FLUX.1 [dev] may be used only for non-commercial purposes unless Black Forest Labs grants a license; its "
                "outputs may be used commercially, within the limits listed."},
    "hunyuan3d-2": {
        "model": "Hunyuan3D-2",
        "licenseName": "Tencent Hunyuan 3D 2.0 Community License",
        "licenseUrl": "https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE",
        "limit": "not licensed in the EU, UK or South Korea",
        "restrictions": [
            "Its territory is worldwide excluding the European Union, the United Kingdom and South Korea; use of the model "
            "or its outputs outside that territory is not licensed.",
            "Outputs placed in any public context must be expressly and conspicuously identified as machine generated.",
            "Outputs may not be used to improve any other AI model.",
            "A licensee whose products or services had over 1 million monthly active users in the calendar month before the "
            "model's release date must request a license from Tencent."],
        "says": "Hunyuan3D-2's license does not cover the European Union, the United Kingdom or South Korea: use there is not "
                "licensed. Its other terms are listed."},
}
_GEN_LICENSE_CLOSE = ("You are responsible for following these terms. Not legal advice; read the license: %s - MifBridge ships "
                      "no model, checks nothing about where or how you use one, and sends nothing.")


def _gen_license_block(key) -> dict:
    """What a reply says about one model: _GEN_LICENSES' row, with the sentence for the assistant to pass on."""
    row = _GEN_LICENSES[key]
    return {"model": row["model"], "licenseName": row["licenseName"], "licenseUrl": row["licenseUrl"],
            "restrictions": list(row["restrictions"]),
            "tellTheUser": "%s %s" % (row["says"], _GEN_LICENSE_CLOSE % row["licenseUrl"])}


_PROJECT_MODE_NOTE = (
    "MifBridge: %s has no generative mode yet, and works as generative until it has one. Ask the user once which mode "
    "it uses - generative: the bl_gen_* tools may create images, meshes and textures with AI models, whose licenses the "
    "user must follow, so name them (" + "; ".join("%s: %s" % (r["model"], r["limit"]) for r in _GEN_LICENSES.values()) +
    "); non-generative: MifBridge only drives the editors, nothing is AI-generated - and record the answer with "
    "mif_call('mif_project_mode', {'mode': 'generative'}) or {'mode': 'non-generative'}.")


def _is_gen_op(name) -> bool:
    return isinstance(name, str) and (name.startswith("gen_") or name.startswith("bl_gen_"))


_LOCATED_MODE = {"key": None, "state": None}     # the located project's file, read again only when it changes


def _located_project_mode():
    """The mode in the MifBridgeProject.json of the project this server was started in (_locate_project_dir), read
    WITHOUT an editor - None when no project folder was found. R85, 2026-11-06: the Unreal project's mode reached the
    gate only after its editor answered a call, so a session whose editor was down or on another port, or that drove
    only Blender, ran the bl_gen_* tools in a project its user had made non-generative. Re-read when the file's mtime
    or size changes, so a mode recorded by hand applies without a restart."""
    try:
        where_from = (os.environ.get("MIF_PROJECT_DIR"), os.getcwd())
    except OSError:
        where_from = (os.environ.get("MIF_PROJECT_DIR"), None)
    if _LOCATED_MODE.get("from") != where_from:          # the folder walk once per working directory, not per call
        _LOCATED_MODE.update({"from": where_from, "dir": _locate_project_dir()[0], "key": None})
    pdir = _LOCATED_MODE["dir"]
    if not pdir:
        return None
    path = os.path.join(pdir, _PROJECT_FILE).replace("\\", "/")
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = (path, None, None)
    if _LOCATED_MODE["key"] != key:
        mode, error = _read_mode_file(path)
        _LOCATED_MODE["key"] = key
        _LOCATED_MODE["state"] = dict({"project": os.path.basename(pdir.rstrip("/\\")) or pdir, "mode": mode,
                                       "where": path, "located": True}, **({"error": error} if error else {}))
    return _LOCATED_MODE["state"]


def _unreal_mode_state():
    """The Unreal project's mode for the gate: the one its editor's project_paths named once an editor answered, else
    the located project's file (_located_project_mode). The editor's answer wins: it knows which project it has open."""
    return _PM.get("unreal") or _located_project_mode()


def _gen_block():
    """Which switch holds the gen tools off on this side - "server", "unreal" or "blender" - or None when none does.
    The add-on's preference is its own (ops_gen._require_generative); bl_status reports it."""
    if not _GENERATIVE:
        return "server"
    for backend in ("unreal", "blender"):
        state = _unreal_mode_state() if backend == "unreal" else _PM.get(backend)
        if (state or {}).get("mode") == "non-generative":
            return backend
    return None


def _gen_off_reply(name, by=None) -> dict:
    by = by or _gen_block() or "server"
    where, enable = _GEN_OFF_BY[by]
    if "%s" in where:
        where %= ((_unreal_mode_state() if by == "unreal" else _PM.get(by)) or {}).get("where") or _PROJECT_FILE
    return {"ok": False, "generativeAllowed": False, "generativeOffBy": by, "enable": enable,
            "error": "%s is generative (it runs a trained image/3D model through ComfyUI) and generative tools are "
                     "off %s, so nothing was sent to Blender or ComfyUI. %s" % (name, where, enable)}


def _generative_refusal(op, params):
    """The refusal for a generative op while a switch holds them off, else None. A batch is refused whole when any of
    its entries is one, as an unknown op refuses it - so none of it runs."""
    named, index = (op, None) if _is_gen_op(op) else (None, None)
    if named is None and op == "batch" and isinstance(params.get("ops"), list):
        for i, entry in enumerate(params["ops"]):
            entry_op = (entry.get("op") or entry.get("endpoint")) if isinstance(entry, dict) else None
            if _is_gen_op(entry_op):
                named, index = entry_op, i
                break
    if named is None:
        return None
    if _gen_block() not in ("server", "unreal"):
        _bl_project_mode(refresh=True)       # the file open NOW: another .blend may have been opened since the last read
    by = _gen_block()
    if by is None:
        return None
    reply = _gen_off_reply(named, by)
    if index is None:
        return dict(reply, endpoint=op)
    reply["error"] = "ops[%d]: %s NOTHING in the batch was run." % (index, reply["error"])
    return dict(reply, endpoint="batch")


def _gen_models_used(op, params):
    """The _GEN_LICENSES keys a gen op runs, from the parameters it was sent (the add-on's default variant: schnell)."""
    flux = "flux-dev" if str(params.get("variant") or "schnell").lower() == "dev" else "flux-schnell"
    return {"gen_status": tuple(_GEN_LICENSES), "gen_image": (flux,), "gen_mesh": ("hunyuan3d-2",),
            "gen_texture": ("hunyuan3d-2",), "gen_asset": (flux, "hunyuan3d-2")}.get(op, ())


def _gen_license_after(op, params, data):
    """Put `license` on a gen op's reply, one block per model that call ran: every success, and gen_status unless a
    generative switch refused it. A bl_batch's reply names the models of the gen ops that succeeded in it."""
    if not isinstance(data, dict) or not isinstance(op, str):
        return
    if op == "batch":
        rows = data.get("results") if isinstance(data.get("results"), list) else []
        done = {r.get("op") for r in rows if isinstance(r, dict) and r.get("ok") is True}
        used = []
        for entry in params.get("ops") or []:
            if isinstance(entry, dict) and entry.get("op") in done:
                used.extend(_gen_models_used(entry["op"], entry.get("params") or {}))
    elif not op.startswith("gen_"):
        return
    elif op == "gen_status":
        # The add-on's own refusal (ops_gen.GENERATIVE_HELP) has no field of its own, so its words are the signal.
        used = () if "generative tools are off" in str(data.get("error") or "") else _gen_models_used(op, params)
    else:
        used = () if data.get("error") or data.get("ok") is False else _gen_models_used(op, params)
    if used:
        data["license"] = [_gen_license_block(key) for key in dict.fromkeys(used)]


def _sync_gen_visibility() -> bool:
    """Take the gen tools out of every list while a switch holds them off, and put them back where they were when none
    does. Kept in _GEN_HIDDEN, never in _UNLOADED, so mif_call and mif_enable_pack cannot bring one back while it is
    off. Returns True when the client's tool list changed (a tools/list_changed is due)."""
    registry = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
    if not isinstance(registry, dict):
        return False
    changed = False
    if _gen_block() is not None:
        for name in _GEN_TOOLS:
            fn = globals().get(name)
            if name in _GEN_HIDDEN or not callable(fn):
                continue
            if name in registry:
                mcp.remove_tool(name)
                _GEN_ORIGIN[name], changed = "registry", True
            elif _UNLOADED.pop(name, None) is not None:
                _GEN_ORIGIN[name] = "unloaded"
            else:
                continue
            _GEN_HIDDEN[name] = fn
        return changed
    for name in list(_GEN_HIDDEN):
        fn = _GEN_HIDDEN.pop(name)
        if _GEN_ORIGIN.pop(name, "registry") == "registry":
            mcp.add_tool(fn)
            changed = True
        else:
            _UNLOADED[name] = fn
    return changed


def _apply_generative_mode():
    """At startup: hide the gen tools when this server's own switch is off. Returns what it did, as text."""
    if _GENERATIVE:
        return ("on (the default): %s offered unless a project's mode or the add-on preference says otherwise; "
                "MIF_GENERATIVE=off hides them" % ", ".join(_GEN_TOOLS))
    _sync_gen_visibility()
    return "off (MIF_GENERATIVE=off): %d gen tools hidden for every project" % len(_GEN_HIDDEN)


def _read_project_doc(path):
    """(doc, error): an Unreal project's MifBridgeProject.json as a dict - {} when there is none."""
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:      # -sig: a file saved by a Windows editor may start with a BOM
            doc = json.load(fh)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as exc:
        return {}, "%s could not be read (%s); recording a setting rewrites it" % (path, exc)
    return (doc, None) if isinstance(doc, dict) else ({}, "%s holds no JSON object; recording a setting rewrites it" % path)


def _read_mode_file(path):
    """(mode, error) from an Unreal project's MifBridgeProject.json: "unset" when there is none or it says nothing."""
    doc, error = _read_project_doc(path)
    if error:
        return "unset", error
    mode = doc.get("generativeMode")
    return (mode, None) if mode in _MODES else ("unset", None if mode is None else
                                                "%s says generativeMode %r, which is not one of %s"
                                                % (path, mode, ", ".join(_MODES)))


def _ue_project_mode(refresh=False):
    """The running editor's project's mode, read from <projectDir>/MifBridgeProject.json once a session (refresh reads it
    again). None when no editor answered."""
    if _PM.get("unreal") is not None and not refresh:
        return _PM["unreal"]
    paths = _post("project_paths")
    if not isinstance(paths, dict) or paths.get("ok") is False:
        return None
    pdir = paths.get("projectDir")
    if not isinstance(pdir, str) or not os.path.isdir(pdir):
        # the editor answered, but its project folder is not on this machine (or it did not say): nothing to read
        _PM["unreal"] = {"project": paths.get("projectName"), "mode": "unknown",
                         "error": "projectDir %r is not a folder this MCP server can read" % pdir}
        return _PM["unreal"]
    path = os.path.join(pdir, _PROJECT_FILE).replace("\\", "/")
    mode, error = _read_mode_file(path)
    stored = _read_project_doc(path)[0]
    _PM["unreal"] = dict({"project": paths.get("projectName") or pdir, "mode": mode, "where": path,
                          "changeCap": stored.get("changeCap"), "roadmap": stored.get("roadmap")},
                         **({"error": error} if error else {}))
    return _PM["unreal"]


_PROJECT_ABOUT = ("MifBridge's settings for this project, chosen by its user. generativeMode - generative: the bl_gen_* "
                  "tools may create images, meshes and textures with AI models; non-generative: MifBridge only drives the "
                  "editors. changeCap - a call that would change more than this many items answers a preview to commit "
                  "instead of writing; 0 is off. roadmap - the plan this project follows, and the steps done by hand; "
                  "mif_roadmap reads it. Not packaged into a cooked build.")


def _ue_set_mode(mode=None, change_cap=None, roadmap=None):
    """Record the Unreal project's generative mode, change cap and/or roadmap ({} clears it). The file is READ AND MERGED, so
    recording one setting keeps the others - and any key a newer MifBridge wrote."""
    state = _ue_project_mode(refresh=True)
    if not state or not state.get("where"):
        return {"ok": False, "error": (state or {}).get("error") or "no Unreal editor answered project_paths"}
    doc = dict(_read_project_doc(state["where"])[0])
    if mode is not None:
        doc["generativeMode"] = mode
    if change_cap is not None:
        doc["changeCap"] = change_cap
    if roadmap:
        doc["roadmap"] = roadmap
    elif roadmap is not None:
        doc.pop("roadmap", None)
    doc.update({"setAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setBy": _agent_tag() or None,
                "about": _PROJECT_ABOUT})
    tmp = state["where"] + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, state["where"])
    except OSError as exc:
        return {"ok": False, "error": "could not write %s: %s" % (state["where"], exc)}
    _PM["unreal"] = {"project": state["project"], "where": state["where"], "changeCap": doc.get("changeCap"),
                     "roadmap": doc.get("roadmap"), "mode": doc.get("generativeMode") if doc.get("generativeMode") in _MODES else "unset"}
    return dict(_PM["unreal"], ok=True)


def _bl_mode_seen(op, data):
    """Every Blender reply passes here: ping and project_mode carry the open file's mode; open_file may change it."""
    if op in _BL_FILE_OPS:
        _PM["blender"] = None
        _PM_TOUCHED.add("blender")
    elif op in ("ping", "project_mode") and data.get("generativeMode") in _MODES + ("unset", "unknown"):
        _PM["blender"] = {"project": data.get("blendFile") or "an unsaved Blender file",
                          "mode": data["generativeMode"], "where": data.get("blendFile") or "the unsaved .blend",
                          "changeCap": data.get("changeCap"), "roadmap": data.get("roadmap")}
    elif _PM.get("blender") is None:
        _PM_TOUCHED.add("blender")


def _bl_set_mode(mode=None, change_cap=None, roadmap=None):
    """Record the open .blend's mode, change cap and/or roadmap through the add-on's project_mode op (_bl_mode_seen notes the
    answer). A helper, not a call in mif_project_mode's body, so gen_packs places that tool by name (misc), not as a
    Blender tool."""
    return _blender("project_mode", mode=mode, changeCap=change_cap, roadmap=roadmap)


def _bl_project_mode(refresh=False):
    """The open .blend's mode, read once a session and after open_file (refresh reads it again). None when Blender did
    not answer."""
    if _PM.get("blender") is not None and not refresh:
        return _PM["blender"]
    before = _PM.get("blender")
    _PM["blender"] = None
    res = _blender("project_mode", _timeout=BLENDER_PROBE_TIMEOUT, _lock_timeout=BLENDER_PROBE_TIMEOUT)
    if _PM.get("blender") is None:
        if isinstance(res, dict) and "unknown endpoint 'project_mode'" in str(res.get("error")):
            _PM["blender"] = {"project": None, "mode": "unknown",
                              "error": "this MifBlender has no project_mode op - update the add-on"}
        else:
            _PM["blender"] = before
    return _PM["blender"]


def _project_mode_label(backend, state):
    if backend == "unreal":
        return "the Unreal project '%s'" % state["project"]
    # an unsaved file's "project" is already a phrase: "the Blender file an unsaved Blender file" was the note's opening
    return "the Blender file %s" % state["project"] if state.get("where") != "the unsaved .blend" else state["project"]


def _project_mode_after_call():
    """(note, changed) after a tool call: read the mode of a project that answered for the first time, word the one
    note for a project whose mode is not chosen, and hide or restore the gen tools. Near free once every answering
    project's mode is read: two set lookups and the visibility check."""
    global _PM_PROBING
    if _PM_PROBING:
        return None, False
    _PM_PROBING = True
    try:
        for backend in sorted(_PM_TOUCHED):
            if _PM.get(backend) is None:
                (_ue_project_mode if backend == "unreal" else _bl_project_mode)()
        _PM_TOUCHED.clear()
        unset = []
        for backend in ("unreal", "blender"):
            state = _PM.get(backend)
            if _GENERATIVE and state and state.get("mode") == "unset" and (backend, state["project"]) not in _PM_NOTED:
                _PM_NOTED.add((backend, state["project"]))
                unset.append(_project_mode_label(backend, state))
        changed = _sync_gen_visibility()
    finally:
        _PM_PROBING = False
    return (_PROJECT_MODE_NOTE % " and ".join(unset) if unset else None), changed


# PICTURES ARE IMAGE BLOCKS (docs/00_ARCHITECTURE.md "Pictures"). A model sees a picture only through an MCP image content
# block: 22 headless runs got 0 of them, because a preview was base64 inside text, or withheld for its size, and a capture
# was a path that a client with no file tool cannot open. _with_pictures is the one place results become blocks, for a
# direct call and for mif_call alike. Which tools, and where each one's reply names its files, measured off each reply:
_PICTURE_TOOLS = {
    "bl_render_still": ("image",), "bl_render_viewport": ("files",), "bl_preview_garment": ("views.file",),
    "capture_camera": ("path",), "capture_viewport": ("file",), "capture_level_map": ("file",),
    "preview_animation": ("frames.file",), "render_thumbnail": ("pngPath",), "preview_widget": ("path",),
    "preview_composite_widget": ("path",), "render_graph_image": ("path",), "mif_view_image": ("file",),
}
_IMAGE_MODE = contextvars.ContextVar("mif_image_mode", default=None)    # None: a handler called outside call_tool
_IMAGE_DEFAULT_PX = 1024              # mif_media.image says why, and fits to these
_IMAGE_HARD_PX = 1568
_IMAGE_WIRE_BYTES = 16_000_000        # what bl_render_still asks the add-on to allow; a 1568 px PNG is well inside it
_IMAGES_PER_CALL = 4                  # preview_animation's frames and a garment's views can be many; the rest stay paths


def _picture_files(value, path):
    """The strings at a dotted path; a path through a list reads each row (_rc_pick's rule)."""
    if isinstance(value, list):
        return [f for row in value for f in _picture_files(row, path)]
    if not path:
        return [value] if isinstance(value, str) and value else []
    head, _, rest = path.partition(".")
    return _picture_files(value.get(head), rest) if isinstance(value, dict) else []


def _with_pictures(name, arguments, blocks):
    """blocks with a picture tool's image blocks added and its base64 out of the text. Anything unexpected is said in
    imageNote, beside the path that is already in the reply: a picture that cannot be shown must not fail a call that worked."""
    tool = arguments.get("tool") if name == "mif_call" else name
    specs = _PICTURE_TOOLS.get(tool.strip()) if isinstance(tool, str) else None
    mode = _IMAGE_MODE.get()
    first = next((b for b in blocks if getattr(b, "type", None) == "text"), None)
    if specs is None or mode == "data" or first is None or first.text[:1] != "{":
        return blocks
    try:
        reply = json.loads(first.text)
    except ValueError:
        return blocks
    limits = reply.pop("_imageLimits", None) or [None, None]
    inline = reply.get("image") if tool == "bl_render_still" else None
    if tool == "bl_render_still":
        files = []
    else:
        files = [f for spec in specs for f in _picture_files(reply, spec)]
    if reply.get("ok") is False or "error" in reply or mode is False or not (files or isinstance(inline, str)):
        if mode is False:
            reply.pop("image", None)
        return [first.model_copy(update={"text": json.dumps(reply)}) if b is first else b for b in blocks]
    import importlib
    from mcp.types import ImageContent
    here = os.path.dirname(_SERVER_PATH)
    if here not in sys.path:
        sys.path.insert(0, here)
    image = importlib.import_module("mif_media.image")
    gif = importlib.import_module("mif_media.gif")
    shown, sent, notes = [], [], []
    if isinstance(inline, str):
        source = reply.get("filePath") or "the render"
        for key in ("image", "imageError", "imageOmitted", "imageBytes", "estimatedBase64Bytes", "imageFormat", "imageWidth",
                    "imageHeight", "previewNote", "downscaledFrom"):
            reply.pop(key, None)
        try:
            shown.append((source, image.fit(base64.b64decode(inline), source, limits[0], limits[1])))
        except (gif.Refusal, ValueError, OSError, zlib.error) as exc:
            notes.append(str(exc))
    for path in files[:_IMAGES_PER_CALL]:
        try:
            shown.append((path, image.fit_file(path, limits[0], limits[1])))
        except (gif.Refusal, ValueError, OSError, zlib.error) as exc:
            notes.append(str(exc))
    for path, fitted in shown:
        sent.append(ImageContent(type="image", data=image.as_base64(fitted), mimeType=fitted["mime"]))
    if shown:
        reply["imageBlocks"] = [{"file": p, "width": f["width"], "height": f["height"], "bytes": len(f["data"])}
                                for p, f in shown]
    if len(files) > _IMAGES_PER_CALL:
        notes.append("%d more are not shown; mif_view_image takes any one of them" % (len(files) - _IMAGES_PER_CALL))
    if notes:
        reply["imageNote"] = "; ".join(notes)
        if tool == "mif_view_image" and not shown:
            reply = {"ok": False, "error": reply["imageNote"]}
    return [first.model_copy(update={"text": json.dumps(reply)}) if b is first else b for b in blocks] + sent


def _note_blocks(note):
    if not note:
        return []
    from mcp.types import TextContent
    return [TextContent(type="text", text=note)]


def _generative_report() -> dict:
    """What self_audit adds: whether the gen tools may run on this side, and each switch's state that is known."""
    modes = {"server": "on" if _GENERATIVE else "off (MIF_GENERATIVE)"}
    for backend, key in (("unreal", "unrealProject"), ("blender", "blenderFile")):
        state = _unreal_mode_state() if backend == "unreal" else _PM.get(backend)
        if state:
            modes[key] = state.get("mode")
    return {"generativeAllowed": _gen_block() is None, "generativeMode": modes}


@mcp.tool()
def bl_call_operator(id: str, properties: dict = None, context: dict = None, describe: bool = None, undo: bool = None) -> dict:
    "Run a registered Blender operator by id ('mesh.primitive_monkey_add'), each property checked against the operator's own ranges and enum items first; describe lists its properties and runs nothing; undo true runs it as the UI does, into the undo stack and Blender's operator log. Gated by Allow run_python, like bl_run_python."
    return _blender("call_operator", id=id, properties=properties, context=context, describe=describe, undo=undo)


@mcp.tool()
def bl_list_recent_operators(limit: int = None) -> dict:
    "Blender's log of the last operators run (WindowManager.operators: id, name, properties) and the calls made through bl_call_operator this session. An operator enters the log when run from the UI or by bl_call_operator with undo true. Gated by Allow run_python."
    return _blender("list_recent_operators", limit=limit)


@mcp.tool()
def bl_enable_addon(module: str = None, file: str = None, enable: bool = None, persistent: bool = None,
                    uninstall: bool = None, repo: str = None) -> dict:
    "Enable or disable a Blender add-on by module, or install a .py, .zip or 4.2+ extension package (into repo, default user_default) and enable it; uninstall removes it. Replies its name, version and classes. Gated by Allow run_python."
    return _blender("enable_addon", module=module, file=file, enable=enable, persistent=persistent, uninstall=uninstall,
                    repo=repo)

# ---------------------------------------------------------------------------
# GAP REPORTS, DRAFTED IN A SESSION (2026-10-11, spec "File a gap report from inside a session").
#
# Reports reach the report loop only when a person files them (docs/12): the issue template on the public tracker, or
# the site's form. This keeps the last failing calls - each backend's own refusals, not a timeout or an unreachable
# editor - and mif_report_draft turns one into the template's exact shape for the PERSON to read and file. Nothing is
# sent from here, ever: the answer is text and two addresses. What is left out is decided here, not by the reporter:
# the token, any key named like a secret, and this machine's own folders (the project, the home folder, temp), each
# replaced by a placeholder - a /Game/ path stays, because the loop rewrites those into scratch before it replays.
# ---------------------------------------------------------------------------
_FAILED_CALLS = []                   # newest last: {at, backend, endpoint, payload, error, extra}
_FAILED_KEEP = 20
# THIS SERVER'S OWN READS (the handshake, the project's mode and paths), never a call a person made: not kept.
_NOT_REPORTED = ("ping", "self_audit", "project_mode", "project_paths")


def _note_failed_call(backend, endpoint, payload, data):
    """Every backend refusal passes here (from _post and _blender): kept for mif_report_draft, newest last."""
    if not isinstance(data, dict) or not (data.get("ok") is False or data.get("error")):
        return
    if endpoint in _NOT_REPORTED:
        return
    _FAILED_CALLS.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": backend, "endpoint": endpoint,
                          "payload": dict(payload or {}), "error": str(data.get("error") or "")[:2000],
                          "errorCode": data.get("errorCode")})
    del _FAILED_CALLS[:-_FAILED_KEEP]

# THE ROADMAP (1.3, docs/30_ROADMAP.md). "Help newbs plan every step to make their mod happen" - and the shallow version
# of that is worse than none: a plan that says "save the map" to a bridge in scratch mode, or names a call that refuses on
# the user's engine, walks a newcomer into a failure they cannot tell from their own mistake. So the catalog
# (roadmap.json) only says WHICH calls a step is made of; whether they work here is derived on every call - from the
# Source (engine refusals, plugin needs, cooked handling, crash refusals: mif_roadmap_ground.scan_source), the shipped
# pack rules, the live editor (self_audit's registry, write gate and undo bucket, enabled plugins, cooked content, the
# generative setting, the catalog's read-only checks) and, in a checkout, the ledger and the verification board. No
# model is called: the assistant plans; this grounds. Nothing is written.
_ROADMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roadmap.json")


def _roadmap_registry():
    """{tool name: description} for every MifBridge tool, registered or held back by the pack selection."""
    out = {}
    try:
        for t in mcp._tool_manager.list_tools():
            out[t.name] = t.description or ""
    except Exception:                                               # noqa: BLE001
        pass
    for name, fn in _UNLOADED.items():
        out.setdefault(name, (getattr(fn, "__doc__", "") or ""))
    return out


# THE PROJECT'S CHOSEN ROADMAP lives in the project (decided 2026-11-02), with its other settings: MifBridgeProject.json beside
# the .uproject, on every scene of a .blend. So it survives a restart and moves with the project, and there is no other file to
# lose or to disagree with. Everything else about a step is still derived on every call. The one thing stored about
# progress is what no check can see: the steps the user says were done by hand.
def _roadmap_plan_of(stored, goals):
    """(steps, why not) for a stored roadmap, which came from a file on disk: a catalog goal's steps, or the plan saved with it."""
    import mif_roadmap_ground as RG
    if not isinstance(stored, dict):
        return None, "it is not an object"
    if stored.get("steps"):
        return RG.normalize_steps(stored["steps"])
    if stored.get("goal") in goals:
        return [dict(s) for s in goals[stored["goal"]]["steps"]], None
    return None, "this server's catalog has no goal %r" % (stored.get("goal"),)


def _roadmap_request(value, current, goals):
    """(the roadmap to store, {} to clear; error). value is mif_project_mode's roadmap: goal words or a catalog id, 'none', or
    {goal, steps, done, undone}. current is what the project holds, so done and undone keep the rest."""
    import mif_roadmap_ground as RG
    if isinstance(value, str):
        value = {} if value.strip().lower() == "none" else {"goal": value}
    if not isinstance(value, dict):
        return None, "roadmap must be goal words, 'none', or {goal, steps, done, undone}, not %r" % (value,)
    if not value:
        return {}, None
    unknown = sorted(set(value) - {"goal", "steps", "done", "undone"})
    if unknown:
        return None, "roadmap takes goal, steps, done and undone, not %s" % ", ".join(unknown)
    doc = dict(current) if isinstance(current, dict) else {}
    if "steps" in value:
        plan, err = RG.normalize_steps(value["steps"])
        if err:
            return None, err
        registry = _roadmap_registry()
        if not any(t in registry for s in plan for t in s.get("tools") or []):
            return None, "no step names a MifBridge tool - find_tools('<words>') finds them"
        doc = {"goal": None, "title": str(value.get("goal") or "") or None, "steps": plan}
    elif value.get("goal") is not None:
        words = str(value["goal"]).strip()
        scored = RG.match_goal(words, goals)
        gid = words if words in goals else RG.goal_matched(scored, words)
        if gid is None:
            return None, ("no catalog plan fits %r. The closest: %s. Name one of those, or send your own steps."
                          % (words, ", ".join(i for _s, i in scored[:3]) or "none"))
        doc = dict(doc, goal=gid, title=goals[gid].get("title")) if doc.get("goal") == gid else {
            "goal": gid, "title": goals[gid].get("title")}
    elif not doc:
        return None, "done and undone mark steps of a chosen roadmap, and this project has none yet"
    plan, why = _roadmap_plan_of(doc, goals)
    if plan is None:
        return None, "the roadmap this project holds cannot be read (%s); choose one again" % why
    ids = [s.get("id") for s in plan]
    for key in ("done", "undone"):
        if not isinstance(value.get(key, []), list):
            return None, "%s must be a list of step ids: %s" % (key, ", ".join(map(str, ids)))
    named = list(value.get("done") or []) + list(value.get("undone") or [])
    missing = [i for i in named if i not in ids]
    if missing:
        return None, "this roadmap has no step %s. Its steps: %s" % (", ".join(map(str, missing)), ", ".join(map(str, ids)))
    marked = set(doc.get("doneByHand") or []) | set(value.get("done") or [])
    marked -= set(value.get("undone") or [])
    doc["doneByHand"] = [i for i in ids if i in marked]
    if value.get("goal") is not None or value.get("steps"):
        doc["chosenAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc["chosenBy"] = _agent_tag() or None
    return doc, None


@mcp.tool()
def mif_project_mode(mode: str = "", project: str = "all", change_cap: int = None, roadmap: Any = None) -> dict:
    "Read or record this project's settings - ask the user. mode: generative | non-generative. change_cap: items one call may change before it previews (0 off). roadmap: the plan it follows. Empty reads. project: all|unreal|blender."
    # SENT WHOLE, under the 240-character budget: it is called about once per project, from the note that asks for it,
    # so the detail - where each mode is stored, what each one means - is in tool_help.json for mif_help.
    want = (mode or "").strip().lower()
    which = (project or "all").strip().lower()
    if which not in ("all", "unreal", "blender"):
        return {"ok": False, "error": "project must be all, unreal or blender, not %r. Nothing was changed." % project}
    if want and want not in _MODES:
        return {"ok": False, "error": "mode must be one of %s (or empty to read it), not %r. Nothing was changed."
                                      % (", ".join(_MODES), mode)}
    # THE CHANGE CAP (docs/15 section 6): a whole number from 0 (off) up. bool is an int in Python, so it is named.
    if change_cap is not None and (isinstance(change_cap, bool) or not isinstance(change_cap, int) or change_cap < 0):
        return {"ok": False, "error": "change_cap must be a whole number from 0 (off) up, not %r. Nothing was changed."
                                      % (change_cap,)}
    writing = bool(want) or change_cap is not None or roadmap is not None
    if which == "all":
        backends = [b for b in ("unreal", "blender") if _PM.get(b) is not None] or ["unreal", "blender"]
    else:
        backends = [which]
    # THE ROADMAP IS WORKED OUT FOR EVERY PROJECT BEFORE ANY IS WRITTEN, so a refusal changes nothing on either. A project that
    # does not answer is left out here and reported unreachable below, as a mode is.
    roadmaps = {}
    if roadmap is not None:
        import mif_roadmap_ground as RG
        goals, err = RG.load_catalog(_ROADMAP_PATH)
        if err:
            return {"ok": False, "error": err}
        for backend in backends:
            state = (_ue_project_mode if backend == "unreal" else _bl_project_mode)(refresh=True)
            roadmaps[backend], err = _roadmap_request(roadmap, (state or {}).get("roadmap"), goals)
            if err:
                return {"ok": False, "error": err + ". Nothing was changed."}
    out = {"ok": True}
    for backend in backends:
        if writing and backend == "unreal":
            res = _ue_set_mode(want or None, change_cap, roadmaps.get(backend))
        elif writing:
            res = _bl_set_mode(want or None, change_cap, roadmaps.get(backend))
        else:
            res = (_ue_project_mode if backend == "unreal" else _bl_project_mode)(refresh=True)
            res = dict(res, ok=True) if res else {"ok": False, "error": "no %s editor answered" % backend}
        if backend == "blender" and isinstance(res, dict) and res.get("ok") is not False:
            res = dict(_PM.get("blender") or {}, ok=True, generativeAllowedByPreference=res.get("allowGenerative"),
                       changeCap=res.get("changeCap"))
        key = "unrealProject" if backend == "unreal" else "blenderFile"
        out[key] = {k: v for k, v in (res or {}).items() if k != "ok" and v is not None}
        if not isinstance(res, dict) or res.get("ok") is False:
            out[key]["reachable"] = False
            if which != "all":
                out["ok"] = False
                out["error"] = "%s: %s" % (backend, (res or {}).get("error"))
    if which == "all" and not any(isinstance(v, dict) and v.get("mode") for k, v in out.items() if k != "ok"):
        out["ok"] = False
        out["error"] = "neither an Unreal editor nor Blender answered, so there is no project to %s." % (
            "record the setting for" if writing else "read")
    out.update(_generative_report())
    if want and out["ok"]:
        out["note"] = ("recorded. MIF_GENERATIVE=off on the server and the add-on preference 'Allow generative tools "
                       "(ComfyUI)' still switch the bl_gen_* tools off whatever a project says.")
    elif change_cap is not None and out["ok"]:
        out["note"] = ("recorded. A project's change cap overrides MIF_CHANGE_CAP in the editor's or Blender's "
                       "environment; self_audit and bl_status show the cap in force.")
    elif writing and out["ok"]:
        out["note"] = ("recorded in the project. mif_roadmap with no words reads it back. On Blender it is on the .blend, "
                       "so save the file to keep it.")
    return out


@mcp.tool()
def bl_gen_status(host: str = "", refresh: bool = True, timeout: int = None) -> dict:
    "Is the local generator usable, and what is installed. FIRST call before any bl_gen_* work. License: Hunyuan3D-2 is not licensed in the EU, UK or South Korea; FLUX.1 [dev] is non-commercial."
    # `refresh` is the addon's own default (ops_gen.py:427 reads it as True), so the tool's default
    # matches rather than quietly answering a different question from a direct call. Exposed here
    # because the endpoint had grown the parameter and NO MCP tool could send it - an agent driving
    # through MCP could not express "ask ComfyUI again", which is the whole point of the parameter:
    # installing a custom node requires restarting ComfyUI, and a cached schema taken before that
    # restart reports every new node as missing.
    #
    # `timeout` THE SAME WAY, 2026-09-28 (the 2026-09-21 pack-rule review): op_gen_status reads it - 20 s by default, short
    # on purpose so "not reachable" does not look like a hang - and no tool could send it, so a slow cold enumerate of a
    # large node install could not be given longer. None leaves the add-on's default; the socket waits past whatever is
    # asked, as bl_gen_image's does, so the transport never gives up before the probe.
    return _blender("gen_status", host=host or None, refresh=refresh, timeout=timeout,
                    _timeout=(float(timeout) + 60.0) if timeout is not None else None)


@mcp.tool()
def bl_gen_image(prompt: str, seed: int = 0, variant: str = "schnell", width: int = 1024,
                 height: int = 1024, steps: int = None, host: str = "",
                 timeout: int = 600) -> dict:
    "Prompt -> reference image via Flux.1, left in ComfyUI's output folder. Stage ONE; bl_gen_mesh takes the returned image name. License: FLUX.1 [schnell] is Apache-2.0, FLUX.1 [dev] is non-commercial."
    return _blender("gen_image", prompt=prompt, seed=seed or None, variant=variant or None,
                    width=width, height=height, steps=steps, host=host or None,
                    timeout=timeout, _timeout=float(timeout) + 60.0)


@mcp.tool()
def bl_gen_mesh(image: str = "", image_path: str = "", prefix: str = "MifGen/mesh",
                name: str = "", seed: int = 0, steps: int = 30, octree: int = 512,
                guidance: float = 5.0, import_result: bool = True, host: str = "",
                timeout: int = 1800) -> dict:
    "Reference image -> untextured mesh via the Hunyuan3D-2 shape DiT. Stage TWO. Takes image (a ComfyUI image ref) or image_path (a local file). License: Hunyuan3D-2 is not licensed in the EU, UK or South Korea."
    return _blender("gen_mesh", image=image or None, imagePath=image_path or None,
                    prefix=prefix or None, name=name or None, seed=seed or None,
                    steps=steps, octree=octree, guidance=guidance,
                    importResult=import_result, host=host or None,
                    timeout=timeout, _timeout=float(timeout) + 60.0)


@mcp.tool()
def bl_gen_texture(mesh_path: str, image: str = "", image_path: str = "",
                   prefix: str = "MifGen/mesh", name: str = "", seed: int = 0,
                   steps: int = 15, view_size: int = 512, import_result: bool = True,
                   host: str = "", timeout: int = 2400) -> dict:
    "Existing mesh + reference image -> PBR textures baked on, via the Hunyuan3D-2 paint path. Stage THREE. License: Hunyuan3D-2 is not licensed in the EU, UK or South Korea."
    return _blender("gen_texture", meshPath=mesh_path, image=image or None,
                    imagePath=image_path or None, prefix=prefix or None, name=name or None,
                    seed=seed or None, steps=steps, viewSize=view_size,
                    importResult=import_result, host=host or None,
                    timeout=timeout, _timeout=float(timeout) + 60.0)


@mcp.tool()
def bl_gen_asset(prompt: str, name: str = "", seed: int = 0, variant: str = "schnell",
                 texture: bool = True, width: int = 1024, height: int = 1024,
                 shape_steps: int = 30, texture_steps: int = 15, octree: int = 512,
                 guidance: float = 5.0, import_result: bool = True, host: str = "",
                 timeout: int = 3600) -> dict:
    "Prompt -> reference image -> mesh -> PBR texture -> imported into the scene. THE ONE CALL that produces something usable. License: Hunyuan3D-2 is not licensed in the EU, UK or South Korea; FLUX.1 [dev] is non-commercial."
    return _blender("gen_asset", prompt=prompt, name=name or None, seed=seed or None,
                    variant=variant or None, texture=texture, width=width, height=height,
                    shapeSteps=shape_steps, textureSteps=texture_steps, octree=octree,
                    guidance=guidance, importResult=import_result, host=host or None,
                    timeout=timeout, _timeout=float(timeout) + 60.0)

_PACKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs.json")
_PACKS_CACHE = None
_UNLOADED = {}      # tool name -> its function, for tools a pack selection held back

_PACK_RULES = (
    "Rules for every MifBridge session:\n"
    "- Failure is the PRESENCE of an `error` key, never the absence of `ok`.\n"
    "- Call mif_help('<tool>') when a description does not tell you enough, before a destructive or engine-level "
    "operation, or after a refusal you do not understand: several tools guard asserts that would terminate the "
    "editor, and the reason is in the help.\n"
    "- describe_endpoint('<name>') is the authority on an endpoint's real parameters.\n"
    "- Call self_audit first: it names the engine and the write mode. Saves, Play-In-Editor, cooks and "
    "console commands are refused unless a person raises the mode - do not try to route around it.\n"
    "- Destructive calls need confirm=true. Say what you are about to delete or rename before you do.\n"
    "- A write is done when its read-back says so. Report what the editor returned, not what you asked for.\n")

_BLENDER_PACK_RULES = (
    "Rules for every MifBlender session:\n"
    "- Failure is the PRESENCE of an `error` key, never the absence of `ok`.\n"
    "- Call bl_status first: it names the Blender that answered, its version and its open file. bl_scene_info and "
    "bl_list_objects say what is already there - read before you write.\n"
    "- bl_describe_op('<op>') is the authority on an op's real parameters: never guess a name or a shape.\n"
    "- A write is done when its read-back says so. Report what Blender returned, not what you asked for.\n"
    "- Destructive ops (bl_clear_scene, bl_delete_object, bl_purge_orphans) - say what you are about to remove first.\n"
    "- self_audit reports both backends; a Blender that is not answering is named there.\n")


def _packs():
    global _PACKS_CACHE
    if _PACKS_CACHE is None:
        try:
            with open(_PACKS_PATH, "r", encoding="utf-8") as fh:
                _PACKS_CACHE = (json.load(fh) or {}).get("packs") or {}
        except Exception as exc:  # noqa: BLE001
            _PACKS_CACHE = {"__error__": str(exc)}
    return _PACKS_CACHE


_ALIASES_CACHE = None


def _pack_aliases():
    """Selection names that stand for several packs (packs.json "aliases"). `blender` was one 194-tool pack until
    2026-09-21 and now names every Blender pack, so a selection or an enable that used it loads what it always did."""
    global _ALIASES_CACHE
    if _ALIASES_CACHE is None:
        try:
            with open(_PACKS_PATH, "r", encoding="utf-8") as fh:
                _ALIASES_CACHE = (json.load(fh) or {}).get("aliases") or {}
        except Exception:  # noqa: BLE001
            _ALIASES_CACHE = {}
    return _ALIASES_CACHE


def _expand_packs(names):
    """What a list of pack names MEANS: aliases expanded, every pack's `requires` added (a Blender workflow pack
    needs blender_core - scene_info, object_info, transform_object). Returns (packs in packs.json order, the names
    that are neither a pack nor an alias)."""
    packs, aliases = _packs(), _pack_aliases()
    got, unknown, todo = set(), set(), [str(n).strip().lower() for n in names if str(n).strip()]
    while todo:
        n = todo.pop()
        if n in aliases:
            todo.extend(aliases[n])
        elif n in packs and n != "__error__":
            if n not in got:
                got.add(n)
                todo.extend(packs[n].get("requires") or [])
        else:
            unknown.add(n)
    return [pid for pid in packs if pid in got], sorted(unknown)


# ---------------------------------------------------------------------------
# PACK RULES - what an agent working one pack is told. Added 2026-09-21 (the request: "split up a math.md/ledger per
# each and train each one").
#
# Each rule is a mistake the confirmed-facts ledger shows that pack's work makes easily - a silent trap, a unit, an
# order that matters, a field that does not prove what it seems to. They are distilled per pack in pack_rules.py
# (dev-only; it keeps the ledger citations) and generated into packs.json as {"rule", "tools"}, so this server only
# reads them. They reach an agent four ways, because no single one reaches every agent:
#   - the pack's prompt (mif-pack-<id>), when a person invokes it;
#   - mif_enable_pack's response, for each pack it loads;
#   - mif_call's response the first time it registers a held-back tool - the path for clients that never refresh
#     their tool list, which never call mif_enable_pack at all and were the ones the UE split lost;
#   - mif_help: '<pack>' gives the pack's rules, '<tool>' adds every rule that names that tool.
# _RULES_SENT keeps the two automatic channels to once per pack per server process (one stdio session).
# ---------------------------------------------------------------------------
_RULES_SENT = set()


def _pack_rules(pid):
    """[{"rule": text, "tools": [names]}] for one pack - [] for a pack without rules, or an unknown one."""
    p = _packs().get(pid) if pid != "__error__" else None
    return [r for r in ((p or {}).get("rules") or []) if isinstance(r, dict) and r.get("rule")]


def _tool_pack(name):
    """The pack a tool belongs to (packs.json places every tool in exactly one), or None."""
    for pid, p in _packs().items():
        if pid != "__error__" and name in (p.get("tools") or []):
            return pid
    return None


def _rules_payload(pids):
    """{pack: [rule texts]} for the packs in `pids` whose rules this session has not been handed yet - and marks
    them handed. Empty when there is nothing new, so a caller adds the key only when it says something. A rule
    mif_call already handed over on its own is not repeated."""
    out = {}
    for pid in pids:
        texts = [r["rule"] for r in _pack_rules(pid) if (pid, r["rule"]) not in _RULE_SENT]
        if texts and pid not in _RULES_SENT:
            out[pid] = texts
        _RULES_SENT.add(pid)
    return out


# RULE BY RULE THROUGH mif_call, decided after benchmark round six (docs/TOKEN_OPTIMIZATION.md 2.10). The first
# tool mif_call registered from a pack carried the pack's whole list: each UMG run took 22 rules, ~6,100 characters
# re-read on ~6 requests, and 5 of them named a tool it called - the 9 ui rules arrived with add_tree_widget and none
# is about a widget tree. Now a tool's first call carries the rules that name it, and a pack's first call those that
# name no tool; each once a session, and none from a pack mif_enable_pack already handed over whole.
_RULE_SENT = set()      # (pack, rule text)


def _tool_rules_payload(tool, pids):
    out = {}
    for pid in pids:
        if pid in _RULES_SENT:
            continue
        for r in _pack_rules(pid):
            key = (pid, r["rule"])
            if key not in _RULE_SENT and (tool in (r.get("tools") or []) or not r.get("tools")):
                out.setdefault(pid, []).append(r["rule"])
                _RULE_SENT.add(key)
    return out


def _rules_block(pid, heading):
    rules = _pack_rules(pid)
    return (heading + "\n" + "".join("- %s\n" % r["rule"] for r in rules)) if rules else ""


def _with_pack_rules(name, out):
    """mif_help's answer for a tool, plus every pack rule that names it - whichever pack's list it is in - and a
    pointer to the tool's own pack's full set. The channel that reaches a session with every tool loaded, which
    never enables a pack and so is never handed a pack's rules any other way."""
    if _PARAM_ALIASES.get(name):                     # the old spellings a call may still use (_renamed_params)
        out["deprecatedParams"] = dict(_PARAM_ALIASES[name])
    if _pair_aliases(name):                          # the other spellings of a cross-tool concept (_SPELLING_PAIRS)
        out["alsoAccepted"] = dict(_pair_aliases(name))
    naming =[r["rule"] for pid in _packs() if pid != "__error__" for r in _pack_rules(pid)
              if name in (r.get("tools") or [])]
    if naming:
        out["packRules"] = naming
    home = _tool_pack(name)
    if home and _pack_rules(home):
        out["pack"] = home
        out["packRulesNote"] = ("%s is in the %s pack, which has %d rule(s) - mif_help('%s') lists them."
                                % (name, home, len(_pack_rules(home)), home))
    return out


# THE DEFAULT IS core, decided on 2026-09-23. the benchmark's core arm (33 runs on 11baf1c7, Claude Code's tool
# search, 0 wrong) cost 227,116 tokens a task against 259,923 for all 711 tools - cheaper on 9 of 11 tasks - and a
# client that loads every schema up front pays ~8k tokens a request for core against ~147k for all
# (docs/TOKEN_OPTIMIZATION.md 3.5). The rest stay reachable: find_tools names a tool's pack, mif_enable_pack loads it
# and mif_call calls any tool by name. MIF_TOOL_PACKS=all loads everything, as the default did until then; an EMPTY
# value means the default, as unset does.
#
# THE DEFAULT IS "lean" SINCE LATER THE SAME DAY, decided again (docs/TOKEN_OPTIMIZATION.md 2.8): only the
# always-loaded tools are registered - find_tools, mif_call and the Python tool (the two became one run_python in 2.9)
# - and every other tool, core's included, is held back where find_tools finds it and mif_call calls it. Measured with Claude Code's own token
# count (ledger CC26): once ANY MCP tool is deferred, Claude Code sends its ToolSearch machinery on every request - 632
# tokens for the first deferred tool, 16 for each after - so core's 16 deferred tools cost 872 tokens a request
# (4,918 against 4,046). MIF_TOOL_PACKS=core loads core as before; "lean" can be named too, alone.
_DEFAULT_PACKS = "lean"


def _apply_tool_packs(selection=None):
    """Keep only the selected packs' tools registered (core with any named pack; the lean default keeps only the
    always-loaded tools). Returns what it did, as text."""
    raw = (os.environ.get("MIF_TOOL_PACKS", "") if selection is None else selection).strip().lower()
    raw = raw or _DEFAULT_PACKS
    if raw == "all":
        return "all tools loaded"
    packs = _packs()
    if packs.get("__error__"):
        return "packs.json unreadable (%s) - all tools left loaded" % packs["__error__"]
    if raw == "lean":
        known, unknown, lean, keep = [], [], False, set(_ALWAYS_LOAD)
    else:
        known, unknown = _expand_packs([p for p in raw.split(",")] + ["core"])
        keep = set()
        for p in known:
            keep.update(packs[p].get("tools") or [])
        lean = _blender_only(known)
        if lean:
            keep -= _core_unreal_tools()
    registry = getattr(mcp, "_tool_manager")._tools
    for name in list(registry):
        fn = globals().get(name) or _EXT_TOOL_FNS.get(name)       # an extension's tool is held back like a pack's
        if name not in keep and callable(fn):
            _UNLOADED[name] = fn
            mcp.remove_tool(name)
    if raw == "lean":
        return ("the lean default: %s loaded; %d held back - find_tools finds them, mif_call calls them, "
                "MIF_TOOL_PACKS=core loads core" % (", ".join(sorted(registry)), len(_UNLOADED)))
    return "packs %s loaded (%d tools); %d held back%s%s" % (
        ",".join(sorted(known)), len(registry), len(_UNLOADED),
        "; a Blender-only selection, so only core's backend-neutral tools" if lean else "",
        "; unknown pack name(s) ignored: %s" % ", ".join(unknown) if unknown else "")


# A BLENDER SESSION DOES NOT CARRY UNREAL'S CORE TOOLS. Thirteen of core's eighteen talk to Unreal only (find_assets,
# get/set_property, batch, the saves, ...), and a selection naming nothing but Blender packs paid ~2,300 tokens for
# them on every request - ~17k a Blender task (the benchmark, 2026-09-22, docs/TOKEN_OPTIMIZATION.md). It was decided then
# that such a selection keeps only core's `neutral` tools, whether or not it also names core, since core is loaded
# either way. The rest stay reachable: find_tools names their pack, mif_call reaches them, and enabling core or any
# pack that is not Blender's brings them back (mif_enable_pack). A selection with any other pack is unchanged.
def _core_unreal_tools() -> set:
    core = _packs().get("core") or {}
    neutral = set(core.get("neutral") or [])
    # AN OLD packs.json HAS NO `neutral` LIST, and subtracting an empty one would drop the WHOLE of core - including
    # the tools that find and load everything else. Without the list nothing is held back.
    return (set(core.get("tools") or []) - neutral) if neutral else set()


def _blender_only(pack_ids) -> bool:
    """True when every pack but core is a Blender pack, and there is at least one."""
    packs = _packs()
    rest = [p for p in pack_ids if p != "core"]
    return bool(rest) and all((packs.get(p) or {}).get("backend") == "blender" for p in rest)


def _register_pack_prompts():
    add = getattr(mcp, "prompt", None)
    packs = _packs()
    if add is None or packs.get("__error__"):
        return 0
    count = 0
    for pid, p in packs.items():
        if pid == "core":
            continue
        title, desc = p.get("title") or pid, p.get("description") or ""
        blender = p.get("backend") == "blender"
        where = "in Blender through MifBlender" if blender else "in Unreal Engine through MifBridge"
        needs = p.get("requires") or []
        also = (" That also loads %s, which this pack needs." % ", ".join(needs)) if needs else ""
        # THE PACK'S OWN RULES FOLLOW THE SHARED ONES, after the rules of what it loads with it (a Blender workflow
        # pack brings blender_core) and, for an Unreal pack, core's - the tools every Unreal session has.
        own = "".join(_rules_block(q, "Rules for %s, which this pack loads with it:" % q) for q in needs)
        if not blender:
            own += _rules_block("core", "Rules for the core tools, in every Unreal session:")
        own += _rules_block(pid, "Rules for this pack (%s):" % pid)

        def make(pid=pid, title=title, desc=desc, where=where, also=also, rules=(
                (_BLENDER_PACK_RULES if blender else _PACK_RULES) + ("\n" + own if own else ""))):
            def pack_prompt() -> str:
                return ("You are working %s, on %s. %s\n\n"
                        "First: if these tools are not in your tool list, call mif_enable_pack('%s') - through "
                        "mif_call('mif_enable_pack', {'pack': '%s'}) when mif_enable_pack is not in it either, as "
                        "on the lean default.%s "
                        "That registers them server-side and sends tools/list_changed, but some clients never "
                        "re-read their list mid-session - so if they still do not appear, call them through "
                        "mif_call(tool, args), which reaches every MifBridge tool without a refresh. A "
                        "restart only helps with MIF_TOOL_PACKS including '%s': restarting without changing "
                        "it starts the same selection again. Then call self_audit.\n\n%s"
                        % (where, title, desc, pid, pid, also, pid, rules))
            return pack_prompt
        add(name="mif-pack-%s" % pid,
            description="Work on %s with MifBridge - loads the %s pack and its rules." % (title, pid))(make())
        count += 1
    # AN ALIAS KEEPS ITS PROMPT: /mif-pack-blender existed before the split and still loads all of Blender.
    for alias, members in _pack_aliases().items():
        if alias in packs:
            continue
        # The alias's BASE is the member the others require (blender_core): its rules hold for every pack the alias
        # loads, so they are printed; each workflow pack's own rules are pointed at rather than all nine inlined.
        base = [m for m in members if any(m in ((packs.get(o) or {}).get("requires") or []) for o in members)]
        base_rules = "".join(_rules_block(m, "Rules for %s, in every Blender session:" % m) for m in base)

        def make_alias(alias=alias, members=tuple(members), base_rules=base_rules):
            def alias_prompt() -> str:
                return ("You are working in Blender through MifBlender, with every Blender pack: %s.\n\n"
                        "First: if the bl_* tools are not in your tool list, call mif_enable_pack('%s') - it loads "
                        "all of them; mif_call('mif_enable_pack', {'pack': '%s'}) reaches it when it is not in your "
                        "list either. That sends tools/list_changed, but some clients never re-read their list "
                        "mid-session - if the tools still do not appear, call them through mif_call(tool, args), "
                        "which needs no refresh. For a lighter session load blender_core and the one workflow pack "
                        "you need instead. Then call bl_status.\n\n%s%s"
                        "\nEach workflow pack has rules of its own: mif_help('<pack>') gives them, and "
                        "mif_enable_pack hands them over as it loads the pack.\n"
                        % (", ".join(members), alias, alias, _BLENDER_PACK_RULES,
                           ("\n" + base_rules) if base_rules else ""))
            return alias_prompt
        add(name="mif-pack-%s" % alias,
            description="Work in Blender with MifBridge - loads every Blender pack (%s)." % ", ".join(members))(
            make_alias())
        count += 1
    return count


# ---------------------------------------------------------------------------
# EXTENSIONS - third-party tools and tool packs from local folders (docs/32_EXTENSIONS.md, 1.3).
#
# mifext.py finds, checks and imports each installed extension; this side registers its tools and makes its pack one of
# the server's packs, so find_tools finds them, mif_call calls them, mif_enable_pack and MIF_TOOL_PACKS load them, and
# the pack prompt and mif_help carry their rules - the machinery every built-in pack uses, not a second copy of it.
# Loaded in main() BEFORE _apply_tool_packs, so the lean default holds extension tools back like any other pack's, and
# never at import: the test suites import this module with a fake FastMCP and no extensions.
#
# An extension tool reaches the editor through api.call -> _post_forwarded, the same request path every tool uses, so
# the editor's safety gate, write modes, change cap, crash journal and parameter checks apply to it unchanged.
# ---------------------------------------------------------------------------
_EXT_TOOL_FNS = {}      # extension tool name -> its function (_apply_tool_packs holds these back like globals)
_EXT_STATE = {"ran": False, "loaded": [], "failed": [], "dirs": [], "disabled": False}


class _ExtTransport:
    """What mifext.Api's call and blender reach: this server's own request paths."""

    def call(self, endpoint, body):
        return _post_forwarded(endpoint, **body)

    def blender(self, op, params):
        # The op name comes from the extension, not this file, so no literal can name it and parity_check cannot hold it
        # to the add-on's table: the add-on's own reject_unknown refuses an unknown op or key. Named apart, as
        # _post_forwarded is, so that check keeps covering every literal _blender call.
        return _blender_forwarded(op, **params)


def _load_extensions():
    """Load every installed extension, register its tools and add its pack. Returns a log line."""
    import mifext       # stdlib only, beside this file
    registry = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
    taken = set(registry or {}) | set(_UNLOADED) | {n for n, f in globals().items() if callable(f)}
    rep = mifext.load_all(_ExtTransport(), taken)
    _EXT_STATE.update(ran=True, failed=list(rep["failed"]), dirs=list(rep["dirs"]), disabled=rep["disabled"], loaded=[])
    packs = _packs()
    for ext in rep["loaded"]:
        m = ext.manifest
        pack = m.get("pack") or {}
        try:
            for name, fn in sorted(ext.tools.items()):
                mcp.add_tool(fn, name=name, description=(fn.__doc__ or "").strip())
                _EXT_TOOL_FNS[name] = fn
        except Exception as exc:  # noqa: BLE001 - one extension's bad schema must not take the server down
            for name in ext.tools:
                _EXT_TOOL_FNS.pop(name, None)
                try:
                    mcp.remove_tool(name)
                except Exception:  # noqa: BLE001
                    pass
            _EXT_STATE["failed"].append({"folder": ext.folder, "name": ext.name,
                                         "error": "registering its tools failed: %s: %s" % (type(exc).__name__, exc)})
            continue
        if ext.tools and isinstance(packs, dict) and not packs.get("__error__"):
            packs[ext.pack_id] = {
                "title": pack.get("title") or ext.name,
                "description": "%s (extension %s %s, from %s)" % (m["description"], ext.name, m["version"], ext.folder),
                "backend": pack.get("backend") or "unreal",
                "tools": sorted(ext.tools),
                "rules": [{"rule": r, "tools": []} for r in pack.get("rules") or []],
                "extension": ext.name}
        _EXT_STATE["loaded"].append(ext.row())
    if rep["disabled"]:
        return "extensions: off (MIF_EXTENSIONS)"
    return "extensions: %d loaded (%s), %d failed, from %s" % (
        len(_EXT_STATE["loaded"]), ", ".join(r["name"] for r in _EXT_STATE["loaded"]) or "-",
        len(_EXT_STATE["failed"]), os.pathsep.join(rep["dirs"]))

# ---------------------------------------------------------------------------
# TEMPLATES - a start that asks the user what they want to make. Added 2026-09-27 (the request: "templates where it like
# asks a user what they wanna make").
#
# A template is DATA, templates.json: what to read first, the questions to ask, the plan to show - each step with the
# calls it is built from - and the rules it keeps. Each registers as an MCP prompt (its `prompt` name) with an optional
# `idea` argument, "a foggy forest clearing at dawn", whose answers are not asked again; mif_help('<id>') gives the
# same text to a client that shows no prompts. A template builds nothing and has no look of its own: the agent asks,
# shows the plan with every value, and calls the tools after the user's yes. Like the pack prompts it costs nothing
# until picked, and registers in main(), never at import (the test stubs' FakeMCP has tool() only). Every tool and
# parameter a template's calls name is checked against this file's own signatures by tools/test_mcp_packs.py, so a
# renamed tool or parameter fails there rather than in a user's session.
# ---------------------------------------------------------------------------
_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates.json")
_TEMPLATES_CACHE = None


def _templates():
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is None:
        try:
            with open(_TEMPLATES_PATH, "r", encoding="utf-8") as fh:
                _TEMPLATES_CACHE = (json.load(fh) or {}).get("templates") or {}
        except Exception as exc:  # noqa: BLE001
            _TEMPLATES_CACHE = {"__error__": str(exc)}
    return _TEMPLATES_CACHE


def _template_id(name):
    """A template's id from its id or its prompt's name, in any case; None for anything else."""
    key = (name or "").strip().lower()
    for tid, t in _templates().items():
        if tid != "__error__" and key in (tid.lower(), str(t.get("prompt") or "").lower()):
            return tid
    return None


# THE SKILLS, served by name. skills/<name>/SKILL.md is the read-before-you-drive text (hazards, call grammar, engine
# versions); it reached an agent only as files, so a client without file access never saw it. mif_help('<name>') returns
# it. The folder ships in the GitHub zip, not the store package, so an absent folder is said out loud rather than
# answered "no tool named". tools/audit_skill_names.py checks every name a skill quotes against this build.
_SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "skills")


def _skills():
    """{name: {"description", "text"}} from skills/*/SKILL.md; {} when the folder is not in this package."""
    out = {}
    try:
        names = sorted(os.listdir(_SKILLS_DIR))
    except OSError:
        return out
    for name in names:
        path = os.path.join(_SKILLS_DIR, name, "SKILL.md")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        desc = ""
        m = _re.match(r"^---\s*\n(.*?)\n---\s*\n", text, _re.S)
        if m:
            d = _re.search(r"^description:\s*(.+)$", m.group(1), _re.M)
            desc = d.group(1).strip() if d else ""
            text = text[m.end():]
        out[name] = {"description": desc, "text": text.strip()}
    return out


def _render_template(tid, idea=""):
    """The text a template's prompt gives, and mif_help('<id>'): what to read first, the questions, the plan with the
    calls each step is built from, and the rules. idea, when given, is quoted so what it answers is not asked again."""
    t = _templates()[tid]

    def calls(step):
        return "; ".join("%s(%s)" % (c["tool"], ", ".join("%s=%s" % kv for kv in (c.get("params") or {}).items()))
                         for c in step.get("calls") or [])

    out = [t.get("intro") or ""]
    idea = (idea or "").strip()
    if idea:
        out.append("THE USER'S IDEA: \"%s\". Whatever it already answers is answered - ask only the rest." % idea)
    out.append("FIRST - these read and change nothing:")
    for n, step in enumerate(t.get("first") or [], 1):
        out.append("%d. %s\n   Calls: %s" % (n, step.get("say") or "", calls(step)))
    questions = t.get("questions") or []
    out.append("THEN ASK, in one message - %d questions, any of which may be answered 'you pick':" % len(questions))
    for n, q in enumerate(questions, 1):
        choices = " / ".join(q.get("choices") or [])
        out.append("%d. %s%s\n   Decides: %s." % (n, q.get("ask") or "", (" (%s)" % choices) if choices else "",
                                                 q.get("use") or ""))
    out.append("THEN PLAN, and build only after a yes. The steps, in order, and the calls each is built from:")
    for n, step in enumerate(t.get("plan") or [], 1):
        out.append("%d. %s\n   Calls: %s" % (n, step.get("say") or "", calls(step)))
    out.append("RULES\n" + "\n".join("- %s" % r for r in t.get("rules") or []))
    return "\n\n".join(out)


def _register_template_prompts():
    """One MCP prompt per template in templates.json, each with an optional `idea`. Called from main()."""
    add = getattr(mcp, "prompt", None)
    templates = _templates()
    if add is None or templates.get("__error__"):
        return 0
    try:
        from typing import Annotated
        from pydantic import Field
    except Exception:  # noqa: BLE001 - the argument then goes out without its description
        Annotated = Field = None
    count = 0
    for tid, t in templates.items():
        about = t.get("argument") or "What you want to make, in your own words (optional)."

        def make(tid=tid, about=about):
            if Annotated is not None:
                def template_prompt(idea: Annotated[str, Field(description=about)] = "") -> str:
                    return _render_template(tid, idea)
            else:
                def template_prompt(idea: str = "") -> str:
                    return _render_template(tid, idea)
            return template_prompt
        add(name=t.get("prompt") or "mif-new-%s" % tid, description=t.get("description") or t.get("title") or tid)(make())
        count += 1
    return count


@mcp.tool()
def mif_list_packs() -> dict:
    "List the tool packs, how many tools each has and whether it is loaded server-side. Load one with mif_enable_pack; if your client never shows the new tools, mif_call reaches any tool by name."
    packs = _packs()
    if packs.get("__error__"):
        return {"error": "packs.json unreadable: %s" % packs["__error__"], "path": _PACKS_PATH}
    try:
        loaded = set(getattr(mcp, "_tool_manager")._tools)
    except Exception:  # noqa: BLE001
        loaded = set()
    rows = []
    for pid, p in packs.items():
        tools = p.get("tools") or []
        here = sum(1 for t in tools if t in loaded)
        rows.append({"pack": pid, "title": p.get("title"), "description": p.get("description"),
                     "backend": p.get("backend"), "requires": p.get("requires") or [],
                     "tools": len(tools), "toolsLoaded": here, "loaded": bool(tools) and here == len(tools),
                     "rules": len(_pack_rules(pid))})
    return {"packs": rows, "aliases": _pack_aliases(),
            "selection": os.environ.get("MIF_TOOL_PACKS", "").strip() or _DEFAULT_PACKS,
            # MIF_DOCS_FIRST (docs/TOKEN_OPTIMIZATION.md 2.13): whether a guessed name is mapped or refused with docs
            "docsFirst": _DOCS_FIRST,
            "blender": "For Blender load blender_core plus the workflow pack you need (a workflow pack brings "
                       "blender_core with it), or 'blender' for every Blender pack. A selection of Blender packs "
                       "only keeps just core's backend-neutral tools; enabling core or any Unreal pack brings "
                       "the rest.",
            "note": "MIF_TOOL_PACKS=core,world in the MCP server's environment loads only those packs at "
                    "startup (core comes with any named pack; unset loads only find_tools, mif_call and run_python, "
                    "and mif_call reaches the rest). mif_enable_pack registers a pack server-side and "
                    "requests tools/list_changed, but the server cannot verify that the client refreshed; "
                    "restart with the pack in MIF_TOOL_PACKS if its tools remain unavailable."}


@mcp.tool()
async def mif_enable_pack(pack: str) -> dict:
    "Register a tool pack server-side and request a client tool-list refresh. The response distinguishes notification delivery from the client actually applying it; restart with MIF_TOOL_PACKS if tools remain unavailable."
    packs = _packs()
    if packs.get("__error__"):
        return {"error": "packs.json unreadable: %s" % packs["__error__"], "path": _PACKS_PATH}
    name = (pack or "").strip().lower()
    resolved, unknown = _expand_packs([name])
    if unknown or not resolved:
        return {"error": "no pack named %r." % pack, "packs": sorted(p for p in packs if p != "__error__"),
                "aliases": sorted(_pack_aliases())}
    # UNREAL WORK IN A SESSION THAT STARTED BLENDER-ONLY needs the Unreal core tools the selection held back (see
    # _core_unreal_tools): core comes along with any pack that is not a Blender pack - and only then, so an ordinary
    # session's enables name exactly the packs they always named.
    if "core" not in resolved and not _blender_only(resolved) and any(t in _UNLOADED for t in _core_unreal_tools()):
        resolved = ["core"] + resolved
    added = []
    total = 0
    for pid in resolved:
        for tool in packs[pid].get("tools") or []:
            if tool in _GEN_HIDDEN:
                continue                     # held off by a generative switch - not loaded, and not "already loaded"
            total += 1
            fn = _UNLOADED.pop(tool, None)
            if fn is not None:
                mcp.add_tool(fn)
                added.append(tool)
    # FROM THE LEAN DEFAULT, A BLENDER PACK BRINGS CORE'S NEUTRAL TOOLS TOO - what MIF_TOOL_PACKS naming it would have
    # loaded at startup, so enabling a pack mid-session and naming it at the start end in the same tool list.
    if _blender_only(resolved):
        for tool in (packs.get("core") or {}).get("neutral") or []:
            fn = _UNLOADED.pop(tool, None)
            if fn is not None:
                mcp.add_tool(fn)
                added.append(tool)
    notified = False
    if added:
        try:
            await mcp.get_context().session.send_tool_list_changed()
            notified = True
        except Exception:  # noqa: BLE001
            notified = False
    refresh_status = ("notification_sent_unverified" if notified else "notification_failed") if added else "not_requested"
    out = {"ok": True, "pack": name, "packs": resolved, "added": len(added),
           "alreadyLoaded": total - len(added),
           # Backward-compatible alias. This has only ever meant that send_tool_list_changed()
           # returned without raising; an MCP server receives no acknowledgement that the client
           # rebuilt the model-visible registry. Claude Desktop and Codex both returned true here
           # while leaving the newly registered tools unavailable across later turns (2026-09-20).
           "clientNotified": notified,
           "clientNotifiedMeaning": "notification_sent",
           "notificationSent": notified,
           "clientRefreshVerified": False,
           "clientRefreshStatus": refresh_status}
    if added:
        out["note"] = ("the tools are registered server-side%s, but the server cannot verify that the client "
                       "refreshed its callable tool list. If they do not appear, call them through "
                       "mif_call(tool, args) - it needs no refresh - or restart the client with "
                       "MIF_TOOL_PACKS including '%s'."
                       % (" and tools/list_changed was sent" if notified else
                          "; tools/list_changed could not be sent", name))
    if resolved != [name]:
        out["packsNote"] = ("'%s' loaded %s - %s." % (name, ", ".join(resolved),
                            "an alias for those packs" if name in _pack_aliases() else "the pack and what it requires"))
    # THE MOMENT AN AGENT CHOOSES THE WORK is when it reads the rules for it. Once per pack per session: a second
    # enable of the same pack adds neither tools nor rules.
    rules = _rules_payload(resolved)
    if rules:
        out["rules"] = rules
        out["rulesNote"] = ("each rule is a mistake this kind of work makes easily - follow them. "
                            "mif_help('<pack>') repeats a pack's rules.")
    return out


# ---------------------------------------------------------------------------
# RESPONSE CONTROLS (2026-10-11, spec "MCP response controls"; docs/00_ARCHITECTURE.md "Response controls").
#
# Results were sized per endpoint (limit, results='errors', compact self_audit) and a long job was polled by hand. These
# reserved keys work on EVERY tool, in a direct call's arguments or inside mif_call's args, and are taken off before the
# tool sees its arguments - no tool has a parameter that starts with "_":
#   _select ["a", "rows.name"]   keep only these dotted paths (a path through a list applies to each element)
#   _omit   ["rows.debug"]       drop these paths
#   _page   50                   return 50 items of the result's longest list (or _pageField's) and a nextCursor
#   _cursor "<nextCursor>"       the next page of THAT result, read from this server's copy - the tool is not called again
#   _wait   120                  on a job-status tool: poll it every 2 s until the job ends or 120 s pass, sending MCP
#                                progress notifications while it runs (a client that sent a progressToken sees them)
#   _image  false                on a tool that shows a picture (_PICTURE_TOOLS): no image block, only the file's path.
#                                'data' keeps bl_render_still's base64 in the text, as a client that reads the bytes wants
# NOTHING IS ADDED TO ANY SCHEMA. Every always-loaded schema is paid for on every request (docs/TOKEN_OPTIMIZATION.md), so
# the keys are named once in mif_call's description and in full in mif_help('mif_call').
# `ok` and `error` always survive _select and _omit: failure is the presence of `error` (instructions, rule 2).
# ---------------------------------------------------------------------------
_RC_KEYS = ("_select", "_omit", "_page", "_pageField", "_cursor", "_wait", "_image")
_RC_ALWAYS = ("ok", "error")
_RC_PAGE_MAX = 1000
_RC_WAIT_MAX = 600.0
_RC_WAIT_POLL = 2.0
_RC_PAGES = {}                 # page id -> {tool, result, field, items, size, expires}
_RC_PAGES_MAX = 32
_RC_PAGE_TTL = 900.0
_RC_PAGE_SEQ = itertools.count(1)


def _rc_split(args):
    """(controls, the tool's own arguments) - the reserved keys taken off a call's arguments."""
    args = dict(args or {})
    controls = {k: args.pop(k) for k in _RC_KEYS if k in args}
    return controls, args


def _rc_paths(value, key):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(p, str) and p.strip(".") for p in value):
        return None, "%s must be a dotted path or a list of them, like [\"actors.name\", \"count\"]" % key
    return [p.strip(".").split(".") for p in value], None


def _rc_check(controls):
    """The reason the controls cannot be used, or None. Checked before the tool runs, so a bad one changes nothing."""
    for key in ("_select", "_omit"):
        if key in controls:
            _paths, why = _rc_paths(controls[key], key)
            if why:
                return why
    if "_page" in controls:
        size = controls["_page"]
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= _RC_PAGE_MAX:
            return "_page must be a whole number of items from 1 to %d" % _RC_PAGE_MAX
    if "_pageField" in controls and "_page" not in controls:
        return "_pageField names the list _page pages, so it needs _page"
    if "_pageField" in controls and (not isinstance(controls["_pageField"], str) or not controls["_pageField"].strip(".")):
        return "_pageField must be the dotted path of a list in the result"
    if "_cursor" in controls:
        if not isinstance(controls["_cursor"], str) or not controls["_cursor"]:
            return "_cursor must be the nextCursor string a page answered with"
        if "_page" in controls or "_wait" in controls:
            return "_cursor continues a paged result as it was paged; it takes no _page or _wait"
    if "_wait" in controls:
        wait = controls["_wait"]
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not 0 < wait <= _RC_WAIT_MAX:
            return "_wait must be a number of seconds above 0 and at most %d" % int(_RC_WAIT_MAX)
    if "_image" in controls and controls["_image"] not in (True, False, "data"):
        return "_image is false (no image block) or 'data' (base64 in the text); leave it out for an image block"
    return None


def _rc_tree(paths):
    tree = {}
    for parts in paths:
        node = tree
        for i, part in enumerate(parts):
            if node.get(part) is True:
                break                                   # a shorter path already takes the whole value
            if i == len(parts) - 1:
                node[part] = True
            else:
                node = node.setdefault(part, {})
    return tree


def _rc_pick(value, tree, found, prefix=""):
    if isinstance(value, list):
        return [_rc_pick(v, tree, found, prefix) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, sub in tree.items():
        if key in value:
            found.add(prefix + key)
            out[key] = value[key] if sub is True else _rc_pick(value[key], sub, found, prefix + key + ".")
    return out


def _rc_drop(value, tree):
    if isinstance(value, list):
        return [_rc_drop(v, tree) for v in value]
    if not isinstance(value, dict):
        return value
    out = dict(value)
    for key, sub in tree.items():
        if key in out and key not in _RC_ALWAYS:
            if sub is True:
                del out[key]
            else:
                out[key] = _rc_drop(out[key], sub)
    return out


def _rc_fields(result, controls):
    """_select then _omit on one result. A selected path that is nowhere in it is named (selectMissing): a typo must not
    read as an empty answer."""
    if "_select" in controls:
        paths, _ = _rc_paths(controls["_select"], "_select")
        found = set()
        picked = _rc_pick(result, _rc_tree(paths), found)
        for key in _RC_ALWAYS + ("page", "waited"):
            if key in result:
                picked[key] = result[key]
        missing = [".".join(p) for p in paths if ".".join(p) not in found]
        if missing:
            picked["selectMissing"] = missing
        result = picked
    if "_omit" in controls:
        paths, _ = _rc_paths(controls["_omit"], "_omit")
        result = _rc_drop(result, _rc_tree(paths))
    return result


def _rc_get(result, parts):
    node = result
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _rc_set(result, parts, value):
    """A copy of result with the list at parts replaced - the dicts on the way copied, the rest shared."""
    out = dict(result)
    node = out
    for part in parts[:-1]:
        node[part] = dict(node[part])
        node = node[part]
    node[parts[-1]] = value
    return out


def _rc_page_out(rec, pid, offset):
    items = rec["items"]
    end = min(offset + rec["size"], len(items))
    out = _rc_set(rec["result"], rec["field"], items[offset:end])
    page = {"field": ".".join(rec["field"]), "offset": offset, "returned": end - offset, "total": len(items),
            "fromOneRead": True}
    if end < len(items):
        page["nextCursor"] = "%s:%d" % (pid, end)
    out["page"] = page
    return out


def _rc_paginate(name, result, controls):
    """The first page of a result, the whole of it kept here for the cursor. Every page comes from this one read, so an
    item is never skipped or repeated because the level or the file changed between pages."""
    size = controls["_page"]
    if "_pageField" in controls:
        parts = controls["_pageField"].strip(".").split(".")
        items = _rc_get(result, parts)
        if not isinstance(items, list):
            return dict(result, page={"error": "_pageField '%s' is not a list in this result" % controls["_pageField"]})
    else:
        lists = [(len(v), k) for k, v in result.items() if isinstance(v, list)]
        if not lists:
            return dict(result, page={"note": "this result holds no list, so there is nothing to page"})
        parts = [max(lists)[1]]
        items = result[parts[0]]
    if len(items) <= size:
        return dict(result, page={"field": ".".join(parts), "offset": 0, "returned": len(items), "total": len(items)})
    now = time.monotonic()
    for old in [p for p, r in _RC_PAGES.items() if r["expires"] < now]:
        del _RC_PAGES[old]
    while len(_RC_PAGES) >= _RC_PAGES_MAX:
        del _RC_PAGES[min(_RC_PAGES, key=lambda p: _RC_PAGES[p]["expires"])]
    pid = "pg%d%s" % (next(_RC_PAGE_SEQ), os.urandom(3).hex())
    _RC_PAGES[pid] = {"tool": name, "result": result, "field": parts, "items": items, "size": size,
                      "expires": now + _RC_PAGE_TTL}
    return _rc_page_out(_RC_PAGES[pid], pid, 0)


def _rc_cursor(name, controls):
    """The page a cursor names, or a refusal. The tool is not called."""
    pid, _, offset = str(controls["_cursor"]).partition(":")
    rec = _RC_PAGES.get(pid)
    if rec is None or rec["expires"] < time.monotonic() or not offset.isdigit():
        return {"ok": False, "error": "_cursor %r is unknown or expired here - pages are kept %d minutes in this server's "
                                      "memory. Call %s again with _page to read it afresh."
                                      % (controls["_cursor"], int(_RC_PAGE_TTL // 60), name)}
    if rec["tool"] != name:
        return {"ok": False, "error": "_cursor %r pages a %s result, not %s." % (controls["_cursor"], rec["tool"], name)}
    out = _rc_page_out(rec, pid, min(int(offset), len(rec["items"])))
    return _rc_fields(out, controls)


def _rc_job(running, progress=None, total=None, message=None):
    return running, progress, total, message


def _wait_state(name, r):
    """(still running, progress, total, message) for one reply of a job-status tool - each read off that endpoint's own
    fields (2026-10-11, from its handler): render_sequence_status percent 0-100 while starting|rendering;
    package_status running and phase; render_movie_capture_status 'capturing'; frame_capture_status warmup|sampling;
    gpu_profile_status waiting; request_status running; mocap_performance_status processing; lighting_build_status
    running; shader_compile_status compiling or numRemainingJobs; ui_scenario_status POSITIONED|WAITING_FOR_STABLE_UI;
    live_coding_status compiling; bl_render_status running; kr_reconstruct_status queued|running."""
    state = str(r.get("state") or "")
    if name == "render_sequence_status":
        return _rc_job(state in ("starting", "rendering"), r.get("percent"), 100, r.get("currentShot") or state)
    if name == "package_status":
        return _rc_job(r.get("running") is True, None, None, r.get("phase"))
    if name == "render_movie_capture_status":
        return _rc_job(state.startswith("capturing"), r.get("filesWritten"), r.get("framesExpected"), state)
    if name == "frame_capture_status":
        return _rc_job(state in ("warmup", "sampling"), r.get("samples"), r.get("wanted"), state)
    if name == "gpu_profile_status":
        return _rc_job(state == "waiting", None, None, state)
    if name == "request_status":
        return _rc_job(state == "running", None, None, state)
    if name == "mocap_performance_status":
        return _rc_job(r.get("processing") is True, r.get("solvedBodyFrames"), r.get("framesInData"), state)
    if name == "lighting_build_status":
        return _rc_job(r.get("running") is True, None, None, "unbuilt objects: %s" % r.get("unbuiltObjects"))
    if name == "shader_compile_status":
        left = r.get("numRemainingJobs")
        return _rc_job(r.get("compiling") is True or (isinstance(left, int) and left > 0), None, None,
                       "shader jobs left: %s" % left)
    if name == "ui_scenario_status":
        return _rc_job(state in ("POSITIONED", "WAITING_FOR_STABLE_UI"), r.get("stableFramesObserved"),
                       r.get("stableFramesRequired"), state)
    if name == "live_coding_status":
        return _rc_job(r.get("compiling") is True, None, None, "compiling" if r.get("compiling") else "idle")
    if name == "bl_render_status":
        return _rc_job(r.get("running") is True, r.get("framesRendered"), r.get("framesExpected"), "rendering")
    if name == "kr_reconstruct_status":
        return _rc_job(state in ("queued", "running"), r.get("functionsDone"), r.get("functionsTotalEstimate"),
                       r.get("phase") or state)
    return None


_WAIT_TOOLS = ("render_sequence_status", "package_status", "render_movie_capture_status", "frame_capture_status",
               "gpu_profile_status", "request_status", "mocap_performance_status", "lighting_build_status",
               "shader_compile_status", "ui_scenario_status", "live_coding_status", "bl_render_status",
               "kr_reconstruct_status")


async def _rc_wait(name, controls, call_once, ctx):
    """Poll a job-status tool until its job ends or _wait seconds pass, with an MCP progress notification per poll.
    A refusal (ok:false) ends the wait at once and is returned as it came."""
    limit = float(controls["_wait"])
    start = time.monotonic()
    polls = 0
    while True:
        result = await call_once()
        polls += 1
        if not isinstance(result, dict) or result.get("ok") is False or "error" in result:
            return result
        running, progress, total, message = _wait_state(name, result)
        waited = round(time.monotonic() - start, 1)
        if not running:
            return dict(result, waited={"seconds": waited, "polls": polls, "ended": True})
        if waited + _RC_WAIT_POLL > limit:
            return dict(result, waited={"seconds": waited, "polls": polls, "ended": False,
                                        "note": "still running after %s s - call %s again (with _wait to keep "
                                                "waiting)" % (waited, name)})
        try:
            if isinstance(progress, (int, float)) and not isinstance(progress, bool):
                await ctx.report_progress(float(progress), float(total) if isinstance(total, (int, float)) else None,
                                          str(message) if message else None)
            else:
                await ctx.report_progress(float(waited), float(limit), str(message) if message else None)
        except Exception:  # noqa: BLE001 - a client that cannot be told still gets the result
            pass
        await asyncio.sleep(_RC_WAIT_POLL)


async def _rc_run(name, controls, call_once, ctx):
    """One call with its response controls: the cursor (no call), the wait, the page, then the fields."""
    if "_cursor" in controls:
        return _rc_cursor(name, controls)
    if "_wait" in controls:
        if name not in _WAIT_TOOLS:
            return {"ok": False, "error": "_wait polls a job-status tool, and %s is not one. It works on: %s. Nothing "
                                          "was called." % (name, ", ".join(_WAIT_TOOLS))}
        result = await _rc_wait(name, controls, call_once, ctx)
    else:
        result = await call_once()
    if not isinstance(result, dict):
        return result
    if "_page" in controls:
        result = _rc_paginate(name, result, controls)
    return _rc_fields(result, controls)


@mcp.tool()
async def mif_call(tool: str, args: dict | None = None) -> Any:
    "Call any MifBridge tool by name, loaded in your client or not - no client refresh needed; args is an object of that tool's own parameters, which mif_help(tool) lists. Any call also takes _select/_omit (paths), _page (n, then _cursor) and _wait (seconds, on a job-status tool): mif_help('mif_call')."
    # THE FIX THAT DOES NOT DEPEND ON THE CLIENT. mif_enable_pack registers a pack server-side and asks
    # the client to re-read its tool list; the handshake now admits the list can change, but whether a
    # given client re-reads it is the client's business, and a user measured Claude and Codex
    # not doing so. This tool is in core, so every client has it from the first turn, and it reaches
    # every other tool through the server - no refresh involved, and the token saving kept.
    #
    # THE SAME GATES AS A DIRECT CALL. Arguments go through the registered tool's own pydantic model
    # (ToolManager.call_tool), and unknown keys are refused first, as _SlimFastMCP.call_tool refuses
    # them for a direct call - a typo must not become a successful call on defaults here either.
    # Only MifBridge tools are reachable: a name must be registered or held back by a pack selection,
    # never an arbitrary module global.
    name = (tool or "").strip()
    if name == "mif_call":
        return {"ok": False, "error": "mif_call cannot call itself."}
    if name in _GEN_HIDDEN:
        refused = _generative_refusal(name[3:], {})       # reads the open .blend again: it may have changed
        if refused:
            return dict(refused, endpoint=name)       # hidden, not unknown: say which switch holds it
        _sync_gen_visibility()
    manager = getattr(mcp, "_tool_manager", None)
    registry = getattr(manager, "_tools", None)
    if not isinstance(registry, dict):
        return {"ok": False, "error": "this FastMCP no longer exposes its tool registry, so mif_call cannot "
                                      "dispatch. Restart the client with the pack named in MIF_TOOL_PACKS."}
    registered_now = False
    if name not in registry:
        fn = _UNLOADED.pop(name, None)
        if fn is None:
            forwarded = _forward_to_newer_editor(name, args)
            if forwarded is not None:
                return forwarded
            return {"ok": False, "error": "no MifBridge tool named %r." % tool,
                    "hint": "find_tools('<words>') searches every tool, loaded or not."}
        mcp.add_tool(fn)
        registered_now = True
    params = (registry[name].parameters or {}).get("properties") or {}
    # A BLIND CALL SPELLS THE OBJECT BOTH WAYS, AND IN THE ADD-ON'S NAMES. mif_call shows no schema, so an agent guesses
    # from the last tool it used: the Blender tools name the object they act on `object` (75 of them) or `object_name`
    # (22), and the benchmark's core arm - core became the default on 2026-09-23 - failed 10 of its 15 unknown-parameter
    # mif_call calls on exactly that, and one more in the add-on's camelCase names, as bl_describe_op reports them
    # (ledger CC16). For a bl_ tool each is renamed to the tool's own parameter: the object's other spelling only when
    # the tool takes it, an add-on name only through the map read off the tool's own call (_op_param_names), neither
    # when the call already gave the target name. Every rename comes back in argsRenamed, so the agent learns the name;
    # anything else unknown is still refused, by name, before anything runs.
    # THE RESPONSE CONTROLS ride inside args (they are not the tool's parameters): taken off, checked, applied below.
    controls, args = _rc_split(args)
    why = _rc_check(controls) if controls else None
    if why:
        return {"ok": False, "tool": name, "error": "%s. Nothing was called." % why}
    if "_cursor" in controls:
        return _first_call_note(_rc_cursor(name, controls), name, registered_now)
    # A DEPRECATED SPELLING (_PARAM_ALIASES) is renamed in both modes: a documented name, not a guess
    args, deprecated, both = _deprecated_renames(name, args)
    renamed = {}                                     # the guesses mapped below; deprecatedParams reports the above
    # a cross-tool spelling (_SPELLING_PAIRS) is not deprecated: it is reported with the other renames, in argsRenamed
    own = _PARAM_ALIASES.get(name) or {}
    for k in [k for k in deprecated if k not in own]:
        renamed[k] = deprecated.pop(k)
    if both:
        return _first_call_note({"ok": False, "tool": name, "error": _both_spellings_error(name, both)},
                                name, registered_now)
    if name.startswith("bl_"):
        op = _bl_tool_op(name)
        forward = _op_param_names("blender", op) if op else {}
        keys = list(forward.values())
        back = {v: k for k, v in forward.items() if keys.count(v) == 1}     # an add-on key two parameters feed: neither
        for key in list(args):
            if key in params:
                continue
            if key in ("object", "object_name"):
                # a guess between the two spellings, which docs-first mode leaves to the refusal (2.13)
                want = None if _DOCS_FIRST else ({"object", "object_name"} - {key}).pop()
            else:
                want = back.get(key)             # the add-on's own name, which bl_describe_op documents: a translation
            if want in params and want not in args:
                args[want] = args.pop(key)
                renamed[key] = want
    for key in [k for k in list(args) if k not in params]:        # every tool: the guesses _arg_alias maps (none in
        want = _arg_alias(key, params, single=True)                # docs-first mode)
        if want is not None and want not in args:
            args[want] = args.pop(key)
            renamed[key] = want
    unknown = sorted(set(args) - set(params))
    if unknown:
        # "YOU SAID X - DID YOU MEAN Y OR Z?" (2026-09-27): each unknown name with the names it most likely
        # meant, in both modes, so the next call is chosen from two names rather than guessed from the whole list
        error, ask = _unknown_params_refusal(name, unknown, params)
        refusal = {"ok": False, "tool": name, "error": error}
        if _DOCS_FIRST:
            refusal = _docs_refusal(name, error + ".")
        if ask:
            refusal["didYouMean"] = ask
        if renamed:
            refusal["argsRenamed"] = renamed    # the names already mapped are taught too, not only on a success
        return _first_call_note(refusal, name, registered_now)
    if _DOCS_FIRST:
        # A WRONG TYPE IS REFUSED THE SAME WAY, with the shape, where it came back as a pydantic dump - five of round
        # eight's as-shipped failures, each on a tool whose find_tools row had given its names but not its types.
        problems = _type_problems(registry[name], args)
        if problems:
            return _first_call_note(_docs_refusal(name, "%s's arguments do not fit its parameters: %s."
                                                  % (name, "; ".join(problems[:6])), typeProblems=problems[:6]),
                                    name, registered_now)
    if controls:
        ctx = mcp.get_context()
        result = await _rc_run(name, controls, lambda: manager.call_tool(name, args, context=ctx, convert_result=False),
                               ctx)
    else:
        result = await manager.call_tool(name, args, context=mcp.get_context(), convert_result=False)
    if renamed and isinstance(result, dict):
        result = dict(result, argsRenamed=renamed)
    if deprecated and isinstance(result, dict):
        result = dict(result, deprecatedParams=deprecated, deprecatedNote=_deprecation_note(name, deprecated))
    return _first_call_note(result, name, registered_now)


_EDITOR_ENDPOINTS = {}     # (base url, editor instance) -> the endpoint names that editor registers
_blender_forwarded = _blender    # an extension's api.blender (_ExtTransport): the op name is the extension's
_post_forwarded = _post    # mif_call's forward of an endpoint this server has no tool for (_forward_to_newer_editor)


def _editor_endpoint_names():
    """The endpoints the running editor registers, read once per editor process (self_audit includeEndpoints), or None
    when it did not answer with a list."""
    key = (BASE, _HS_UE_SEEN.get("instance"))
    if key in _EDITOR_ENDPOINTS:
        return _EDITOR_ENDPOINTS[key]
    res = _post("self_audit", summaryOnly=True, includeEndpoints=True, _timeout=_HS_TIMEOUT)
    names = res.get("endpoints") if isinstance(res, dict) and res.get("ok") is not False else None
    if not isinstance(names, list):
        return None
    found = {n for n in names if isinstance(n, str)}
    _EDITOR_ENDPOINTS[(BASE, _HS_UE_SEEN.get("instance"))] = found    # keyed after the read: its reply names the instance
    return found


def _camel(key):
    head, *rest = str(key).split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _forward_to_newer_editor(name, args):
    """mif_call of a name this server has no tool for, which the EDITOR registers (FOUND 2026-10-14: a server started
    before a plugin update refused add_landscape_edit_layer until it restarted). The call goes to the endpoint as it is:
    args are the endpoint's own JSON keys (snake_case keys are sent camelCase, the server's convention, and named in
    argsRenamed), the editor's own guard refuses an unknown key with the list it accepts, and the reply says the server is
    older than the plugin. None when the name is not the editor's either, or it could not be asked."""
    if not name or name.startswith("bl_") or not all(c.isalnum() or c == "_" for c in name):
        return None
    names = _editor_endpoint_names()
    if not names or name not in names:
        return None
    controls, body = _rc_split(args)
    body = dict(body or {})
    renamed = {}
    for key in list(body):
        if "_" in key.strip("_"):
            want = _camel(key)
            if want not in body:
                body[want] = body.pop(key)
                renamed[key] = want
    # THE ONE CALL SITE WITH A NAME FROM OUTSIDE THIS FILE, on purpose: the name is one the editor registers and this
    # server does not, so no literal can name it and parity_check cannot hold it to a wrapper. It goes through
    # _post_forwarded so that check (literal _post names only) keeps covering every other call.
    res = _post_forwarded(name, **body)
    out = dict(res) if isinstance(res, dict) else {"ok": False, "error": "the editor answered %r" % (res,)}
    out["forwarded"] = True
    provider = out.get("providedBy")
    if isinstance(provider, str) and provider and provider != "MifBridge":
        # AN EXTENSION'S NATIVE ENDPOINT (docs/32_EXTENSIONS.md): its plugin registered it and no tool was written for it,
        # which is normal - this server is not older than anything. The editor checked the keys against its declaration.
        out["serverOlderThanPlugin"] = False
        out["forwardNote"] = ("%r is an extension endpoint registered by %s; it has no MCP tool of its own, so the call "
                              "was sent to the endpoint as it is (its keys are the endpoint's own: describe_endpoint %r "
                              "lists them, as declared by the extension)." % (name, provider, name))
    else:
        out["serverOlderThanPlugin"] = True
        out["forwardNote"] = ("this MCP server has no tool named %r, but the editor registers it - the server is older "
                              "than the plugin, so the call was sent to the editor's endpoint as it is, with no typed "
                              "wrapper (its keys are the endpoint's own: describe_endpoint %r lists them). Update or "
                              "restart the MCP server to get the typed tool." % (name, name))
    if renamed:
        out["argsRenamed"] = renamed
    if controls:
        out["controlsIgnored"] = sorted(controls)
    return out


def _first_call_note(result, name: str, registered_now: bool):
    """The answer to the mif_call that registered a held-back tool, with its pack's rules (mifCall.packRules).

    AN AGENT ON mif_call NEVER CALLS mif_enable_pack - its client does not refresh, so it reaches the pack one tool at a
    time, and each tool's first call carries the pack rules that name it, and the ones that name no tool (see
    _tool_rules_payload), so this path is taught what the enable path is, as it needs it. A REFUSED FIRST CALL CARRIES
    THEM TOO: until 2026-09-27 only a successful one did, and the next call found the tool registered already, so a
    tool whose first call was refused never delivered its rules - rarely, while guesses were mapped, and on every guess
    once docs-first mode refuses them (docs/TOKEN_OPTIMIZATION.md 2.13)."""
    if not registered_now or not isinstance(result, dict) or "mifCall" in result:
        return result
    note = {"tool": name, "registeredServerSide": True}
    home = _tool_pack(name)
    rules = _tool_rules_payload(name, _expand_packs([home])[0]) if home else {}
    if rules:
        note["packRules"] = rules
        note["packRulesNote"] = ("the %s pack's rules for %s, each once; mif_help('%s') lists them all."
                                 % (home, name, home))
    return dict(result, mifCall=note)


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="MifBridge MCP server")
    parser.add_argument("--debug", action="store_true", help="log request/response to stderr")
    args = parser.parse_args()
    DEBUG = args.debug or os.environ.get("MIF_BRIDGE_DEBUG", "").lower() in ("1", "true", "yes")
    _log(f"starting; ue={BASE} timeout={TIMEOUT}s token={'set' if TOKEN else 'empty'}")
    # Reported, never dialled: connecting to Blender at startup would block the
    # MCP handshake whenever Blender is closed. bl_* tools connect lazily.
    _log(f"           blender={BLENDER_HOST}:{BLENDER_PORT} connect={BLENDER_CONNECT_TIMEOUT}s "
         f"probe={BLENDER_PROBE_TIMEOUT}s work={BLENDER_TIMEOUT}s (lazy connect)")
    _log("generative tools: " + _apply_generative_mode())    # first: the packs must never hold a hidden gen tool back
    _log(_load_extensions())                                 # before the packs, so extension tools are held back too
    _log("tool packs: " + _apply_tool_packs())
    _log("docs first: " + _apply_docs_first())
    _log("pack prompts registered: %d" % _register_pack_prompts())
    _log("template prompts registered: %d" % _register_template_prompts())
    global _SERVING
    _SERVING = True
    mcp.run()


@mcp.tool()
def bl_collect_vdb_frames(destination: str, object: str = None, folder: str = None, prefix: str = None,
                          start: int = None, end: int = None, base_name: str = None, digits: int = None,
                          first_number: int = None, sample_ranges: bool = None) -> dict:
    "Copy the frames of a baked OpenVDB sequence into a clean folder, renumbered from 0 - COPIED, never moved, so the cache is untouched. The source is a Blender Mantaflow DOMAIN (object=, its cache folder's data/ is found for you) or any folder of .vdb files (folder=, prefix= when it holds more than one sequence). The kept frames default to the frames the scene plays (scene start to the domain's cache end) for a domain and to every non-negative frame for a folder - Mantaflow writes pre-roll frames with a minus sign (fluid_data_-0001.vdb), which Unreal's importer reads as frame 1 and aliases onto the real frame 1. start/end choose the source frames to keep; a gap inside them is refused (Unreal's importer stops at the first gap, and renumbering would hide it). destination must hold no .vdb that would join the new sequence. base_name names the files (and so the Unreal asset: fire -> fire_0000.vdb -> asset fire). Reports every grid (name, type, channels, voxel size) read by Blender's own OpenVDB reader and, with sample_ranges (default true), each grid's value range on the first, middle and last frame through Blender's bundled openvdb module when it has one - the numbers a material's multipliers depend on (Mantaflow temperature runs to about 3, not kelvin). ueImport carries the next call."
    return _blender("collect_vdb_frames", destination=destination, object=object, folder=folder, prefix=prefix,
                    start=start, end=end, baseName=base_name, digits=digits, firstNumber=first_number,
                    sampleRanges=sample_ranges, _timeout=900.0)


@mcp.tool()
def mif_help(tool: str = "") -> dict:
    "Get the FULL documentation for a MifBridge tool, or a tool pack's rules - the traps, engine citations and failure modes left out of the one-line tool descriptions. Call it when a description does not tell you enough to call a tool, before a destructive or engine-level operation you have not done before, or after a refusal you do not understand: several endpoints guard engine asserts that would terminate the editor, and the reason lives here rather than in the one-line summary. A tool's answer also carries every pack rule that names it; pass a pack name (mif_list_packs lists them) for all of that pack's rules. A template's name - environment - gives the questions and plan its mif-new-<name> prompt gives, for a client that shows no prompts. A skill's name - mifbridge-hazards, mifbridge-driving - gives that skill's text: what can take the editor down and how to verify a result, worth reading before the first call. Pass no argument to list every tool that has extended help. For an endpoint's real accepted parameters as the LIVE editor sees them, use describe_endpoint instead - that reads the running plugin and is the authority when this server and the plugin disagree."
    store = _tool_help()
    if store.get("__error__"):
        return {"error": "tool help unavailable: %s" % store["__error__"],
                "path": _TOOL_HELP_PATH}
    if not tool:
        return {"tools": sorted(k for k in store if not k.startswith("__")),
                "count": len([k for k in store if not k.startswith("__")]),
                "templates": sorted(t for t in _templates() if t != "__error__"),
                "skills": {k: v["description"] for k, v in _skills().items()},
                "note": "pass one of these as `tool` for its full documentation, a template's name for its questions, "
                        "or a skill's name for its text."}
    name = tool.strip()
    if name in store:
        # THE SIDECAR IS A SNAPSHOT, AND 115 OF ITS 446 ENTRIES HAD FALLEN BEHIND THEIR TOOLS. It was cut
        # from the descriptions once, on 2026-08-30; descriptions kept growing in place after that, and this
        # answered from the snapshot alone - so the newer text went out neither inline (clients get the lead
        # sentence) nor here. Measured 2026-09-21: bl_import_mesh's entry still said "FBX ONLY" while the
        # tool took glTF, and its warning that clear_scene defaults TRUE and deletes every object in the file
        # reached no client. So when the tool's own description says anything the entry does not, the
        # description comes first - it is the one edited beside the code - and the entry follows, marked.
        own = _registered_description(name)
        if own and not _says_everything(store[name], own, name):
            return _with_pack_rules(name, {
                "tool": name, "olderNotesFollow": True,
                "help": own + "\n\nOLDER NOTES, written before the description above was last revised - "
                              "where the two disagree, the description above is the current "
                              "behaviour:\n" + store[name]})
        return _with_pack_rules(name, {"tool": name, "help": store[name]})

    # "NO SUCH TOOL" AND "REAL TOOL, NOTHING EXTRA TO SAY" ARE DIFFERENT ANSWERS, and this used to
    # return the same `error` for both. 382 of 496 tools have an extended entry; the other 114 are
    # tools whose one-line description was already the whole story, which is a fine thing to be. An
    # agent that asks for help on add_branch and gets an `error` back has been told something false
    # about add_branch - and the house rule here is that failure is the PRESENCE of error, so it
    # reads as a failure whatever the prose says.
    #
    # The tool list is taken from FastMCP's registry when it exposes one, and from this module's own
    # globals otherwise. Every @mcp.tool in this file is a module-level function and the decorator
    # returns it unchanged, so globals() carries them all; the leading-underscore skip keeps the
    # private helpers out.
    known = set()
    try:
        known = set(getattr(mcp, "_tool_manager")._tools)          # FastMCP's registry
    except Exception:
        known = {k for k, v in globals().items()
                 if callable(v) and not k.startswith("_")}
    # A HELD-BACK TOOL IS A TOOL. The registry holds only what the pack selection loaded, so a tool in a pack that was
    # not loaded, with no extended-help entry, was answered "no tool named 'batch'" - false, and the house rule reads
    # `error` as failure. Rare while core was the default; on the lean default every one of core's own tools without an
    # entry would have met it (tools/test_mcp_packs.py P124, found building that default, 2026-09-23).
    known |= set(_UNLOADED)
    if name in known:
        # A TRIMMED DESCRIPTION IS EXTENDED HELP. Since 2026-09-11 list_tools sends any description
        # over _DESC_BUDGET as its lead sentence plus "Full notes: mif_help('<name>')" - so this
        # answer must return the full text, or that pointer leads nowhere. It is read from the
        # registry, which keeps the description exactly as the tool declared it.
        full = _registered_description(name)
        if len(full) > _DESC_BUDGET:
            return _with_pack_rules(name, {
                "tool": name, "help": full, "hasExtendedHelp": True,
                "source": "the tool's own description; clients are sent its lead sentence only"})
        return _with_pack_rules(name, {
            "tool": name, "help": None, "hasExtendedHelp": False,
            "note": "%s exists and has no extended help - its one-line description is the whole "
                    "of it. Extended help is written for the tools that guard an engine assert "
                    "or carry a trap worth reading first; a tool without an entry is not an "
                    "undocumented tool." % name})

    # A PACK NAME, after every tool name - a tool keeps the name if the two ever collide. It answers with the rules
    # (the packs'-worth of "what goes wrong here") rather than tools; mif_list_packs is the place for counts.
    packs = _packs()
    key = name.lower()
    if key in packs and key != "__error__":
        p = packs[key]
        needs = p.get("requires") or []
        out = {"pack": key, "title": p.get("title"), "description": p.get("description"),
               "tools": len(p.get("tools") or []), "rules": [r["rule"] for r in _pack_rules(key)]}
        if needs:
            out["requires"] = needs
            out["requiredRules"] = {q: [r["rule"] for r in _pack_rules(q)] for q in needs if _pack_rules(q)}
        if not out["rules"]:
            out["note"] = "the %s pack has no rules of its own - its tools' mif_help entries carry the detail." % key
        return out
    if key in _pack_aliases():
        members = _pack_aliases()[key]
        return {"alias": key, "packs": members,
                "note": "'%s' names several packs - mif_help('<pack>') gives each one's rules: %s."
                        % (key, ", ".join(m for m in members if _pack_rules(m)) or "none has rules yet")}

    # A TEMPLATE, by its id or its prompt's name, after every tool and pack name: the text its prompt gives, for a
    # client that shows no prompts.
    tid = _template_id(key)
    if tid:
        t = _templates()[tid]
        return {"template": tid, "title": t.get("title"), "prompt": t.get("prompt"), "text": _render_template(tid)}

    # A SKILL, by its folder name (mifbridge-hazards), last: the hyphenated names cannot collide with a tool.
    skills = _skills()
    if key in skills:
        return {"skill": key, "description": skills[key]["description"], "text": skills[key]["text"]}
    if key.startswith("mifbridge-") and not skills:
        return {"error": "%r looks like a skill, and this package has no skills/ folder - the skills ship in the "
                         "repository zip, not the store package (looked in %s)." % (name, _SKILLS_DIR)}

    near = sorted(k for k in store if name.lower() in k.lower())[:12]
    return {"error": "no tool named %r. Either the name is wrong, or it is an ENDPOINT name rather "
                     "than an MCP tool name - describe_endpoint {name} answers for those." % name,
            "didYouMean": near}

if __name__ == "__main__":
    main()
