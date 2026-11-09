"""MifBlender ops: local text/image -> 3D generation, driven through ComfyUI.

Nothing here calls a paid API and nothing leaves the machine. The chain is:

    Flux.1              prompt  -> a clean reference image
    Hunyuan3D-2 shape   image   -> an untextured mesh
    Hunyuan3D-2 paint   mesh    -> PBR textures baked from multiview renders
    this module         result  -> imported into the open Blender scene

WHY THIS LIVES IN THE ADDON. Generation used to sit in a mod's tools/ directory, which
meant the mesh was produced somewhere else and imported by hand. Blender is where the
result actually gets inspected and fixed, so the generator belongs on the same side of the
socket as the mesh ops that clean it up.

WHY IT ASKS THE SERVER FOR THE SCHEMA. Node inputs are validated against ComfyUI's own
/object_info rather than hardcoded from reading the custom node source. Custom nodes change
between commits, and a workflow built from stale assumptions fails deep inside the run with
a KeyError on a tensor - long after the interesting part started. Failing at submit time
with "node X has no input Y" is worth the extra request.

TEXTURE IS NOT OPTIONAL FOR A USABLE ASSET. The shape DiT alone returns bare geometry, and
a bare mesh needs a human to author materials before it is worth anything. The paint path
below (delight -> uv wrap -> multiview render -> sample -> bake -> PBR) is what turns a
generation into something that can be dropped straight into a level.

DELIGHT MATTERS MORE THAN IT SOUNDS. Flux renders its reference image with lighting -
speculars, shadows, a key light. Painting straight from that bakes those highlights into
the albedo, and the asset then looks wrong under every other light in the game.
Hy3DDelightImage strips them first. Skipping it is the single most common reason a
generated texture looks "off" in a way nobody can name.
"""

import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import bpy

from .ops_common import (MifOpError, reject_unknown, take, take_bool, take_float, take_int)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1:8188"

# Flux comes in two variants that are NOT interchangeable at the sampler. schnell is
# guidance-distilled: it runs at cfg 1.0, ignores the negative prompt, and is tuned for four
# steps. dev takes a real FluxGuidance node and needs ~20 steps - sampling it at schnell's
# four produces mush, which reads as a bad model rather than a bad configuration.
FLUX = {
    "schnell": {"ckpt": "flux1-schnell-fp8.safetensors", "steps": 4, "guidance": None},
    "dev": {"ckpt": "flux1-dev-fp8.safetensors", "steps": 20, "guidance": 3.5},
}

# The reference image wants to read like a product shot, not a render. A three-quarter
# perspective view produces a subtly sheared mesh that looks like a model failure but is
# really a framing failure, so the framing is part of the prompt rather than advice.
PROMPT_SUFFIX = (
    ", full object centred in frame, orthographic side profile, plain flat background, "
    "even diffuse studio lighting, no dramatic shadows, no motion blur, sharp focus, "
    "single object, entire object visible"
)


# ---------------------------------------------------------------------------
# The generative mode: on by default, off by the preference or by the file
# ---------------------------------------------------------------------------
#
# Decided 2026-10-01. These five ops are the only part of the add-on that runs a trained model (Flux,
# Hunyuan3D-2). They were made opt-in that morning, then reversed to generative by default, as a choice
# users are aware of and each project makes. So they run unless one of two switches here says no:
#   - the add-on preference allow_generative (default True), for every file this Blender opens;
#   - the open FILE's mode, 'non-generative', recorded by the project_mode op below - the user's choice
#     for that project, asked for by the assistant the first time MifBridge meets the file.
# The MCP server has its own switch (MIF_GENERATIVE=off) and refuses before a call reaches here; an
# Unreal project's mode holds these ops off on the server's side too. The check is the FIRST statement of
# every op, ahead of reject_unknown, so a refused call reads no parameter and contacts nothing. ping
# reports both switches, the sidebar shows them, and every journal start record carries `generative` -
# whether the ops could run at that call - so a session's own log shows how it ran.
#
# WHERE A FILE'S MODE LIVES: a custom property on the file's scenes (MODE_KEY). It is saved in the .blend
# and travels with it - copied, renamed or opened on another machine - which a file beside it would not,
# and an unsaved file has one too, kept when it is first saved. Written on EVERY scene, because the file is
# the project and any scene may be the active one; read conservatively: if any scene says non-generative
# the file is, whatever another scene appended from elsewhere says.

MODE_KEY = "mifbridge_generative_mode"
MODES = ("generative", "non-generative")
ROADMAP_KEY = "mifbridge_roadmap"    # file_roadmap: the plan this project follows, as JSON text on every scene

GENERATIVE_HELP = (
    "%s is generative (it runs a trained image/3D model through ComfyUI) and generative tools are off %s, "
    "so nothing was sent to ComfyUI. %s")
_OFF_BY = {
    "preference": ("in this Blender's add-on preferences",
                   "The user turns them back on with Edit > Preferences > Add-ons > MifBlender > "
                   "'Allow generative tools (ComfyUI)'."),
    "file": ("for this file (its project mode is non-generative)",
             "Only the user changes a project's mode: project_mode with mode 'generative' (the MCP server's "
             "mif_project_mode) when they ask for it."),
}


def _addon_prefs():
    """(preferences or None, readable). An add-on loaded without being enabled has no preferences entry -
    the preference's default applies; a context that cannot be read at all is (None, False)."""
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        return (addon.preferences if addon else None), True
    except Exception:  # noqa: BLE001 - a restricted context: the caller reads it as off, never as on
        return None, False


def preference_allows():
    """The add-on preference allow_generative: its default, True, when the add-on has no preferences entry
    (loaded without being enabled); False when they cannot be read at all."""
    prefs, readable = _addon_prefs()
    if not readable:
        return False
    return bool(getattr(prefs, "allow_generative", True)) if prefs is not None else True


def file_mode():
    """The open file's project mode: 'generative', 'non-generative', 'unset' (not chosen yet), or 'unknown'
    when bpy.data cannot be read here (a restricted context) - which off_by holds off, never lets through."""
    seen = set()
    try:
        for scene in bpy.data.scenes:
            value = scene.get(MODE_KEY)
            if value in MODES:
                seen.add(value)
    except Exception:  # noqa: BLE001 - bpy.data unreadable here
        return "unknown"
    if "non-generative" in seen:
        return "non-generative"
    return "generative" if seen else "unset"


def off_by():
    """Which switch holds the gen ops off - 'preference' or 'file' - or None when they may run."""
    if not preference_allows():
        return "preference"
    return "file" if file_mode() in ("non-generative", "unknown") else None


def generative_allowed():
    """True when neither the preference nor the file's mode holds the gen ops off."""
    return off_by() is None


def _require_generative(op):
    by = off_by()
    if by is not None:
        raise MifOpError(GENERATIVE_HELP % ((op,) + _OFF_BY[by]))


def _host(params):
    host = take(params, "host", "server", default=None)
    return str(host or os.environ.get("MIF_COMFY_HOST") or DEFAULT_HOST)


def _post(host, path, payload=None, timeout=60):
    url = "http://%s/%s" % (host.rstrip("/"), path.lstrip("/"))
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise MifOpError(
            "ComfyUI rejected %s: HTTP %s %s. The workflow was refused before anything ran - "
            "usually a node input name that changed in a wrapper update." % (path, exc.code, detail))
    except TimeoutError:
        # A BARE TimeoutError is the READ half: urlopen sends inside the block that wraps OSError into
        # URLError and reads the response outside it, so reaching here means the connection WAS made.
        raise MifOpError(_silent(host, path, timeout))
    except urllib.error.URLError as exc:
        # A TIMEOUT WRAPPED IN URLError IS THE CONNECT HALF, and it is NOT the situation _silent
        # describes - no connection was ever accepted. This arm used to send it to _silent anyway
        # ("the two spellings of the same situation"), and they are not the same situation.
        #
        # MEASURED 2026-09-16 on Windows 11: a connect to a CLOSED loopback port is refused only
        # after 2.02 s, because Windows retries the SYN before it reports the RST. So with timeout=2
        # and NOTHING listening, urlopen raised URLError(TimeoutError) and the caller was told
        # ComfyUI "accepted the connection" and was "probably running and still loading a model" -
        # wrong on both counts, and the advice (wait, retry) is the opposite of the remedy (start
        # it). The same call with timeout=5 came back ConnectionRefusedError after 2.04 s.
        if isinstance(exc.reason, TimeoutError):
            raise MifOpError(_absent(host, "no connection was accepted within %ss - nothing may be "
                                           "listening, or something is dropping the connection. On "
                                           "Windows a refused connection takes about 2 s to report, so "
                                           "a timeout that short reads as this rather than as a refusal"
                                           % timeout))
        raise MifOpError(_absent(host, exc.reason))
    return json.loads(body) if body else {}


def _absent(host, reason):
    """Nothing is listening: the remedy is to start it.

    NO COMMAND LINE IN THE ADVICE. Until 2026-10-01 this told every user to run a python.exe from
    a ComfyUI venv at a path on the machine this was written on, which no user has. ComfyUI is
    installed many ways (portable build, desktop app, a venv anywhere), so the message names what
    MifBridge expects instead: an address, and the setting that changes it. audit_shipped_wording
    keeps that folder out of shipped text.
    """
    return ("cannot reach ComfyUI at %s (%s). Start ComfyUI the way you installed it, listening on "
            "that address - MifBridge does not start or install it. The address comes from the call's "
            "'host' parameter, else MIF_COMFY_HOST in Blender's environment, else %s."
            % (host, reason, DEFAULT_HOST))


def _silent(host, path, timeout):
    """Something ACCEPTED the connection and then said nothing.

    A different situation from _absent and it needs different advice - ComfyUI is most likely
    there and still loading a checkpoint (Flux fp8 is 16.6 GB and takes minutes on a cold cache),
    so "start it" would be wrong. Telling the caller to start a thing that is already running is
    how somebody ends up with two of them fighting over the port.

    THIS PATH USED TO HAVE NO MESSAGE AT ALL. urlopen raises a bare TimeoutError on a read
    timeout - CPython calls h.getresponse() outside the block that wraps OSError into URLError -
    so only _post's URLError arm existed and a silent ComfyUI surfaced to the caller as
    "TimeoutError: timed out", naming neither ComfyUI, nor the address tried, nor the timeout it
    waited out.
    """
    return ("ComfyUI at %s accepted the connection but did not answer %s within %ss. It is "
            "probably running and still loading a model - watch its console and retry, or pass a "
            "larger 'timeout'. If its console shows nothing, something else is holding that port."
            % (host, path, timeout))


def _get(host, path, timeout=60):
    return _post(host, path, None, timeout)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_schema_cache = {}


def _object_info(host, refresh=False, timeout=120):
    """ComfyUI's own node schema, cached per host.

    NOT set-and-forget, and it used to be. Installing a custom node requires restarting
    ComfyUI, so any session that had already called a gen_* op held a schema that could
    never mention the new nodes: gen_status went on reporting every Hy3D node missing
    while ComfyUI was plainly serving all 32 of them, and only restarting BLENDER - the
    process that was right - cleared it. That sends people to reinstall a wrapper that
    is already installed. Callers that can afford a refetch pass refresh=True.
    """
    if refresh:
        _schema_cache.pop(host, None)
    if host not in _schema_cache:
        _schema_cache[host] = _get(host, "object_info", timeout=timeout)
    return _schema_cache[host]


def _require_nodes(host, names):
    """Fail at submit time, naming the missing node, rather than deep inside the run."""
    info = _object_info(host)
    missing = [n for n in names if n not in info]
    if missing:
        # Re-ask before believing it. The overwhelmingly likely cause of a "missing" node
        # is a schema cached from before ComfyUI was restarted with the wrapper installed,
        # not an absent node - so refusing without a refetch blames the wrong thing.
        info = _object_info(host, refresh=True)
        missing = [n for n in names if n not in info]
    if missing:
        raise MifOpError(
            "ComfyUI is running but these nodes are not installed: %s. "
            "The Hunyuan3D wrapper provides them - check custom_nodes/ComfyUI-Hunyuan3DWrapper "
            "is present and that ComfyUI logged no import error for it."
            % ", ".join(sorted(missing)))
    return info


def _check_inputs(info, class_type, inputs):
    """Reject an input the installed node does not declare, AND a required one we omit.

    Both halves matter and only the first existed at first. Drift moves in two
    directions: the wrapper can DROP an input we still send (Hy3DGenerateMesh lost
    attention_mode to Hy3DModelLoader) or ADD a required one we do not send
    (DownloadAndLoadHy3DPaintModel and ...DelightModel both gained a required
    'model'). Checking only for unknown inputs caught the first and sailed straight
    past the second, which then failed at ComfyUI queue time -- minutes into the run,
    after the shape stage had already been paid for. Cheap check, expensive omission.
    """
    spec = info.get(class_type, {}).get("input", {})
    required = set((spec.get("required") or {}).keys())
    accepted = set(required)
    accepted.update((spec.get("optional") or {}).keys())
    if not accepted:
        return
    unknown = [k for k in inputs if k not in accepted]
    if unknown:
        raise MifOpError(
            "node '%s' has no input(s) %s. It accepts: %s. The wrapper changed under us - "
            "this is exactly the drift /object_info exists to catch."
            % (class_type, ", ".join(sorted(unknown)), ", ".join(sorted(accepted))))
    absent = sorted(required - set(inputs))
    if absent:
        raise MifOpError(
            "node '%s' requires input(s) %s which this workflow does not set. The wrapper "
            "changed under us - this is exactly the drift /object_info exists to catch."
            % (class_type, ", ".join(absent)))


# ---------------------------------------------------------------------------
# Queue + wait
# ---------------------------------------------------------------------------

def _submit(host, workflow):
    res = _post(host, "prompt", {"prompt": workflow}, timeout=120)
    pid = res.get("prompt_id")
    if not pid:
        raise MifOpError("ComfyUI accepted the request but returned no prompt_id: %s"
                         % json.dumps(res)[:300])
    return pid


def _wait(host, prompt_id, timeout_s):
    """Poll /history. ComfyUI has no completion callback, so polling is the contract."""
    deadline = time.time() + timeout_s
    stalls = 0
    while time.time() < deadline:
        # A STALLED READ IS "NOT YET", NOT A FAILURE (2026-09-13). ComfyUI's HTTP server stops answering for
        # longer than 60 s while the texture stage's paint and delight models saturate the GPU, and one timed-out
        # GET of /history used to raise straight out of a run that was still going - the retest that proved the
        # image handoff fix died this way in its third stage. The op's own deadline is the only clock that ends it.
        try:
            hist = _get(host, "history/%s" % prompt_id, timeout=60)
        except (TimeoutError, OSError, ValueError) as exc:
            stalls += 1
            if stalls <= 3 or stalls % 10 == 0:
                print("[MifBlender] gen: /history did not answer (%s: %s) - still waiting, %ds left"
                      % (type(exc).__name__, str(exc)[:80], int(deadline - time.time())))
            time.sleep(5.0)
            continue
        entry = hist.get(prompt_id)
        if entry:
            status = (entry.get("status") or {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                msgs = json.dumps(status.get("messages") or [])[:600]
                raise MifOpError("ComfyUI run failed: %s" % msgs)
            if entry.get("outputs"):
                return entry
            # A node that writes its file itself and registers no UI output leaves
            # outputs {} on a run that succeeded perfectly well - Hy3DExportMesh is one,
            # so EVERY mesh run landed here. Completion is the terminal signal; outputs
            # are a bonus. WAITED ON outputs UNTIL 2026-09-09, which turned every good
            # gen_mesh into a timeout against a file that was already on disk, and sent
            # the caller hunting a hang that did not exist.
            if status.get("completed") is True or status.get("status_str") == "success":
                return entry
        time.sleep(2.0)
    raise MifOpError(
        "ComfyUI did not finish within %ds. A first run pays for model load (the shape DiT is "
        "2.4GB, Flux fp8 is 16.6GB) - raise 'timeout' and try again before assuming a hang."
        % timeout_s)


def _outputs_of(entry, key):
    """Collect one kind of output across every node that produced it."""
    found = []
    for node_out in (entry.get("outputs") or {}).values():
        for item in (node_out.get(key) or []):
            found.append(item)
    return found


# ---------------------------------------------------------------------------
# Workflow fragments
# ---------------------------------------------------------------------------

def _wf_text_to_image(info, prompt, seed, variant, width, height, steps=None):
    cfg = FLUX.get(variant)
    if cfg is None:
        raise MifOpError("unknown flux variant '%s' - use 'schnell' or 'dev'." % variant)
    steps = steps or cfg["steps"]

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": cfg["ckpt"]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt + PROMPT_SUFFIX, "clip": ["1", 1]}},
        # cfg stays 1.0 for BOTH variants: Flux is guidance-distilled, so real
        # classifier-free guidance is not what steers it. dev is steered by FluxGuidance.
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    }
    if cfg["guidance"] is not None:
        wf["7"] = {"class_type": "FluxGuidance",
                   "inputs": {"conditioning": ["2", 0], "guidance": cfg["guidance"]}}
        wf["5"]["inputs"]["positive"] = ["7", 0]
    for nid, node in wf.items():
        _check_inputs(info, node["class_type"], node["inputs"])
    return wf


def _wf_shape(info, image_ref, prefix, seed, steps, octree, guidance):
    """Image -> mesh.

    octree_resolution is the detail dial. 384 is the wrapper's comfortable default; 512
    resolves noticeably more surface without changing the silhouette, and there is no point
    generating detail only for a later remesh to throw it away - so the default here is 512
    and the postprocess below is deliberately conservative.
    """
    wf = {
        "10": {"class_type": "Hy3DModelLoader",
               # attention_mode moved here from Hy3DGenerateMesh in a wrapper update. It is
               # selected on the LOADER now, so setting it on the sampler is a hard reject.
               "inputs": {"model": "hunyuan3d-dit-v2-0-fp16.safetensors",
                          "attention_mode": "sdpa"}},
        "11": {"class_type": "Hy3DGenerateMesh",
               # guidance_scale 5.0 is Tencent's value in both the upstream pipeline and
               # their gradio app; the wrapper defaults to kijai's 5.5.
               "inputs": {"pipeline": ["10", 0], "image": image_ref,
                          "guidance_scale": guidance, "steps": steps, "seed": seed}},
        "12": {"class_type": "Hy3DVAEDecode",
               "inputs": {"vae": ["10", 1], "latents": ["11", 0],
                          "box_v": 1.01, "octree_resolution": octree,
                          "num_chunks": 8000, "mc_level": 0.0,
                          "mc_algo": "mc", "enable_flash_vdm": True}},
        "13": {"class_type": "Hy3DPostprocessMesh",
               "inputs": {"trimesh": ["12", 0], "remove_floaters": True,
                          "remove_degenerate_faces": True, "reduce_faces": False,
                          "max_facenum": 200000, "smooth_normals": False}},
        "14": {"class_type": "Hy3DExportMesh",
               "inputs": {"trimesh": ["13", 0], "filename_prefix": prefix,
                          "file_format": "glb", "save_file": True}},
    }
    for node in wf.values():
        _check_inputs(info, node["class_type"], node["inputs"])
    return wf


def _wf_texture(info, mesh_ref, image_ref, prefix, seed, steps, view_size):
    """Mesh + reference image -> PBR-textured mesh.

    Order is not negotiable: delight the reference first, UV the mesh second, then render
    the mesh from several cameras, sample new views conditioned on the delit reference, bake
    those back down, and finally lift PBR channels out of the bake.
    """
    wf = {
        # Both download nodes gained a REQUIRED 'model' in a wrapper update; an empty
        # inputs dict is now a queue-time rejection, not a "use the default".
        "20": {"class_type": "DownloadAndLoadHy3DDelightModel",
               "inputs": {"model": "hunyuan3d-delight-v2-0"}},
        "21": {"class_type": "Hy3DDelightImage",
               "inputs": {"delight_pipe": ["20", 0], "image": image_ref,
                          "steps": 50, "width": 512, "height": 512,
                          "cfg_image": 1.5, "seed": seed}},
        # 'hunyuan3d-paint-v2-0-turbo' is the faster alternative if bake time ever hurts.
        "22": {"class_type": "DownloadAndLoadHy3DPaintModel",
               "inputs": {"model": "hunyuan3d-paint-v2-0"}},
        "23": {"class_type": "Hy3DMeshUVWrap", "inputs": {"trimesh": mesh_ref}},
        "24": {"class_type": "Hy3DCameraConfig",
               # Six orbit views plus top/bottom-ish elevations. Fewer views leaves seams on
               # whatever face no camera saw; more costs time without adding coverage.
               "inputs": {"camera_azimuths": "0, 90, 180, 270, 0, 180",
                          "camera_elevations": "0, 0, 0, 0, 90, -90",
                          "view_weights": "1, 0.5, 0.5, 0.5, 0.05, 0.05",
                          "camera_distance": 1.45, "ortho_scale": 1.2}},
        "25": {"class_type": "Hy3DRenderMultiView",
               "inputs": {"trimesh": ["23", 0], "camera_config": ["24", 0],
                          "render_size": view_size, "texture_size": 2048}},
        "26": {"class_type": "Hy3DSampleMultiView",
               "inputs": {"pipeline": ["22", 0], "ref_image": ["21", 0],
                          "normal_maps": ["25", 0], "position_maps": ["25", 1],
                          "camera_config": ["24", 0], "view_size": view_size,
                          "steps": steps, "seed": seed}},
        "27": {"class_type": "Hy3DBakeFromMultiview",
               "inputs": {"images": ["26", 0], "renderer": ["25", 2],
                          "camera_config": ["24", 0]}},
        "28": {"class_type": "Hy3DMeshVerticeInpaintTexture",
               "inputs": {"texture": ["27", 0], "mask": ["27", 1], "renderer": ["27", 2]}},
        "29": {"class_type": "CV2InpaintTexture",
               "inputs": {"texture": ["28", 0], "mask": ["28", 1],
                          "inpaint_radius": 3, "inpaint_method": "ns"}},
        "30": {"class_type": "Hy3DApplyTexture",
               "inputs": {"texture": ["29", 0], "renderer": ["27", 2]}},
        # prefix IS the export prefix, "_textured" suffix included - the caller builds it, because the
        # caller also has to derive the written filename from it (see op_gen_texture).
        "31": {"class_type": "Hy3DExportMesh",
               "inputs": {"trimesh": ["30", 0], "filename_prefix": prefix,
                          "file_format": "glb", "save_file": True}},
    }
    for node in wf.values():
        _check_inputs(info, node["class_type"], node["inputs"])
    return wf


# ---------------------------------------------------------------------------
# Import + quality gate
# ---------------------------------------------------------------------------

def _import_glb(path, name_hint):
    if not os.path.isfile(path):
        raise MifOpError(
            "ComfyUI reported success but '%s' is not on disk. If ComfyUI runs on another "
            "machine the file is on ITS filesystem - copy it over, or pass import_result=false."
            % path)
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    new = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if meshes and name_hint:
        meshes[0].name = name_hint
    return meshes


def _quality_report(objs):
    """What a generated mesh has to survive before it is worth anyone's time.

    Reported, never enforced: a mesh that fails one of these is often still the best
    starting point available, and silently rejecting it would waste the run. Loud is enough.
    """
    report = []
    for obj in objs:
        mesh = obj.data
        tris = sum(max(0, len(p.vertices) - 2) for p in mesh.polygons)
        dims = tuple(round(d, 4) for d in obj.dimensions)
        report.append({
            "object": obj.name,
            "triangles": tris,
            "vertices": len(mesh.vertices),
            "materials": len(mesh.materials),
            "uvLayers": len(mesh.uv_layers),
            "hasUVs": len(mesh.uv_layers) > 0,
            "dimensions": dims,
            "openEdges": _boundary_edges(mesh),
            "warnings": _warnings(tris, mesh, dims),
        })
    return report


def _boundary_edges(mesh):
    """Count edges used by fewer than two faces - a watertight mesh has none.

    Counted from polygon.edge_keys, which is one pass over the faces. The obvious version -
    for each edge, scan every polygon - is O(edges x faces) and turns a 200k-triangle
    generation into a hang. MeshVertex/MeshEdge also have no link_faces (that is bmesh), so
    there is no cheap adjacency to borrow.
    """
    used = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            used[key] = used.get(key, 0) + 1
    return sum(1 for count in used.values() if count < 2)


def _warnings(tris, mesh, dims):
    out = []
    if not mesh.uv_layers:
        out.append("no UVs - texturing and lightmaps will both fail until it is unwrapped")
    if not mesh.materials:
        out.append("no material - shape-only generation, run gen_texture to paint it")
    if tris > 250000:
        out.append("%d triangles is far above a game budget - decimate before use" % tris)
    if tris < 200:
        out.append("only %d triangles - the shape model probably failed to find an object" % tris)
    if max(dims) <= 0.0:
        out.append("zero-sized bounds - the import produced nothing usable")
    return out


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def op_gen_status(params):
    """Is the generator usable, and what is installed."""
    _require_generative("gen_status")
    reject_unknown(params, ("host", "server", "refresh", "timeout"), "gen_status")
    host = _host(params)
    # SHORT BY DEFAULT, unlike every other caller of _object_info. This is the op you reach for
    # to find out WHETHER the generator is usable, so the one thing it must not do is behave like
    # a hang: on the shared 120s it was the slowest op in the file at exactly the moment the
    # answer was "not reachable". 20s still clears a cold enumerate on a large node install, and
    # a caller who knows theirs is slower can say so.
    timeout = take_int(params, "timeout", default=20)
    # Default TRUE: gen_status exists to answer "what can the generator do right now",
    # and answering that from a cache taken before the last ComfyUI restart is the one
    # way it can be actively misleading.
    #
    # take_bool, BEFORE THE CONNECTION: bool(params.get("refresh")) read "false" as true and let a typo
    # through to the network - {refresh: "flase"} went on to try ComfyUI rather than being refused
    # (measured on 5.0.1, 2026-09-21).
    refresh = take_bool(params, "refresh", default=True)
    info = _object_info(host, refresh=refresh, timeout=timeout)

    have = {n: (n in info) for n in (
        "Hy3DModelLoader", "Hy3DGenerateMesh", "Hy3DGenerateMeshMultiView",
        "Hy3DVAEDecode", "Hy3DExportMesh", "DownloadAndLoadHy3DPaintModel",
        "DownloadAndLoadHy3DDelightModel", "Hy3DMeshUVWrap", "Hy3DRenderMultiView",
        "Hy3DSampleMultiView", "Hy3DBakeFromMultiview", "Hy3DApplyTexture",
        "CheckpointLoaderSimple")}

    ckpts = []
    try:
        ckpts = ((info.get("CheckpointLoaderSimple", {}).get("input", {})
                  .get("required", {}).get("ckpt_name") or [[]])[0])
    except Exception:  # noqa: BLE001
        pass
    shapes = []
    try:
        shapes = ((info.get("Hy3DModelLoader", {}).get("input", {})
                   .get("required", {}).get("model") or [[]])[0])
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "host": host,
        "nodeCount": len(info),
        "capabilities": {
            "shape": have["Hy3DGenerateMesh"],
            "shapeMultiView": have["Hy3DGenerateMeshMultiView"],
            "texture": have["DownloadAndLoadHy3DPaintModel"] and have["Hy3DBakeFromMultiview"],
            "delight": have["DownloadAndLoadHy3DDelightModel"],
            "textToImage": have["CheckpointLoaderSimple"] and bool(ckpts),
        },
        "nodesPresent": have,
        # THE TOTALS, because both lists stop at 20 and said nothing about it. This endpoint
        # answers "what can this install generate", so a truncated list is not a shortened
        # answer, it is a wrong one: somebody with 30 checkpoints was told about 20 and had no
        # way to know theirs was simply past the cut. nodeCount is len(info) and is a different
        # quantity - it never covered these.
        "checkpointCount": len(ckpts),
        "shapeModelCount": len(shapes),
        "checkpoints": list(ckpts)[:20],
        "shapeModels": list(shapes)[:20],
    }


def op_gen_image(params):
    """Prompt -> reference image, left in ComfyUI's output folder."""
    _require_generative("gen_image")
    reject_unknown(params, ("prompt", "seed", "variant", "width", "height", "steps",
                            "host", "server", "timeout"), "gen_image")
    host = _host(params)
    prompt = take(params, "prompt", required=True)
    seed = take_int(params, "seed", default=0)
    variant = str(take(params, "variant", default="schnell"))
    width = take_int(params, "width", default=1024)
    height = take_int(params, "height", default=1024)
    steps = take_int(params, "steps", default=None)
    timeout = take_int(params, "timeout", default=600)

    info = _require_nodes(host, ("CheckpointLoaderSimple", "KSampler", "VAEDecode"))
    wf = _wf_text_to_image(info, prompt, seed, variant, width, height, steps)
    wf["99"] = {"class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "MifGen/ref"}}
    entry = _wait(host, _submit(host, wf), timeout)
    images = _outputs_of(entry, "images")
    return {"ok": True, "images": images, "variant": variant, "seed": seed,
            "steps": steps or FLUX[variant]["steps"]}


def op_gen_mesh(params):
    """Reference image -> untextured mesh. Accepts a ComfyUI image ref or a local file."""
    _require_generative("gen_mesh")
    reject_unknown(params, ("image", "imagePath", "prefix", "seed", "steps", "octree",
                            "guidance", "host", "server", "timeout", "importResult",
                            "import_result", "name"), "gen_mesh")
    host = _host(params)
    prefix = str(take(params, "prefix", default="MifGen/mesh"))
    # Unique per run so the counter ComfyUI appends restarts at 1. That is what makes the
    # written filename derivable when the export node reports no output at all.
    run_prefix = "%s_%s" % (prefix, uuid.uuid4().hex[:8])
    seed = take_int(params, "seed", default=0)
    steps = take_int(params, "steps", default=30)
    octree = take_int(params, "octree", default=512)
    guidance = take_float(params, "guidance", default=5.0)
    timeout = take_int(params, "timeout", default=1800)
    do_import = take_bool(params, "importResult", "import_result", default=True)
    name = take(params, "name", default=None)

    image = take(params, "image", default=None)
    image_path = take(params, "imagePath", default=None)
    if not image and not image_path:
        raise MifOpError("gen_mesh needs 'image' (a ComfyUI image name) or 'imagePath' "
                         "(a file on this machine). Use gen_asset to go from a prompt.")

    info = _require_nodes(host, ("Hy3DModelLoader", "Hy3DGenerateMesh",
                                 "Hy3DVAEDecode", "Hy3DExportMesh"))
    if image_path:
        image = _upload_image(host, str(image_path))
    wf = _wf_shape(info, [str(image), 0] if not isinstance(image, list) else image,
                   run_prefix, seed, steps, octree, guidance)
    # LoadImage feeds the shape node when the caller supplied a filename.
    if not isinstance(image, list):
        wf["9"] = {"class_type": "LoadImage", "inputs": {"image": str(image)}}
        wf["11"]["inputs"]["image"] = ["9", 0]

    entry = _wait(host, _submit(host, wf), timeout)
    path, meshes = _resolve_mesh(host, entry, run_prefix)
    out = {"ok": True, "meshOutputs": meshes, "path": path,
           "octree": octree, "steps": steps, "seed": seed}
    if do_import and path:
        objs = _import_glb(path, name)
        out["imported"] = [o.name for o in objs]
        out["quality"] = _quality_report(objs)
    return out


def op_gen_texture(params):
    """Existing mesh + reference image -> PBR textures baked on."""
    _require_generative("gen_texture")
    reject_unknown(params, ("meshPath", "mesh_path", "image", "imagePath", "prefix",
                            "seed", "steps", "viewSize", "host", "server", "timeout",
                            "importResult", "import_result", "name"), "gen_texture")
    host = _host(params)
    mesh_path = take(params, "meshPath", "mesh_path", required=True)
    prefix = str(take(params, "prefix", default="MifGen/mesh"))
    run_prefix = "%s_%s" % (prefix, uuid.uuid4().hex[:8])
    seed = take_int(params, "seed", default=0)
    steps = take_int(params, "steps", default=15)
    view_size = take_int(params, "viewSize", default=512)
    timeout = take_int(params, "timeout", default=2400)
    do_import = take_bool(params, "importResult", "import_result", default=True)
    name = take(params, "name", default=None)

    image = take(params, "image", default=None)
    image_path = take(params, "imagePath", default=None)
    if not image and not image_path:
        raise MifOpError("gen_texture needs the reference image the mesh was generated "
                         "from ('image' or 'imagePath') - the painter conditions on it.")

    info = _require_nodes(host, ("DownloadAndLoadHy3DPaintModel", "Hy3DMeshUVWrap",
                                 "Hy3DRenderMultiView", "Hy3DSampleMultiView",
                                 "Hy3DBakeFromMultiview", "Hy3DApplyTexture",
                                 "Hy3DExportMesh", "Hy3DUploadMesh"))
    if image_path:
        image = _upload_image(host, str(image_path))

    # Hy3DUploadMesh takes a name from ComfyUI's INPUT dir, not a path - so a mesh that
    # exists on disk has to be uploaded first. A value that is NOT an existing file is
    # passed straight through, so a caller who already knows an input-dir name still works.
    mesh_ref = str(mesh_path)
    if os.path.isfile(mesh_ref):
        mesh_ref = _upload_mesh(host, mesh_ref)

    wf = {"5": {"class_type": "Hy3DUploadMesh", "inputs": {"mesh": mesh_ref}},
          "9": {"class_type": "LoadImage", "inputs": {"image": str(image)}}}
    # ONE NAME FOR THE EXPORT AND THE FETCH (2026-09-14). _wf_texture used to append "_textured" to the prefix
    # itself, so ComfyUI wrote <run_prefix>_textured_00001_.glb while _resolve_mesh derived <run_prefix>_00001_.glb
    # from the bare run_prefix - and Hy3DExportMesh registers no output, so the derived name is the ONLY way the
    # file is found. A texture job ComfyUI finished cleanly (21 minutes on a 559k-face mesh, the file on disk)
    # ended in "would not serve ... HTTP 404" and no import, for every gen_texture and every textured gen_asset.
    # Built once here and handed to both sides, so they cannot disagree again.
    export_prefix = run_prefix + "_textured"
    wf.update(_wf_texture(info, ["5", 0], ["9", 0], export_prefix, seed, steps, view_size))
    entry = _wait(host, _submit(host, wf), timeout)
    path, meshes = _resolve_mesh(host, entry, export_prefix)
    out = {"ok": True, "meshOutputs": meshes, "path": path, "steps": steps, "seed": seed}
    if do_import and path:
        objs = _import_glb(path, name)
        out["imported"] = [o.name for o in objs]
        out["quality"] = _quality_report(objs)
    return out


def op_gen_asset(params):
    """Prompt -> reference image -> mesh -> PBR texture -> imported into the scene.

    The one call that produces something usable. Set texture=false to stop at geometry.
    """
    _require_generative("gen_asset")
    reject_unknown(params, ("prompt", "name", "seed", "variant", "steps", "shapeSteps",
                            "textureSteps", "octree", "guidance", "texture", "width",
                            "height", "host", "server", "timeout", "importResult",
                            "import_result"), "gen_asset")
    host = _host(params)
    prompt = take(params, "prompt", required=True)
    name = take(params, "name", default=None)
    seed = take_int(params, "seed", default=0)
    variant = str(take(params, "variant", default="schnell"))
    want_texture = take_bool(params, "texture", default=True)
    do_import = take_bool(params, "importResult", "import_result", default=True)
    timeout = take_int(params, "timeout", default=3600)

    prefix = "MifGen/%s" % (str(name) if name else "asset")

    img = op_gen_image({"prompt": prompt, "seed": seed, "variant": variant,
                        "width": take_int(params, "width", default=1024),
                        "height": take_int(params, "height", default=1024),
                        "host": host, "timeout": timeout})
    image_name = _first_name(img.get("images"))
    if not image_name:
        raise MifOpError("the text-to-image stage produced no image; nothing to build from.")
    # THE IMAGE LIVES IN THE OUTPUT FOLDER AND LoadImage READS THE INPUT FOLDER (2026-09-13). gen_image writes
    # output/MifGen/ref_NNNNN_.png, and the bare filename handed to the shape workflow's LoadImage was refused
    # by ComfyUI before anything ran - "Invalid image file: ref_00002_.png" - after a text-to-image stage that
    # had worked. The error this op raised guessed at a renamed node input; it was the folder. Pulled back
    # through /view and uploaded to the input folder, exactly as gen_mesh treats a local imagePath; the name
    # ComfyUI returns from the upload is the one both later stages use.
    first = next((it for it in (img.get("images") or []) if isinstance(it, dict) and it.get("filename")), {})
    local_ref = _download_output(host, first["filename"], first.get("subfolder") or "")
    image_name = _upload_image(host, local_ref)

    mesh = op_gen_mesh({"image": image_name, "prefix": prefix, "seed": seed,
                        "steps": take_int(params, "shapeSteps", default=30),
                        "octree": take_int(params, "octree", default=512),
                        "guidance": take_float(params, "guidance", default=5.0),
                        "host": host, "timeout": timeout,
                        "importResult": do_import and not want_texture, "name": name})

    result = {"ok": True, "prompt": prompt, "seed": seed, "image": image_name,
              "shape": {"path": mesh.get("path")}}
    if not want_texture:
        result["imported"] = mesh.get("imported")
        result["quality"] = mesh.get("quality")
        return result

    if not mesh.get("path"):
        raise MifOpError("the shape stage produced no mesh file, so there is nothing to texture.")
    tex = op_gen_texture({"meshPath": mesh["path"], "image": image_name, "prefix": prefix,
                          "seed": seed,
                          "steps": take_int(params, "textureSteps", default=15),
                          "host": host, "timeout": timeout,
                          "importResult": do_import, "name": name})
    result["textured"] = {"path": tex.get("path")}
    result["imported"] = tex.get("imported")
    result["quality"] = tex.get("quality")
    return result


# ---------------------------------------------------------------------------
# helpers that need the HTTP layer
# ---------------------------------------------------------------------------

def _upload_image(host, path):
    """A local image file -> ComfyUI's input dir. Returns the name to use in LoadImage."""
    return _upload_file(host, path, "image")


def _upload_mesh(host, path):
    """A local mesh file -> ComfyUI's input dir. Returns the name to use in Hy3DUploadMesh.

    Hy3DUploadMesh's 'mesh' input is an ENUM over ComfyUI's INPUT directory (/object_info
    reports {"mesh": [[]]} on a clean install), so handing it an absolute path - e.g. the
    one Hy3DExportMesh just wrote into the OUTPUT dir - is an unconditional HTTP 400 at
    queue time. That made gen_texture unusable for every realistic input, and gen_asset
    with it, since gen_asset routes through op_gen_texture.

    /upload/image is not image-only despite the name: verified 2026-08-15 by uploading a
    9.3 MB .glb, which returned {"name": "...glb", "type": "input"} and landed in the
    input dir. It is ComfyUI's generic ingest endpoint.
    """
    return _upload_file(host, path, "mesh")


def _upload_file(host, path, kind):
    """ComfyUI's /upload/image is multipart, not JSON - hand-rolled to avoid a dependency.

    ONE implementation for both images and meshes: the two differed only in the error
    string, and a second hand-rolled multipart body is a second place for the boundary
    handling to be wrong. The form FIELD stays "image" for both - that is the field name
    the endpoint requires, regardless of what the bytes are.
    """
    if not os.path.isfile(path):
        raise MifOpError("no such %s: %s" % (kind, path))
    boundary = "----MifBlender%d" % int(time.time())
    name = os.path.basename(path)
    with open(path, "rb") as handle:
        blob = handle.read()
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="image"; filename="%s"\r\n' % name).encode(),
        b"Content-Type: application/octet-stream\r\n\r\n", blob, b"\r\n",
        ("--%s\r\n" % boundary).encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        ("--%s--\r\n" % boundary).encode(),
    ])
    req = urllib.request.Request(
        "http://%s/upload/image" % host.rstrip("/"), data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read() or b"{}")
    except TimeoutError:
        # THE READ HALF escapes URLError (see _post), so without this arm a slow reply surfaced as a bare
        # "TimeoutError: timed out" naming neither ComfyUI nor the file.
        raise MifOpError("uploading '%s' to ComfyUI at %s: the upload was sent and no reply came within "
                         "120 s. ComfyUI is probably busy on the GPU - retry once it is idle." % (path, host))
    except urllib.error.URLError as exc:
        raise MifOpError("uploading '%s' to ComfyUI failed: %s" % (path, exc))
    return res.get("name") or name


def _first_name(items):
    for item in (items or []):
        if isinstance(item, dict) and item.get("filename"):
            return item["filename"]
        if isinstance(item, str):
            return item
    return None


def _first_path(items):
    for item in (items or []):
        if isinstance(item, dict):
            for key in ("path", "fullpath", "filename"):
                if item.get(key):
                    return item[key]
        elif isinstance(item, str):
            return item
    return None


def _output_subpath(prefix):
    """A ComfyUI filename prefix -> (subfolder, basename)."""
    sub, _, base = str(prefix).replace("\\", "/").rpartition("/")
    return sub, base


def _download_output(host, filename, subfolder):
    """Pull a file ComfyUI wrote back through /view, into Blender's temp dir.

    The mesh nodes write into the OUTPUT dir themselves and register no UI output, so
    /history carries no path to import from. /view serves the bytes by NAME, which also
    means this works when ComfyUI runs on another machine - where a path would be on ITS
    filesystem and useless here.
    """
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder,
                                    "type": "output"})
    url = "http://%s/view?%s" % (host.rstrip("/"), query)
    try:
        with urllib.request.urlopen(url, timeout=300) as resp:
            blob = resp.read()
    except TimeoutError:
        # The read half escapes URLError, as in _post and the upload above.
        raise MifOpError("ComfyUI reported success and then did not finish serving '%s' within 300 s"
                         % filename)
    except urllib.error.URLError as exc:
        raise MifOpError(
            "ComfyUI reported success but would not serve '%s' from %s: %s"
            % (filename, subfolder or "the output root", exc))
    dest = os.path.join(bpy.app.tempdir or tempfile.gettempdir(), filename)
    with open(dest, "wb") as handle:
        handle.write(blob)
    return dest


def _resolve_mesh(host, entry, run_prefix):
    """The written mesh as a LOCAL path: from /history when it is there, else by name.

    Hy3DExportMesh registers no output, so entry["outputs"] is {} even on a clean run.
    run_prefix is unique per run, so ComfyUI's per-prefix counter starts at 1 and the
    name is derivable rather than a guess.
    """
    reported = _outputs_of(entry, "mesh") or _outputs_of(entry, "result")
    path = _first_path(reported)
    if path and os.path.isfile(path):
        return path, reported
    subfolder, base = _output_subpath(run_prefix)
    return _download_output(host, "%s_00001_.glb" % base, subfolder), reported


# project_mode sits HERE, after the shared helpers, not beside the switch it sets: param_reach reads the take()
# aliases declared above the FIRST op (_host's host/server) as every gen op's, so an op defined above them hides
# those aliases and reads five gen ops' `server` as unreachable.
def file_roadmap():
    """The roadmap this file follows, as the MCP server recorded it, or None. Each scene holds it as JSON text, as each holds the
    mode, so it travels with the .blend and a scene deleted later takes no copy of it away. Scenes appended from another
    file can disagree, so the newest chosenAt wins. A value that does not parse is left out here; the server reports a
    stored roadmap it cannot use, and recording one replaces it."""
    rows = []
    try:
        values = [s.get(ROADMAP_KEY) for s in bpy.data.scenes]
    except Exception:  # noqa: BLE001 - bpy.data unreadable here
        return None
    for raw in values:
        try:
            doc = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            rows.append(doc)
    return max(rows, key=lambda d: str(d.get("chosenAt") or "")) if rows else None


def op_project_mode(params):
    """Read, or record, the open file's project settings, the user's choices: mode (generative or non-generative), changeCap -
    how many things one call may change before it answers a preview instead (cap.py; 0 is off) - and roadmap, the plan this
    project follows as mif_project_mode built it ({} clears it). Its MCP tool
    is mif_project_mode (project 'blender'), one tool for both backends' settings, so no bl_* tool wraps this op: the
    server reaches it through _bl_set_mode and _bl_project_mode, and reads the mode back after open_file."""
    reject_unknown(params, ("mode", "changeCap", "roadmap"), "project_mode")
    from . import cap as _cap
    want = take(params, "mode", default=None)
    want_cap = params.get("changeCap")
    want_roadmap = params.get("roadmap")
    if want_roadmap is not None:
        if not isinstance(want_roadmap, dict):
            raise MifOpError("project_mode: roadmap must be an object, or {} to clear it, not %r. Nothing was changed."
                             % (want_roadmap,))
        if not bpy.data.scenes:
            raise MifOpError("project_mode: this file has no scene to record the roadmap on. Nothing was changed.")
    if want is not None:
        if want not in MODES:
            raise MifOpError("project_mode: mode must be one of %s, not %r. Nothing was changed."
                             % (", ".join(MODES), want))
    if want_cap is not None:
        _cap.record(want_cap)            # checked and recorded before the mode, so a bad cap changes nothing
    if want is not None:
        scenes = list(bpy.data.scenes)
        if not scenes:
            raise MifOpError("project_mode: this file has no scene to record the mode on. Nothing was changed.")
        for scene in scenes:
            scene[MODE_KEY] = want
    if want_roadmap:
        text = json.dumps(want_roadmap)
        for scene in bpy.data.scenes:
            scene[ROADMAP_KEY] = text
    elif want_roadmap is not None:
        for scene in bpy.data.scenes:
            if ROADMAP_KEY in scene:
                del scene[ROADMAP_KEY]
    mode = file_mode()
    out = {"generativeMode": mode, "blendFile": bpy.data.filepath or None,
           "allowGenerative": preference_allows(), "generativeAllowed": generative_allowed(),
           "changeCap": _cap.change_cap(), "roadmap": file_roadmap()}
    if want_cap is not None:
        out["recordedChangeCap"] = want_cap
    if want is not None:
        out["recorded"] = want
    if want is not None or want_cap is not None or want_roadmap is not None:
        out["note"] = ("recorded on %d scene(s) of this file; it is kept when the file is saved%s."
                       % (len(bpy.data.scenes), "" if bpy.data.filepath else
                          " - this file has never been saved, so save it to keep the choice"))
    return out


OPS = {
    "project_mode": op_project_mode,
    "gen_status": op_gen_status,
    "gen_image": op_gen_image,
    "gen_mesh": op_gen_mesh,
    "gen_texture": op_gen_texture,
    "gen_asset": op_gen_asset,
}
