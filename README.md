<!-- mcp-name: io.github.mifsopo1/mifblender -->
<p align="center">
  <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_hero.jpg" alt="A plated crab walker on a desert stage at sunset, built and rendered in Blender" width="100%">
</p>

<h1 align="center">MifBlender</h1>

<p align="center"><b>Your AI assistant, working inside a running Blender.</b><br>
428 typed tools for modeling, materials, UVs, rigging, animation, physics, geometry nodes and rendering.<br>
Each one checks its parameters, does the work on Blender's main thread and reads the result back.</p>

<p align="center">
  <a href="https://pypi.org/project/mifblender-mcp/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mifblender-mcp"></a>
  <img alt="Free" src="https://img.shields.io/badge/price-free-brightgreen">
  <img alt="Blender 3.6 to 5.2" src="https://img.shields.io/badge/Blender-3.6.23%20%C2%B7%204.2.17%20%C2%B7%204.4.0%20%C2%B7%205.0.1%20%C2%B7%205.2.1-orange">
  <img alt="Windows" src="https://img.shields.io/badge/tested%20on-Windows-blue">
</p>

## Start in one line

With [uv](https://docs.astral.sh/uv/) installed and Blender closed:

```
uvx mifblender-mcp setup
```

It finds your Blender installs and AI apps, shows what it will do and asks first. Then it installs and enables the add-on,
and adds MifBlender to the apps you pick. Restart your AI app, open Blender, and ask:

> Call bl_status, then add a rounded cube with a brushed metal material and render a preview.

Free for any use, including commercial work. No account. Everything runs on your computer.

## See what it can do

<p align="center">
  <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_walk.gif" alt="The crab walker's walk cycle" width="100%">
</p>

The crab walker: 515 pieces, an interior, a rig and a walk cycle, built with these tools in a long, guided session.

| The form, in clay | The topology |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_clay.jpg" alt="Clay render of the crab walker"> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_wire.jpg" alt="Wireframe close-up of a crab walker leg"> |

| The cabin | Every angle |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_interior.jpg" alt="The crab walker's lit wooden cabin"> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_turntable.jpg" alt="Eight views around the crab walker"> |

<p align="center">
  <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/crab_rig.jpg" alt="The crab walker's game rig in wireframe through one walk cycle" width="100%">
</p>

### From images to a model

The creature was built from one six-view character sheet in eight staged sessions, from blockout to a rigged walk. Each
session started fresh, with only the saved file and a running journal to go on, and every round was checked against the sheet
with `bl_compare_to_reference`.

<p align="center">
  <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/creature_walk.gif" alt="The creature's walk cycle from a three-quarter camera and from the side" width="100%">
</p>

The finished model: 910,104 vertices with a painted tone on each vertex, and a 77-bone rig with IK on the arms and legs. In the
walk loop, rendered in Cycles, the planted feet slide at most 1.9 mm per frame, the last frame matches the first exactly, and the
claws stay within 0.008 mm of the fingers.

| Reference and model | The face |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/creature_pair.jpg" alt="The creature from the front, beside its reference sheet"> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/creature_head.jpg" alt="Close-up of the creature's head beside its reference"> |

<p align="center">
  <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/creature_views.jpg" alt="The finished creature from eight angles" width="100%">
</p>

**What the build taught:**

- **A good score is not a good look.** The outline matched the sheet at .88 to .90 while the body still read as a balloon. Numbers
  catch the shape, and someone still has to look at the picture.
- **Stop when the rounds stop helping.** When the scores had been flat for half an hour, more rounds bought nothing, so the work
  moved to the next stage.
- **Reference views can disagree.** On a painted sheet, the 3/4 views did not match the front, side and back, so the build held to those three.
- **Honest numbers can fall.** Real fingers scored lower than the mitts they replaced, because the mitts had covered the painting by
  accident. Silhouette fit after the surface stage: front .85, side .89, back .84.
- **A dense mesh needs another way to weight.** Blender's heat weighting failed on a mesh this size, so the weights were computed
  from capsules around each bone instead.

### One prompt, one model

Each of these was built from the one prompt under it:

| | |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_robot.jpg" alt="A boxy robot on tank treads"><br><sub>"Model a cute boxy robot with antenna, round eyes and tank treads, lit for a nice render."</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_bonsai.jpg" alt="A bonsai tree in a shallow pot"><br><sub>"Model a bonsai tree in a shallow ceramic pot with a few small rocks, lit for a nice render."</sub> |
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_lowpoly_car.jpg" alt="A low-poly cartoon car"><br><sub>"Model a low-poly cartoon car with round wheels, headlights and a little spoiler, lit for a nice render."</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_teapot.jpg" alt="A teapot and cup on a wooden tray"><br><sub>"Model a cute ceramic teapot with a cup and saucer on a small wooden tray, lit for a nice render."</sub> |
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_castle_tower.jpg" alt="A stone castle tower with a flag"><br><sub>"Model a stone castle tower with battlements, arrow slits and a flag on top, lit for a nice render."</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/prompt_chest.jpg" alt="An open treasure chest spilling gold coins"><br><sub>"Model a wooden treasure chest with iron bands, its lid open, and gold coins spilling out, lit for a nice render."</sub> |

### Simulation, shading and scale

Scripted scenes, each built entirely from typed calls:

| | | |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_newyork.jpg" alt="Manhattan as a scale model"><br><sub>Manhattan as a scale model</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_liquid.jpg" alt="Liquid poured over a head"><br><sub>A baked liquid sim</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_pile.jpg" alt="125 toys poured into a crate"><br><sub>125 rigid bodies</sub> |
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_robotarm.jpg" alt="A robot arm carrying a block"><br><sub>IK with a baked carry</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_facepaint.jpg" alt="27 cubies painted face by face"><br><sub>27 cubies, painted face by face</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_staircase.jpg" alt="A spiral stair"><br><sub>A spiral stair from one tread</sub> |
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_geodesic.jpg" alt="A geodesic lit from inside"><br><sub>A geodesic lit from inside</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_fur.jpg" alt="A furred sphere"><br><sub>30,000 hairs from one call</sub> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/scene_passes.jpg" alt="One render shown as beauty, normals and mist"><br><sub>Render passes</sub> |

### Game props

| Clean to worn, from one setting | Topology you can check |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/props_wear.jpg" alt="A table, crate, chair and barrel, clean and worn"> | <img src="https://raw.githubusercontent.com/mifsopo1/MifBlender/v1.3.1/assets/props_wire.jpg" alt="Five game props in clay and wireframe"> |

## What is in it

| Area | Examples |
|---|---|
| **Modeling** | primitives, mesh editing, bevels, booleans, modular kit pieces, rocks and trees built by rule |
| **Materials and UVs** | shader node graphs, texture sets, baking, unwrapping |
| **Rigging and animation** | armatures, Rigify, poses, keyframes, walk cycles |
| **Simulation** | rigid bodies, cloth, particles, geometry nodes |
| **Rendering** | lights, cameras, Cycles and EEVEE stills and sequences, compositing |
| **Checking** | mesh quality reports, comparing a render to a reference image, reading results back |

A lean default sends only `find_tools`, `mif_call` and `run_python` up front and finds the rest on demand, so a long session
stays small. Set `MIF_TOOL_PACKS=blender` to load all 428 tools, or name the packs you want; there are 13.

## AI apps

`setup` knows Claude Code, Claude Desktop, Cursor, Windsurf, VS Code and Codex. For any other MCP client, start this as a stdio
server:

```
uvx mifblender-mcp
```

Useful setup options: `--dry-run` shows the plan and changes nothing, `--check` reports what is set up, `--upgrade` moves the
add-on and the apps to this version together, and `--uninstall` removes them. If several Blenders are found, it asks which.

## Settings

Set these in the client's `env` block when you need them:

| variable | default | |
|---|---|---|
| `MIF_BLENDER_PORT` | `8792` | the port the add-on listens on |
| `MIF_BLENDER_TOKEN` | set by `setup` | must match the add-on's **Token** preference |
| `MIF_TOOL_PACKS` | `lean` | `lean`, `blender`, or a list of packs |
| `MIF_BLENDER_TIMEOUT` | `180` | seconds one call may take |

## Safety

- Everything stays on your computer. The add-on listens on `127.0.0.1` only, and `setup` generates a private token that links
  it to your AI apps. Neither makes any network call of its own.
- **`run_python` is on by default and runs any Python inside Blender.** Untick **Allow run_python** in the add-on's preferences
  to limit the assistant to the typed tools.
- The `bl_gen_*` tools are generative: they send a prompt to a ComfyUI you run yourself, and do nothing without one. Every other
  tool builds from explicit parameters. Set `MIF_GENERATIVE=off`, or untick **Allow generative tools**, to switch them off.

## Tested on

Blender 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, on Windows. macOS and Linux have not been tested yet; reports are welcome.

## License

**Free, source available.** The add-on is MIT. The server is free for any use, including commercial work, and you may share it
unchanged through the channels its license names; you may not sell it or publish changed copies. It is not an open source
license. See [LICENSE.txt](LICENSE.txt). Some comments cite internal documents that are not published.

## MifBridge

MifBlender is the Blender half of [MifBridge](https://mifshowcase.store/mifbridge), a paid product that adds Unreal Engine 5.3 to
5.8 and the tools that move models between the two. MifBlender stands on its own and needs neither.

Questions and bug reports: [open an issue](https://github.com/mifsopo1/MifBlender/issues).
