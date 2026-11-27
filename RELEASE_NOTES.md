MifBlender 1.5.0.

469 tools (1.4.0 had 437), tested on Blender 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 on Windows.

New in this release:

- **Undo and redo as tools.** `bl_undo`, `bl_redo` and `bl_undo_history`: every call that changes the scene is one
  Blender undo step, and the answer says what the scene gained and lost.
- **Kit pieces.** `bl_build_stairs`, `bl_build_railing`, `bl_build_sign` and `bl_build_decal`, with `bl_kit_audit` and
  `bl_kit_preview` to check a set of pieces together.
- **Checking a model.** `bl_ray_cast`, `bl_vertex_info`, `bl_silhouette_stats`, `bl_self_clipping`, `bl_pose_seams`,
  `bl_render_pixel_probe` and `bl_look_studio` for a neutral studio look at the result.
- **Modelling and UVs.** `bl_sdf_union`, `bl_split_plates`, `bl_conform_sheet`, `bl_cell_pattern`, `bl_uv_to_cell`,
  `bl_unwrap_organic`, `bl_export_mirrored_side`, `bl_compose_maps`, `bl_normal_map_convention` and
  `bl_bake_modifier_texture`.
- **Roblox checks for a model made in Blender.** `bl_roblox_preflight`, `bl_roblox_ugc_check`, `bl_roblox_kit_check`,
  `bl_roblox_fit_check`, `bl_roblox_pbr_bake`, `bl_roblox_bone_clip` and `bl_roblox_action_samples`.
- **Scenes and view layers.** `bl_set_current_scene`, `bl_delete_scene`, `bl_list_view_layers`, `bl_link_objects` and
  `bl_unlink_objects`.
- **Finding tools.** `bl_describe_op` describes any op's parameters before a call; `mif_help`, `mif_list_packs` and
  `mif_enable_pack` show and load the tool packs.
- Many fixes to existing tools, each read back after it runs: among them duplicate_object keeping its rotation and scale
  in a collection hidden in the viewport.

The add-on zip is attached below as `MifBlender-1.5.0.zip`. To install or upgrade with uv: `uvx mifblender-mcp setup`
(or `--upgrade`).
