# Changelog

## 0.1.1 (2026-10-04)

Security, after a review of the published code:
- **Director window:** values from a workflow or a composition file (layer kind, blend mode, shape type, number
  fields, view names) are escaped before they become HTML. A file from someone else could run script in the ComfyUI
  page before.
- **scene render `blender_path`:** only `blender.exe` (or the folder that holds it) on a local drive is started.
  Another program is only taken from `blender = ...` in kubakub.ini.
- **save folders** (region video sampler, scene render, regions from sam3, versions, regions to svg / pdf / dxf): a
  folder typed into a node stays inside ComfyUI's output folder; `save_anywhere = on` in kubakub.ini allows any
  folder again. The video sampler's `filename_prefix` is a file name only. `kubakub/save_paths.py`,
  `tests/test_save_paths.py`.
- **Director routes:** with ComfyUI started with `--listen`, the window's routes no longer read videos, images or
  HDRIs by path (`remote_paths = on` allows it); an HDRI on a network share is never opened from a route; the
  projection mask preview has a size limit; errors return their first line only.
- **Saved workflows** no longer carry the LoRA folder listing in the director node (it comes back with the next run).
- **.blend files** are opened with their scripts off, explicitly.
- **Setup tool:** the Hugging Face token is not sent along when a download is redirected to another host; an
  incomplete download is kept as `.part` instead of being renamed.
- **Publish action:** pinned to a commit, read-only permissions.

Fixes:
- Folders with characters outside the Windows code page (a user name like Zażółć): scene render stopped, the
  director showed no pictures, ID passes were not written. All image files go through `kubakub/imio.py` now.
- The director's `regions` output keeps the scope of the regions it got.
- regions from sam3: an object under `min_area` no longer removes every other object with the same name.
- keyframe clips: `crossfade` uses at most 12 frames, the number the director keeps with skip_h3_frames.
- relief field: an alpha channel is not averaged into the depth.
- 16-bit greyscale files (depth passes, After Effects masks) keep their range in load images from a folder and as a
  versions start / style image; the folder loader's name filter treats only `*` and `?` as wildcards.
- onnx style transfer: overlap as large as the tile size, a side shorter than the overlap, RGBA input.
- regions to mask: a `select` that matches nothing is an error instead of an empty mask.

## 2026-10-04
- New layout: the node files are in `nodes/<group>/`, one folder per menu group (project, regions, sketch,
  generate, director, motion, post, scene3d, from_renders, fabricate, lab, utils); the logic they call is in
  `kubakub/` (was `kuba_regions/`, now with `facade_core.py` and `relief_core.py`). Node ids, display names and
  categories are unchanged, saved workflows load as before. File paths in older entries below name the old places.
  Tests import a node module as `kubapack.nodes.<group>.<module>` (`tests/_pack.py`).
- `.comfyignore`: `tests/` and `.github/` stay out of the Comfy Registry package.
- kubakub region video sampler: **loop**. Every clip returns to the still and ends exactly on its first frame, so
  the video can repeat (a still only, not into a background video). The last latent frame is held like the first;
  its 8 decoded frames are faded onto frame 0, because the LTX VAE drifts while decoding them.
- kubakub region video sampler: **audio**. A sound becomes each clip's audio latent, held through both stages, so
  the regions are sampled together with it; with an audio-reactive LoRA on the model the regions move with the
  sound while the facade stays pixel exact. The clip's stretch of the sound is part of its cache key; a short
  sound is padded with silence. New inputs are last, so saved workflows keep their values.
  `kuba_regions/video.py` (`latent_mask(loop=)`, `close_loop`, `audio_range`), `tests/test_video.py`, docs/director.md.
- First steps for a new install:
  - starter 01 needs nothing any more: its vector node (regions as svg) needed the optional package shapely and
    stopped the workflow on a clean ComfyUI; the note now points to the node instead;
  - a missing optional package (shapely, scikit-image, PyMuPDF, onnxruntime) says what to install and how
    (`kuba_regions/optional.py`), and an unreadable EXR ID mask says what it needs;
  - README: install by hand or by Git URL (the pack is not in the Manager list yet), the templates are listed under
    the folder name; a table of what each part needs (nothing, a package, models, Blender, your own files);
  - the older example workflows (facade_regions_diffusion, frame_in_frame, regions_from_masks_patch,
    regions_sam3_masks, regions_to_vector) start from the sample facade instead of a matrix file and a mask folder,
    and name model files without subfolders;
  - rarely needed inputs of the large nodes are advanced inputs now (region video sampler, versions, keyframe
    clips, scene relight, scene render, regions from matrix / masks, regions from sam3, region sampler); order and
    names are unchanged;
  - tooltips and docs use the current node names.

## 2026-10-01
- kubakub versions: **[image]** in the sheet, the picture a version starts from (a file, a folder, or `input`
  for the connected image), so versions can rotate between the clay render, a wireframe, a canny map, a
  cryptomatte. A batch of several images on the `image` input gives the same list. Same view as the regions;
  another size is scaled, another shape is refused.
- kubakub versions: **generate = whole picture**. One free sample of the whole picture from an empty latent,
  the starting image as reference image 1 and the style image as image 2, no masks (regions: every region
  inside its own mask, as before). For large forms that cross the facade's elements; about 33 s per draft at
  1.5 MP with Klein 9B, a 3200x2160 final from it 5.5 min (12 tiles).
- kubakub versions: **unify**, one more pass over the whole picture after the regions (a denoise, 0 = off):
  ties regions with different prompts, e.g. a random idea per region with `{a|b|c}`, into one structure.
- kubakub versions: **picks_folder**. The node now saves its versions itself (`save_folder`, default
  `output/kubakub/versions`; drafts `r<run>_v<number>_<label>.png`, finals in `final/`), and every file holds its
  version (rules, LoRA chain, style image, seed). Copy the drafts you like into a folder, paste the path into
  `picks_folder`, quality final: each image becomes a final with what it was made with, independent of the sheet
  and of `pick`. Older drafts written by Save Image are recognised as long as the run's files are still where
  they were saved and the copy keeps its name. New inputs are last, so saved workflows keep their values; remove
  the Save Image node of the versions from older workflows (the example workflow has none any more).
- New node **kubakub versions** (2d / generate): many versions of one facade from one queue. A versions sheet
  lists what changes, one option per line: `[look]`, `[region <selector>]`, `[rotate a | b | c]` (prompt
  rotation), `[lora]` (name : strength : words), `[reference]` (style images, a file or a folder), `[set <setting>]`,
  `[seed]`; modes rotate / combine / one by one. Draft = a numbered contact sheet (canvas scaled down, regions
  with the same treatment sampled together: 6-8 s per version on a 3200x2160 facade with Klein 4B, the region
  sampler needs 32 s); final = the picked numbers: the draft scaled up and resampled tile by tile at 1:1
  (about 95 s each), so a final is its draft with full-size detail. Unchanged versions come from the region
  cache. `kuba_regions/versions.py`, `kuba_regions/nodes_versions.py`, `tests/test_versions.py`,
  `example_workflows/versions.json`, docs/generate.md.
- kubakub versions: `scheduler` = auto (the model's own schedule: Flux 2 uses core's Flux2Scheduler), any core
  scheduler (simple, beta, sgm_uniform ...) as in KSampler, or the Qwen turbo LoRA schedules. LoRA chains in the
  sheet: `a : 1.0 + b : 0.5`, with a short name `pair = a : 1.0 + b : 0.5`. A reference folder can give a few
  images spread over it: `folder | 12`, `folder | 12 | 5`. For a negative prompt set `cfg = 2` and `negative = ...`
  in the plan.
- Region plan: `reference = style` (the region's crop plus a style image as a second reference; used by kubakub
  versions); the plan keeps its rules text. Regions can be scaled to another canvas size (`Regions.resized`).
  `seams.refine_tiles`: the whole canvas tile by tile with blended tile borders.

## 2026-09-29
- Settings with plain names in `kubakub.ini`, section `[settings]` (`region_cache_mb`, `cond_cache_mb`,
  `director_seq_cache`, `cache_folder`, `cache_gb`, `blender`, `blender_worker`, `blender_idle`; see
  `kubakub.ini.example`). The old environment variables still work. `settings.py`, `tests/test_settings.py`.
- README, docs and changelog describe nodes and connections by their names instead of internal ids / type names.
- The optional setup tool saves the Blender path in kubakub.ini instead of a user environment variable.
- Director window: Alt + drag moves a copy of the layer (the original stays).
- New menu group **3d / pieces** (a small MOPS): kubakub scene pieces, kubakub pieces falloff (wave, pulse,
  stagger, noise), kubakub pieces transform (push, move, tilt / turn / roll, random rotate, scale, jitter) and
  kubakub pieces render (Cycles, clay or the matrix projected on the moving pieces, every frame cached).
  `kuba_regions/scene3d/pieces.py`, `kuba_regions/nodes_pieces.py`, the relight job moves piece vertices per frame
  (`blender_export.py`), `tests/test_pieces.py`.
- Pieces: regions as the source (every part votes for its region: one block per region, or only the parts inside
  the selected regions), kubakub pieces beat falloff (beats / bars / markers of the director timeline, ripple by
  order), and the audience view in pieces render (a camera in front of the building; the projector stays).
- Review fixes (pieces): every frame is cached on its own (a longer clip renders only the new frames; identical
  frames render once); renders go to Blender in chunks of 40 (kept as they finish, Stop works between chunks, no
  15-minute limit for long clips); radial wave / pulse at the set speed; a looped wave waits for its start; random
  direction = a random ripple; region pieces keep the size filter and cap; seeds stay fixed (a random seed after
  every run re-rendered everything); the audience inputs moved to the end and are optional (older saves and API
  graphs keep working); Blender puts the scene back after an error; a pieces cache purge no longer breaks a render;
  tooltips on every input. Director window: a plain Alt click copies nothing, Alt drag = one undo step.
- Fix: kubakub keyframe clips (H3) and the region video sampler with the prompt enhancer and two or more clips
  crashed ComfyUI on the second enhance (CUDA 'scatter gather index out of bounds'): core's between-node cleanup is
  now run after every enhance inside the node (`kuba_regions/runtime.py`). GPU-checked: two H3 clips, enhancer on.

## 2026-09-28
- Simpler install: `setup_kubakub.bat` removed; README says ComfyUI Manager or git clone + requirements.txt (OpenCV
  only) + Blender for the 3D nodes. The model downloader stays as an optional `tools/setup_kubakub.py` (docs/MODELS.md).
- Display names of TrimToExactDuration and TimecodeFilenamePrefix now start with "kubakub " (node ids unchanged).
- README split: a short front page (quick start, groups, install, requirements, documentation table, credits,
  license) and one page per menu group in `docs/` (regions, sketch, generate, director, post, scene3d,
  from-renders, fabricate). Dev status, dates and single-machine notes removed; timings say "12 GB laptop GPU".
- Removed two private lab experiments (their nodes, workflows and test) from the pack; 55 nodes.
- `setup_kubakub.bat` + `tools/setup_kubakub.py`: guided setup (packages after a pip dry run, ffmpeg, Blender path
  -> the blender setting, model sets downloaded into ComfyUI/models, resumable, Hugging Face token for gated files);
  `--check` only reports. Finds models through extra_model_paths.yaml. `docs/MODELS.md`, `tests/test_setup.py`.
- License: GPL-3.0 (`LICENSE`, the text shipped with ComfyUI), README "License", `pyproject.toml` license field.
- Merged the two nodes of the old Comfy_KubakubNodes pack (node ids unchanged, saved workflows keep working):
  **Trim To Exact Duration (LTXV-safe)** (TrimToExactDuration, kubakub/2d/motion, `ltxv_trim_exact_Kub.py`): audio
  of a constant sample count whatever the start (loop / silence_pad / clamp), so the LTX AV guide mask never sees a
  shorter clip; **Timecode Filename Prefix** (TimecodeFilenamePrefix, kubakub/utils, `timecode_filename_Kub.py`):
  clip start in seconds -> `name_mm-ss-ff` for sortable files. `tests/test_trim_timecode.py` (model free).
- Node menu regrouped into 2d and 3d: kubakub/project, 2d/{regions, generate, director, motion, post},
  3d/{scene, from renders, fabricate}, lab, utils. Only categories changed; node ids are the same, saved workflows
  keep working.
- New **kubakub project settings** (kubakub/project): name, matrix size (presets or a connected matrix),
  fps, facade width and the audience in metres -> width / height / fps / name / viewer outputs to link into the other
  nodes. Logic in `kuba_regions/project.py`.
- `kubakub.ini` (see `kubakub.ini.example`) switches menu groups off, e.g. `3d = off` (`menu_switches.py`).
- Removed the unused old `nodes.py` (ReliefForge copy; `relief_forge_Kub.py` is the loaded one).
- `tests/test_project.py` (model free).
- New **kubakub brightness compensation** (kubakub/3d/scene): evens the projector's
  light over the building (brightness map of scene measure -> gain in linear light, grazing faces ignored for the
  level, optional one gain per region, lift for dim parts). `kuba_regions/scene3d/compensate.py`,
  `tests/test_compensate.py`; added to the scene3d example workflow.
- Sketch tools (kubakub/2d/sketch, `kuba_regions/sketch.py`, `nodes_sketch.py`): **scan to line** (paper
  search, perspective, paper / shadow / grain removal, pencil or ink, batches), **regions from sketch** (gap bridging
  of hand strokes, coloured marker dots as tags / names), **line overlay**. Sample facade got a `sketch_photo`
  output (a synthetic phone photo of a pencil sketch); starter 05 sketch to render; `tests/test_sketch.py`.
  Marker dots (compact colour blobs) are names: left out of the line output (`dots_in_line` keeps them).
- New **kubakub regions from cryptomatte** (kubakub/3d/from renders): objects by
  their scene names, groups from the names, material layer as tags, the render picture. Own numpy EXR reader
  (`kuba_regions/exr.py`: scanline / tiled, NONE / RLE / ZIPS / ZIP, all header attributes, plus a writer), other
  compressions through Blender's OpenImageIO; `kuba_regions/cryptomatte.py`; `tests/test_cryptomatte.py`.
- Post nodes (kubakub/2d/post, `kuba_regions/post.py`, `nodes_post.py`): **colour match** (mkl / mean_std, one
  transform for all frames, bake to .cube), **apply lut** (.cube 1D / 3D, trilinear), **deflicker** (zone means
  smoothed over time in linear light), **retime** (DIS optical flow in-betweens, blend, nearest), **burn in**
  (name / frame / timecode). `tests/test_post.py`.

## 2026-09-23
- Repo created; baseline commit of the existing Kub nodes.
- Added kubakub facade mask atlas (kubakub/facade): color_regions, line_drawing and
  mask_folder modes, small region merge/drop, shape and colour-name grouping,
  JSON region table and labelled preview. Pure logic in `facade_core.py`.
- Added `tests/test_facade_masks.py` (model free).
- Kuba Regions M1 step 1, geometry: `kuba_regions/geometry.py` (canvas plan per model
  family, uniform downscale + grid padding, restore with exact crop, frame rules) and
  nodes kubakub canvas plan, kubakub canvas to work, kubakub canvas restore (kubakub/regions).
  `tests/test_geometry.py` (67 checks, model free).
- M1 step 2, atlas for mask folders: kubakub facade mask atlas reads subfolders (`recursive`),
  takes `scope_masks` (not regions, limit all regions), `split_masks` (one region per
  separate shape, `<stem>_01..`), `tag_only_masks`, `group_by` stem/folder, an optional
  `scope` MASK in any mode; regions get `tags` (covering masks). New outputs `scope` and
  `regions` (label map, `kuba_regions/types.py`); `output_masks` switch.
  Existing inputs, outputs and defaults unchanged except recursive (on; same result on flat
  folders). New test `test_mask_folder_groups`.
- M1 step 3, kubakub region plan (kubakub/regions): INI-like rules with name / group / tag /
  region selectors (wildcards, AND, OR, NOT), priority by specificity, `+=` and `{prompt}`
  composition, `{a|b}` seeded choices, optional rules_file, report and preview. Logic in
  `kuba_regions/plan.py`, plan type (`RegionPlan`). `tests/test_plan.py` (32 checks).
  `docs/rules_example.txt`.
- M1 step 4, kubakub region sampler (kubakub/regions): model adapters (`kuba_regions/adapters.py`:
  Flux 2 with reference latents and core Flux2Scheduler sigmas, generic KSampler fallback),
  S1 sequential inpaint (`strategies.py`) with exact uniform crop scaling on the 16 px grid,
  scope-limited masks, mean_std / mkl colour match, feathered paste (`ops.py`).
  `tests/test_sequential.py` (26 checks, fake adapter). Tested with Klein 9B fp8 on the
  a 3840x2160 matrix: ~10 s per region.
- M1 step 5, S4 frame in frame (stills) in kubakub region sampler: own scene per region from an
  empty latent at the region's aspect, uniform cover + centre crop, placed through the exact
  mask, optional harmonize ring (fif_harmonize, fif_border_px). `strategies.run_plan` now
  dispatches inpaint / frame_in_frame in plan order; `Adapter.empty_latent`. Tests extended
  (fake adapter). Tested with Klein 9B on two floor 1 windows (~14 s each).
- M1 step 6, kubakub seam pass (S5): soft linear band along changed region borders (blend on, keep
  and scope respected), 1:1 tiles only where seams are, low denoise with reference, optional
  differential diffusion (core function copied). `kuba_regions/seams.py`, `tests/test_seams.py`.
  Tested with Klein 9B on a 3840x2160 matrix (~14 s per tile). Known limit: one prompt per tile.
- Qwen Image 2.1 adapter (`QwenImage21Adapter`): per-region conditioning with the crop through the
  Qwen3-VL vision tower plus reference latent, grid 32, 25 steps. `Adapter.conds_for` replaces the
  separate cond / add_reference calls in the strategies. Region Sampler and Seam Pass `steps` / `cfg`
  now default to 0 = the model's default. `qwen_image21` geometry for Canvas Plan. Warning when the
  Qwen cache is on auto. Tested on facade windows (int8 model, w4a8 encoder, cache gpu).
- `schedule` option on Region Sampler and Seam Pass: `qwen21_turbo_5` for the Viggle Qwen-Image-2.1
  turbo LoRA (fixed 5 trained timesteps, dynamic shift, partial denoise from the matching step).
  `kuba_regions/schedules.py`, `tests/test_schedules.py`. LoRA key mapping checked offline against
  the int8 checkpoint (227/227). GPU test 2026-09-24: fif ~25 s (was ~130 s), inpaint ~20 s (was ~75 s).
- Region Sampler report shows the schedule.

## 2026-09-24
- **M1 finished.** Example workflow `example_workflows/facade_regions_diffusion.json`: Mask Atlas -> Region Plan ->
  Region Sampler Klein 9B (`only = !W_F1_*`, 67 regions) -> Region Sampler Qwen 2.1 + turbo LoRA
  (`only = W_F1_*`, 7 frame-in-frame windows) -> Seam Pass (Klein) -> Save. A full run with the same
  graph: 1025 s (Klein 9.8 s/region, Qwen 27 s/window, seams 12 tiles x 12 s), registration exact.
- M2 step 1, **kubakub regions from masks** (kubakub/regions, `kuba_regions/nodes_sources.py`): MASK batch +
  names -> regions, or patched into `base_regions` (`on_top` cuts out, `underneath` fills gaps);
  new regions inherit the tags of the region they were cut from. `facade_core.label_masks()` now
  holds the mask labelling shared with the mask folder atlas (atlas results unchanged, 9/9 tests);
  `atlas_from_masks()` and `parse_mask_names()` are new. `tests/test_regions_from_masks.py` (8 tests;
  60 masks into a 4K base in 1.6 s). Example `example_workflows/regions_from_masks_patch.json`.
- M2 step 2, **kubakub regions sam3 masks** (kubakub/regions): named masks from core SAM3 Detect with
  SAM 3.1. Text lines `name = text : N` (default 50 hits, every hit its own mask, whole-row hits
  dropped), KJNodes Points Editor points (one object per point) and boxes, square full-resolution
  detail pass for points and boxes, scope, min_area, optional save of `<name>.png` into a project
  folder. Outputs masks + names for Regions From Masks. Helpers in `kuba_regions/sam_prompts.py`,
  `tests/test_sam_prompts.py` (7 tests). Test facade: 41 windows, 5 balustrades, 3 clicks, 1 box in 24 s.
  Example `example_workflows/regions_sam3_masks.json`.
- M2 step 3, **kubakub regions from illustrator** (kubakub/regions): regions from the layers of a festival
  .ai (PDF compatible) or .pdf via PyMuPDF. All layers read (hidden too, switched on in memory), each
  rasterised by replaying its paths; roles lines / shapes / outside / scope / tag / reference / ignore
  guessed from content or set per layer; artboard mapped with one uniform scale; outputs regions, line
  image, placed photo, layer masks, report. Logic in `kuba_regions/illustrator.py`,
  `tests/test_illustrator.py` (6 tests on a generated layered PDF). A 3200x2160 festival .ai: 795 regions
  in 4 s, MASK layer = the festival's MASK export to 99.96 %. Example `example_workflows/regions_from_illustrator.json`.
- M2 step 4, **kubakub regions from id maps** (kubakub/regions): regions from 3D ID renders with legends
  (`ids_<pass>.png/.txt`, optional `mask_<name>.exr/png`, clay render). One pass gives regions, the
  others tags; linear legend vs sRGB file detected; exact colours only, AA edges filled from neighbours;
  depth order of overlapping object masks learned from the ID map (or smaller_wins); parts split and
  numbered in reading order. EXR via imageio/FreeImage (OpenCV's EXR stays off inside ComfyUI), PNG copies
  as fallback. Logic in `kuba_regions/idmaps.py`, `tests/test_idmaps.py` (4 tests). Houdini test renders:
  365 regions in 12 s. Example `example_workflows/regions_from_id_maps.json`.
- **Frame in frame 2, step F1** (plan section "Frame in frame 2"): viewer camera geometry
  (`kuba_regions/viewer.py`: s = d / (d + D), rooms behind frames, parasites in front, window hull with
  mullions in front, corner-pin placement), nodes kubakub audience viewpoint and kubakub frame guide
  (`kuba_regions/nodes_frames.py`, a viewer socket), `tests/test_viewer.py` (6 tests). Test:
  guide + Region Sampler (Klein, denoise 0.95, reference self) gives rooms in the viewer's perspective.
- **Frame in frame 2, step F2**: Region Plan keys `fif_depth_m`, `fif_source` (generate / media / world),
  `fif_media`; `viewer.compose()` and node kubakub frame compose (media corner-pinned onto back walls, a shared
  world behind the facade, walls tinted with the content); `test_viewer.py` now 7 tests. Example workflow
  `example_workflows/frame_in_frame.json` (courtyard: 40 m facade, viewer 15 m).
- **Frame in frame 2, step F3** (parasites): cast shadow on the facade (`viewer.cast_shadow`, Frame Compose
  inputs shadow_strength / angle / length), hidden window pixels of a parasite count as wall, `extend_plan`
  adds a region `<name>_out` per parasite with the frame's settings; Frame Compose outputs this plan and a
  shadow mask. `test_viewer.py` 8 tests.
- **M2 step 5, kubakub regions to vector** (M2 finished): outline mode (pixel-edge tracing with holes,
  corners kept, Catmull-Rom Beziers, parts touching at a corner kept) and centerline mode (skeleton lines,
  spur removal, KD-tree nearest-neighbour order); SVG (groups / ids), PDF (one layer per group), DXF R12;
  mm scale and kerf. `kuba_regions/vector.py`, `nodes_vector.py`, `tests/test_vector.py` (6 tests), example
  `example_workflows/regions_to_vector.json`. Tested on a 3840x2160 facade (75 regions) and a festival line drawing
  (42 000 centerlines).
- **Video V1 + V2 (work in progress), kubakub region video sampler**: plan keys `animate`, `video_prompt`
  (`+=` works), `t_start`, `t_end`, `motion`; animated regions are grouped into clips, sampled with LTX 2.5
  (masked still video, optional x2 latent upscaler stage, tiled VAE decode) and pasted back into the
  untouched still frame; PNG sequence output. `kuba_regions/video.py`, `nodes_video.py`, `tests/test_video.py`.
  Model-free tests pass; the first GPU run crashed the laptop at the start of stage 2 (1.8 MP x 121 frames),
  so the resolution still has to come down.

## 2026-09-25
- Region Sampler / Seam Pass: new `schedule = pruna_qwen21_8` for the Pruna Qwen-Image-2.1 8-step
  LoRA (fixed card sigmas, denoise on the same axis as `qwen21_turbo_5`); the sampler input stays
  active (euler, res_2s, deis_2m ...). `schedules.pruna_qwen21_8_sigmas`, tests in `test_schedules.py`.
- Example workflow example_workflows/facade_regions_diffusion.json now loads the texture-fix Qwen 2.1
  VAE (decoder-only finetune, removes the checkerboard speckle). README: test results of both VAEs.
- 3D-1: new kubakub scene render (kubakub/scene3d): .blend / .fbx / .obj / .abc / .glb / .usd -> projection view
  from the file's camera (clay, metric depth, normals, coverage) and ID passes shelves / layers / facing /
  planes / parts / objects / materials / collections, written in the From ID Maps folder format. Headless
  Blender subprocess (`scene3d/bridge.py`, `scene3d/blender_export.py`), passes in numpy
  (`scene3d/scene_ids.py`), FBX camera fixes (render size from the camera name, flipped lens shift).
  `tests/test_scene3d.py`, workflow `example_workflows/scene3d_to_regions.json`, plan section "3D plan".
- 3D-2 kubakub scene measure: per pixel / per region distance to the projector, incidence angle, mm per matrix
  pixel, relative brightness, offset from the wall, area; `grazing` tag; a viewer from the real projector
  frame on the wall and an audience spot. 3D-4 kubakub scene preview: matrix (image or frames) projected onto
  the model and seen from audience spots, projection-shadow masks. `scene3d/scene_view.py`.
  kubakub scene render: optional `unit_scale`, units warning, new last output `scene`.
  Blender export: audience 'view' cameras, unit scale. Tests extended; example workflow extended.
- kubakub scene walkthrough: previz video along a keyframed camera path, matrix frames in sync, optional
  background; kubakub scene preview got the background input. Blender export renders many views per session.
- kubakub scene render `min_size_m` (default 0.5 m): pieces the crowd cannot read merge into neighbours
  (`scene_ids.merge_small`). kubakub regions from id maps 127 s -> 5.5 s on 450 parts (per-colour decode,
  ID-map-only fast path, bounding-box slices), identical output.
- kubakub director (W1, part 1): `kuba_regions/director/` - `blend.py` (W3C blend modes incl. hue / saturation /
  color / luminosity and the CSS filter colour grade, exactly as the browser canvas does them) and `render.py`
  (document format v1, full-resolution render: placement with rotation / flips, front / back order with facade
  holes, layer masks by shape / brightness / cut out, clipping to the layer below, region clips, adjustment layers;
  outputs per-layer seen masks, changed mask with edge band, regions with placed layers cut in, plan rules).
  `tests/test_director.py`. The node and the window follow.
- kubakub director (W1, part 2): node (`kuba_regions/nodes_director.py`, category kubakub/director)
  and the window `web/kubakub_director.js` (plain ES module, pack WEB_DIRECTORY, Hanken Grotesk shipped in
  `web/fonts`, OFL): base image + layer images / masks + regions + scene views as preview copies in temp, the
  document in the node's hidden 'document' widget, imports and paint layers uploaded to input/kuba_director,
  apply = save + queue. Outputs image, layer masks, changed, regions, plan rules, document, report.
- Scene export fixes: FBX cameras with a 1e-5 near clip made the Workbench clay lose recessed details (clip
  range now clamped, noted in the report); imported meshes are flat shaded without custom normals; the bridge
  makes the cache path absolute (Blender runs in its own working directory).
- Review fixes (2026-09-25, two reviews): director rules and regions share one naming (a layer named like an
  existing region no longer re-diffuses that region), prompts go into the rules on one line without '//',
  clip / holes match exact region names first (spaces, ':' ...), input region ids and fields are kept (placed
  layers appended), invalid documents raise instead of silently rendering the base, subgraph node ids are made
  safe for Windows paths, preview files are cleaned after an hour, paint layers are named by content.
- Load Images From Dir + Names: new outputs image_batch / mask_batch (one batch, optional batch_fit) next to the
  unchanged list outputs; reruns when files in the folder change. ONNX Style Transfer: no black border
  (tile weights never 0), no 8 bit round trip for tiles, model_path relative to models/onnx (no drive default),
  keep_loaded (off frees VRAM). kubakub save trimesh glb: 'sub/name' filenames work. kubakub scene render: depth_gray
  output (linear grey depth). New kubakub regions mask (regions + plan selector -> MASK).
  `tests/test_legacy_nodes.py`, director regression tests.
- All display names are now "kubakub <name>" and all categories "kubakub/<area>" (regions, scene3d, director,
  facade, relief, mosaic, io, style, 3d); node ids are unchanged, saved workflows keep working. The
  naming rule in CLAUDE.md was updated with Kuba.
- kubakub director for large matrices: every layer lives in its own bounding box, full-canvas work (base,
  colour grades) runs in 256-row strips, region statistics are counted strip by strip. 10240x5760 with 8
  layers, grade, holes and a mask: peak 2.5 GB instead of 17.9 GB, 14 s instead of 54 s; the node on a 10k
  matrix +5.4 GB for the whole ComfyUI process including loading and full-size outputs. The window's working
  copies are 2048 px on the long side (was 1600).
- 3D-3: new kubakub scene relight: the model rendered in Blender Cycles from the projection
  camera with Blender's built-in HDRIs (night, city, courtyard, sunset, sunrise, forest, studio, interior) or an
  own .hdr/.exr, point / area / spot / sun lights written in facade metres, and masks that turn the faces they
  cover into emissive material (glowing windows that light the stone around them); shadows and bounces, OIDN
  denoise, AgX. The test facade at 3200x2160, 64 samples: ~20 s on the GPU. `scene_view.parse_lights / lights_to_world /
  emissive_faces`, relight mode in blender_export.py, tests in test_scene3d.py. The example
  example_workflows/director_scene.json now starts from a relit night view (glowing arcade glass).
- kubakub director window: snapping (canvas edges / centre, other layers, region borders; lavender guides; S
  toggles, Ctrl suspends), side handles for one-direction scaling, corners scale proportionally from the opposite
  corner (Shift free, Alt from the centre, also on rotated layers), background removal per image layer with
  ComfyUI core's BiRefNet (preview through POST /kubakub/director/remove_bg, full-size cutout in the node,
  cached). `kuba_regions/director/bg.py`.
- kubakub director window: a 'regions' view (every region in its own colour) with the region names drawn on the
  canvas (also on the ids view, N toggles them anywhere; small regions get their label when zoomed in); clip to
  and holes are text fields with suggestions (names, group:NAME, wildcards) instead of dropdowns.
- kubakub director window: autosave (a draft 2.5 s after each change, paint layers uploaded, in ComfyUI's user data
  user/default/kubakub_director/drafts and in the node); on opening a newer, unapplied draft is offered
  ("continue where you left off?"); apply clears it. 'file' menu: save as… (Ctrl S) / open… named compositions
  (user/default/kubakub_director/compositions), export / import as a .kubakub.json file.
- Robustness: kubakub scene render and kubakub director run again when their cache in ComfyUI/temp is gone (every
  ComfyUI start empties temp - a second instance started on the same install deleted a running session's scene
  cache), instead of passing on a folder that no longer exists; clearer error in From ID Maps.
- Director: clip to / holes use an own region picker. The browser suggestion list only offered entries matching the
  current value, so once a region was set no other could be chosen. The picker lists every group and region (current
  one highlighted), filters as you type, arrow keys + Enter pick, Escape restores; typed patterns (W_F1_*) still work.
- kubakub director: light layers (+ layer → light, Shift L). A Cycles relight of the connected scene as a layer:
  HDRI environment, point / area / spot / sun lamps dragged on the facade (double click adds one), glow from layers
  or regions, clay / roughness, outside black / environment / transparent. Previews render in the window through
  POST /kubakub/director/relight (~3 s at 800 px); apply renders at full size. Relight core shared with kubakub scene
  relight (`nodes_scene3d.relight_scene`, `scene_view.rig_from_doc`, `render.glow_masks`); tests in test_director.
- Director fixes from a frontend review: the preview scale in the manifest was overwritten by a loop variable
  (large matrices got full-size or oversized canvases); dialogs get their keys (Esc no longer closes the window
  behind them, Enter saves); a paste without an image no longer pastes nodes into the graph; hover highlights no
  longer fill memory; undo shares unchanged paint pixels; drawing coalesced per frame and layer thumbnails cached;
  duplicated director nodes no longer share a draft; deleted node inputs stay deleted (+ layer → bring back);
  late background removal answers are dropped; selectors in the window follow the plan syntax (',' or, space and,
  '!', tag:, region:3-7); region boxes scaled when the regions have another size; pointer cancel ends drags;
  sliders no longer swallow shortcuts; a pending draft is saved on close; the remove_bg route reuses its mask.
- Light layers: projector. The layers below (or the base, or one layer, even hidden) cast from the projection
  camera, either as light (textured spot at the camera, radius 0, frame mapped via view_frame incl. lens shift;
  power from the projector distance so brightness 1 = the image's own brightness) or as paint (the image as the
  model's colour through window coordinates, lit by the lamps / HDRI with shadows). Tone AgX / Standard. The window
  sends its own composite of what is cast with each preview; apply composes it at full size. kubakub scene relight
  got the same as inputs (projector image, projector_brightness, projector_mode, view).
- Director: diffuse from the window. The diffuse button (D) on an image / paint layer queues a small graph of its own
  (UNETLoader, CLIPLoader flux2, VAELoader, up to 3 LoraLoaderModelOnly, new node kubakub director diffuse): the
  layer's area from the window's composite + its shape as mask (rediffuse = shape + band, edges = band), layer prompt
  and denoise; the result lands in input/kuba_director and comes back as a new layer above with a soft edge.
  'diffusion' popover: klein 4b (default) / 9b presets resolved against the installed files, LoRA slots (matching
  model first), steps, size, context, edge band, seed, reference; saved in the document. Prefers the
  flux2_full_encoder_small_decoder VAE. Progress / errors from the queue's websocket events.
- Director timeline (T1): keyframes per value path ('x', 'adjust.hue', 'light.lights.#p1.power'; lamps and glows got
  stable ids) with smooth / linear / hold, colours mixed in RGB. Timeline strip under the canvas: play, scrub, length,
  fps, record (auto-key), key, keyframe rows of the selected layer (drag, Alt click ease, Del). The layers always hold
  the values at the current time; apply renders that frame.
- Director sequence (T2): render_sequence / sequence_scale inputs, frames / fps outputs (appended). render.animate /
  key_value evaluate keyframes exactly like the window (tested with the same numbers). Light layers render per frame
  in one Blender session (blender_export apply_rig per frame, fixed Cycles seed = no sampling flicker), identical
  frames once; frames are read from disk on demand (nodes_scene3d.relight_sequence).
- Director timeline T4, sound: load / drop an audio file (input/kuba_director), waveform row, playback in sync
  (Web Audio is the clock), beat detection in the window (onset envelope, autocorrelation tempo with octave check,
  least-squares phase; tested 90-174 bpm within 1.5 bpm / 10 ms), bpm field, ÷2 / ×2, "beat here", bar.beat readout;
  markers (M, drag, rename, delete); snapping of scrub / keyframes / markers to beats, markers and keys (Ctrl =
  free). New director output 'audio' (core AUDIO, trimmed / padded to the timeline, offset and gain; silence
  without a sound), appended after frames / fps.
- Timeline T3, LTX in the director: layers get 'video' (on, prompt, t_start, t_end, motion) -> plan rules
  animate / video_prompt / t_start / t_end / motion (render.rules_text; tests). Window: 'animate (ltx)' section with
  from / to (here = playhead), motion, the range drawn in the timeline, ' · ltx' in the layer list.
- kubakub region video sampler: 12 GB safe - stage1_mp default 0.2, new stage2_max_mp (0.6) caps the x2 second
  stage, VRAM freed between stages / before decoding; new optional 'background' video (e.g. the director's frames):
  each clip covers its t_start..t_end, starts from that background frame, is pasted into those frames only.
  GPU-tested on the laptop: stage 1 49 frames 73 s; two stages 121 frames 130 s (peak 11.8 GB, no crash);
  director sequence + LTX 215 s.
- Example workflow example_workflows/director_video_ltx.json (director timeline -> plan -> LTX sampler -> Create Video + sound -> Save).
- Scene render: architecture passes elements (windows, window_frames, columns, cornices, relief, wall, roof),
  sections (left / centre / right at the wall steps) and floors (ground_floor, floor_1 …, roof), from the pixels'
  3D data (scene3d/architecture.py, synthetic test in test_scene3d). Default passes start with them. 3200x2160 test facade: 27
  windows (arcade split per arch), left / centre / right exactly at the avant-corps, 3 floors + roof.
- Director: the clip to / holes picker lists groups, tags (sections, floors, other tag passes) and combinations
  ('tag:left group:windows', 'tag:floor_1 group:windows').
- The director example workflows use elements as regions (354 instead of 904 regions, 7 groups) with sections / floors /
  shelves / facing / layers as tags; the relight glow is 'tag:ground_floor group:windows'.
- Performance (from a performance review): core Model Attention Backend = comfy kitchen INT8
  attention in the director's diffuse graph, example_workflows/director_video_ltx.json, the M1 and frame in frame
  workflows (after every UNETLoader). Relight: a Blender that stays open for relight jobs (blender_export.py
  --serve, bridge._Worker; the scene stays loaded while the file is the same, quits after 2 idle minutes, falls
  back to one process per job): previews 2.2 s -> 0.6 s. Cycles persistent data + GPU denoising for sequences;
  sequence frames composited in parallel threads, thread-safe frame reader. Prompt encodings cached in the diffuse
  node (second diffuse 8 s -> 6 s with kitchen attention) and encoded once per run in the region video sampler.
- Director diffuse presets: klein 4b nvfp4 (official BFL NVFP4 + Comfy-Org qwen_3_4b_fp4) is the default when
  installed (measured: 3.0 s vs 6.0 s per diffuse loaded, same composition, LoRAs work), klein 9b nvfp4 when
  installed; the bf16 / fp8 presets exclude fp4 files (name matching picked the nvfp4 file otherwise).
- Region Sampler with Qwen Image 2.1: every region's crop is encoded (text + vision) before any sampling, so the
  text encoder loads once; at cfg 1 (turbo LoRAs) the negative is not image-encoded (never evaluated). 3
  regions: encoding 10 -> 5 s, run 35-37 -> 32 s. The encoder sees the crops before earlier regions are pasted
  (differs only where regions overlap). Measured: Qwen 2.1 NVFP4 DiT ~ int8 speed here, the NVFP4 text encoder is
  slower than w4a8 (more swapping on 12 GB) - keep int8 / w4a8 for Qwen.
- H3 clips (MiniMax H3, local): timeline.clips in the director (h3 clips popover, + clip from the markers around
  the playhead, bars on the ruler) and the node kubakub keyframe clips (h3): keyframe clips (fl2va, start / end frame
  of the director sequence) and reference clips (ref2va: the sequence stretch as reference video, the timeline music,
  director layers), crossfaded joins, H3 sound mixed into the timeline sound. H3 prompt format built automatically
  (look, anchors / tags, timeline with named markers as beats, locked-off camera, Audio, avoid); prompts output; raw
  mode. keyframes.py + tests/test_keyframes.py (clip parsing, 17k+5 lengths, 24 fps mapping, audio mix, prompt
  builder, plausibility check). GPU-tested: keyframe clip 56 s, reference clip 263 s (1120x768).
- Prompt enhance (local, core Generate Text) for H3 clips and the LTX region video sampler via enhance_clip: the
  Qwen 3.5 9B prompt-enhancer encoders write well; the Gemma 4 LTX int8 encoder returns garbage (detected, the draft
  is used instead).

## 2026-09-26
- H3 prompt enhance: the draft's anchor / reference lines are restored word for word when the enhancer rewords them
  (the Qwen 3.5 9B turned "image 1" into "<Picture 1>", so the whole rewrite was thrown away before);
  `keyframes.restore_anchors`, stricter system prompt (never add tags), enhance time logged. Tests added.
- Enhancer comparison on a keyframe clip: qwen3.5 9B pe_i2i 9 s, Qwen3.8 27B w4a8 81 s and a weaker rewrite.
  9B stays the recommended enhance_clip.
- Speed (director + H3, keyframe clip, 100 frames): cold run 165 -> 142 s, re-run after a clip change 60+ -> 5 s.
  - kubakub director: new input skip_h3_frames (on): frames that H3 keyframe clips replace completely are not
    relit / composited (a dissolve stands in; 12 frames kept at each clip end for crossfades). Sequence 56 -> 36 s.
  - Relight frames written as uncompressed PNG (temp files): ~0.07 s less per frame.
  - kubakub keyframe clips (h3): rendered clips are cached (content hash of anchors / references, prompt draft,
    size, steps, seed; same model objects). Changing one clip no longer renders (and enhances) the others.
  - Fingerprint fixes, every queue used to re-run these: Regions From ID Maps with a linked folder hashed
    ComfyUI's working directory (ComfyUI passes only constants to fingerprints); scene render and director
    flipped 'cache missing' -> 'present' after the first run. New bridge.folder_token (tests added).
- Fix: director sequence and region video output sizes are rounded to even numbers (H.264 / yuv420 refused
  737x1140 on a portrait festival template). That template (.blend scan, 855k faces, 2048x3168) runs end to end:
  scene render 18 s, 195 regions, H3 keyframe clip.
- Director video layers, step 1 (backend): an image layer whose source is a video file plays on the timeline.
  `media` = segments (start, in / out trim, speed, length with loop / hold fill, per-segment file = cuts),
  volume, blend_frames (smooth slow motion / fps conform). Frames decoded with the bundled ffmpeg (ProRes, HAP,
  DNxHD, H.264/5, VP9, alpha) at the layer's size, once per needed frame; the still shows the timeline's current
  time; the sequence gets every frame; hidden outside its segments. All image-layer features apply per frame
  (clip to regions, masks, blend modes, opacity, keyframes, grade). Video sound is mixed into the audio output
  (speed 1 segments). kuba_regions/director/media.py (adapted in part from VideoHelperSuite, see Credits),
  tests/test_media.py (synthetic videos, 28 checks). Window support comes in step 2.
- Director video layers, step 2 (window): import videos (menu / Ctrl O / drop; over 100 MB: "video from a path",
  source 'path:' read in place, never copied). Route /kubakub/director/media makes a small proxy once per file
  version (H.264, keyframe every 5 frames; VP9 with alpha for alpha sources). The proxy plays in a <video> synced to
  the timeline: exact frame when scrubbing, native playback at the segment's speed (with its sound) when playing,
  corrected on drift. Timeline clip row for the selected video: drag to move, left grip trims in, right grip trims
  out, Alt = speed (stretch), Shift = length (loop / hold fill); Ctrl K cuts at the playhead; Del removes a piece.
  Inspector video panel: pieces (at, speed, in, out, length, then hold / loop), cut, timeline = video, fit to
  canvas / clip regions / native, sound volume, frame blending. Region masks stay cached during playback (they were
  rebuilt every frame): a composite takes ~0.3 ms at preview size. window.kubakubDirector() for tests.
  Headless UI test (Playwright + Chrome): ui_test_video_layers.py, 15 checks.
- Bug hunt (ultracode workflow: 4 reviewers + 3 Fable UI testers, 59 findings, all verified). Backend fixes:
  - media.py: rotated phone clips (displaymatrix) probed in displayed size (decode was sheared, proxy squashed);
    23.98/29.97 read as the exact NTSC rate; variable-frame-rate clips decoded on the k/fps grid (fps filter);
    missing container Duration measured from the stream; frames counted from the video when the sound is longer;
    decode always returns every requested index (the still crashed with KeyError); zero-span loop guard;
    proxies written to <key>.tmp.<ext> and renamed (never a half-written or failed proxy).
  - nodes_director.py: MediaFrames plans once and decodes per 50-frame window (bounded RAM at delivery size);
    hidden video layers used as mattes / projector / glow sources are decoded, a matte outside its segments is an
    empty picture (as in the window); light previews see video layers; proxy route re-checks inside its lock and
    only serves finished files (PROXY_VERSION in the key); 'path:' refuses UNC / device paths; sequence renders at
    the exact scaled size and crops the parity pixel (no resampling at scale 1).
  - render.py: image_only mode (sequence frames, projector composites skip masks / bands / regions: ~30 % faster
    per frame with rediffuse layers); mask_cache for region selections across a sequence; director regions' bbox
    now x, y, w, h like every regions table. blend.py: float32 colour matrices (no float64 upcast).
  - Tests: test_media +12 (rotation, NTSC, VFR, no duration, long sound, past-the-end, proxy temp), test_director +3.
- Bug hunt, window fixes (kubakub_director.js; 42 of the testers' / verifiers' repro scripts now print NOT REPRODUCED,
  the rest check old code text or were measurements):
  - video: frame-exact seeks (the frame the render picks, no off-by-one at frame boundaries); no seek storm while a
    clip holds its last frame; playback nudged towards the playhead (lag 0.1 s -> 0.01 s); >16x speeds shown frame by
    frame; sound through a gain node (0-200 % like the render); hidden videos used as mattes / projector / glow still
    play; videos sync on open, open composition, duplicate; deleted / undone / replaced video layers go silent at once,
    every element released on close; proxies cached by the browser, video layers load in parallel.
  - timeline: one play loop only (P P P no longer doubles speed / work; auto-repeat ignored); markers before H3 clip
    bars in hit testing; at a cut the left piece's end and the right piece's start both grabbable; right grip of a
    clip running past the timeline end works; a cut inside a hold keeps holding; Del acts only on the timeline when it
    has focus and on what was clicked last; undo covers markers, length and fps; clicks that only select no longer
    cost an undo step (canvas and timeline); the timeline scrolls instead of being squashed (hit testing was off).
  - window: region boxes read as x, y, w, h (fit regions, region names and snapping were broken); brightness masks
    as a GPU filter (no getImageData per frame); region masks stay cached across refreshes (LRU); Esc in a field only
    leaves the field; Home / End on a slider stay with the slider; keys ignored while loading; Space-pan released on
    window blur; 'clip to' typing survives re-renders and commits to its own layer; diffusion jobs follow the layer by
    id through undo; nothing is saved or inserted after close; newest sound load wins; the path dialog has its text
    field; a glow on regions shows its pattern field; dangling mask / glow / projector references are cleared; a bad
    composition file no longer leaves 'loading' up; apply runs once (double click / Ctrl Enter twice).
- Director projection mask slot: the building's silhouette at the delivery size, always on top. Source = the 3D
  scene's silhouette, a template mask file (upload) or a path; the usual template forms are read automatically
  (opaque-black overlay with a transparent building as festivals deliver it, a cutout, or white on black), invert,
  grow / shrink and feather in delivery pixels. Window views black / dim / off (one cached drawImage per frame);
  the still, every sequence frame and the new 'projection_mask' output (MASK, last output) apply it. Route
  /kubakub/director/pmask reads the mask exactly as the render does. kuba_regions/director/projmask.py,
  tests/test_projmask.py (13 checks); headless UI test ui_test_projmask.py (11 checks).
- Director shape layers (kind "shape"): solid colour (a rect over the canvas), rectangle, ellipse, polygon (click the
  corners, Enter / double click finishes, Esc cancels); colour and feather (a soft edge that spreads outside, sigma in
  delivery px) in the inspector; they move / scale / rotate / keyframe like images (plus colour and feather keys),
  clip to regions, blend, and a hidden shape is a shape mask for other layers. render.py shape_source rasterises
  the same shape with OpenCV (window vs render within 1/255 in the UI test). Tests: test_director +6,
  ui_test_shapes.py (8 checks).
- Director layer effects: blur, sharpen (unsharp mask + radius), glow (threshold, radius, strength) and grain (amount,
  size, new every frame) on image / video / paint / shape / base layers (after the region clip, before masks by other
  layers) and on colour-grade layers (then on everything below). All keyframeable (fx.* paths). Window: one SVG filter
  per layer for blur / sharpen / glow, grain as cached noise frames blended with the canvas overlay mode (an SVG
  feTurbulence was cached by Chrome and cost 40 ms). Render: kuba_regions/director/fx.py (premultiplied like SVG;
  window vs render within 1/255 for blur, sharpen and glow; grain matches in strength and size, seeded per frame).
  scale_doc now also scales effect distances and shape feathers for the smaller sequence render (the feather was not
  scaled before). Tests: tests/test_fx.py (11 checks), ui_test_fx.py (7 checks).
- Director video: reverse per piece (panel switch, ◀ on the clip bar): plays from out to in (a hold keeps 'in', a loop
  replays backwards, the sound backwards too); the window steps frame by frame for backwards pieces (browsers cannot
  play backwards); cutting a backwards piece keeps both halves backwards. media.py source_time / layer_audio,
  window mediaTime twin; test_media +4, ui_test_video_layers 16 checks.
- Director colour management. Working space: display-referred sRGB / Rec.709, float end to end, blends in that
  space like the browser and After Effects. director/colour.py: imported images with an embedded ICC profile (Adobe
  RGB, P3, ProPhoto, CMYK ...) are converted to sRGB with LittleCMS (relative colorimetric, as the browser shows them;
  e.g. Adobe RGB (100,150,200) -> sRGB (66,151,203), ComfyUI's loaders read it raw); the report names the profile.
  media.py: untagged HD YUV is read as Rec.709 like every player (bars within 3 levels; ffmpeg's Rec.601 default was
  20 off), untagged SD as Rec.601; 10 / 12-bit sources decode at 16 bits (880 levels in a 10-bit ProRes gradient
  instead of 256); proxies are converted and tagged Rec.709 (PROXY_VERSION 3). tests/test_colour.py (11 checks,
  Windows' own AdobeRGB1998 / FOGRA39 profiles).
- Director lamp reaction: image / video / paint / shape layers can be 'shaded by' a light layer: x (the lamps' light on
  the clay / the clay albedo), brighter where the lamps hit, darker in shadow, tinted by coloured lamps, with an amount
  (keyframeable). The light layer may stay hidden (it still renders, in the still, the sequence and the window
  preview). Window: canvas multiply + brightness(1 / albedo); render: render.py light_react (window vs render
  within 3/255 in the UI test). 'Projected and lit' = the light layer's projector in paint mode with a video as its
  source (per frame in the sequence; the window previews the playhead's frame). test_director +3,
  ui_test_react.py (6 checks).
- Director delivery export: director inputs export (png8, png16, prores4444, prores422hq, h264, h264_444, h265,
  preview), export_scale (1 = delivery size), export_alpha (layers without the base, alpha = coverage x projection
  mask), export_name. The timeline is rendered window by window and streamed into the writer (no RAM limit), into
  output/kubakub_director/<name>_<time>/; Rec.709 conversion and tags explicit (MOV colr atom written), the timeline
  sound muxed (a .wav next to PNG sequences). New node kubakub export writes any IMAGE batch the same
  way (after the H3 clip node). render.py with_alpha (coverage). director/export.py; tests/test_export.py (all
  formats read back: sizes, frames, tags, alpha, sound, colour error 0.1-1.5/255); end-to-end
  run_export_test.py (4K H.264: 50 frames in 18 s; ProRes 4444 alpha correct). README section.
- H3 pinned frames: a timeline marker inside a keyframe clip can pin its frame ("pin frames" in the H3 clips popup,
  a diamond on the marker); kubakub keyframe clips guides H3 through the director's picture there (core
  MiniMaxH3AddGuide), the pins are part of the clip cache key, skip_h3_frames still renders pinned frames.
  keyframes.clip_anchors + tests. (GPU run still to do.)
- example_workflows/director_h3_clips.json: scene -> regions -> director (render_sequence, skip_h3_frames) -> keyframe clips
  (fl2va + 4-step LoRA, ref2va + 8-step LoRA, 9B enhancer) -> Create Video, plus a bypassed kubakub export.
- Director window: export button. Format / size / alpha / name, then only the director node runs
  (partial_execution_targets) with those settings in the queued prompt; the node's widgets stay as they are and
  the window stays open, a status line follows the job. ui_test_export_button.py (12 checks).
- Behaviours (step 1 of the motion engine): layer "motion" list, procedural motion on top of the keyframes.
  wiggle, oscillate (sine / triangle / square / saw / circle orbit), drift / spin, random steps, loop keyframes
  (cycle / pingpong / continue); any numeric animatable value plus position (x y) and scale (% about the centre);
  range, fade, on / off, several add up. kuba_regions/director/motion.py and its twin in the window (integer hash
  noise, bit-identical); render.animate applies them per frame, the still render at the playhead. Window: inspector
  "behaviours" section with presets (wiggle, breathe, orbit, sway, spin, drift, flicker, pulse …), timeline rows
  with a wave over each behaviour's range, dashed loops; the saved document keeps the rest values (a drag moves the
  rest), undo / duplicate copy them. tests/test_motion.py runs the same 56 cases through Python and the window's
  code (node.js); ui_test_motion.py (11 checks, node render = window within 1.5 px).
- Behaviours step 2, beat and sound drivers: pulse (a kick on beats / bars / markers / every n s, every nth,
  attack, decay) and audio (the loudness of the timeline sound, 100 Hz RMS curve normalised to its 98th
  percentile, peak with a release; past the file's end the tail still releases). The node rebuilds beats from the
  saved tempo and loads the curve once per file (motion_context, cached). Presets: pulse on beats (scale), kick on
  markers (up), flash on bars (glow), sound -> scale / opacity. Timeline: spikes at the triggers, the level curve.
  test_motion: 28 more twin cases, beat times and the loudness curve identical in both;
  ui_test_motion_audio.py (6 checks, click track; node = window within 1.6 px).
- Behaviours step 3, mograph over the building: stagger (a layer clipped to regions lights them one after another:
  sequence in / out, chase, wave, random flicker; order left / right / top / bottom / centre / outside / size /
  random; step, fade, hold, period, density) and repeat (copies of a box layer in a line / grid / ring, rotation /
  scale / opacity per copy, a delay per copy replays the layer's animation later: echo trails). The node orders
  regions by the same rounded canvas boxes the window gets (table canvas_bbox) and weights the clip mask per region;
  animate() expands the copies (ids "<id>~k", below the layer). Window: weighted clip mask rewrites only the
  clipped pixels per frame, presets, staircase row. test_motion: 257 stagger + 144 repeat twin cases;
  test vectors now go to the temp folder (tests/motion_vectors.json removed). ui_test_motion_step3.py
  (14 checks: 354 regions, window = node IoU 0.98-0.99, composite 4-7 ms).
- Behaviours review (ultracode: 5 reviewers + 5 adversarial verifiers, 21 confirmed findings, all fixed):
  keys / K / deleted keys never bake a behaviour's offset in (L._off per value); delayed copies no longer write
  into the layer (light_react, mask copied); a loaded / opened document shows its behaviours at once; mask "below"
  with repeat copies masks by the real layer below (as in the window); the projector cast and the light preview
  use the playhead (stagger, behaviours); the window's sound level is the node's own curve (/kubakub/director/level);
  type switching gives a valid path / mode / defaults, fields show the engine's defaults, loop offers only keyed
  values, effect presets create the effect values, Space works on focused checkboxes, the projector preview
  follows cast layers' behaviours.
- Performance: stagger mask per frame only rewrites regions whose weight changed and uploads the changed rectangle,
  shared by all copies (50 clipped staggered copies 269 -> 3 ms per composite, a staggered layer 6 -> 0.3 ms);
  node: stagger LUT once per render, per-frame documents without keys (sequence memory ~5 MB -> 0.13 MB per frame),
  keys sorted once per frame and one-level copies for repeats (200 delayed copies: 4.3 -> 2.2 ms per frame).

## 2026-09-27
- Director performance pass (all changes pixel-identical to before, checked per step):
  window: composite only redraws each layer's bounding box, playback capped at the timeline fps, grain noise cached
  per size and prefilled when idle, panels rebuilt only when their inputs change, deferred panel updates while
  scrubbing, stagger data kept across undo, paint dirty-flag race fixed;
  node: premultiplied sources cached per render, fast fx path for opaque layers (glow r30 at 3200x2160 2.6 -> 1.2 s),
  in-place composite and soft light (soft light strip 134 -> 85 ms), still rendered in parallel strips (4.5 -> 3.6 s),
  cached full-size paint / shape pixels, grade via cv2.transform (up to 3x faster);
  export: rendering and encoding overlap (bounded queue, encoder errors raised instead of hanging), scene views and
  base / label proxies reused between runs (base-only re-run 2.0 -> 1.0 s).
- Director window: menus, popups and hover helpers were see-through (colour variables only on the window root, the
  popups live on document.body) - white text on the image; now they have their panel colours in light and dark.
  The shortcuts popup scrolls instead of running off the screen.
- Director window design pass (review with screenshots at 1280 / 1600 / 1920, light and dark): the layer's own
  controls (shape, brush, grade, light) come right after name / blend / opacity / clip / mask; the layer list scrolls
  on its own (drag the handle below it for more / fewer rows) and keeps the selected layer in view; the side panel
  is wider on big screens and can be dragged wider / narrower; the timeline can be dragged taller (double click a
  handle resets it; sizes are kept per browser); undo / redo are icons and the draft stamp hides below 1440 px, so
  the top bar fits one line at 1280; effects show only the ones in use, "+ effect" adds blur / sharpen / glow /
  grain as groups with clear labels (strength, radius, threshold …) and × removes one; behaviour fields accept 0,5
  and 0.5; only "behind" layers get a tag; the projection mask is one line until opened; no hover helper over an
  open menu; lighter slider tracks with an orange fill, lavender checkboxes; the layer drop line is right in a
  scrolled list.
- New node **kubakub director sequence**: the video side of the director. kubakub director
  gets a 12th output **director** (appended, older links keep their slots); the sequence node renders every frame
  (frames / fps / audio), the delivery export, frames off = export only. Runs only when it is in the graph, and
  changing its settings re-runs only it (the director comes from the cache). The director's render_sequence /
  export / sequence_scale / skip_h3_frames inputs stay for older workflows (advanced section) and the window's export
  button. Identical frames, sound and export files (tests/test_director_sequence.py). Example workflows
  example_workflows/director_video_ltx.json and example_workflows/director_h3_clips.json now use it.
- Fix: kubakub director with an empty document (a fresh node) failed in render() since 53e310e (json.loads("")).
- Compatibility: requirements.txt (OpenCV, which ComfyUI does not ship), pyproject.toml (registry metadata,
  requires-comfyui >= 0.37.0, license still open), README "Requirements" (tested-with line, a table of every package /
  program, what needs it, what happens without it; replaces the stale ReliefForge install note), and a startup
  self-check kub_env.py: silent when everything is there, else one [Kub] line per missing piece (ComfyUI too old,
  required / optional packages with the pip name, Blender, ffmpeg). Looks things up only, never imports or starts them.
- Director **flow** (new last input, and a hold / auto pill in the window next to apply): "hold until apply" (default)
  holds every node after the director while no composition is applied (ComfyUI ExecutionBlocker, silent): the
  director still shows its preview and a "waiting for apply" report, nothing heavy runs on an empty composition;
  "always" = the old behaviour. The example video / H3 workflows no longer need node 5 bypassed.

## 2026-09-27 (review round: flow bugs, speed, ease of use)
- Nothing heavy runs for nothing:
  - The LTX region video sampler and the H3 keyframe clips take their models as lazy inputs: models load only when
    there is a clip to render. They never load while the director holds, or when nothing is animated / no clip is
    set; then the frames pass through ("no LTX / H3 model loaded").
  - The LTX sampler no longer fails with "no region to animate".
  - Seeds of both default to fixed.
  - The director sequence skips frames for H3 only when a keyframe clips node takes them, and that option now
    defaults to off.
  - Director previews are kept per workflow (two tabs with director node 3 no longer mix scenes).
  - regions to svg / pdf / dxf runs on its own (output node); a subfolder in its filename_prefix works.
  - The batch loader's "next image per queue" loop moves on again; an edited mesh re-runs mesh to relief field;
    an edited HDRI re-runs the director and the relight.
- Caches (details and settings in README "Speed and caches"):
  - region sampler: per region and per prompt encoding;
  - director sequence: its frames and identical exports;
  - LTX / H3: per clip and the enhanced prompt, each model loaded once for all clips;
  - 3D: scene render ID passes, relight per frame and walkthrough views in a persistent cache in the user folder
    with a size cap; one Blender worker stays loaded;
  - the sequence node has a light_scale option.
- Speed, same output:
  - scene ID build 25.8 -> 8.2 s, architecture analysis 9.5 -> 2.1 s, ID-map regions 26.9 -> 12.1 s
    (bounding-box crops, bincount instead of unique, threads);
  - region sampler CPU per region 37 -> 2 ms;
  - LTX frame writing in threads;
  - director: imports decoded once, previews written only when changed, video decoded ahead.
- Fixed on the way:
  - the walkthrough waited for a file Blender never writes;
  - the Blender worker's idle timer could kill a long relight;
  - LTX with enhance_prompt on crashed after the first clip.
- Ease of use:
  - Node menu in numbered groups: 1 regions, 2 diffusion, 3 director, 4 video, 5 3d scene, 6 fabrication,
    7 projects, tools. Clearer display names (node ids unchanged) and search aliases. The old frame guide is marked
    deprecated; the window-only diffuse node is hidden.
  - New node kubakub sample facade (a synthetic facade, its colour matrix, silhouette and mask folder) and four
    starter workflows that need no files: 01 regions in 30 seconds, 02 one prompt per region (Klein 4B),
    03 director playground, 04 animate the windows (LTX).
  - Workflows moved to example_workflows/ (the Templates menu lists and opens them) with neutral names and empty
    path fields that the notes explain.
  - README: a new front page with quick start, node groups and caches.
- Privacy: tests/check_private.py flags machine paths, user folders and e-mail addresses (plus a git-ignored list of
  private words); git pre-commit / pre-push hooks run it.
