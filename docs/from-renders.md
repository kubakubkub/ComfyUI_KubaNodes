# Regions from renders (3d / from renders)

Your own 3D renders as the start: plain render passes, an ID pass from any renderer, or a cryptomatte EXR.
All three nodes run on a built-in sample when their folder / file field is empty.

## kubakub render passes

A folder of passes written by any 3D tool as PNG (also JPG, TIFF, WebP; 16 bit keeps its range):

- `<render>_beauty.png`: the picture (also `_rgb`, `_combined`, `_color`, `_clay`, `_render`).
- `<render>_depth.png`, `<render>_normal.png`: come out as pictures (black when missing).
- every other pass of that render is a **mask**, named by its pass: `<render>_cut.png`, `<render>_windows.png`.
  White = inside; `invert` lists the ones where black is inside. A mask kept in the alpha channel works too.
- a file that belongs to no render (`facade_mask.png` next to `facade_a_beauty.png`, `facade_b_beauty.png`) is a
  mask for every render of the folder. Plain names (`beauty.png`, `windows.png`) work for a folder with one render.
- several renders in one folder: `render` picks one (empty = the first; the report lists them).
- colour pictures among the passes (an ID map) are left out with a note: **regions from id renders** reads those.
  EXR files and files starting with `_` are not read.

Outputs: `beauty`, `depth`, `normal`, `masks` (a batch) and `names` (one per line), which go straight into
**kubakub regions from masks** (`masks`, `names`, `matrix` = beauty). With `split_masks = cut` every separate
shape of the cut mask becomes its own region (`cut_01`, `cut_02` ...); where masks overlap the smaller one wins,
so the openings are cut out of a wall mask around them. Then a region plan and the region sampler (or versions)
repaint the regions; outside them your render stays as it is (a region grows over its edge by the plan's
`dilate_px`, 4 pixels by default).

Example workflow `example_workflows/render_passes_to_final.json`: the places you opened or redesigned in 3D
(a `cut` pass) each get their own idea, the wall around them one calm material first. Test without a model:
`tests/test_render_passes.py`.

## kubakub regions from id renders

Regions from ID renders of the 3D model (a Houdini setup; any renderer
that writes the same files works). The `folder` holds:

- `ids_<pass>.png`: ID map of one pass, every object in a flat colour.
- `ids_<pass>.txt`: its legend, one line per object, `<name> rgb r g b` (0..1;
  also `name = #rrggbb` or 0..255). Every legend makes a pass.
- `mask_<name>.exr` (optional, per object, the mask is the alpha), or
  `mask_<name>.png`, `png/mask_<name>.png`.
- `*clay*` / `*beauty*` render (optional): the `reference` output, tone mapped.

One pass gives the regions (`regions_pass`, default `elements`); the others
(`tag_passes`, default all) tag every region they cover by half or more, so
`zone_left`, `level_1`, `bay_3` become rule tags: `[group:windows tag:level_1]`.

Things this handles:
- **Linear legend, sRGB file**: renderers write the legend linear, 8 bit PNGs
  are sRGB encoded (the test legends match 0 px as written). `legend_colors =
  auto` takes whichever reading matches more pixels.
- **Anti-aliased ID maps**: only exact colours are read; blended edge pixels
  take the label of the nearest exact pixel (a blend of two colours can be
  close to a third one).
- **Object masks ignore occlusion** (`face_front` covers the whole facade behind
  everything). With `overlap = id_map` the depth order is learned from the ID
  map: where two masks overlap, the one the ID map shows is in front; masks are
  then painted back to front at full resolution. Elements the ID map never shows
  (a recessed clock) end up behind and are listed in the report;
  `smaller_wins` puts every smaller mask in front instead.
- **Half-size ID maps**: the full-res masks give the edges; a pass without mask
  files is cut from its ID map (upscaled).
- **EXR**: ComfyUI loads OpenCV before this pack, so OpenCV's EXR codec stays
  off. EXRs are read with imageio's FreeImage backend (FreeImage.dll, often
  present in Windows System32; nothing is ever downloaded); where it is missing,
  the PNG copies are used and the report says so.
- `split_parts` (names, wildcards; default `*`): separate parts become
  `windows_01, windows_02 ..` in reading order, group `windows`; slivers under
  `min_region_area` merge into a neighbour.

Outputs: `regions`, `preview`, `reference`, `regions_json` (with the passes),
`report` (passes, depth order, hidden elements, notes).

3200x2160 test facade: 4 passes (elements 17, level 5, zone 4, bay 7), 365 regions in
12 s (280 of them glass panes between the window bars), the full-res result
agrees with the ID map on 98 % of interior pixels; hidden in the ID map:
clock, entrance_windows, fe_glass, windowback.

Example workflow `example_workflows/regions_from_id_maps.json`.

Test without a model (writes a small Houdini-like folder and reads it):

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_idmaps.py
```

## kubakub regions from cryptomatte

Render a **cryptomatte** pass in your 3D tool (Blender: View Layer → Passes → Cryptomatte → Object and Material;
Houdini Karma / Mantra, C4D, Maya with Arnold, Redshift, V-Ray, Octane all write it) as a 32-bit multilayer EXR, and
every object becomes a region under its real name from the scene. Objects named alike (`window_left_01`, `_02` ...)
share a group (`window_left`), so `[group:window_left]` in the plan picks them all. `tag_layers = material` adds
each object's material as a tag (`[tag:glass]`). `exclude` leaves helper objects out (wildcards). The `render`
output is the picture of the same EXR (Combined pass, sRGB). Connect your `matrix` to get its size.

EXR support: built in (no extra packages) for scanline and tiled files with no, RLE, ZIPS or ZIP compression,
which is what Blender and most renderers write by default. PIZ, DWA, B44, PXR24 and multipart files are read
through the OpenImageIO module inside Blender 4.x (found automatically, a few seconds), or the `OpenEXR` Python
module if you have it. Cryptomatte is Psyop's open standard (github.com/Psyop/Cryptomatte).

Speed: the ID map is decoded per distinct colour instead of per pixel and legend
  entry, and a pass with an ID map but no mask files (Scene Render's) uses the decoded labels directly
  (no per-name masks, no depth order). The test facade, 450 parts: 127 s -> 5.5 s, identical regions and tags.
