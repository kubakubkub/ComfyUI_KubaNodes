# Regions (2d / regions)

A facade becomes a partition of named regions: every window, door, cornice and floor is its own region with a name, a group and tags, so rules, samplers and the director can pick them. Sources: a colour matrix, a line drawing, a folder of After Effects masks, any MASK batch, SAM 3, an Illustrator file. 3D sources are in [from-renders.md](from-renders.md) and [scene3d.md](scene3d.md), hand drawings in [sketch.md](sketch.md).

## kubakub regions from matrix / masks (the facade mask atlas)

Splits the matrix into a partition of regions: every pixel belongs to exactly
one region (unless small regions are dropped). Region ids follow reading order
(rows top to bottom, left to right) and equal the index in the MASK batch.

| input | default | what it does |
|---|---|---|
| matrix | | The facade matrix image, e.g. the 3840x2160 contest PNG. |
| mode | color_regions | `color_regions`: flat fills in the matrix. `line_drawing`: closed cells between lines. `mask_folder`: one region per PNG in a folder. |
| min_region_area | 400 | Regions smaller than this many pixels are merged or dropped. Anti-aliasing specks and gaps end up here. |
| merge_small_regions | on | On: a small region joins the neighbour it shares the longest border with. Off: it is dropped and its pixels stay unassigned (never diffused). |
| color_tolerance | 24 | color_regions: RGB distance (0-255 units) under which two colours count as one fill. Edge pixels further than this from every fill are given to the spatially nearest region, so anti-aliased rims don't become regions. |
| split_disconnected | on | color_regions: on = each separate patch of a colour is its own region (every window); off = one region per colour. |
| group_tolerance | 0.08 | Regions share a group_id when their bbox width and height differ by at most this fraction and their shapes overlap with IoU >= 1 - this (same colour required in color mode). |
| color_names | empty | color_regions: `#2b3a55 = windows` or `43,58,85 = windows`, one per line. Every region of that colour gets the name as group_id; unnamed colours are grouped by shape (g1, g2, ...). Comments start with `//`. |
| mask_folder | empty | mask_folder: folder of PNGs (After Effects export). Alpha is used when it varies, otherwise brightness > 50 %. `windows_03.png` becomes region `windows_03` in group `windows`. Where masks overlap the smaller one wins, so windows can be cut out of a full wall mask. Quotes from "Copy as path" are fine. Masks of another size are resized to the matrix. |
| recursive | on | mask_folder: also read PNGs in subfolders. Sources are then relative paths like `Windows/W_F1_C03.png`. |
| scope_masks | empty | mask_folder: names or wildcards (comma or one per line, matched against the file name, stem or relative path, any case) of masks that are **not regions** but the area regions may use. Their union is the scope; nothing outside it is ever a region. |
| split_masks | empty | mask_folder: masks holding several separate shapes. Each 8-connected shape becomes a region `<stem>_01`, `_02` ... in reading order, all in group `<stem>`. Specks under min_region_area are dropped. |
| tag_only_masks | empty | mask_folder: masks that only tag regions (see `tags`) and are no region themselves. |
| group_by | stem | mask_folder: `stem` = file name without its trailing number (`windows_03` -> `windows`); `folder` = the subfolder name (`Windows`). Split masks always group by their own name. |
| scope | | Optional MASK, any mode: pixels outside it are never part of a region. Must be the matrix size. |
| output_masks (advanced) | on | Off: `masks` is a single empty mask, to save RAM when only `regions` is used. |
| line_threshold (advanced) | 0.5 | line_drawing: brightness separating lines from paper. |
| line_polarity (advanced) | auto | line_drawing: `dark_lines`, `light_lines`, or `auto` (the minority is the lines). |
| line_gap_close_px (advanced) | 1 | line_drawing: lines are thickened by this before filling, closing gaps up to about twice this wide. Line pixels are then given back to the nearest cell. |

| output | |
|---|---|
| masks | MASK batch, one per region, same size as the matrix. Float 0/1. RAM: regions x width x height x 4 bytes (100 regions at 4K = 3.3 GB); a warning is logged above 2 GB. |
| regions_json | Region table, see below. Feed it to kubakub region plan. |
| preview | One colour per group, black borders, region id and group id drawn inside each region. Also shown on the node. |
| scope | MASK, 1 where regions may be (all 1 without a scope). |
| regions | int32 label map + this table + scope, for the region nodes (33 MB at 4K instead of one float mask per region). |

regions_json:

```json
{
 "format": "kubakub.facade.atlas", "version": 1,
 "width": 3840, "height": 2160, "mode": "color_regions",
 "regions": [
  {"region_id": 3, "name": "r3", "group_id": "windows", "bbox": [120, 180, 110, 150],
   "area": 16500, "centroid": [174.5, 254.5], "color": "#283c5a"}
 ],
 "groups": {"windows": [3, 4, 5]},
 "unassigned_px": 0,
 "notes": ["1 region(s) under 400 px merged into a neighbour"]
}
```

`bbox` is `[x, y, width, height]` in matrix pixels. In mask_folder mode every
region also has `tags`: the other masks covering at least half of it, broadest
first (e.g. `["M_Facade_Except_Glass", "M_FLOOR_F1"]`), which is the hierarchy
the overlapping masks describe. With a scope the table has `scope_px` and
`unassigned_in_scope_px`.

a typical After Effects mask setup (`Masks` folder with `Groups/` and `Windows/`):
recursive on, scope_masks `M_Projection_Range`, split_masks
`M_Pilasters, M_Stone_Frames, M_Lintels` (add `M_Cornices_Portals` to separate
the cornice from the three portal frames), group_by `folder`. Result: 75
regions (4 floor walls, cornice+portals, 15 pilasters, 20 stone frames, 5
lintels, 30 windows), nothing outside the projection range, portal openings
unassigned, about 8 s. `color` appears in
color_regions mode, `source` (the PNG filename) in mask_folder mode.

### Test without a model

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_facade_masks.py
```

Builds a synthetic facade (sky, cornice, wall, 4x8 windows, ground floor, door)
and checks all three atlas modes; `--full` also times a 3840x2160 run. Preview
PNGs land in `tests/test_output/`.

## Canvas plan / to work / restore: the geometry contract

For projection mapping the matrix is the rule. No node scales non-uniformly.
The work canvas is `target / k` for one uniform factor k, padded to the model's
size grid (never stretched), and after generation the padding is cropped away
and the canvas is scaled back to the exact target. This matters because core
silently center-crops off-grid VAE encodes (Flux2, LTX) and stretches H3 first
frames.

```
matrix 3840x2160 ─► Canvas Plan (flux2, k 2) ─► width 1920 / height 1088 ─► empty latent
          │                     │
          └─► Canvas To Work ◄──┤   image 1920x1088 (4 px replicate padding top and bottom), scope mask
                                │
   decode ─► [model upscaler 4x] ─► Canvas Restore ─► 3840x2160, registered to the matrix
```

**kubakub canvas plan**: `target_width/height` (or `size_from` an image),
`model_family` (flux2 divisor 16; ltxav divisor 32, 8k+1 frames; minimax_h3
divisor 32, 17k+5 frames, 24 fps; sd_8), `k` as a number, a fraction (`15/4`)
or `auto` (picks from `budget_mp`, the model's trained budget when 0).
A k that doesn't divide the target into whole pixels is refused, with the
nearest exact factors listed. `target_frames` (video) is rounded up to the frame
rule. Outputs the plan, the grid `width`/`height` for the empty latent,
`frames`, `fps`, the plan as JSON and a readable report (also shown on the node).

**kubakub canvas to work**: any image of the target's aspect (matrix, start
image, reference, first frame) and an optional mask onto the grid canvas.
`aspect_mode` error / center_crop / pad for images of another aspect.
`pad_mode` replicate / reflect / gray / black. `scope` is 0 on the padding.

**kubakub canvas restore**: accepts the padded canvas, the cropped work canvas or
the target at any uniform scale (so a 4x model upscaler can sit before it),
crops the padding, resizes to the target, trims extra video frames and checks
the size. A non-uniformly resized input is refused.

For the 3840x2160 matrix: flux2 k 2 gives 1920x1088, k 3 gives 1280x720 exact;
ltxav k 4 gives 960x544; minimax_h3 k 3 gives 1280x736. Full table in the plan.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_geometry.py
```

## kubakub regions from masks

Masks from anywhere (SAM, painted, Illustrator layers, ID passes) become regions,
either a new atlas or a patch into an existing one.

- `masks`: a MASK batch, one region per mask (values over 0.5 are inside).
- `names`: one per line in batch order, or one comma separated line. `name = group`
  sets the group; otherwise the group is the name without its trailing number
  (`W_F1_02` -> `W_F1`). Missing names become `mask_01..`, repeated ones get `_2`.
- Overlapping new masks: the smaller one wins (as in the mask folder atlas).
- `base_regions` + `patch`: **on_top** cuts every new mask out of the regions
  below (a window out of a wall; a base region that loses all its pixels is
  removed, with a note). **underneath** only fills pixels no base region has.
  The base keeps its names, groups, tags and scope; new masks are clipped to
  that scope. Each new region is tagged with the base region it was cut from
  plus that region's tags, so `[tag:M_FLOOR_1]` or `[tag:W_F1_*]` rules reach it.
- `matrix` (optional): preview background, and the size when there is no base.
  Masks of another size are resized (nearest), with a note.
- `scope`, `min_region_area`, `merge_small_regions`, and (advanced) `scope_masks`,
  `split_masks`, `tag_only_masks` work as in regions from matrix / masks, matched
  against the names.

Outputs: `regions` (for kubakub region plan), `region_masks`,
`regions_json`, `preview`. The logic is `facade_core.atlas_from_masks()`, shared
with the mask folder atlas (`label_masks()`).

Example workflow `example_workflows/regions_from_masks_patch.json`: a mask
folder (the sample facade's) as base, a new shape patched on top as `balcony_01`, then the region plan. With your
own painted PNG (several shapes in one mask) every shape becomes its own region.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_regions_from_masks.py
```

## kubakub regions from sam3

Named masks from SAM 3.1 for Regions From Masks. Load `sam3.1_multiplex_fp16` with
Load Checkpoint (MODEL + CLIP). The node calls core SAM3 Detect once per object:

- `prompts`: one line per kind of object, `name = text : N` or just `text`. Core
  finds only 1 object per prompt unless `:N` is given, so the default here is 50.
  Every hit is its own mask `name_01, name_02 ...` in reading order.
  `drop_groups` drops hits that cover two or more other hits of the same prompt
  (SAM often adds one mask for a whole row of balustrades). A window surround and
  its glass (one inside one) are both kept.
- Points from the KJNodes **Points Editor** (width/height = image size, or
  `normalize` on): with `point_mode` = one region per point, each positive
  point is one object and negative points apply to all. Boxes (ctrl+drag) are
  one object each. `point_names` names them in click order, then the boxes.
  Only the connected parts under the click are kept (no stray specks).
- `detail` (points and boxes): SAM sees every image at 1008x1008, so on a 4K
  matrix a balustrade is a few dozen pixels. The detail pass segments the
  object's box again on a square crop at full resolution and pastes it back;
  the edges then follow balusters and frames. If the result is under 0.3x or
  over 3x the first mask, the first one is kept (noted in the report).
- `scope` clips the masks, `min_area` drops specks, `save_folder` writes every
  mask as `<name>.png` (white on black, image size) into e.g. the project
  Masks folder; existing files are kept unless `overwrite`.

Outputs: `masks` (batch) and `names` (one per line) go straight into Regions
From Masks; `preview`; `report` (area, source, score per object and notes).

Test: `window : 60` and `stone balustrade : 10` gave 41 windows and 5
balustrades (plus one whole-row hit, dropped), 3 clicks and 1 box with detail,
~24 s in total. Text hits come as closed outlines; clicked objects with detail
cut out the gaps between balusters.

Example workflow `example_workflows/regions_sam3_masks.json`: matrix -> Points Editor
-> SAM3 Masks -> Regions From Masks patched on top of a mask-folder atlas.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_sam_prompts.py
```

## kubakub regions from illustrator

Regions straight from the festival's Illustrator file. `.ai` files are PDF
compatible by default (Illustrator: Save As .ai, "Create PDF Compatible File"),
so PyMuPDF reads their layers, paths and placed images; a `.pdf` works too.
The file on disk is never changed.

- **Every layer is read, hidden ones too** (the layers are switched on in
  memory before reading). Each layer is rasterised by replaying its own paths,
  so curves, compound shapes (even-odd holes) and stroke widths are exact.
- **Artboard -> matrix** with one uniform scale. With a `matrix` input the
  artboard must have its aspect (else refused); without one, `scale` pixels per
  point (1 = artboard size, e.g. 3200x2160 pt = 3200x2160 px).
- **Roles** per layer, guessed from the content and listed in the report,
  or set in `layer_roles` (`CENTRE, COUR, JAR = lines`, wildcards ok):

  | role | guessed when | becomes |
  |---|---|---|
  | lines | mostly stroked paths | the closed cells between the lines, `<LAYER>_001..` in reading order, grouped by shape `<LAYER>_g<n>` (repeated windows share a group), tagged `<LAYER>` |
  | shapes | mostly filled paths | one region per separate shape `<LAYER>_01..` on top (cuts into the cells), group `<LAYER>`, tagged with the fill colour and the cell it covers |
  | outside | name has mask/range/scope/matte and the fill touches the border | never a region (the scope is everything else) |
  | scope | same name, not touching the border | regions only inside |
  | tag | only when set | tags the regions it covers by half or more |
  | reference | only placed images | the `reference` output (the photo, placed at its artboard position) |
  | ignore | a fill covering the whole artboard, no paths (text), names starting with `_` or text/label/guide/grid | nothing |

- `min_region_area`, `merge_small_regions`, `line_gap_close_px` work as in the
  line_drawing atlas; `split_shapes` off = one region per shape layer.

Outputs: `regions`, `preview`, `lines` (the line layers white on black, a
control image), `reference` (the placed photo), `layer_masks` + `layer_names`
(every layer with paths as a mask), `scope_mask`, `regions_json` (with a
`layers` table) and a `report`. The file's date and size are part of the
cache key, so saving the .ai again re-runs the node.

3200x2160 test facade (8 layers, 5094 paths, 3 of the line layers hidden):
roles all guessed right (MASK outside, CENTRE / COUR / JAR lines, OBSTACLES
shapes, _IMG reference, TEXT and Background ignore); the MASK layer matches the
festival's own MASK_1.png export to 99.96 %; 795 regions in 4 s. Rules like
`[tag:COUR]` and `[group:OBSTACLES] strategy = keep` work directly. Known
limits: dense ornament (the balustrade) becomes many small cells (raise
`min_region_area`); a few cells along the edge lie outside every layer's
outline and are named `lines_…`; only the first artboard is read; live text is
not a path (outline it in Illustrator if it should be a region).

Example workflow `example_workflows/regions_from_illustrator.json`.

Test without a model (writes a small layered PDF with PyMuPDF and reads it back):

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_illustrator.py
```

## kubakub regions to svg / pdf / dxf

Regions (or any MASK) as vector files, in two modes sharing one tracer:

- **outline**: each region as filled paths with its holes, traced along the
  pixel edges (a 40 x 40 px square is exactly 1600 px). Vertices that turn
  more than `corner_deg` stay corners, so straight edges stay straight; the
  rest become cubic Beziers (Catmull-Rom through the simplified points).
  Parts that touch only at a corner pixel are kept as separate outlines.
- **centerline**: the skeleton as single open lines (spurs under `spur_px`
  removed), ordered nearest-neighbour from the top-left for short travel
  (KD tree: 42 000 lines of a festival line drawing in 4 s). For plotters, lasers,
  engraving.

Files (`svg`, `pdf`, `dxf` switches; `folder` empty = output/kuba_vector; a
counter never overwrites): SVG with one group per region group and the region
names as path ids; PDF with one layer (optional content group) per group, which
Illustrator shows as layers; DXF R12 with one layer per group (closed
POLYLINEs for outlines, open ones for centerlines, y flipped, curves flattened).
`width_mm` > 0 writes real millimetres (one uniform scale), `kerf_mm` grows or
shrinks every outline by kerf / 2 (mitred corners) for cutting. `select` picks
regions (`W_F1_*`, `group:Windows`, `tag:...`).

Tested: a mask-folder atlas, 75 regions -> 223 paths in 5 layers (traced area = pixel
area for every region); festival line layers as centerlines over 40 x 27 m.
Example `example_workflows/regions_to_vector.json`; test `tests/test_vector.py`.
