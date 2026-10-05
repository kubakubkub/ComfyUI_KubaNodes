# Generate (2d / generate)

A prompt and settings per region (region plan), diffusion region by region with Klein or Qwen Image 2.1 (region sampler), soft seams between them (seam pass), and rooms or worlds behind the windows in the audience's perspective (frame in frame). Regions come from [regions.md](regions.md).

## Example workflow: `example_workflows/facade_regions_diffusion.json`

Open it from the templates. It runs the whole chain on the sample facade (1920x1080), no files needed:

```
kubakub sample facade -> regions from matrix / masks (its mask folder) -> region plan (example rules)
   -> region sampler, Klein 9B, only = !W_F1_*               every region but the floor 1 windows
   -> region sampler, Qwen 2.1 + turbo LoRA, only = W_F1_*   the floor 1 windows, frame in frame
   -> seam pass (Klein) over both changed masks -> Save (output/KubaRegions)
```

With your own facade: a Load Image node for the matrix and the path of your mask folder in `mask_folder`
on the regions node. The rules are the text of [rules_example.txt](rules_example.txt), pasted into the region
plan so the workflow stands alone; rules for names your masks do not have match nothing. To iterate on a few
regions, set `only` on one sampler (e.g. `W_F2_*`) and mute the other.

## kubakub region plan

Gives every region its prompt and settings with a small rule text. Input:
`regions` from any of the regions nodes. Output: `plan` (for the samplers),
`plan_json`, a readable `report`, and a `preview` where regions of the same
strategy and prompt share a colour (`p1`, `fif2`, `keep`).

```ini
// comments start with //
[plan]
order = largest_first            // plan | reading | largest_first | smallest_first

[default]                        // every region
prompt = weathered sandstone palace facade at night
denoise = 0.55

[group:M_Pilasters]
prompt += carved vines climbing the stone      // += appends to the inherited prompt

[W_F1_*]                         // a bare selector is a region name; wildcards * ? [..]
strategy = frame_in_frame
prompt = an underwater room with {jellyfish|a sunken piano|koi}   // one pick per region

[tag:M_FLOOR_F3 group:Windows]   // space = AND
prompt = candle lit attic room behind old glass

[region:0, M_Cornices_Portals]   // comma = OR; region:3-7 ranges; !name = NOT
strategy = keep
```

Selectors: `default` or `*`, `name:` (or a bare name), `group:`, `tag:`
(the covering masks from the atlas), `region:<id>` or `region:<a>-<b>`. Case
does not matter. **Priority:** region > name > group > tag > default; an AND of
several terms beats a single term of the same kind; among equals the later
rule wins. Settings merge key by key, so a window rule can change the prompt and
keep the denoise from the group rule.

In values: `{prompt}` is the inherited prompt, `{name}`, `{group}`, `{id}` the
region's own, `{a|b|c}` one option chosen per region from `seed` (repeatable).

| setting | default | |
|---|---|---|
| prompt, negative | empty | `+=` appends |
| strategy | inpaint | `inpaint` (repaint in place), `frame_in_frame` (own scene set into the region), `keep` (untouched; `skip` works too) |
| denoise | 0.6 | 0..1 |
| steps, cfg | 0 | 0 = backend default |
| seed | -1 | -1 = plan seed + region id |
| order | 0 | lower runs earlier |
| context_px | 64 | canvas around the region the model sees (target pixels) |
| dilate_px, feather_px | 4, 8 | inpaint mask growth, paste-back softness |
| color_match | mean_std | none, mean_std, mkl |
| blend | on | include the region's borders in the seam pass |
| reference | self | none, self, style (the crop plus the style image of kubakub versions) |
| lora | empty | `name:strength; ...` (per-region hook LoRA) |
| region_mp | 0 | work MP for the region crop, 0 = backend default |
| z | 0 | stacking for frame in frame |
| fif_width, fif_height | 0 | frame in frame sub-generation size, 0 = from the region box |
| fif_harmonize, fif_border_px | 0.35, 24 | frame in frame border pass |

Mistakes are reported with the line number (`line 7: unknown setting
'denoize'; did you mean 'denoise'?`); rules that match no region are listed as
notes in the report. `rules_file` loads rules from a .txt (read only, e.g. in
the project folder); the widget text is applied after it and can override it.
[rules_example.txt](rules_example.txt) is a starting point for a mask-folder facade.

Test without a model: `tests/test_plan.py`.

## kubakub region sampler

Runs a region plan on an image with one model (sequential inpaint). Inputs: `model`, `clip`, `vae` (Flux 2 Klein 4B or 9B is detected
automatically; other models use the generic KSampler path), `plan`, and
`image`, the canvas, exactly the size of the regions (never resized to fit).

Per region, in plan order:
1. A box around the region plus `context_px` is scaled **uniformly and exactly**
   (scale = 16 / block for a whole block, the box grows to whole blocks) to about
   `region_mp` on the 16 px grid, at most `max_upscale` times. A 270x310 window
   box becomes 864x992 at x3.2; a whole-floor wall is scaled down.
2. The region, grown by `dilate_px` and cut to the scope, is the inpaint mask.
   The crop's own latent is attached as a reference latent (`reference = self`).
3. Flux 2 is sampled with core's Flux2Scheduler sigmas for the crop size;
   denoise works as in KSampler.
4. The result is scaled back by the exact inverse, colour matched to the
   original crop (`color_match`), and pasted with a `feather_px` soft edge that
   never reaches outside the inpaint mask or the scope. The canvas updates, so
   later regions see earlier ones.

`reference` is the main creative control: `self` keeps the structure of what is
there (Klein then behaves like an editor; good for materials, weathering, added
details like vines), `none` lets the region be freely repainted inside its mask
(a window becomes a lit room). `denoise` matters much more with `none`.

| input | default | |
|---|---|---|
| steps, cfg | 0, 0 | used where the plan says 0; 0 = the model's default (Klein 4 / 1, Qwen 2.1 25 / 1) |
| sampler_name | euler | |
| scheduler | simple | generic models only; Flux 2 uses its own schedule |
| region_mp | 1.0 | work megapixels per region crop |
| max_upscale | 4.0 | small regions are scaled up at most this much |
| only | empty | process only regions matching these selectors (plan syntax), to iterate on a few |
| seed_offset (advanced) | 0 | added to every region seed |
| adapter (advanced) | auto | flux2 / generic |

**frame_in_frame** regions get their own scene instead: it is generated
from scratch with the region's prompt at about `region_mp` in the region box's
aspect (or `fif_width` x `fif_height`), scaled uniformly to cover the box,
centre-cropped, and placed through the exact region mask (feathered inward by
`feather_px`, never outside the region or the scope). With the glass
masks the window bars and frames stay intact and the scene shows only in the
panes. Then, if `fif_harmonize` > 0, a ring `fif_border_px` wide along the
region border is inpainted at that denoise (with the crop as reference) so the
scene settles into its frame. `color_match` and `reference` don't apply to the
scene itself. Scenes and inpaints run in the same plan order, so regions after
a scene see it. Klein 9B: about 12-16 s per window scene including the border
pass.

Outputs: `image`, `changed` (where pixels were replaced), `report` (per region:
box, scale, work size or scene size, seconds). Klein 9B fp8 on a 12 GB laptop GPU: about 10 s per region
at 1 MP, so the 74 regions of a 3840x2160 facade take roughly 12 minutes.

Test without a model: `tests/test_sequential.py` (a fake adapter checks crop
planning, masks, scope, pasting and registration).

## Qwen Image 2.1 in the region sampler

Detected automatically (core 0.37 `QwenImage21`). Loaders: UNETLoader
`qwen_image_2.1_int8_convrot`, CLIPLoader `qwen3vl_8b_w4a8` (or int8_convrot) with
type **qwen_image**, VAELoader **`texture_fix_vae_for_qwen_image_2.1_bf16`**
(or the original `qwen_image_2.1_vae_bf16`). The `pe_t2i` / `pe_i2i`
files are optional prompt enhancers for core's Generate Text node, not encoders.

The texture-fix VAE (huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) is a
decoder-only finetune with the same encoder, so it replaces the original everywhere. Tested
on facade crops: the fine checkerboard speckle on flat plaster drops 4-14x
(same latent decoded with both), edges and structure unchanged. Both example Qwen workflows use it.

- **Put core `Qwen Image 2.1 Cache` between the loader and the sampler with
  `device = gpu` and `dtype = default`.** In testing `auto` (pinned RAM)
  aborted ComfyUI and `dtype = int8` failed with an aimdo error. The adapter logs a warning when the cache is on auto.
- Grid 32: crops are scaled exactly by 32 / block, so an edit's output matches its
  reference size.
- With `reference = self` the crop goes through the Qwen3-VL vision tower and in as
  a reference latent, as core Text Encode Qwen Image 2.1 does, so every region is
  encoded on its own. Prompts may address it as `<image1>` ("turn the glass in
  <image1> into an aquarium").
- Defaults: 25 steps, cfg 1, euler / simple (sampler `steps` / `cfg` = 0 picks the
  model default). Measured on a 12 GB laptop GPU with the w4a8 encoder: about 75 s
  per inpaint region at 1 MP and about 130 s per frame-in-frame window (scene plus
  border pass), roughly 10x Klein. Frame-in-frame scenes are photographic and
  detailed; use Qwen for the few hero windows and Klein for the many small regions
  (one backend per sampler node, chain two samplers with `only`).

### Qwen Image 2.1 turbo LoRA (Viggle)

`Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors` (huggingface.co/Viggle/
Qwen-Image-2.1-viggle-turbo) loads with plain core **LoraLoaderModelOnly**: all 227
target layers map onto Qwen 2.1 through core's own key mapping (including the fused
`img_mlp.gate_up`). Chain: UNETLoader -> LoraLoaderModelOnly (strength 1) -> Qwen Image
2.1 Cache (gpu, default) -> region sampler / seam pass with **`schedule =
qwen21_turbo_5`**: the LoRA's five trained timesteps 1, 0.875, 0.75, 0.5, 0.25 with the
model's size-dependent shift (mu 0.5 at 256 tokens to 0.9 at 8192, 0.69 at 1 MP), cfg 1.
Partial denoise starts at the matching trained step: 0.75 runs 3 steps, 0.5 two, the
seam pass at 0.3 one. `steps` is ignored with this schedule.

Measured (12 GB laptop GPU, int8 model, w4a8 encoder, cache gpu): frame-in-frame window
**~25 s** (was ~130 s at 25 steps), inpaint region **~20 s** (was ~75 s). Same seed gives
nearly the same scene as the 25-step run, a little softer. No LoRA keys failed to load.

### Qwen Image 2.1 8-step LoRA (Pruna)

`pruna_qwen_image_2.1_8step_v0.1.safetensors` (huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1),
same chain as above (LoraLoaderModelOnly strength 1, cfg 1) with **`schedule =
pruna_qwen21_8`**: the card's fixed sigmas 1, 14/15, 6/7, 10/13, 2/3, 6/11, 0.4, 2/9
(shift 2 on eight linear steps, no size-dependent shift). `denoise` is read on the same
unshifted axis as `qwen21_turbo_5`, so both LoRAs start at about the same noise for the same
denoise: 0.75 -> 6/7 (6 steps), 0.875 -> 14/15 (7 steps). Trained at 1K; a 1568x640 crop
worked as well.

The sampler input is used as set. Tested on facade crops (1 MP):

| sampler | time | look |
|---|---|---|
| Viggle 5-step, euler | 11-18 s | reference |
| Pruna, euler | 17-31 s | nearly the same image as Viggle at the same seed |
| Pruna, deis_2m | 24-38 s | more material detail (stone, stains, ivy), invents a bit more |
| Pruna, res_2s | 29-50 s | most detail, also most invention (windows become doors) |
| Pruna, res_2s_ode | 29-50 s | close to euler |

res_2s / deis_2m come from RES4LYF (there is no deis_2s). For a restyle that keeps the
facade layout, denoise ~0.875 works; 0.75 hardly changes the facade. A full-denoise edit with
the facade only as a reference image invents a new building with every LoRA and sampler.

## kubakub versions

Many versions of one facade from one queue: a numbered contact sheet of fast drafts, then only the versions you
pick at the full canvas size. Example workflow: `example_workflows/versions.json`.

Inputs: `model`, `clip`, `vae`, the `plan` (from kubakub region plan: what every version has in common) and the
`image` every version starts from (the clay render, a photo). The **versions sheet** lists what changes, one
option per line:

```
// 'short name = text' gives an option its label on the sheet
[look]                            // added to every region's prompt
copper = oxidised copper plates, verdigris, rivets
paper = folded white paper, soft daylight
none                              // the plan as it is

[region windows]                  // added to some regions: any plan selector (name, group:, tag:, region:)
warm rooms, people moving inside
aquariums, fish, blue light

[rotate windows | wall | roof]    // prompt rotation: the ideas move on by one region per version
molten glass
woven textile
moss and ivy

[lora]                            // name : strength : words added to the prompt
none
My_Style_v1 : 0.8 : my style
My_Style_v1 : 1.2 + Detail_v2 : 0.5   // a chain: several LoRAs on one version
late and early = My_Style_v1_6000 : 1.1 + My_Style_v1_3000 : 0.65   // a chain with a short name for the sheet

[reference]                       // style images: a file, or a folder (every image in it)
none
my_project/refs/styles
my_project/refs/abstract | 12      // 12 images spread over a big folder; | 12 | 5 = another 12

[image]                           // the picture a version starts from: a file, a folder, or 'input'
input                             // the image connected to the node
wireframe = my_project/renders/facade_wire_on_clay.png
my_project/renders/maps           // every image in the folder (canny, cryptomatte ...)

[set denoise]                     // any plan setting; [set denoise windows] for some regions only
0.7
0.9

[seed]
1
2
```

**Whole picture.** With `generate = regions` every region is repainted inside its own mask: the facade keeps
its elements, a window stays a window. With `generate = whole picture` the picture is made in one free sample
from nothing, at about `draft_mp` megapixels: the starting image is only the model's reference image 1, the
style image is image 2, and there is no mask. Forms can grow across the whole facade, break through it and leave
it; the background is whatever the prompt says. This is the way to large sculptural interventions. A free sample
comes back a few pixels moved or scaled; the node puts it back onto the starting image by itself (as **kubakub
align to source** does, see [post.md](post.md)), and leaves it alone when that fit is not certain. Three things
decide the result: the prompt must say what happens (a prompt that only asks for a texture repaints the facade),
the style image sets most of the look, and `draft_mp` around 1.5 gives the model room. Rules for single regions
only add their words to the one prompt. The final scales the draft up and adds full-size detail inside the
regions, as usual; picks from a folder remember how they were made.

**A random idea per region.** In the plan, `prompt += {tiles|folded cloth|porous stone|mesh}` gives every
region one of the ideas (which one depends on the seed and the region). With a `[seed]` list in the sheet every
version deals them out anew. Regions that got different ideas are sampled one after the other, so the picture
can fall apart into patches: set `unify` to about 0.3 and the whole picture gets one more pass that ties them
into one structure (0.5 and more lets the regions grow into each other).

**Starting images.** `[image]` rotates the picture the versions start from, the model's image 1: the clay
render, the wireframe over the render, a canny map, a cryptomatte. They must show the same view as the regions
(another size is scaled, another shape is refused). A batch of several images on the `image` input does the
same without the sheet. The regions stay the same for every starting image.

A version is the plan plus that version's rules, a LoRA, a style image, a starting image and a seed. The report prints the rules
of every rendered version, so one version can be continued by hand in a region plan.

| input | default | |
|---|---|---|
| mode | rotate | **rotate**: version n takes option n of every list (short lists start over). **combine**: every combination. **one by one**: the first option of everything, then one change at a time (which LoRA? which look?) |
| quality | draft | **draft**: small and fast. **final**: the picked versions at the full canvas size |
| pick | empty | version numbers as on the sheet, `3, 7-9`; empty = all |
| max_versions | 12 | the sheet never gives more than this (combine grows fast) |
| seed | 0 | added to the plan's seed |
| draft_mp | 0.5 | work megapixels per region of the draft |
| final_denoise | 0.4 | final: how much the scaled-up draft is resampled; higher = more new detail, further from the draft |
| draft_long_edge | 1600 | the picture is made on the canvas scaled down to this long edge |
| scheduler | auto | **auto**: the model's own schedule (Flux 2: core's Flux2Scheduler for the crop size). simple, beta, sgm_uniform ...: that core scheduler, as in KSampler. qwen21_turbo_5 / pruna_qwen21_8: only with the Qwen turbo LoRAs |
| references | optional | style images as a batch: more options of the reference list |
| picks_folder | empty | final: a folder with the drafts you like; every image in it becomes a final. The sheet and `pick` are not used |
| generate | regions | **regions**: every region is repainted inside its own mask. **whole picture**: one free sample of the whole picture, see below |
| unify | 0 | one more pass over the whole picture after the regions, at this denoise; 0 = off |
| save_folder | kubakub/versions | where the versions are saved, inside ComfyUI's output folder (a full path needs `save_anywhere = on` in kubakub.ini); empty = nothing is saved |

**Picking by folder.** The node saves every draft as `r<run>_v<number>_<what it is made of>.png` (for example
`r07_v03_isometric_soft-pair_ref12.png`). Each file holds its own version: the rules, the LoRAs, the style image
and the seed. Copy the drafts you like into a folder of your own, paste its path into `picks_folder`, set
`quality = final` and queue. Every image in that folder is scaled up to the full canvas and gets its full-size
detail with exactly what it was made with, whatever the sheet says now; the finals land in
`<save_folder>/final` under the name of their draft. Drafts of several runs and sheets can sit in one picks
folder. An image that does not say which version it is (a renamed older draft, a picture from elsewhere) is
still scaled up and detailed, with the plan as it is (no look, no LoRA, no style image); the report names it.

How it works:
- **Draft**: canvas and regions are scaled down to `draft_long_edge`; regions that get the same prompt and
  settings in a version are sampled together (one look over seven regions = one sample). Measured on a
  3200x2160 facade with 7 regions, Klein 4B, 12 GB: 6-8 s per version (the region sampler at full size: 32 s).
  A style image adds about 6 s.
- **Final**: the same draft (from the region cache when it was just rendered) is scaled up to the full canvas and
  resampled in 1024 px tiles at 1:1 with `final_denoise`, each tile with the prompts of the regions in it and
  itself as reference. So a final is its draft with full-size detail, not a new picture. About 95 s per version
  at 3200x2160 (12 tiles). Pixels outside the regions and `keep` regions stay as in the input image.
- **Cache**: results are kept across queues; changing one line of the sheet re-renders only the versions that
  use it, and a second queue of the same sheet takes seconds.
- **LoRAs** are applied to the image model only (not the text encoder), by file name or an unambiguous part of
  it; versions of one LoRA run after each other.
- **Style images** go in as a second reference next to the region's own crop (`reference = style`). The style
  image alone, without the crop, loses the building at a high denoise. Tested with Klein; with Qwen Image 2.1
  the style image is passed as a reference latent only (not tested yet).
- For a look that should really change the material, give the plan `color_match = none` and a `denoise` of
  0.8-0.9: the default colour match pulls every region back to the colours of the input.

Test without a model: `tests/test_versions.py`.

## kubakub seam pass

Runs after the region sampler (same model, clip, vae and plan; `image` and
`changed` from the sampler). A soft band `seam_px` wide (linear falloff, 1 on
the border, 0 at the band edge) is built along every border between two regions
where both have `blend = on` and at least one side changed. `keep` regions and
pixels outside the scope are never in the band. The canvas is covered by tiles
of `tile_px` at 1:1 (no loss of detail); only tiles containing seams are
sampled, with the band as a soft inpaint mask, the crop as reference and a low
`denoise` (0.3). With `differential` on (core DifferentialDiffusion, copied
into the pack), the band edge is denoised less than its centre, step by step.
Each tile uses the prompt of the region most present in its seams, unless
`prompt` is set.

Outputs: `image`, `seam_mask` (the band, to inspect or to feed core nodes) and a
`report` (tiles, prompts, seconds). Klein 9B: about 14 s per 1024 px tile. The
sampler's feathered paste already hides most edges, so the seam pass matters most
where neighbouring regions differ strongly in style or light.

## Frame in frame: audience viewpoint, rooms behind windows

Rooms behind windows and things sticking out of the facade, in the perspective
of the audience.

**kubakub audience viewpoint**: the facade's real width (`facade_width_m`, the matrix maps onto
it with one scale), the height of the matrix's bottom edge above the ground,
and where the viewer stands (`viewer_x_m`, -1 = centre; `eye_height_m`;
`distance_m`). A plane D metres behind the facade appears scaled towards the
viewer's foot point by `d / (d + D)`; D < 0 (in front) scales away from it.

**kubakub frame guide**: regions + viewer + `depth_rules` (`W_F1_* = 4`,
`group:Windows = 3`, `tag:M_FLOOR_F0 = -1.5`; later lines win) ->
- `guide`: the matrix with every frame drawn as the viewer sees it: a room
  behind a window (back wall, shaded side walls, ceiling / floor, edges; the
  facade hides the rest, the mullions of a multi-pane window stay in front) or a
  parasite in front of the facade (front face and side walls over the facade),
- masks `area` (what generation may change), `back` (back / front faces, where
  media or a world goes), `walls`, and `frames_json` with the corner-pin quads.
`dim` 0.55 for looking, 0 when the guide is the image to generate on.

Tested on a 3840x2160 facade (region sampler, Klein 9B, guide as image and reference,
**denoise 0.95**): five floor-1 windows became libraries with the ceilings seen
from below and each vanishing point pulled towards the viewer. At 0.8 the grey
guide survives; at 1.0 without reference the perspective is lost and extra
window bars appear.

**kubakub frame in frame**: the frames come from the region plan, so all
settings live in one rule text:

```
[W_F1_*]
fif_depth_m = 3            // metres behind the facade (< 0 = sticks out)
fif_source = generate      // perspective guide, painted by the region sampler
strategy = inpaint
denoise = 0.95
reference = self
[W_F2_*]
fif_depth_m = 0.4
fif_source = media         // one of the media images on the back wall
fif_media = rose           // a name from media_names, an index, or empty = in turn
[W_F3_*]
fif_depth_m = 0.3
fif_source = world         // the world image behind the whole facade
```

Inputs: plan, viewer, matrix, `media` (a batch, cover-fitted and corner-pinned
onto each back wall), `media_names`, `world` (it maps 1:1 onto the facade
through the viewer projection, so every window shows its own part of one
continuous scene). Side walls of media / world frames are the guide's shading
tinted with the content's colour. Outputs: the composite (feed it to the Region
Sampler as `image`), masks `area`, `generate`, `back`, `walls`, frames JSON.

A courtyard (facade ~40 m, audience ~15 m, from the site photos): the
viewer looks steeply up, so on the upper floors a deep room shows mostly
ceiling. Keep depths shallow there: 0.3-0.5 m is a picture just behind the
glass or a window reveal onto a world. Example: `example_workflows/frame_in_frame.json`.

**Parasites**: a negative `fif_depth_m` makes the frame stick out
towards the audience (front face scaled away from the viewer's point, side
walls, drawn over the facade). frame in frame casts its shadow on the facade
(`shadow_strength`, `shadow_angle_deg` 0 = right / 90 = down, `shadow_length`
per metre it sticks out) and outputs an extended `plan`: every parasite is a new
region `<name>_out` (frame + walls + front face) with the frame's settings, so
the region sampler paints the whole object. Connect frame in frame's `plan` (not
the region plan's) to the sampler.

Test without a model: `tests/test_viewer.py`.
