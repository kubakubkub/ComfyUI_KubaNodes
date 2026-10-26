# kubakub utils

Small helpers: crop by drawing, put an edited crop back, compare two pictures or videos.
Example: [`example_workflows/crop_and_compare.json`](../example_workflows/crop_and_compare.json) (no models).

## kubakub crop

Crop the way you do in Photoshop: on the picture, with the mouse. Nothing is typed.

The node shows the picture as soon as the node before it has one (a *Load Image* at once; anything else after
one run).

| on the picture | does |
|---|---|
| drag | a new box |
| drag inside the box | move it (it snaps to the picture's edges) |
| drag a corner or an edge | size it · **Shift** keeps the shape · **Alt** sizes from the centre |
| drag past the picture's edge | more canvas (see *new area*) |
| double click | all of the picture again |
| arrow keys | 1 px (**Shift** = 10 px) |

The readout on the picture says what the run will cut: `1120 × 528 · 2.12:1 → 1024 × 480`.

| pills | |
|---|---|
| **ratio** | free · original (the picture's own shape) · picture (the shape of the picture on `ratio_of`, e.g. the facade matrix; only shown when one is connected) · 1:1 · 4:5 · 3:2 · 16:9 · 9:16 · 21:9 · ⇄ turns the box (wide ↔ tall) |
| **size** | as cut · 1024 · 1536 · 2048 · 3840 (the long side, the shape kept) · **16 px steps** (on: width and height are multiples of 16, as Flux and most models want) |
| **new area** | how canvas added past the edge is filled: edge (the border pixels stretched) · mirror · black · white · grey · **reset** |

| input | |
|---|---|
| `image` | a picture, or the frames of a video (every frame gets the same box) |
| `ratio_of` | optional: a picture whose shape the box takes (connected = the *picture* pill) |

| output | |
|---|---|
| `image` | the crop |
| `box_mask` | white where the box lies on the picture (the picture's size): to diffuse only that part |
| `new_area` | white on the canvas added past the edge (the crop's size): the mask for an outpaint; black when nothing was added |
| `crop_box` | where the crop came from, for *kubakub uncrop* |

The box is kept as a part of the picture (not in pixels), so a box drawn on a 1024 draft cuts the same part of
the 3840 final.

## kubakub uncrop

Puts an edited crop back where *kubakub crop* cut it from.

| input | |
|---|---|
| `original` | the picture the crop came from (or the same picture at another size: the box goes to the same spot) |
| `crop` | the edited crop, any size (it is scaled to the box) |
| `crop_box` | from *kubakub crop* |
| `soft_edge` | the width of the blend at the box's edges (px of the original); edges on the picture's border stay hard |
| `mask` | optional: paste only where it is white (a mask of the crop, or of the whole original) |

Canvas that was added past the edge is left out: only the part over the picture comes back. One original and
many crops (or the other way round) give one picture per frame.

## kubakub compare

Two pictures or two videos on the node, always on the same frame.

| mode | |
|---|---|
| swipe | drag the line |
| side by side | each side fitted to its half |
| difference | what changed lights up · drag sideways = brighter |
| blink | a and b in turn |
| onion | b over a · drag sideways = how much |

`a` and `b` take a picture, a batch of frames or a video (*Load Video*), in any mix: a still against a video
holds the still. Play, the scrub bar, `,` / `.` for one frame; the corner button **⤢** opens the big window:
wheel = zoom at the mouse, drag = pan, double click = fit, Space = play, 1-5 = the modes (only in the big window, so a stray key on the node never switches it), Esc = close. Sides of
another shape are each fitted, never stretched (the hint under the picture says so).

The pills on the right (`1280` · `1920` · `full`) set the size the frames are shown at, from the next run: full =
every pixel, more to load. Up to 3000 frames per side are shown. `fps` (under *more settings*) is the speed of a
batch of frames; a video plays at its own.

The same compare is in the director (`\`), see [director](director.md#crop-and-compare-in-the-window).

## Tests

    python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_crop.py

Model free: the box, ratios and 16 px steps, every padding fill, the paste back, the three nodes, the director's
layer crop, and (with node installed) that the box drawn on the node gives exactly the pixels the node cuts.
