# Masks (2d / masks)

Masks that are more than on / off, and masks that move. Two nodes that work together or on their own:

```
regions ─ kubakub mask field ─ field ─┐
                        └──── mask ──┼─ kubakub mask animate ─ masks / fps / audio ─ any node that takes a mask
                     Load Audio ─ audio ┘                                              (or MaskToImage ─ Create Video)
```

Example: `example_workflows/mask_animate.json` (runs with nothing connected: the sample facade and a built-in beat).

## kubakub mask field

Regions as a grey ramp: black is where a move starts, white where it ends. It is the path a reveal, a band or rings
run along in kubakub mask animate. It also works on its own as a soft mask, or as a time offset map in After Effects,
Houdini or TouchDesigner.

| input | default | what it does |
|---|---|---|
| regions | | From any kubakub regions node. Empty (and no mask) = the windows of the sample facade. |
| mask | | Instead of regions: a mask or a mask batch. Each mask is a region; with `split_shapes` every separate shape is one. |
| select | `*` | Which regions, in plan syntax: `W_F1_*`, `group:Windows`, `tag:front`, comma = or, space = and, `!` = not. |
| field | edge distance | See the table below. |
| per | each region | `each region`: every region runs from black to white on its own (all windows open at once). `all together`: one ramp over all of them (a wipe across the facade). |
| angle | 0 | direction: 0 = left to right, 90 = top to bottom, 180 = right to left. |
| order | left | region order: left / right / top / bottom, centre (from the middle out), outside, size (the biggest first), random. |
| centre_x, centre_y | 0.5 | radial with `all together`: the centre, 0 to 1 across the picture. |
| noise_px | 64 | noise: the size of the blobs in pixels. |
| seed | 1 | noise and order = random: another pattern. |
| invert | off | The move runs the other way. |
| split_shapes (advanced) | on | mask input: every separate shape is its own region. |

| field | what you get | a reveal along it looks like |
|---|---|---|
| edge distance | black on the edge of a region, white at its deepest point | the outline first, then it closes to the middle (invert: it grows from the middle) |
| direction | black to white along `angle` | a wipe |
| radial | black at the centre, white farthest out | an iris |
| region order | one flat grey per region | the regions switch on one after another |
| noise | soft random greys, every grey equally often | a dissolve |

Outputs: `field` (0 to 1 inside the regions, 0 outside; also shown on the node), `mask` (1 on the regions; connect
it to `mask` of kubakub mask animate), `report`.

## kubakub mask animate

One mask per frame. The **effect** says what moves, the **curve** says how it runs over time. The curve is drawn on
the node: the buttons set its type, and dragging left / right on it makes it faster or slower. After a run the node
plays the masks.

| effect | what moves | value means |
|---|---|---|
| reveal, hide | the mask appears (or goes) along the field, from black to white | 0 = nothing yet, 1 = all of it |
| band | a stripe of `width` travels through the field | 0 = before the start, 1 = past the end |
| rings | `rings` stripes that keep running; loops without a jump | 0 to 1 = one step on |
| move x, move y | the mask itself | picture widths / heights (0.5 = half a picture to the right / down) |
| rotate | the mask itself | turns, clockwise (1 = once around) |
| scale | the mask itself | size (1 = as it is, 0 = gone) |
| opacity | the mask itself | 0 to 1 (with a square curve: a strobe) |

| curve | how the value runs |
|---|---|
| ramp | from `value_from` to `value_to` once, then it stays |
| saw | again and again |
| triangle | there and back |
| sine | a soft there and back |
| square | on / off; `duty` = the part of the cycle it is on |
| random | a new value every cycle |
| noise | wanders softly |
| sound level | follows the loudness (`listen`: everything, low, mid, high; `gain`, `release`) |
| beats, bars, low / mid / high hits | each one starts a ramp of `cycle` seconds |

| input | default | what it does |
|---|---|---|
| mask | | What is animated: a mask, or a mask batch (one per frame). For reveal / hide / band / rings it limits the result. |
| field | | For reveal / hide / band / rings. Empty = a left to right ramp over the mask. |
| cycle | 2 | Speed: seconds for one cycle (for beats and hits: seconds the ramp takes after each one). |
| value_from, value_to | 0, 1 | The bottom and the top of the curve, in the effect's unit. |
| easing | linear | ramp, saw, triangle, beats and hits: how the ramp starts and stops. |
| soft | 0.05 | reveal / hide / band / rings: how soft the moving edge is. |
| width, rings | 0.2, 3 | band / rings: how wide a stripe is, how many stripes. |
| trail | 0 | Seconds a place keeps glowing after the mask has moved on. |
| fps, seconds | 25, 0 | Length: 0 = the length of the sound, else of the mask batch, else 4 s. |
| scale | 1 | Size of the masks relative to the input. RAM: 250 masks at 3840x2160 are 8 GB. |
| audio | | For the sound curves; it is passed on, cut to the frames. A sound curve without a sound runs on a built-in beat. |
| advanced | | `phase`, `duty`, `seed`; `listen`, `gain`, `release` (sound level); `nth`, `steps`, `threshold`, `gap`, `bpm` (beats and hits); `start`; `pivot` (rotate / scale: the mask's centre or the picture's); `wrap` (move: out on one side, in on the other). |

Outputs: `masks`, `fps`, `audio` (silence without a sound), `frame_count`, `report`.

Things to try:

- **Two moves at once**: chain two nodes, `masks` of the first into `mask` of the second (a reveal that also swings on a sine).
- **One more region per beat**: field `region order`, effect `reveal`, curve `beats`, `steps` = the number of regions.
- **Strobe**: effect `opacity`, curve `square`, a short `cycle`, `duty` 0.1 for flashes.
- **Light running over the windows**: field `direction` with `all together`, effect `band`, curve `saw`, a `trail`.

## The same in the kubakub director

In the window the field effects are a behaviour: clip a layer to regions (`clip to`), then **behaviours → + behaviour
→ field: …** (wipe across, iris from the middle, outlines close in, region by region, dissolve, running band, rings,
band on beats, opens with the sound). The card has the same choices as the two nodes: along (the field), per, effect,
soft edge, trail, curve, cycle, from / to, easing, and for the sound curves every nth, steps, hit strength. The
timeline shows the curve over the behaviour's range. It is the same engine as the nodes, so the preview is what
renders (`tests/test_motion.py` runs both and compares them per pixel). On beats it also takes your markers.

Moving, rotating, scaling and flashing a mask needs no field in the window: use a shape or video layer as another
layer's mask and put wiggle, oscillate, pulse or sound level on it.

One difference: the node's `trail` fades frame by frame, the window's is built from 16 earlier moments, so a fast
band leaves slightly stepped ghosts there.

Limits: move, rotate and scale act on the whole mask, not on each region by itself. The curve on the node draws the
sound curves on an example beat (120 bpm); the real sound decides when the node runs.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_maskfields.py
```
