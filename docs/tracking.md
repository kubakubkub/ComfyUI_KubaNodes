# kubakub tracker and kubakub track draw (2d / motion)

Find what moves in a video, follow it with ids, and draw it as a diagram: the look of TouchDesigner blob tracking,
with more to find, depth, and more ways to draw. Examples:
[`tracker_diagram.json`](../example_workflows/tracker_diagram.json) (no models: the tracker runs a built-in sample
clip when nothing is connected) and [`tracker_depth_layers.json`](../example_workflows/tracker_depth_layers.json)
(your video and Depth Anything 3).

## kubakub tracker

Put a *Load Video* (or any batch of frames) on `video`, pick the finders on the pills, Queue. The node then shows
the video with everything it found; scrub or play it.

| finder | finds | needs (on your disk, never downloaded) |
|---|---|---|
| motion | what moves against the still background | nothing |
| colour | things of one colour: **pick on the video** with the eyedropper, *how close* = tolerance | nothing |
| lights | bright spots, *brighter than* = the level | nothing |
| people & things | people, cars, bikes, birds, umbrellas … (YOLO; pick the kinds on the pills, *sure* = confidence). A **pose** file (`yolo26x-pose`) adds 17 body points to every person in one pass, a **seg** file (`yolo26x-seg`) the exact outline: both fast, also in crowds | a YOLO file in `models/ultralytics` |
| words | anything you name: "umbrella, pigeon, head" (Grounding DINO SwinT or SwinB) | `models/grounding-dino` and bert-base-uncased in the Hugging Face cache |
| body | 33 points of each person: head, shoulders, hands, hips, feet (MediaPipe; inside every YOLO person, so a crowd works) | MediaPipe (ships its models) |
| hands | 21 points per hand | MediaPipe |
| face | the face as a mesh outline and its key points | MediaPipe |
| motif | anything you **click on the video** (Shift click = not this; objects 1-4), followed through the whole clip with its exact outline | SAM 2.1 in `models/sam2` (tiny is fast) |
| outlines | the exact outline inside the boxes of the kinds you choose | SAM 2.1 |

Several at once is the point: each thing keeps the kind that found it, so track draw can treat people, lights and
a motif differently. Below the finders: *look at* (the size the finders see the video at: 640 · 960 · 1280 ·
full), *smooth* (the paths are smoothed both ways, so there is no lag) and *keep lost* (frames a thing may vanish
and keep its id).

**try this frame**: after one run, scrub to a frame, change a setting and press it: only that frame is looked at
again (about a second; dashed on the node). Queue to run the whole clip.

**depth**: connect any depth node's frames to `depth` (Depth Anything 3 per frame, DA3 Streaming, Depth Anything 2,
Marigold, Video Depth Anything …). Every tracked thing gets its depth per frame (z: 0 far … 1 near), and the depth
goes on, one scale for the whole clip, to track draw. *near is* bright / dark on the node: bright for Depth Anything,
dark for Marigold. Which model: Depth Anything 3 per frame is fast (about 7 frames a second at 640 x 360 on a
12 GB card) and may flicker a little (the tracker smooths each thing's z); DA3 Streaming is steady over the whole
clip (about 3 frames a second); Marigold is the finest and the slowest (stills rather than long clips).

**Tracked once**: the tracks of every run are kept on disk (`kuba_tracks` in the pack's cache folder, setting
`cache_folder`, default `user/kubakub_cache`; together with the other caches at most `cache_gb`). The same video
with the same finder settings comes back from there at once, also after a restart of ComfyUI (the report says so);
change a finder and it tracks again. Changing only the look of track draw never tracks again.

**Chaining trackers**: put one tracker's `tracks` into the next tracker's `tracks` (same video): the second one adds
its finds to the first one's. So you can track two colours, motion at two sensitivities, or people with YOLO and a
motif with SAM 2 with their own settings each, and still get one set of tracks: ids go on, a kind found twice gets a
number (colour, colour 2), and track draw can treat them apart and join them with lines. A tracker on another video
is refused with the reason.

**sound**: connect the video's sound or music to `audio` (Load Audio, a video's audio, or kubakub sound on `sound`).
The tracks then carry how loud it is and where the beats are, frame by frame, and track draw reacts to them (its
*sound* part, below). The sound plays with the video on the node (and on track draw's node), and goes on, cut to
the clip, to `audio` for Create Video.

Outputs: `tracks` (for track draw, the shader and the director), `found` (a mask per frame: white on what was found,
its outline when there is one), `near` (a soft glow around everything found: the shader's `field`, a mask for mask
animate, the region sampler's `change`), `frames` and `fps` (the video, for track draw and Create Video), `report`,
`depth` (near = white), `audio` (the sound cut to the clip; silence without one).

Limits: body points need people at least a few hundred pixels tall in real footage (MediaPipe does not read drawn
figures or tiny walkers; YOLO pose does better there). The tracker tells ultralytics to stay offline (no usage
data, no automatic installs) for its own use.

## kubakub track draw

Plays the look **live on the node** as soon as the tracker has run, before anything is queued. Start from a preset,
then **edit the look** (a big preview, Space plays, `,` / `.` a frame, Esc closes):

| part | choices |
|---|---|
| presets | **diagnostic** (thin boxes, numbers, state words, lines fanning to the corners of the biggest) · **constellation** (gold circles on black, ids, a network, the video only inside the boxes) · **stamp trail** (each thing leaves its own pixels along its path) |
| under it | video · black · transparent (the alpha output) |
| marks (any number, stacked) | box · brackets · outline · circle · cross · dot · label · skeleton · points; colour (palette, own colour of each thing, any), line, scale, fill, opacity, *near = bigger* (with depth), on which kinds |
| labels | your text with live values: `{id}` `{id_hex}` `{id_long}` `{state}` `{label}` `{kind}` `{x}` `{y}` `{z}` `{speed}` `{size}` `{data}` `{frame}` `{t}` (click them in); where: top left / top right / below / bottom left / centre |
| lines (any number) | nearest (how many) · within a distance · network · tree · hull · hub (biggest, first, centre, a corner) · chain (in the order they appeared) · trail (length) · corners (to the corners of the biggest box); straight / curved / right angle, dashed, fade with distance, same kind only, same depth only, colour, width, opacity |
| inside | reveal the video · pixels (block size) · freeze (the first look of each thing travels with it) · invert; in the box or the exact outline |
| depth (with the tracker's depth) | **layers** (the picture in depth bands, coloured from a near to a far colour) · **lines** (depth contours, like a topographic map) · **far = earlier** (far layers show older frames: depth becomes time) · **fog** (far things fade into a colour) · **slice** (only a depth range stays) |
| stamp trail | every N pixels walked a patch is left behind: its size (the thing's, or fixed), the pixels of now or of the first moment, how long stamps stay (0 = for good), a thin frame |
| time | every mark, every set of lines, every inside and depth part and the stamp trail has its own **time**: a slider with two dots, where it starts and where it ends in the clip (the preview jumps there while you drag; the playhead shows on it), and a fade in and out in seconds. The stamp trail only leaves stamps inside its time |
| blend | every mark, every set of lines, every inside and depth part and the stamp trail has its own blend mode: normal, multiply, screen, overlay, darken, lighten, color dodge, color burn, hard light, soft light, difference, exclusion, hue, saturation, color, luminosity (the director's modes) |
| state words | what a thing is called while new / moving / still / lost (default INPUT · PROCESS · STABLE · ARCHIVED), and when it counts as moving, new or lost |
| sound (with a sound on the tracker) | *beat → marks pulse* (the marks grow with the loudness and jump on the beat) · *beat → lines flash* (the lines are dimmer between beats and light up on them) · *loud → more lines* (lines *within a distance* reach further when it is loud) · *loud → thicker* |

**render the look** (on the node and in the editor) renders it at the video's size together with whatever saves it
after track draw (Create Video, Save ...), and nothing else of the workflow runs; the tracking comes from the cache.
Queueing the whole workflow does the same for track draw (`images`, `alpha`). The preview and the render decide the same
things (states, labels, lines, stamps, depth per thing; checked by the test); letters are drawn with a slightly
different font, and the depth layers of the preview are worked out on a coarser grid.

## Tracks in other nodes

- **kubakub shader** `tracks`: the shader follows what was tracked. Its arrow is the biggest thing's way, frame by
  frame, and when no field is connected the field is a glow around everything tracked, so field presets (scan band,
  rings, contours …) run from the tracked things. The clip is as long as the tracks.
- **kubakub director** `tracks`: the behaviour **follow a tracked thing** moves a layer with it (the biggest, or one
  you pick by its id) from the behaviour's start time on, scaled to the canvas; **grow with a tracked thing** scales
  it with the thing's size. *amount* 1 = with it, −1 = mirrored. See [director](director.md).
- The `found` and `near` masks plug into anything that takes a mask: mask animate, the shader's mask or field, the
  region sampler's `change`.

## Tests

    python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_tracking.py

The finders on the sample clip, ids over a missed frame, all presets, depth and blend modes, both nodes, the
preview against the render (with node installed), the shader and the director's follow, and YOLO (pose, seg),
MediaPipe, Grounding DINO and SAM 2 when their files are there, with the network blocked.
