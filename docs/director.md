# kubakub director (2d / director)

The layer window of the Kuba nodes: run the workflow once, then **open director** on the node. Place, scale and
rotate images over the facade, drag layers to decide what is in front of what (below **base** = behind the facade,
seen through its **holes**), mask layers with other layers, paint, colour grade, import images, blend modes;
**?** shows every shortcut. **apply** saves the composition into the node and queues; the node renders it at full
size and returns the image, per-layer masks, a changed mask, regions with the placed layers and plan rules.
Example: `example_workflows/director_scene.json`.

## Timeline: animate layers and lights

Under the canvas: **play** (P), scrub by clicking / dragging, length and fps. Keyframes are per value: position, size,
rotation, opacity, visibility, colour grade values, and in light layers the HDRI strength / rotation, exposure,
projector brightness and every lamp's position, power, colour, size, cone and on / off.
- **● record** (Shift K): every change you make at the current time becomes a keyframe (the first one also keys the
  old value at 0 s, so it animates from there). A value that already has keyframes always keys when you change it.
- **◆ key** (K): keyframe the selected layer (or the selected lamp) as it is now.
- The selected layer's keyframes show as ◆ rows: drag them in time, Alt click = smooth / linear / hold, Del removes
  the selected one; , / . = frame back / forward, < / > = previous / next keyframe, Home = start.
- Light layers re-render their preview when you stop (not while playing).

**Sound**: ♪ audio (or drop an mp3 / wav / ogg onto the window) loads a sound into input/kuba_director. Its waveform
shows under the ruler and it plays with P in sync (the sound is the clock). The tempo and beat grid are found
automatically (bars every 4 beats on the ruler, the time readout shows bar.beat); correct it with the bpm field,
÷2 / ×2, or "beat here" on a known beat. "starts at" = seconds into the file where the timeline begins.
**Markers**: M (or + marker) at the playhead, drag them, double click to rename, Del removes. With **snap** on,
scrubbing and dragged keyframes / markers snap to beats, markers and keyframes (hold Ctrl to move freely);
< / > jump between keyframes and markers. The node's **audio** output is the sound trimmed to the timeline
(silence without one): frames + fps + audio into core Create Video.

**flow** (node setting, and the **hold / auto** pill next to apply in the window): with *hold until apply* (the
default) a director that has nothing applied yet only shows its preview; every node after it waits, so queueing a new
workflow never starts LTX / H3 / the sequence on an empty composition. Open the director, compose, **apply**: the
workflow runs on. *always* passes the plain base on right away (the old behaviour).

**apply** renders the frame you are on. For video, connect the director's **director** output to
**kubakub director sequence**: it renders every frame (outputs **frames**, **fps**, **audio**,
e.g. into core Create Video / Save Video) at its **sequence_scale** of the matrix size, and can write the delivery file
(**export**: PNG 8/16, ProRes 4444 / 422 HQ, H.264, H.265; **frames** off when you only export). It only runs when it is
in the graph (bypass it for still-only work), and changing its settings does not re-render the director's still.
The director's old render_sequence / export inputs still work for older workflows (advanced section); the window's
export button uses them.
Light layers render all their frames in one Blender session and frames with the same rig only once: on a 12 GB laptop GPU
~1.1 s per light frame at 800x540 (16 samples), 10 frames of the 3200x2160 test facade in 17 s.

## Animate layers with LTX

An image or paint layer's **animate (ltx)** section: on, a video prompt (what happens over time), from / to
(seconds; "here" takes the playhead) and motion. The director writes it into its plan rules (animate,
video_prompt, t_start, t_end, motion). Chain: director → director sequence → region plan (the director's regions +
rules) → **kubakub region video sampler** with the sequence's **frames** as **background** → Create Video with the
sequence's **fps** and **audio**. Each clip covers its time range, starts from the background frame at t_start and
keeps the background everywhere else (lights keep moving). Example: `example_workflows/director_video_ltx.json`.

For 12 GB cards the sampler samples stage 1 small (stage1_mp 0.2) and caps stage 2 (stage2_max_mp 0.6; the x2 upscaler
makes stage 2 four times the pixels, and 1.8 MP x 121 frames runs out of memory) and frees VRAM between the stages.
Measured on a 12 GB laptop GPU: 121 frames two stages 130 s (960x640 clip, peak VRAM 11.8 GB, 89 °C); a 4 s director
sequence with a moving lamp + an animated arcade 215 s.

Two more inputs of the region video sampler, for clips that run on their own (a still, no background video):

- **loop**: every clip comes back to the still and ends on its first frame, so the video repeats without a jump.
  The model holds the clip's last 8 frames at the still, and the node fades them onto the first frame (the LTX VAE
  drifts while it decodes them). The motion has to return inside the clip, so it calms down near the end.
- **audio**: a sound the clips are sampled with, held as it is. Put an audio-reactive LoRA on the model (Load
  LoRA, strength 1.2 to 1.5) and start each `video_prompt` with the LoRA's own words, e.g. `sound-driven video,
  audio-reactive motion, continuous visual flow`: the regions then move in hits near the note changes while the
  facade stays still. The frames carry no sound; connect the same audio to Create Video. A sound shorter than
  the clip is padded with silence.

## H3 clips: MiniMax H3 animates stretches of the timeline (with sound)

In the window: **h3 clips** (timeline bar) → **+ clip** (between the markers around the playhead, else the next 3 s).
Per clip: from / to, type, prompt (what happens), sound, raw, on. Project-wide: look, camera, sound, avoid. Clips
show as bars on the ruler (orange = keyframes, lavender = reference); click one to edit it.
- **keyframes** (fl2va model): the frames at the clip's start and end are H3's first and last frame; H3 animates
  the way between them. Joined into the sequence with a short crossfade (crossfade frames).
- **reference** (ref2va model): references instead of anchors - the sequence stretch itself (reference video with
  its sound: keeps the layout, e.g. light lines that follow the real cornices), the timeline music, director layers.
The node **kubakub keyframe clips (h3)** after **kubakub director sequence** renders them: frames / fps / audio from the
sequence, document from the director, H3 models (fl2va + LoRA; ref2va + LoRA optional), the qwen3vl 32b H3 encoder,
video + audio VAE. It builds H3's prompt format for you (look, anchors / tags in connection order, a timeline with
the named markers inside the clip as beats, locked-off camera, Audio line, avoid) - the output **prompts** shows
them; 'raw' sends your text as it is. H3's sound is mixed into the timeline sound (h3_audio).
**Prompt enhance** (optional, local): connect **enhance_clip** = a text encoder that can write, e.g.
`qwen3.5_9b_qwen_image_2.1_pe_i2i` (CLIPLoader type qwen_image); it looks at the clip's first frame and expands the
prompt. The Gemma 4 LTX int8 encoder cannot generate text (garbage) - unusable output is detected and the draft is
used. The region video sampler (LTX) has the same enhance_prompt / enhance_clip with core's LTX instructions.
Measured (12 GB laptop GPU, 1120x768): keyframe clip 2 s, 4 steps (flashgen LoRA) ~56 s; reference clip 2.5 s, 8 steps
(Ref2VA Acc-8Step LoRA) with the sequence as reference ~263 s.

## Diffuse from the window

Select an image or paint layer, write its **prompt**, set **denoise** and **re-diffuse / edges**, press **diffuse**
(or D). The window queues its own small graph (your workflow is untouched): Klein loader, text encoder, VAE, your
LoRAs, then **kubakub director diffuse** on the layer's area as the window shows it (plus context). Progress shows
under the button; the result comes back as a new layer "<name> · diffused" right above, cut to the diffused area
with a soft edge (compare with the eye, delete, undo). Paint layers are the sketch-and-prompt route: paint where
something should appear, prompt it, diffuse.

The **diffusion** pill holds the settings, saved with the composition: model (klein 4b default, klein 9b), three
LoRA slots with strength (the ones made for the chosen model are listed first, e.g. Kubakub_FluxKlein4B), steps,
size (megapixels the model works at), context, edge band, seed (-1 = new each time), reference (the crop as
reference latent). VAE: flux2_full_encoder_small_decoder when installed (fast decode), else flux2-vae.
Speed on a 12 GB laptop GPU (klein 4b + one LoRA, 1 MP, 4 steps): ~15 s the first time (loading), ~8 s after.

## Light layers: relight the scene inside the director

With a scene connected (kubakub scene render → director `scene`), **+ layer → light** (Shift L) adds a light
layer: the 3D model relit in Blender Cycles from the projection camera, no render engine to open.

- **environment**: Blender's built-in HDRIs (night, city, courtyard, sunset …) or an own .hdr / .exr, with
  strength, rotation and exposure.
- **lamps**: point, area, spot and sun. The dots on the facade are the lamps: drag them, or double click the facade
  to add a point light there. Position in metres: across (from the centre of the projector frame), height (above
  the ground), out (in front of the wall); power, colour, size / softness, spot cone, sun direction.
- **glow**: a layer's shape (e.g. a hidden paint layer or a mask input) or regions (`group:Windows`, `W_F1_*`)
  become emissive faces that light the stone around them.
- **projector**: the matrix as light. **casts** the layers below the light layer, the base, or any single layer
  (also a hidden one). **as light**: a spot light at the projection camera throws the image onto the model like
  the real projector (falloff, surface angle, bounce light; brightness 1 = the image at about its own brightness on
  a frontal wall). **as paint**: the image becomes the building's colour (your layers, texture masks) and the
  lamps and HDRI light it, with their shadows. Checked pixel-exact against the camera frame (lens shift included).
- **material**: clay brightness and roughness (in paint mode: the brightness of the painting); outside the building
  black, the HDRI or transparent; **tone** AgX (soft highlights) or Standard (projected colours as they are).

Every change renders a preview in the window (about 3 s at 800 px on a 12 GB laptop GPU, "fine" = 1600 px); **apply**
renders the layer at full matrix size with the "on apply" samples and resolution. The light layer is an ordinary
layer: blend it (multiply over the matrix = the content lit by the lamps, screen, soft light …), lower its opacity,
clip it to regions or mask it. It does not count as a placed object (no region, no changed mask). The standalone
**kubakub scene relight** node does the same from a text list of lamps.

## Video mapping in the director

- **Video layers**: + layer → import image or video (or drop it); files over 100 MB: *video from a path…* (read
  where it is, never copied). ProRes, HAP, DNxHD, H.264/5, VP9 and alpha are read by the ffmpeg ComfyUI ships. The
  window plays a small preview copy in real time, synced to the timeline; the render and export use the file.
  The clip row on the timeline: drag = move, left grip = trim in, right grip = trim out, **Alt** = speed,
  **Shift** = length (then loop / hold), **Ctrl K** = cut at the playhead, **Del** = remove a piece. The video
  panel: at / speed / in / out / length, then hold / loop, **backwards**, fit (canvas / clip regions / native),
  sound volume (0-200 %), frame blending. Videos clip to regions, mask other layers (a hidden video = a track
  matte), blend, fade and animate like images.
- **Projection mask** (top of the side panel): the building's silhouette at the delivery size, from the 3D scene or
  the template's mask file (opaque-black overlay, cutout or white on black are all read), invert, grow, feather;
  views black / dim / off. Applied to the still, the sequence, the export (as alpha) and the `projection_mask`
  output.
- **Shapes**: + layer → solid colour, rectangle, ellipse, polygon (click the corners, Enter). Colour and feather;
  hidden and used as another layer's mask = a shape mask.
- **Effects** (inspector): blur, sharpen, glow, grain, keyframeable, on a layer or on a colour-grade layer
  (then on everything below).
- **React to light**: *light → shaded by …* multiplies a layer by the lamps of a light layer (which may stay
  hidden); for physically lit projection use the light layer's projector in paint mode with the video as source.
- **Colour**: one working space, display-referred sRGB / Rec.709. Embedded ICC profiles (Adobe RGB, P3, CMYK …)
  are converted on import (ComfyUI's loaders ignore them); untagged HD video is read as Rec.709; 10/12-bit video
  keeps its precision; every exported video is tagged Rec.709.
- **Export** (director inputs `export`, `export_scale`, `export_alpha`, `export_name`): the whole timeline at the
  delivery size, streamed frame by frame (no RAM limit), into `output/kubakub_director/<name>_<time>/`: PNG sequence
  8/16 bit (+ .wav), ProRes 4444 (alpha) / 422 HQ, H.264 4:2:0 or 4:4:4 10 bit, H.265 10 bit, preview. With
  `export_alpha` the base is left out and the layers (x the projection mask) become the alpha. **kubakub export**
  writes any frames (e.g. after the H3 clip node) the same way.

## Behaviours: motion graphics in the director

Procedural motion on top of the keyframes, like Cavalry behaviours or After Effects' wiggle / loopOut. Select a
layer, **behaviours → + behaviour** in the inspector, pick a preset, tune it in its card. Several behaviours add up;
each has a time range (from / to), a fade and an on / off switch. The timeline shows a row per behaviour.

| behaviour | what it does | on |
|---|---|---|
| wiggle | smooth noise (amount, per second, seed) | any number: position, scale %, rotation, opacity, effects, grades, lamp power / colour position … |
| oscillate | sine / triangle / square / saw, or a circle orbit for position (amount, period, phase, angle) | the same |
| drift | constant speed, spin on rotation (held after the range) | the same |
| random steps | a new random value every n seconds | the same |
| loop keyframes | cycle / ping-pong / continue after the last key | a keyed value |
| pulse | a kick on beats, bars, markers or every n s (attack, decay, every nth) | the same numbers |
| sound level | the loudness of the timeline sound (release); `listen to`: everything, or only its low (kick, bass), mid or high (hats, clicks) frequencies | the same numbers |
| stagger | the regions a layer is clipped to light up one after another: sequence in / out, chase, wave, random; order left / right / top / bottom / centre / outside / size / random | a layer clipped to regions |
| swap | layers trade their masks in time with the sound (see below) | layers clipped to regions |
| repeat | copies in a line / grid / ring, rotation / scale / opacity per copy, a delay per copy replays the animation later (echo trails) | image and shape layers |

Presets: wiggle position / rotation, breathe, orbit, sway, spin, drift, flicker, pulse, pulse on beats, kick on
markers, flash on bars (glow), sound → scale / opacity, regions on in order, chase, wave across, random flicker,
repeat in a row / grid / ring, echo trail, swap masks on beats / low hits / high hits, sound → opacity (low).

### Swap masks: layers trade their masks with the sound

Several layers, each clipped to its own regions (`clip to`): a look for the top windows, another for the middle
floor, a third for the arches. **swap masks** makes them trade places: at every step each picture moves on to the
mask of the next layer. The pictures stay where they are on the facade; only the masks move, so nothing slides.

1. Give each layer its own `clip to`.
2. Select one of them, **behaviours → + behaviour → swap masks: on beats** (or on low hits / on high hits).
3. Tick the other layers in the card. The status line counts the steps it found.

| setting | what it does |
|---|---|
| order | loop (on to the next mask), there and back, random (a new shuffle every step, never the same twice) |
| step on | beats, bars, markers (M in the timeline), every … s, or low / mid / high hits of the timeline sound |
| every nth | only every nth beat, bar, marker or hit |
| hit strength, min gap | low / mid / high: how strong a hit has to be (0-1) and the shortest time between two steps |
| fade s | 0 = the picture just appears in the new mask; above 0 it fades |
| fade style | crossfade (the new picture fades in over the old one, the base never shows through) or out, then in |
| from s / to s | before `from` every layer has its own mask; after `to` the last arrangement holds |

Soft edges: **clip feather** under `clip to` in the inspector (pixels of the delivery size) softens the clip of any
layer, swapping or not. It belongs to the layer, so the picture takes its soft edge along into every mask it holds.

The layers trade in the order of the layer list, top down, whichever of them carries the behaviour. A layer can be
in one swap. A stagger on a layer works on the mask it holds at that moment. The saved document keeps every
layer's own `clip to`; the node works the swap out per frame, exactly like the window (same engine, 1300 cases in
tests/test_motion.py). The hits come from the node (kubakub/sound.py), so the preview is what renders.

A control wav or a list of times as the trigger is in the node below; in the director, put markers (M) where the
steps should be and use `step on = markers`.

For the same thing without the window, see **kubakub sound mask swap** below.

How it stays exact: the window and the node run the same engine (kubakub/director/motion.py and its twin in
web/kubakub_director.js, integer-hash noise, identical rounding); tests/test_motion.py runs ~500 cases through both.
The saved document keeps the values without the behaviours (a drag moves the rest position, keys never contain a
behaviour's offset); the node adds them per frame (render.animate), the still render at the playhead. Beats come from
the saved tempo, the loudness curve from the sound file (the window fetches the node's own curve), the stagger order
from the same region boxes. A drag of a moving layer moves its rest position.

Test in the UI: three shapes of different colours, each clipped to another floor of windows, a song (♪ audio),
"swap masks: on beats" on one of them, tick the other two, play. A shape + "orbit" or "breathe", play; a layer clipped to `*` + "stagger: regions on in order";
a song (♪ audio) + "pulse on beats (scale)"; "echo trail" on a keyed shape.

## kubakub sound mask swap (2d / motion)

One node that does the swap without the director: masks in, moving masks and finished frames out.

```
Load Audio ─ audio ┐
picture ─ background ┤
mask batch ─ masks ┼─ kubakub sound mask swap ─ frames / fps / audio ─ Create Video ─ Save Video
image batch ─ pictures ┘                      └ mask (the moving mask of one layer)
```

Nothing connected runs a sample: the sample facade, its windows as four layers (one per floor) and a built-in
beat. Example: `example_workflows/sound_mask_swap.json`.

| input | what it is |
|---|---|
| audio | the sound; it also sets the length |
| background | the picture behind the layers; it gives the size |
| masks | one mask per layer (a mask batch, at least two) |
| pictures | one whole picture per layer (an image batch; repeated when there are fewer). Empty = a colour per layer |
| order | loop, pingpong (there and back), random |
| step_on | beats, bars, every, low / mid / high (hits in those frequencies), signal, list |
| nth | every nth beat, bar or hit |
| fade, fade_style | 0 = the picture just appears; above 0 a crossfade, or out, then in |
| feather | soft mask edges in pixels of the background |
| threshold | how strong a hit has to be (0-1); for `signal` the level it has to rise through |
| fps, seconds, scale | frames per second, length (0 = the whole sound), size of the frames |
| mask_of | which layer's moving mask goes out on `mask` |
| signal | step_on = signal: a control track as a wav (gate, trigger, LFO, envelope, CV recorded from a synth). A step each time it rises through `threshold`; it has to fall below half of it before the next |
| step_times | step_on = list: times in seconds, e.g. `0.5, 1, 1.75, 3` |
| start, bpm, every, gap, seed (advanced) | where in the sound the frames start, a tempo you know (0 = found), the fixed time, the shortest time between two steps, the shuffle |

The first mask of the batch is the top layer. `mask` is what is seen of that layer, so during a crossfade it is
the part the layer above has not covered yet. `report` says what was found: tempo or hits, steps, frames.

RAM: 250 frames at 1600x1080 are about 7 GB; lower `scale` or `seconds` for long sounds (the node says so before
it starts). Test without ComfyUI: `tests/test_soundswap.py`.
