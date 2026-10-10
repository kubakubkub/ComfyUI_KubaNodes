# Shader (2d / shader)

Moving pictures and masks drawn by a small program that runs on the graphics card. Pick a preset, or paste the code
of a shader from shadertoy.com. You do not need to know how to write one. No AI model, and a 4K frame takes a
fraction of a second.

```
kubakub mask field ─ field ─┐
                └── mask ──┼─ kubakub shader ─ images / masks / fps / audio ─ Create Video, or any node
       Load Audio ─ audio ─┤
       Load Image ─ image0 ┘
```

Example: `example_workflows/shader.json` (runs with nothing connected: two bands of light close in on the windows
of the sample facade and flash on a built-in beat).

## kubakub shader

| input | default | what it does |
|---|---|---|
| preset | warped noise | A built-in shader, used while `code` is empty. **pick a preset by its picture** right under it shows all of them as small pictures: click one (it also empties `controls`, which belong to the old preset). The line **draws** under the button says what the node will draw, before you queue: the preset, or your pasted code and which pictures it reads, or why pasted code cannot run. |
| code | | A shader from shadertoy.com instead of the preset: see [Pasting a Shadertoy](#pasting-a-shadertoy). |
| controls | | Values for the shader's controls: `scale = 5, warp = 2` (commas or lines). Empty = the defaults. The report lists every control with its range. |
| width, height | 0 | 0 = the size of the first connected picture or mask, else 1280 x 720. |
| fps, seconds | 25, 0 | Length: 0 = the length of the sound, else of the longest picture or mask batch, else 4 s. |
| speed | 1 | How fast the shader's clock (`iTime`) runs. 0 = a still, negative = backwards. For pasted shaders. |
| loops | 1 | How many times a preset runs through within the clip. More = faster. |
| loop | on | The last frame meets the first. The presets loop by themselves; a pasted shader is rendered a little longer and its end fades into its start (`loop_fade`). |
| image0 - image3 | | Pictures for the shader (`iChannel0` - `iChannel3`). A batch plays as a clip, one picture per frame. |
| mask | | Where the shader shows. With `clip` the pictures and masks are black outside it. A batch plays one mask per frame, e.g. from kubakub mask animate. |
| field | | A grey ramp from kubakub mask field: the path a band runs along. Empty = left to right. |
| audio | | What the shader hears. Passed on to `audio`. A shader that listens without a sound runs on a built-in beat (120 bpm). |
| advanced | | `time_offset` (the clock at the first frame), `loop_fade` (seconds, 0 = no fade), `mask_from` (auto / alpha / brightness), `clip`, `bpm` (a tempo you know), `gain`, `release` (sound levels), `unconnected` (what a shader reads on a picture input with nothing connected: auto, noise, sample facade, black). |

Outputs: `images`, `masks` (how bright the picture is; the alpha of a plain fragment shader that writes one), `fps`, `audio`,
`frame_count`, `report`. After a run the node plays the frames.

## The presets

| preset | what it draws | works on | controls |
|---|---|---|---|
| mapping grid | a line-up grid for the projector: squares, the centre cross, circles and the diagonals; a scan line runs through |  | cells, line, scan |
| scan band | a band of light that runs along the field (connect kubakub mask field; without one: left to right) |  | width, soft, bands, beat, warm |
| warped noise | slow folding smoke: noise warped by noise. Loops without a jump |  | scale, warp, drift, contrast, sound |
| caustics | light on the floor of a pool: bright moving ridges. Loops without a jump |  | scale, sharp, speed, tint, sound |
| voronoi cells | cells with glowing borders that slowly trade places. Loops without a jump |  | cells, border, move, fill, sound |
| rings | rings that run out from a point (or in). Loops without a jump; a beat can flash them |  | rings, width, soft, cx, cy, inward, beat |
| chromatic split | the picture on image0 with its colours pulled apart; a wave runs through and the sound pushes it | a picture | amount, wave, lines, sound |
| glass | the picture on image0 seen through rippled glass. A normal pass on image1 is the glass; without one it is made of ripples | a picture | bend, ripples, split, shine, flow |
| clouds | soft clouds drifting across a sky. Loops without a jump |  | cover, scale, drift, dusk |
| fire | flames rising from the bottom edge. Loops without a jump; the low end of a sound makes them taller |  | height, scale, rise, sound |
| ripples | rings of water spreading from a few drops and crossing each other. Loops without a jump |  | drops, waves, speed, contrast |
| plasma | the classic plasma: soft colour waves folding through each other. Loops without a jump |  | scale, speed, warm, sound |
| tunnel | a flight through a striped tunnel. Loops without a jump; a beat can flash the rings |  | rings, segments, twist, fog, beat |
| starfield | stars flying towards you. Loops without a jump |  | stars, speed, streak, colour |
| aurora | curtains of northern light swaying across the top. Loops without a jump |  | height, folds, sway, green |
| blobs | soft blobs of light that melt into each other, like a lava lamp. Loops without a jump |  | blobs, size, edge, sound |
| kaleidoscope | a turning kaleidoscope of coloured shards. Loops without a jump |  | mirrors, detail, turn, warm |
| hexagons | a honeycomb whose cells light up one after another. Loops without a jump; a beat can flash them |  | cells, gap, lit, beat |
| brick wave | a wall of bricks; a wave of light runs through them along the field (or left to right). Loops without a jump |  | rows, ratio, mortar, width, waves |
| stripes | stripes that run across at an angle. Loops without a jump; a beat can flash them |  | stripes, angle, width, soft, speed, beat |
| spiral | a turning spiral, hypnotic. Loops without a jump |  | arms, tight, width, soft, turn |
| glitch bars | jumping digital bars and blocks, a broken signal. Loops; the highs of a sound add more |  | rows, amount, steps, colour, sound |
| rain | streaks of rain, or light falling down the wall. Loops without a jump |  | columns, length, speed, slant, warm |
| fireflies | small lights wandering in the dark, each on its own path. Loops without a jump |  | count, size, wander, twinkle, sound |
| radar | a sweeping beam like a radar or a lighthouse, with a fading tail. Loops without a jump |  | beams, tail, cx, cy, rings, turn |
| light rays | rays of light fanning out from a point above, slowly shifting. Loops without a jump |  | rays, cx, cy, reach, warm |
| truchet | a maze of quarter circles that rearranges itself, with light running along the paths. Loops without a jump |  | tiles, line, flow, warm |
| sound bars | bars that jump with the music: lows on the left, highs on the right. Needs a sound (a built-in beat without one) |  | bars, gap, mirror, glow |
| flow lines | fine lines flowing like wood grain or a topographic map. Loops without a jump |  | lines, scale, thin, drift, warm |
| edge glow | the outlines of the picture on image0 drawn in glowing light; a pulse runs through them | a picture | thick, gain, keep, pulse, sound |
| mosaic | the picture on image0 as tiles that change size with the loop or the sound | a picture | tiles, breathe, gap, round, sound |
| contour lines | the picture on image0 as height lines, like a map of its light and dark; the lines climb. Loops without a jump | a picture | lines, thin, speed, smooth, keep |
| region contours | lines of light that follow the outline of every region inwards (connect kubakub mask field, edge distance). Loops without a jump |  | rings, thin, speed, outline, beat |
| wireframe | the picture on image0 as a glowing wire mesh, its lines bent by depth (a depth pass on image1, else by brightness). A wave of light runs through | a picture | rows, columns, relief, thin, wave, keep |
| blueprint | the picture on image0 as an architect's blueprint: white outlines on blue, a drawing grid, and a pen of light that redraws it | a picture | thick, gain, grid, pen, paper |
| depth scan | a sheet of light that travels through the depth of the picture on image0 (a depth pass on image1, else its brightness). Loops without a jump | a picture | sheets, width, passes, keep, warm |
| slit scan | slit scan: a slit travels across and paints what the clip on image0 shows at that moment, so moving things bend and stretch. Needs a clip (a moving pattern without one) | its last frame | passes, upright, show |
| datamosh | datamosh: the clip on image0 breaks into blocks that smear along its own shapes until a clean frame snaps back. Needs a clip (a moving pattern without one) | its last frame | blocks, smear, keys, heal, bleed |
| trails | everything that moves in the clip on image0 leaves a fading trail that drifts, turns or zooms away. Needs a clip (a moving pattern without one) | its last frame | fade, zoom, turn, drift_y, tint |
| grid flight | an endless flight over a glowing grid towards a striped sun. Loops without a jump; a beat can flash the lines |  | horizon, lanes, speed, sun, beat |
| endless zoom | frames within frames within frames: an endless zoom into the wall. Loops without a jump |  | depth, gap, border, round, turn |
| corridor | an endless walk down a corridor of lit panels. Loops without a jump; the low end of a sound brightens the lights |  | panels, width, lights, fog, sound |
| point cloud | the picture on image0 as a cloud of points that sways in depth (a depth pass on image1, else by brightness), like a 3D scan | a picture | points, size, sway, depth_size, glow |
| halftone | the picture on image0 printed in dots, big where it is bright; the raster slowly turns. Loops without a jump | a picture | dots, angle, turn, colour, soft |
| eight bit | the picture on image0 as on an old game console: big pixels, a handful of colours, dither patterns | a picture | pixels, palette, dither, shimmer, contrast |
| characters | the picture on image0 typed in characters, dense signs where it is bright, like text-mode art; the signs flicker | a picture | columns, colour, flicker, contrast |
| old screen | the picture on image0 on an old tube television: a curved glass, scan lines, coloured dots and a bar that rolls through | a picture | lines, curve, dots, roll, split, noise |
| pixel smear | the bright parts of the picture on image0 melt downwards in long streaks, like sorted pixels; the amount breathes with the loop | a picture | length, level, breathe, sideways, sound |
| vhs tape | the picture on image0 as a worn VHS tape: soft, colours bleeding sideways, a wobbling edge, snow, a noisy band and the torn lines of the head switch at the bottom | a picture | wear, bleed, wave, snow, tracking, head, ring, grain, lines |
| composite video | the picture on image0 sent through a composite video cable: colour fringes and rainbows crawling along fine detail, dots on colour edges, soft colour | a picture | carrier, fringe, dots, crawl, colour, lines |
| scan modulation | the picture on image0 as scan lines that the brightness pushes out of place, like a signal on a tube; the lines drift | a picture | rate, depth, steps, drift, soften, tint, light |
| cartoon | the picture on image0 as a cartoon: a few flat tones, black ink outlines and hatching in the dark parts that drifts. High contrast, made for architecture | a picture | tones, contrast, ink, thick, detail, hatch, lines, drift, two_colours |
| hard shadow | a light circles the picture on image0 and everything that stands out throws a hard shadow (a depth pass on image1, else its brightness). A fake shadow, no 3D needed. Loops without a jump | a picture | sun, reach, turns, angle, dark, soft, shade, tint |

All of them loop without a jump and stay inside the mask. `sound` and `beat` are 0 in most presets: raise them in
`controls` to make the picture follow the music (sound bars follows it by itself).

**Works on a picture**: connect yours to `image0` (a still or a clip); with none connected the sample facade is
used. wireframe, depth scan and point cloud read a depth pass on `image1` when there is one (kubakub render
passes, scene render) and take the brightness of the picture as depth when there is none; glass reads a normal
pass there.

**Works on its last frame** (slit scan, datamosh, trails): these build every frame on the one before, which is
what makes a slit scan bend moving things and a datamosh smear. They want a **clip** on `image0` (Load Video, or
the frames of another kubakub shader); with nothing connected they run on a moving pattern of their own so you
see what they do. Their loop is closed by a fade (`loop_fade`). They exist on the node only, not as a fill in
the director.

For projection mapping: mapping grid lines the projector up; region contours, scan band and brick wave follow
your regions (connect kubakub mask field); contour lines, wireframe, blueprint, edge glow and depth scan trace
the building's own shapes from a render of it.

The presets are plain text files in `kubakub/shaders/`, written for this pack. Copy one into `code` to change it.

## Mixing two shaders

**kubakub shader mix** takes two shaders (presets, or pasted code under *advanced*) and one **mix** value: 0 is
shader A, 1 is shader B. In between they meet in the shaders, not as two finished pictures laid over each other.
`how` says in which way:

| how | what happens between 0 and 1 |
|---|---|
| b works on a | B takes A's picture as its own picture, so B is an effect on A: plasma seen through eight bit, clouds on a VHS tape. Pick a B that works on a picture |
| a bends b | A's picture pushes the places B draws at (red sideways, green up and down), strongest half way: B melts out of A. `strength` sets how far |
| a cuts b | B shows first where A is bright and takes over from there: a wipe in the shape of A's pattern. `soft` is its edge |
| blend | a plain fade |

`curve` runs the mix over time (a ramp from A to B, a wave, the sound level, a ramp on every beat), `cycle` is its
length and `mix_from` its low end. A picture or a clip on `image` is what both shaders work on; without one, B
takes A's picture. `controls_a` and `controls_b` set each shader's own controls. Shaders that build on their own
last frame (slit scan, datamosh, trails) cannot be mixed here: render them with kubakub shader and connect the
result as `image`. Example: `example_workflows/shader_mix.json`. For your own code: a shader that is bent reads the
pushed place in `kUV` and `fragCoord`; one that uses `gl_FragCoord` directly is not bent.

## Drawing the direction

Setting a direction or a point in numbers is guesswork, so the node takes an **arrow**: drag on the picture under **draw the direction** (after one run the picture is what the node drew), or type it into `vector` as x, y of the start and x, y of the end, each 0 to 1 of the picture. What a preset does with it:

- presets that run along the field (scan band, region contours and others that read `kField`) run from the arrow's start to its end, when no field is connected
- **stripes**, **rain** and **pixel smear** take its direction
- **rings**, **ripples** and **radar** start at its start

The report says whether the shader used it. Your own code reads `kFrom`, `kTo`, `kDir` and `kVector`. **kubakub mask field** takes the same arrow for its direction and radial fields, and the kubakub director has it on shader fills and on field behaviours.

## Pasting a Shadertoy

You do not need to read or understand the code.

1. Open the shader on shadertoy.com.
2. Click into the code on the right, press **Ctrl+A** (everything), then **Ctrl+C**.
3. Click into `code` on the node, press **Ctrl+V**, queue.

If the shader has a **Common** tab next to **Image**, paste Common first and Image below it. Pasting the link does
not work (the node downloads nothing, and says so).

The report then tells you in plain words what the shader uses. The most common case: a Shadertoy expects a noise
picture on one of its channels. When nothing is connected to `image0` - `image3` the node puts its own **built-in
noise** there, laid out the way those shaders expect, so most of them simply run. If a shader looks wrong, it may
expect a photo or another picture: connect one to the input the report names.

What cannot work, and the node says so instead of showing a graphics error:

- **Buffer A - D tabs** (each one is a shader of its own that feeds the next)
- **cubemaps** and 3D textures
- **sound shaders**, the keyboard and the webcam
- the mouse: `iMouse` reads as never pressed

A Shadertoy's alpha is ignored, as on shadertoy.com (many write 0 there and would be invisible otherwise): its mask
is how bright the picture is.

`loop` renders a pasted shader a little longer and fades its end into its start, so it loops even when it was not
written to. Pictures you connect are clamped at their edges, not repeated; the built-in noise repeats.

A mistake in the code is reported with the line number of what you pasted: `line 12: 'foo' : undeclared identifier`.

## Writing your own

Two forms work. The Shadertoy form:

```glsl
uniform float speed;   // 1 0 10   the comment is: default min max, then anything
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float v = 0.5 + 0.5 * sin(uv.x * 20.0 + 6.2831853 * kLoop * speed);
    fragColor = vec4(vec3(v) * kMask(uv), 1.0);
}
```

Or a complete fragment shader with its own `main()`, as for ComfyUI's GLSL Shader node (`u_resolution`,
`u_image0` - `u_image3`, `v_texCoord`, `fragColor0`). This form can write an alpha: it becomes the mask, and in the
director the layer is see-through there. The language is GLSL ES 3.00, the same as WebGL 2.

| name | what it is |
|---|---|
| `iResolution`, `iTime`, `iTimeDelta`, `iFrame`, `iFrameRate` | as on Shadertoy |
| `iChannel0` - `iChannel3`, `iChannelResolution[n]` | the pictures on `image0` - `image3` and their sizes |
| `kLoop` | 0 to 1 through the clip (`loops` times). Build on it and the clip loops: `sin(6.2831853 * kLoop)` |
| `kSeconds` | the length of the clip |
| `kLevel`, `kLow`, `kMid`, `kHigh` | the sound right now, 0 to 1: all of it, kick and bass, the middle, hats and clicks |
| `kBeat` | 1 on a beat, falling to 0 before the next |
| `kBeats` | how many beats have passed |
| `kUV` | this pixel, 0 to 1 across the picture (0, 0 is bottom left, as on Shadertoy) |
| `kPrev(uv)` | the frame before this one (black at the first frame): trails, slit scan, datamosh, anything that builds up. Node only |
| `kField(uv)`, `kMask(uv)` | the grey ramp and the mask connected to the node |
| `kFrom`, `kTo`, `kDir`, `kVector` | the drawn arrow: its start and its end (0 to 1, as `kUV`), the way it points (length 1), and 1 when one is drawn. Without a connected field `kField` runs from 0 at its start to 1 at its end |
| `uniform float name; // default min max` | a control of your own, set in `controls` |

Things to try:

- **A band per beat along the windows**: preset scan band, `controls` = `beat = 1`, a field from kubakub mask field
  (`edge distance`), your music on `audio`.
- **Glass over a render**: kubakub render passes, `beauty` into `image0`, `normal` into `image1`, preset glass.
- **A fake shadow on the building**: `beauty` into `image0`, `depth` into `image1`, preset *hard shadow*. A light
  circles once per loop and cornices, window frames and anything else that stands out throw shadows across the
  wall; `sun` low gives long shadows, `turns` 0 holds the light at `angle`. Without a depth pass the brightness
  of the picture is taken as height, which reads as an embossed print rather than a real shadow. A depth pass
  saved in 8 bit shows its steps as fine stripes on slanted faces: raise `soft` or `sun`.
- **The building as a wire mesh or a point cloud**: `beauty` into `image0`, `depth` into `image1`, preset
  wireframe, point cloud or depth scan.
- **Slit scan or datamosh of a clip**: Load Video into `image0`, preset slit scan or datamosh, `seconds` = the
  length of the clip. Chain two shaders: the frames of clouds or tunnel into `image0` of datamosh.
- **An 8 bit facade**: a picture into `image0`, preset eight bit, `controls` = `palette = 1, pixels = 96`.
- **A texture for the region sampler or the director**: `images` is a normal picture batch; `masks` cuts another
  layer to the shader's shapes.

## The same in the kubakub director

In the window a shader is the **fill** of a shape layer: + layer → solid colour (or rectangle, ellipse, polygon),
then inspector → shape → **fill** → a preset or **own code**. It plays live while you scrub or press play, and the
node renders the same picture (measured on two shader layers at 1920x1088: the window and the render differ by
0.1 % on average). Clip the layer to regions and the shader runs only in the windows; feather, masks, blend modes,
effects and behaviours work as on any shape. The sliders are the shader's controls and take keyframes and
behaviours. `loops` is counted over the timeline's length.

The **pictures** button next to `fill` shows the presets as small pictures (slit scan, datamosh and trails are
left out: they build on their last frame, which only the node can do).

**An arrow instead of numbers.** A shader that can follow a direction has an **arrow** row: press **draw** and drag on the layer. Presets that run along the field (scan band, region contours ...) then run from the arrow's start to its end; stripes, rain and pixel smear take its direction; rings, ripples and radar start at its start. The arrow belongs to the layer and moves with it. **clear** removes it.

**A shader as an effect.** A preset that works on a picture (eight bit, characters, mosaic, halftone, old
screen, pixel smear, vhs tape, composite video, scan modulation, cartoon, hard shadow, contour lines, wireframe, blueprint, edge glow, depth scan, point cloud, chromatic split,
glass) works here on **everything under the layer**: put a solid with fill = eight bit on top and the whole
composition below turns 8 bit; make it an ellipse with a feather, or clip it to the windows, and only that part
does. Stack two for two effects. Your own code has a `picture` choice for this: *the layers below*, or the
*built-in noise* that most pasted Shadertoys expect (the default for own code). Without a depth pass, wireframe,
depth scan and point cloud take the brightness of the picture as depth. A pasted Shadertoy works here too
(own code): the built-in noise sits on its channels, in the window and in the render alike.

Not in the window: pictures on `image1` - `image3` (a depth or normal pass) and a field (left to right there).
Use the node for those and bring its frames in as a video layer. A broken shader shows its mistake under the
code, with the line, and the layer shows its colour until it is fixed. Every control has a hover text that says
what it does.

## Speed

The shader compiles once per run and pictures that do not change are uploaded once. On an RTX 5070 Ti a simple
preset takes about 14 ms per frame at 1920x1080 and 57 ms at 3840x2160; warped noise 32 ms and 112 ms. Most of
what is left is copying the frames into memory, so RAM is the limit: 250 frames at 3840x2160 are 33 GB (the node
refuses a clip that does not fit and says how much it needs).

Limits: one pass only. The mask and field are shared by the whole picture, there is no value per region. On the
node the controls are fixed for a run and what moves is the loop, the clock and the sound; in the director the
controls can be keyframed.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_shadertoy.py
```
