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
| preset | warped noise | A built-in shader, used while `code` is empty; the list is sorted by type. **browse the presets** right under it shows all of them as small pictures under their type (patterns, 3d scenes, on your regions, picture looks, screen and tape, picture with depth, over time): rest on one to see it move, type a word to find one, click one to use it (it also empties `controls`, which belong to the old preset). The line **draws** under the button says what the node will draw, before you queue: the preset, or your pasted code and which pictures it reads, or why pasted code cannot run. |
| code | | A shader from shadertoy.com instead of the preset: see [Pasting a Shadertoy](#pasting-a-shadertoy). |
| controls | | Values for the shader's controls: `scale = 5, warp = 2` (commas or lines). Empty = the defaults. The report lists every control with its range. |
| width, height | 0 | 0 = the size of the first connected picture or mask, else 1280 x 720. |
| fps, seconds | 25, 0 | Length: 0 = the length of the sound, else of the longest picture or mask batch, else 4 s. |
| speed | 1 | How fast the shader's clock (`iTime`) runs. 0 = a still, negative = backwards. For pasted shaders. |
| loops | 1 | How many times a preset runs through within the clip. More = faster. |
| loop | on | The last frame meets the first. The presets loop by themselves; a pasted shader is rendered a little longer and its end fades into its start (`loop_fade`). |
| image0 - image3 | | Pictures for the shader (`iChannel0` - `iChannel3`). A batch plays as a clip, one picture per frame. `image0` is your picture: the presets that work on a picture redraw it, every other shader is laid over it (`on_picture`). |
| on_picture | screen | A picture on `image0` and a shader that does not read it (the patterns, the 3d scenes, most pasted code): how the shader is laid over the picture. See [Whatever you connect does something](#whatever-you-connect-does-something). |
| mask | | Where the shader shows. With `clip` the pictures and masks are black outside it. A batch plays one mask per frame, e.g. from kubakub mask animate. |
| field | | A grey ramp from kubakub mask field: the path a band runs along. Empty = left to right. |
| audio | | What the shader hears. Passed on to `audio`. Connecting a sound makes a preset follow it (its `sound` or `beat` control goes to 1 unless you type a value). A shader that listens without a sound runs on a built-in beat (120 bpm). |
| advanced | | `time_offset` (the clock at the first frame), `loop_fade` (seconds, 0 = no fade), `mask_from` (auto / alpha / brightness), `clip`, `bpm` (a tempo you know), `gain`, `release` (sound levels), `picture_amount` (how much of the shader is laid over the picture, 0 = the picture alone), `unconnected` (what a shader reads on a picture input with nothing connected: auto, noise, sample facade, black). |

Outputs: `images`, `masks` (how bright the picture is; the alpha of a plain fragment shader that writes one), `fps`, `audio`,
`frame_count`, `report`, `depth` (how near each pixel is, near = white, from a shader that is a 3D scene: see
[3D scenes](#3d-scenes-raymarching); one black picture from any other shader). After a run the node plays the frames.

## Adjusting a preset, and saving your own

**adjust** (the button under the pictures) opens the shader's controls as sliders next to a picture of it, so you
do not have to read code or type into `controls`:

- A colour of the shader is a swatch: click it to pick another, or type `#rrggbb` next to it.
- Every other control is a slider with its name, its range and a line on what it does. Moving one redraws the picture
  at once (one frame while you drag, the moving loop when you stop) and writes the node's `controls` field, which
  stays the truth: a workflow saved this way opens anywhere. A name turns orange when it is off its default;
  double-click the name to reset it, **reset** for all.
- The picture is drawn by the node with nothing connected (the sample facade for presets that work on a picture),
  so it is what a queue gives.
- **show the code** puts the preset's code into `code` and opens it in **the big code window**, where you can
  change anything: colours, shapes, counts. The sliders keep working on your changed code. **back to the preset**
  empties it again.

**The big code window.** The `code` field on the node is only a few lines high. **open the code in a big window**
(the button under it) shows the code large, with line numbers; drag its corner to resize it, its head to move it.
What you type goes into the node's `code` field. On the right the node draws what the code gives as you type, or
says what is wrong and on which line, and lists the names a shader can use. **start from the preset** loads the
picked preset's code, **adjust** opens the sliders for it, Tab indents, Esc or **done** closes.
- **save as my preset**: type a name and the shader, with the sliders as they stand as its new defaults, is saved
  as a preset of your own. It appears first in the `preset` list and in the picker under **my presets**, with its
  own moving picture. Saving under the same name again replaces it; **delete** removes it. A built-in preset
  cannot be overwritten or deleted.

Your presets are plain `.glsl` files in ComfyUI's user folder, under `kubakub/shaders`, so they survive an update
of the pack and can be copied to another machine. A workflow that uses one needs that file there.

## Whatever you connect does something

Every preset works with every input, so you can connect first and read later:

| you connect | what happens |
|---|---|
| a picture on `image0` | The types *picture looks*, *screen and tape* and *picture with depth* redraw it. Every other shader (patterns, 3d scenes, on your regions, pasted code that reads no picture) is laid over it, the way `on_picture` says: **screen** and **add** put its light on top, **multiply** makes the shader the light that falls on the picture (dark where the shader is dark: a pattern as a projection on your facade), **overlay** gives more contrast, **difference** inverts where it is bright, **displace** lets its light and dark bend the picture like glass, **off** gives the shader alone (the picture then only sets the size). `masks` and `depth` stay those of the shader, so you can still cut or relight with them. |
| a `mask` | Every shader shows inside it only. With a picture under the shader, the picture stays as it is outside the mask. |
| a sound on `audio` | A preset follows it: its `sound` or `beat` control, which rests at 0, goes to 1. Type `sound = 0.3` in `controls` for less, `sound = 0` for none. |
| a `field` | The presets of *on your regions* run along it. Others have no use for it, and the report says so. |
| pictures on `image1` - `image3` | A depth or normal pass for the presets that read one (picture with depth, glass). The report names a picture the shader does not read. |

The report always says what was done with each input.

## The presets

The presets are sorted by type, in the `preset` list and in the picker (**browse the presets** right under it),
which shows every preset as a small picture under these headings, plays one when the pointer rests on it, and
finds by a word (`fire`, `vhs`, `depth`).

### patterns

Moving pictures made from nothing: smoke, cells, stripes, light. They need no input.

| preset | what it draws | works on | controls |
|---|---|---|---|
| warped noise | slow folding smoke: noise warped by noise. Loops without a jump |  | scale, warp, drift, contrast, sound |
| caustics | light on the floor of a pool: bright moving ridges. Loops without a jump |  | scale, sharp, speed, tint, sound |
| voronoi cells | cells with glowing borders that slowly trade places. Loops without a jump |  | cells, border, move, fill, sound |
| rings | rings that run out from a point (or in). Loops without a jump; a beat can flash them |  | rings, width, soft, cx, cy, inward, beat |
| clouds | soft clouds drifting across a sky. Loops without a jump |  | cover, scale, drift, dusk |
| fire | flames rising from the bottom edge. Loops without a jump; the low end of a sound makes them taller |  | height, scale, rise, sound |
| ripples | rings of water spreading from a few drops and crossing each other. Loops without a jump |  | drops, waves, speed, contrast |
| plasma | the classic plasma: soft colour waves folding through each other. Loops without a jump |  | scale, speed, warm, sound |
| aurora | curtains of northern light swaying across the top. Loops without a jump |  | height, folds, sway, green |
| blobs | soft blobs of light that melt into each other, like a lava lamp. Loops without a jump |  | blobs, size, edge, sound |
| kaleidoscope | a turning kaleidoscope of coloured shards. Loops without a jump |  | mirrors, detail, turn, warm |
| hexagons | a honeycomb whose cells light up one after another. Loops without a jump; a beat can flash them |  | cells, gap, lit, beat |
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
| endless zoom | frames within frames within frames: an endless zoom into the wall. Loops without a jump |  | depth, gap, border, round, turn |

### 3d scenes

A space you look into: forms, columns, corridors, fog. They need no input; they cost more to draw, and some give a depth.

| preset | what it draws | works on | controls |
|---|---|---|---|
| tunnel | a flight through a striped tunnel. Loops without a jump; a beat can flash the rings |  | rings, segments, twist, fog, beat |
| starfield | stars flying towards you. Loops without a jump |  | stars, speed, streak, colour |
| grid flight | an endless flight over a glowing grid towards a striped sun. Loops without a jump; a beat can flash the lines |  | horizon, lanes, speed, sun, beat |
| corridor | an endless walk down a corridor of lit panels. Loops without a jump; the low end of a sound brightens the lights |  | panels, width, lights, fog, sound |
| melting forms | soft 3D forms that melt into each other and out of a wall, lit from the side with real shadows. Loops without a jump; the low end of a sound makes them swell. Gives a depth |  | forms, size, melt, blocks, light, sound, steps |
| colonnade | an endless walk between two rows of columns under an open sky; the sun lays their shadows across the floor. Loops without a jump. Gives a depth |  | bays, width, thick, arches, sun, sway, fog, sound, steps |
| fog light | a beam of light swinging through drifting fog, like a projector in a hazy hall. Loops without a jump; a sound brightens the beam |  | density, scale, beam, spread, rays, sweep, warm, sound, steps |

### on your regions

Light that follows your masks and regions: connect kubakub mask field (the sample facade without one). For projection mapping.

| preset | what it draws | works on | controls |
|---|---|---|---|
| mapping grid | a line-up grid for the projector: squares, the centre cross, circles and the diagonals; a scan line runs through |  | cells, line, scan |
| scan band | a band of light that runs along the field (connect kubakub mask field; without one: left to right) |  | width, soft, bands, beat, warm |
| brick wave | a wall of bricks; a wave of light runs through them along the field (or left to right). Loops without a jump |  | rows, ratio, mortar, width, waves |
| region contours | lines of light that follow the outline of every region inwards (connect kubakub mask field, edge distance). Loops without a jump |  | rings, thin, speed, outline, beat |
| extrude | the regions rise out of the wall as blocks with sloped edges and throw shadows on it (connect the field of kubakub mask field, edge distance; nothing connected = the windows of the sample facade); a picture on image0 is painted on them. Loops without a jump; a beat can lift them. Gives a depth |  | height, bevel, round, rise, wave, sway, sun, turns, angle, dark, shade, beat, steps |

### picture looks

Your picture on image0, redrawn: cartoon, blueprint, dots, big pixels (the sample facade without one).

| preset | what it draws | works on | controls |
|---|---|---|---|
| chromatic split | the picture on image0 with its colours pulled apart; a wave runs through and the sound pushes it | a picture | amount, wave, lines, sound |
| glass | the picture on image0 seen through rippled glass. A normal pass on image1 is the glass; without one it is made of ripples | a picture | bend, ripples, split, shine, flow |
| edge glow | the outlines of the picture on image0 drawn in glowing light; a pulse runs through them | a picture | thick, gain, keep, pulse, sound |
| mosaic | the picture on image0 as tiles that change size with the loop or the sound | a picture | tiles, breathe, gap, round, sound |
| contour lines | the picture on image0 as height lines, like a map of its light and dark; the lines climb. Loops without a jump | a picture | lines, thin, speed, smooth, keep |
| blueprint | the picture on image0 as an architect's blueprint: white outlines on blue, a drawing grid, and a pen of light that redraws it | a picture | thick, gain, grid, pen, paper |
| halftone | the picture on image0 printed in dots, big where it is bright; the raster slowly turns. Loops without a jump | a picture | dots, angle, turn, colour, soft |
| eight bit | the picture on image0 as on an old game console: big pixels, a handful of colours, dither patterns | a picture | pixels, palette, dither, shimmer, contrast |
| characters | the picture on image0 typed in characters, dense signs where it is bright, like text-mode art; the signs flicker | a picture | columns, colour, flicker, contrast |
| pixel smear | the bright parts of the picture on image0 melt downwards in long streaks, like sorted pixels; the amount breathes with the loop | a picture | length, level, breathe, sideways, sound |
| cartoon | the picture on image0 as a cartoon: a few flat tones, black ink outlines and hatching in the dark parts that drifts. High contrast, made for architecture | a picture | tones, contrast, ink, thick, detail, hatch, lines, drift, two_colours |

### screen and tape

Your picture on image0 as on old video gear: a VHS tape, a tube television, a composite cable.

| preset | what it draws | works on | controls |
|---|---|---|---|
| old screen | the picture on image0 on an old tube television: a curved glass, scan lines, coloured dots and a bar that rolls through | a picture | lines, curve, dots, roll, split, noise |
| vhs tape | the picture on image0 as a worn VHS tape: soft, colours bleeding sideways, a wobbling edge, snow, a noisy band and the torn lines of the head switch at the bottom | a picture | wear, bleed, wave, snow, tracking, head, ring, grain, lines |
| composite video | the picture on image0 sent through a composite video cable: colour fringes and rainbows crawling along fine detail, dots on colour edges, soft colour | a picture | carrier, fringe, dots, crawl, colour, lines |
| scan modulation | the picture on image0 as scan lines that the brightness pushes out of place, like a signal on a tube; the lines drift | a picture | rate, depth, steps, drift, soften, tint, light |

### grade and lens

Your picture on image0, finished: blur, colour grading, glow, and what a lens does to it (bend, swirl, mirrors). Most hold still unless you ask them to move.

| preset | what it draws | works on | controls |
|---|---|---|---|
| blur | the picture on image0 out of focus: a soft, even blur. The loop can pull the focus, a loud sound sharpens it | a picture | radius, breathe, sound, taps |
| motion blur | the picture on image0 smeared in one direction, as if it rushed past; a drawn arrow sets the direction. The loop or a beat can kick it | a picture | length, angle, pulse, beat |
| zoom blur | the picture on image0 rushing out of a point, like a zoom during a long exposure; a drawn arrow's start is the point. The loop or a beat can kick it | a picture | amount, spin, cx, cy, pulse, beat |
| grade | the picture on image0 graded like a photograph: exposure, contrast, colour, warmth, and where black and white sit. A still look; the loop can let the light breathe | a picture | exposure, contrast, saturation, warmth, tint, black, white, gamma, breathe, sound |
| duotone | the picture on image0 in two colours of your choice: one for the shadows, one for the lights, like a risograph or a poster. The loop can swing the balance | a picture | shadows, lights, contrast, balance, amount, swing, sound |
| vignette | the picture on image0 with darkened corners and film grain, as through an old lens. The grain moves; everything else holds still | a picture | amount, size, soft, round, edge, grain, grain_size, beat |
| lens bend | the picture on image0 through a strong lens: it bulges like a fish eye or pinches in, with colour fringes towards the corners. The loop can let it breathe | a picture | bend, fringe, zoom, dark, breathe, sound |
| swirl | the picture on image0 wound up around a point like cream in coffee, and back again; a drawn arrow's start is the point. Loops without a jump | a picture | angle, size, cx, cy, turn, sound |
| under water | the picture on image0 seen through moving water: rings run out from a point and bend it, with a glint on their crests; a drawn arrow's start is the point. Loops without a jump | a picture | waves, amount, speed, cx, cy, shine, sound |
| little planet | the picture on image0 rolled into a little planet, its ground in the middle and its sky all around (or the other way: a tunnel). It turns; loops without a jump | a picture | amount, turn, zoom, tunnel, mirror, curve |
| glow | the bright parts of the picture on image0 bleed light into their surroundings, like lamps in fog or a soft-focus lens. The loop or a sound can swell the glow | a picture | level, radius, strength, glow_colour, breathe, sound, taps |
| mirrors | the picture on image0 in a turning kaleidoscope: a wedge of it mirrored around a point; a drawn arrow's start is the point. Loops without a jump | a picture | mirrors, turn, zoom, cx, cy, slide |

### picture with depth

Your picture on image0 gets depth: relief, shadows, a wire mesh, points. A depth pass on image1 makes it true; without one its brightness is the depth.

| preset | what it draws | works on | controls |
|---|---|---|---|
| wireframe | the picture on image0 as a glowing wire mesh, its lines bent by depth (a depth pass on image1, else by brightness). A wave of light runs through | a picture | rows, columns, relief, thin, wave, keep |
| depth scan | a sheet of light that travels through the depth of the picture on image0 (a depth pass on image1, else its brightness). Loops without a jump | a picture | sheets, width, passes, keep, warm |
| point cloud | the picture on image0 as a cloud of points that sways in depth (a depth pass on image1, else by brightness), like a 3D scan | a picture | points, size, sway, depth_size, glow |
| hard shadow | a light circles the picture on image0 and everything that stands out throws a hard shadow (a depth pass on image1, else its brightness). A fake shadow, no 3D needed. Loops without a jump | a picture | sun, reach, turns, angle, dark, soft, shade, tint |
| facade relief | the picture on image0 as a real relief: what stands out (a depth pass on image1, else its brightness) rises out of the wall, the view leans from side to side so near parts hide far ones, and a circling light throws true shadows. Loops without a jump. Gives a depth | a picture | height, sway, nod, sun, turns, angle, dark, soft, shade, steps |

### over time

They build every frame on the one before: a clip that smears and leaves trails, a pattern that grows. On the node only.

| preset | what it draws | works on | controls |
|---|---|---|---|
| slit scan | slit scan: a slit travels across and paints what the clip on image0 shows at that moment, so moving things bend and stretch. Needs a clip (a moving pattern without one) | its last frame | passes, upright, show |
| datamosh | datamosh: the clip on image0 breaks into blocks that smear along its own shapes until a clean frame snaps back. Needs a clip (a moving pattern without one) | its last frame | blocks, smear, keys, heal, bleed |
| trails | everything that moves in the clip on image0 leaves a fading trail that drifts, turns or zooms away. Needs a clip (a moving pattern without one) | its last frame | fade, zoom, turn, drift_y, tint |
| growth | a living pattern that grows from scattered seeds into spots, worms and mazes, like coral or lichen spreading over the wall (reaction and diffusion); stays inside the mask. Gives a depth | its last frames | feed, kill, size, seeds, warm |

Most presets of the first three types and several others also have two colours, `colour_a` (the cool one) and
`colour_b` (the warm one): pick them in **adjust**, or type `colour_a = #2040ff` in `controls`.

All of them loop without a jump and stay inside the mask. `sound` and `beat` are 0 in most presets and go to 1
when you connect a sound; set them in `controls` to make the picture follow the music more or less (sound bars
follows it by itself).

**3D scenes** (melting forms, colonnade, facade relief, extrude, fog light) are raymarched: for every pixel the
shader walks a ray through a scene. They cost more than the flat presets; `steps` in `controls` trades clean edges
for speed, and the report says so when a frame takes long. The first four also give a `depth`. extrude shows the
wall and the shadows on it, so it is not cut to its own regions: connect only the `field`, or the `mask` as well to
keep the blocks alone.

**Works on a picture**: connect yours to `image0` (a still or a clip); with none connected the sample facade is
used. wireframe, depth scan and point cloud read a depth pass on `image1` when there is one (kubakub render
passes, scene render) and take the brightness of the picture as depth when there is none; glass reads a normal
pass there.

**Works on its last frame** (slit scan, datamosh, trails, growth): these build every frame on the one before, which is
what makes a slit scan bend moving things, a datamosh smear and a growth spread. growth needs no clip: it starts
from seeds and grows for as long as the clip is (try 10 s or more, a `mask` to keep it on the windows, and its
`depth` into facade relief or kubakub height animate). The other three want a **clip** on `image0` (Load Video, or
the frames of another kubakub shader); with nothing connected they run on a moving pattern of their own so you
see what they do. Their loop is closed by a fade (`loop_fade`). They exist on the node only, not as a fill in
the director.

For projection mapping: mapping grid lines the projector up; region contours, scan band and brick wave follow
your regions (connect kubakub mask field); contour lines, wireframe, blueprint, edge glow and depth scan trace
the building's own shapes from a render of it.

The presets are plain text files in `kubakub/shaders/`, written for this pack. **adjust > show the code** puts one
into `code` to change it, and **save as my preset** keeps the result (see
[Adjusting a preset](#adjusting-a-preset-and-saving-your-own)).

## 3D scenes (raymarching)

A shader can draw a 3D scene without any 3D software: the scene is a function that says how far the nearest
surface is from a point (a signed distance function, SDF), and the shader walks along a ray for every pixel until
it arrives. Most 3D shaders on shadertoy.com work this way, and a single-tab one runs here when pasted.

For your own scene the node brings the usual parts. Write `float map(vec3 p)` (the distance from p to the nearest
surface) and use these names; a shader gets the ones it uses, and a pasted shader that brings its own `sdBox` keeps
its own.

| name | what it is |
|---|---|
| `sdSphere(p, r)`, `sdBox(p, half)`, `sdRoundBox(p, half, r)`, `sdTorus(p, vec2(R, r))`, `sdCylinder(p, halfHeight, r)`, `sdCapsule(p, a, b, r)`, `sdPlane(p, normal, h)` | the distance from p to a ball, a block, a rounded block, a ring, an upright cylinder, a rod from a to b, a plane |
| `opSmoothUnion(a, b, k)`, `opSmoothSubtraction(a, b, k)`, `opSmoothIntersection(a, b, k)` | two forms melted together, a cut out of b, or only where both are; k = how soft the join is. Hard joins are `min(a, b)`, `max(-a, b)`, `max(a, b)` |
| `opRep(p, vec3 cell)` | the space repeated every `cell`: one column becomes an endless row |
| `kRot(angle)` | a turn, as a mat2: `p.xz *= kRot(a);` |
| `kCamera(fragCoord, eye, target, zoom)` | the ray through this pixel for a camera at `eye` that looks at `target` (zoom 1 = wide, 2 = a long lens) |
| `kMarch(eye, ray, far, steps)` | how far along the ray the first surface is; -1 when there is none within `far` |
| `kNormal(p)` | which way the surface faces at p |
| `kShadow(p, toLight, far, k)` | 1 = lit, 0 = in shadow; k = how hard the edge is (8 soft, 64 hard) |
| `kAO(p, normal)` | 1 = open, darker in corners and creases |
| `kDepth` | write how near this pixel is, 0 far to 1 near: it becomes the node's `depth` output. `kDepthOf(t, near, far)` turns the distance from kMarch into it |

```glsl
uniform float melt;    // 0.3 0.01 1
float map(vec3 p) {
    float ball = sdSphere(p - vec3(sin(6.2831853 * kLoop), 0.0, 0.0), 0.6);
    float block = sdRoundBox(p, vec3(0.5), 0.1);
    return opSmoothUnion(opSmoothUnion(ball, block, melt), p.y + 0.6, 0.1);      // ... and a floor
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 eye = vec3(0.0, 1.0, 3.5), ray = kCamera(fragCoord, eye, vec3(0.0), 1.5);
    float t = kMarch(eye, ray, 20.0, 80);
    vec3 col = vec3(0.0);
    if (t > 0.0) {
        vec3 p = eye + ray * t, n = kNormal(p), sun = normalize(vec3(0.6, 0.8, 0.4));
        col = vec3(0.9, 0.6, 0.45) * (0.15 * kAO(p, n) + max(dot(n, sun), 0.0) * kShadow(p + n * 0.01, sun, 10.0, 16.0));
    }
    kDepth = kDepthOf(t, 2.0, 8.0);
    fragColor = vec4(sqrt(col), 1.0);
}
```

**The depth** is a normal grey picture batch, near = white, the same way round as the depth of kubakub form height.
It goes into `image1` of another kubakub shader (wireframe, point cloud, depth scan, hard shadow, facade relief),
into kubakub height animate, or into a depth ControlNet. Example: `example_workflows/shader_3d.json` (melting forms,
and sheets of light through its depth).

**A relief of your building.** facade relief is the preset for a render of the real facade: `beauty` into `image0`
and `depth` into `image1` (kubakub render passes, scene render, or the height of kubakub form height). Unlike hard
shadow it moves the view, so a cornice really covers what is behind it. Without a depth pass the brightness is
taken as height.

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
- **rings**, **ripples**, **radar** and **light rays** start at its start; so do swirl, zoom blur, under water and mirrors
- every other pattern and 3d scene follows it with its whole picture: what rises runs along the arrow (**fire**,
  aurora, sound bars), what drifts or lies sideways turns to it (**clouds**, warped noise, flow lines, glitch bars
  and the rest), and what has a middle moves its middle to the arrow's start (**starfield**: the stars fly out of
  that point; tunnel, spiral, kaleidoscope, grid flight, corridor, colonnade, melting forms, fog light). A mask stays
  where it is
- the presets that redraw a picture have no use for an arrow unless listed above (the report says so)

The report says whether the shader used it. Your own code reads `kFrom`, `kTo`, `kDir` and `kVector`; or it says
in a line `// arrow: up` (or `right`, or `centre`) which way it runs by itself, and the node turns or moves it. **kubakub mask field** takes the same arrow for its direction and radial fields, and the kubakub director has it on shader fills and on field behaviours.

## Pasting a Shadertoy

You do not need to read or understand the code.

1. Open the shader on shadertoy.com.
2. Click into the code on the right, press **Ctrl+A** (everything), then **Ctrl+C**.
3. Click into `code` on the node, press **Ctrl+V**, queue.

If the shader has a **Common** tab next to **Image**, paste Common first and Image below it. Pasting the link does
not work (the node downloads nothing, and says so).

**A Shadertoy with Buffer tabs** (Buffer A - D: simulations, trails, feedback) works too, on the node. Paste every
tab under a line that names it:

```glsl
// Common
(the Common tab, if there is one)
// Buffer A
(the Buffer A tab)
// Image
(the Image tab)
```

On shadertoy.com each tab's `iChannel0` - `iChannel3` are set by clicking, not in the code, so the node cannot see
them. It assumes the usual wiring: `iChannel0` is Buffer A, `iChannel1` is Buffer B and so on, and a channel
without a buffer of that letter is the node's picture of that number (the built-in noise when nothing is
connected). Where the Shadertoy is wired differently, say it on the tab's line: `// Buffer A: 0 = A, 1 = noise`,
`// Image: 0 = B, 1 = image0`. The report lists what every tab reads. A tab that reads itself sees the frame
before, as on shadertoy.com; the first frame starts from black, and `loop` closes the clip with a fade. Such a
shader builds on its own last frames, so it does not run as a fill in the director or in kubakub shader mix:
render it with the node and bring the frames in.

The report then tells you in plain words what the shader uses. The most common case: a Shadertoy expects a noise
picture on one of its channels. When nothing is connected to `image0` - `image3` the node puts its own **built-in
noise** there, laid out the way those shaders expect, so most of them simply run. If a shader looks wrong, it may
expect a photo or another picture: connect one to the input the report names.

What cannot work, and the node says so instead of showing a graphics error:

- **cubemaps** and 3D textures
- **sound shaders**, the keyboard and the webcam; a text longer than 400 000 characters
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
uniform vec3 ink;      // #f18a58  a colour to pick: a hex colour, then anything
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float v = 0.5 + 0.5 * sin(uv.x * 20.0 + 6.2831853 * kLoop * speed);
    fragColor = vec4(ink * v * kMask(uv), 1.0);
}
```

A `uniform vec3` with a hex colour in its comment is a colour control: a swatch in **adjust** and in the director,
`ink = #2040ff` in `controls`. (Inside, it is three numbers `ink_r`, `ink_g`, `ink_b`, which is why each can have
keyframes of its own in the director.)

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
| `kDepth`, `sdBox` ..., `kMarch` ... | for a 3D scene: see [3D scenes](#3d-scenes-raymarching) |
| `kFrom`, `kTo`, `kDir`, `kVector` | the drawn arrow: its start and its end (0 to 1, as `kUV`), the way it points (length 1), and 1 when one is drawn. Without a connected field `kField` runs from 0 at its start to 1 at its end |
| `uniform float name; // default min max` | a control of your own, set in `controls` |
| `uniform vec3 name; // #rrggbb` | a colour of your own, picked in adjust or set in `controls` |

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

The `fill` menu lists the presets under their types, and the **pictures** button next to it opens the same picker
as on the node: small pictures by type, moving under the pointer (the type *over time* is left out: slit scan,
datamosh, trails and growth build on their last frames, which only the node can do).

**The shape as the field.** A shader that runs along a field (scan band, region contours, brick wave, extrude)
has a **field** row: *left to right*, or *the shape's edges*. With the edges the field runs from the outline of
the shape inwards, and when the layer is clipped to regions every region is a piece of its own: region contours
then draws its lines around every window, and extrude lifts every window as a block (it paints the layers below
on them). extrude starts with the edges; the others keep left to right until you choose, so saved compositions
look as before. A drawn arrow still comes first.

**An arrow instead of numbers.** A shader that can follow a direction has an **arrow** row: press **draw** and drag on the layer. Presets that run along the field (scan band, region contours ...) then run from the arrow's start to its end; stripes, rain and pixel smear take its direction; rings, ripples and radar start at its start. The arrow belongs to the layer and moves with it. **clear** removes it.

**A shader as an effect.** A preset that works on a picture (eight bit, characters, mosaic, halftone, old
screen, pixel smear, vhs tape, composite video, scan modulation, cartoon, hard shadow, facade relief, contour lines, wireframe, blueprint, edge glow, depth scan, point cloud, chromatic split,
glass) works here on **everything under the layer**: put a solid with fill = eight bit on top and the whole
composition below turns 8 bit; make it an ellipse with a feather, or clip it to the windows, and only that part
does. Stack two for two effects. Your own code has a `picture` choice for this: *the layers below*, or the
*built-in noise* that most pasted Shadertoys expect (the default for own code). Without a depth pass, wireframe,
depth scan and point cloud take the brightness of the picture as depth. A pasted Shadertoy works here too
(own code): the built-in noise sits on its channels, in the window and in the render alike.

The 3D presets play in the window too (melting forms, colonnade, fog light as a fill; facade relief on the
layers below, with their brightness as height). They are heavier than the flat ones: lower `steps` if scrubbing
stutters.

Not in the window: pictures on `image1` - `image3` (a depth or normal pass), a field from kubakub mask field
(the window has its own: left to right, or the shape's edges) and the depth output.
Use the node for those and bring its frames in as a video layer. A broken shader shows its mistake under the
code, with the line, and the layer shows its colour until it is fixed. Every control has a hover text that says
what it does.

## Speed

The shader compiles once per run and pictures that do not change are uploaded once. On an RTX 5070 Ti a simple
preset takes about 14 ms per frame at 1920x1080 and 57 ms at 3840x2160; warped noise 32 ms and 112 ms. Most of
what is left is copying the frames into memory, so RAM is the limit: 250 frames at 3840x2160 are 33 GB (the node
refuses a clip that does not fit and says how much it needs). The 3D presets take about 100 ms per frame at
1920x1080 and 400 ms at 3840x2160 with their default `steps`; a clip with a depth needs about twice the RAM. When a
frame takes longer than a quarter of a second the report says so.

Limits: a shader with Buffer tabs runs on the node only. The mask and field are shared by the whole picture, there is no value per region. On the
node the controls are fixed for a run and what moves is the loop, the clock and the sound; in the director the
controls can be keyframed.

Test without a model:

```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tests\test_shadertoy.py
```
