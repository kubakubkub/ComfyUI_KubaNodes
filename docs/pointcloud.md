# kubakub pointclouder (2d / motion)

A picture or a video lifted into points by its depth and filmed with a camera move: up to two million points, live
on the node. Example: [`point_cloud.json`](../example_workflows/point_cloud.json) (your video, Depth Anything 3 and
the tracker). The node alone, with nothing connected, shows a sample facade with its own depth.

kubakub shader's preset *point cloud* is the quick look of the same idea: the picture sways in depth in the flat
picture, inside the shader and the director. This node is the real cloud: a camera that moves through it, millions
of points, pictures that morph, tracked things in 3D, the sound, a .ply.

## Inputs

| input | what it does |
|---|---|
| `image` | a picture, a batch of pictures or a video (*Load Video* directly, or any frames). A batch of up to 16 pictures is a set that **morphs**: every point travels to the point of the next picture with the same brightness. More is a video (`batch` on the node changes that). Nothing connected: the sample facade. |
| `depth` | the depth of the same picture(s) or video from any depth node: Depth Anything 3 per frame, DA3 Streaming (steady over a long clip), Depth Anything 2, Marigold, Video Depth Anything. *near is* bright / dark on the node: bright for Depth Anything, dark for Marigold. Nothing connected: a soft relief from the brightness (the hint on the node says so). |
| `field` | a mask that pushes the points out where it is white, frame by frame: the tracker's `near` (the points burst around what moves), kubakub mask animate, a mask from kubakub shader. How far: *push*. |
| `regions` | from any kubakub regions node (regions from matrix / masks / sam3 / id renders ...), or a mask: its groups become forces, set per group on the node (*forces*: attract, repel, curl + / −). Connected as it is, the group with the most regions (the windows, as a rule) attracts. A mask: every separate white part is a region, all in one group *mask*. |
| `tracks` | from kubakub tracker on the same video: a ring on every tracked thing at its place in the cloud, joined by lines (network, tree, within a distance: the same joins as track draw). |
| `audio` / `sound` | a sound the cloud follows (Load Audio, a video's sound, or kubakub sound on `sound`): the points burst out on the beat, swell with the bass, move more when it is loud (how much: *sound* on the node). It plays with the cloud on the node. Pictures last as long as the sound. |
| `seconds` | how long the render is. 0 = a video's own length; pictures as long as the sound, else 3 s each (one picture 4 s). Longer than the video: it loops. |
| `fps` *(advanced)* | frames per second for pictures; a video brings its own. |
| `long_side` *(advanced)* | the render's long side in pixels; 0 = the picture's own size. |
| `batch` *(advanced)* | auto / video / pictures (see `image`). |
| `save_ply` | also writes the first frame's points as a binary .ply to `output/kubakub/` (x y z and colour), for Blender, Houdini, MeshLab, TouchDesigner. |

Outputs: `images` (the filmed cloud), `alpha` (where points are, to put it over something), `fps` (for Create Video),
`ply` (the saved file's path), `report` and `audio` (the sound cut to the render, for Create Video; silence without
one).

## On the node

Queue once; the cloud then plays on the node from that run's frames and depth, and everything below changes live,
without running again.

- **Look around:** drag to turn, Shift-drag (or the right button) to move sideways, the wheel to go nearer.
- **The camera move:** find a view and press **end = view**, then one for **start = view**; **see the move** plays
  it over the render's length (**smooth**, **linear**, or **still** = the start view only). **picture view** is the
  picture's own view: from there the points cover the picture exactly (a far point is also spread wider), so only a
  camera move shows the depth.
- **Moves** from the view you see: **push in**, **pull out**, **pan** (slides sideways), **tilt**, **orbit**,
  **crane** (rises and looks down). **handheld** adds a little shake on top of any move (a slider for how much).
- **● record**: the clip plays and records how you drag and scroll on the picture, until its end (or **■ stop**);
  that is the camera move then, in the render too. **forget the recording** goes back to start → end.
- **settings ▾** unfolds every part below (each part also opens on its own by its title):
  - **look:** colour (true, mono, tint between two colours you pick, negative); blend: *glow* (light adds up, on
    black, like a projector), *ink* (on white, colours multiply), *solid* (the nearest point wins, no see-through).
  - **points:** *points* (across the long side; 1920 = about 2 million, 3072 = about 5 million), *size* (in grid
    cells: 1 = they just touch from the picture's view; near points grow), *opacity* (above 1 even a point smaller
    than a pixel shines), *jitter* (0 = an exact grid), *depth* (how deep the picture goes), *far cut* (leaves out
    what is farther: the sky, the background). A point smaller than a pixel and a half shares its light with the
    pixels around it by where it falls, so many fine points read as a smooth fine picture, not as grain.
  - **fine motion** (the main motion, each point on its own, like a scanner's points): *shiver* (every point
    trembles along its own line of sight at its own rhythm; 1 = about two point spacings; 0.3 by default),
    *dropout* (which share of the points blinks out now and then, each on its own clock), *scan lines* (the points
    gather into horizontal rows, the colour stays theirs).
  - **flow** (off by default): a share of the points streams along smooth swirls like a liquid. Each flowing point
    is born at its place in the picture, flows for its *life*, fades out and is born again at its place, so the form
    keeps coming back while it dissolves. *flow* (how fast; 0 = none), *of points* (which share flows; the others
    hold the form), *swirl size*, *life* (seconds), *where*: **everywhere**, **field** (where the `field` mask is
    white), **tracked** (near the tracked things, as far as *reach*), **forces** (near your curves and regions). *where* also
    decides where the beat hits. The swirls have no sources and no sinks, so the points stream around each other
    instead of bunching up; with *trails* they draw their flow lines.
  - **motion** (formulas of the time, nothing is simulated, so every frame is the same in the render as on the
    node): *speed* (of everything), *drift* (every point wanders a little on its own: a shimmer; 0 by default),
    *breathe* (the cloud swells along the view, in waves from near to far), *wind* (sideways, the far points more),
    *curl* (the cloud sways on big curly loops), *scan* (a bright layer that goes through the depth), *push* (with a
    field), *swarm* (pictures: how far the points fly out on their way to the next picture).
  - **trails:** each point's own way over the last seconds drawn behind it as a fine line: *length* (seconds),
    *of points* (which share of the points draws one), *opacity*. With the flow they are its flow lines; a point
    that holds draws none.
  - **forces:** what the flowing points do near a curve you draw or near a group of regions.
    - **draw curves:** drag on the picture with the mouse (the view holds on the picture while you draw, where
      they act is lit faintly). What a new curve does, the pills under it: **wind** (the points stream along it,
      the way you drew it), **attract** (they are pulled onto it), **repel** (pushed away), **curl +** / **curl −**
      (they turn round it anticlockwise / clockwise, with small eddies). A curve that ends where it began is a loop:
      curl turns round its middle like a whirlpool, and inside it counts as near. Up to 8 curves, **undo**,
      **clear**. An older look's arrows are wind curves.
    - **regions:** with `regions` connected, every group has its row: **off**, **attract** (the points near its
      regions stream into them and gather there), **repel** (they leave them and keep away), **curl +** / **curl −**
      (they turn round each region's middle). So all windows can attract while the piers repel.
    - *strength* (of the curves and regions), *reach* (how far from a curve or region it still acts). With *where* =
      forces the swirls and the beat act only there; the curves and regions act whatever *where* says.
  - **sound** (with a sound): *spectrum → depth* (the depth hears the song band by band: far = the bass, near =
    the highs; each slice comes forward and lights up with its band), *beat → sweep* (on every beat a thin bright
    plane rushes through the depth), *highs → shiver*, *loud → fill* (when it is loud the blinked out points come
    back), *beat → burst* (the parts where the flow acts pop out and kick along the swirls; 0 by default),
    *bass → swell*, *loud → motion* (drift, breathe, wind and curl get bigger), *loud → size*, *highs → sparkle* (a
    few points light up); **hear it** plays the sound with the cloud.
  - **tracked things** (with `tracks`): rings on / off, lines off / within / network / tree, colour, opacity, ring size.
- **render** renders only the cloud and what saves it after it (Create Video, Save ...); the depth and the tracks are
  not made again.

## Speed and size

The render draws every frame on the graphics card: 10 to 40 ms a frame for a million points at 1920 x 1080 (a 12 GB
card; the flow and the trails take more: each flowing point takes 12 steps through the swirls, a trail 13 times
that), so most of the time goes into the depth node. A video is read once and kept, so a second
render (only the look changed) starts at once. The live view draws the same points with WebGL2 in the browser; 2
million points play smoothly on a recent graphics card, lower *points* (or *of points* for the trails) on a laptop.

The live view uses the frames at 1280 px and its own pixel grid for the morph partners, so a still from the node and
the render can differ in the grain, never in where things are.
