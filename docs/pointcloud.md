# kubakub point cloud (2d / motion)

A picture or a video lifted into points by its depth and filmed with a camera move: up to two million points, live
on the node. Example: [`point_cloud.json`](../example_workflows/point_cloud.json) (your video, Depth Anything 3 and
the tracker). The node alone, with nothing connected, shows a sample facade with its own depth.

kubakub shader's preset *point cloud* is the quick look of the same idea: the picture sways in depth in the flat
picture, inside the shader and the director. This node is the real cloud: a camera that moves through it, millions
of points, pictures that morph, tracked things in 3D, a .ply.

## Inputs

| input | what it does |
|---|---|
| `image` | a picture, a batch of pictures or a video (*Load Video* directly, or any frames). A batch of up to 16 pictures is a set that **morphs**: every point travels to the point of the next picture with the same brightness. More is a video (`batch` on the node changes that). Nothing connected: the sample facade. |
| `depth` | the depth of the same picture(s) or video from any depth node: Depth Anything 3 per frame, DA3 Streaming (steady over a long clip), Depth Anything 2, Marigold, Video Depth Anything. *near is* bright / dark on the node: bright for Depth Anything, dark for Marigold. Nothing connected: a soft relief from the brightness (the hint on the node says so). |
| `field` | a mask that pushes the points out where it is white, frame by frame: the tracker's `near` (the points burst around what moves), kubakub mask animate, a mask from kubakub shader. How far: *push*. |
| `tracks` | from kubakub tracker on the same video: a ring on every tracked thing at its place in the cloud, joined by lines (network, tree, within a distance: the same joins as track draw). |
| `seconds` | how long the render is. 0 = a video's own length; pictures 3 s each (one picture 4 s). Longer than the video: it loops. |
| `fps` *(advanced)* | frames per second for pictures; a video brings its own. |
| `long_side` *(advanced)* | the render's long side in pixels; 0 = the picture's own size. |
| `batch` *(advanced)* | auto / video / pictures (see `image`). |
| `save_ply` | also writes the first frame's points as a binary .ply to `output/kubakub/` (x y z and colour), for Blender, Houdini, MeshLab, TouchDesigner. |

Outputs: `images` (the filmed cloud), `alpha` (where points are, to put it over something), `fps` (for Create Video),
`ply` (the saved file's path) and `report`.

## On the node

Queue once; the cloud then plays on the node from that run's frames and depth, and everything below changes live,
without running again.

- **Look around:** drag to turn, Shift-drag (or the right button) to move sideways, the wheel to go nearer.
- **The camera move:** find a view and press **end = view**, then one for **start = view**; **see the move** plays
  it over the render's length (**smooth**, **linear**, or **still** = the start view only). **picture view** is the
  picture's own view: from there the points cover the picture exactly (a far point is also spread wider), so only a
  camera move shows the depth.
- **colour:** true, mono, tint (dark to bright between two colours you pick), negative.
- **blend:** *glow* (light adds up, on black, like a projector), *ink* (on white, colours multiply), *solid* (the
  nearest point wins, no see-through).
- **points:** *points* (across the long side; 2048 = up to about 2.3 million), *size* (in grid cells: 1 = they just
  touch from the picture's view; near points grow), *opacity*, *jitter* (0 = an exact grid), *depth* (how deep the
  picture goes), *far cut* (leaves out what is farther: the sky, the background).
- **motion** (formulas of the time, nothing is simulated, so every frame is the same in the render as on the node):
  *speed*, *drift* (every point wanders a little), *breathe* (the cloud swells along the view, in waves from near to
  far), *wind* (sideways, the far points more), *scan* (a bright layer that goes through the depth), *push* (with a
  field), *swarm* (pictures: how far the points fly out on their way to the next picture).
- **tracked things** (with `tracks`): rings on / off, lines off / within / network / tree, colour, opacity, ring size.
- **render** renders only the cloud and what saves it after it (Create Video, Save ...); the depth and the tracks are
  not made again.

## Speed and size

The render draws every frame on the graphics card: about 30 ms for a million points at 1920 x 1080 (a 12 GB card),
so most of the time goes into the depth node. The live view draws the same points with WebGL2 in the browser; 2
million points play smoothly on a recent graphics card, lower *points* on a laptop.

The live view uses the frames at 1280 px and its own pixel grid for the morph partners, so a still from the node and
the render can differ in the grain, never in where things are.
