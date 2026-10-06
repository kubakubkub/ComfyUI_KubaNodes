# 3D scene (3d / scene)

The festival's 3D file, read by Blender in the background: the projection view with ID passes and real distances, previz from the audience, and even brightness on the building.

## kubakub scene render

A festival's 3D file -> the projection view, metric depth and ID passes that the region tools
already read. Example: `example_workflows/scene3d_to_regions.json`.

- **`file` empty** = the built-in sample building (kubakub sample model with its default size), so the node
  runs as it is; paste the path of your own file to replace it.
- **Files**: .blend .fbx .obj .abc .glb .gltf .stl .ply .usd(a/c/z). Blender 4.x runs headless as a
  subprocess (found in `C:\Program Files\Blender Foundation`, or the `blender_path` input / `blender = ...` in kubakub.ini);
  it is not imported into ComfyUI's Python. **.c4d** cannot be read outside Cinema 4D: ask for FBX or
  Alembic. The source file is only read, never saved.
- **Camera**: a connected **kubakub projector** (below), else `camera` by name, else the file's active camera,
  else the first one, else a front camera framing the model: it stands in front of the side the model is widest
  across (a facade is wide and shallow), level with its middle. Resolution 0 = the file's render size; FBX / OBJ / Alembic carry none, so a size
  in the camera name (`RENDER_CAM_3200_2160PX`) is used, else 1920x1080 with a note. FBX round trips
  can flip the lens shift: when the imported camera sees almost none of the model and the flipped shift
  frames it, the flipped one is used and the report says so.
- **What Blender does** (one call, cached per file / camera / size / frame in `ComfyUI/temp/kuba_scene3d`):
  flattens everything render-visible into world-space meshes (modifiers applied, instances real),
  renders a Workbench clay view, and one Cycles sample without AA that stores, per pixel, the face it
  sees, its world position and normal. The test facade (20k faces, 3200x2160): 4.5 s on the GPU (OptiX).
- **ID passes** are computed in numpy from that (no re-render when a threshold changes):
  | pass | what |
  |---|---|
  | `shelves` | the facade's depth steps found from the depth histogram along the facade normal, named by their offset from the main wall: `shelf_+0.35m` = 35 cm in front |
  | `layers` | fixed depth bands around the main plane: deep_recess / recess / facade / relief / projection (`layer_bands_m`) |
  | `facing` | front / top / underside / side / back (relative to the facade normal and world up) |
  | `planes` | connected coplanar face patches (`plane_angle_deg`, `plane_offset_m`) |
  | `parts` | loose parts: every separately modelled stone, frame, cornice ... |
  | `objects` / `materials` / `collections` | from the file, written when more than one is visible |
  The main plane = the dominant direction of the camera-facing pixels at the depth where most of them lie.
- **Outputs**: clay, the `preview_pass` ID map, depth (near = bright), normals, a coverage mask (scope for
  the region nodes), the ID-map folder (-> **kubakub regions from id maps**, same format as the Houdini
  renders), `scene_json` (camera matrices, lens, shift, bbox, shelves) and a report. `save_folder` copies
  ID maps, legends, clay and scene.json for Houdini / After Effects.
- Tested on a 3200x2160 station facade: the .blend and the .fbx give the same result (camera
  RENDER_CAM_3200_2160PX, 16 shelves from -0.46 to +0.81 m, 450 parts); `parts` as regions gives 423
  regions (every stone, voussoir, frame, pediment). From ID Maps needs ~2 min for that many parts,
  ~9 s for shelves.
- Model free test: `tests/test_scene3d.py` (synthetic facade through the passes and From ID Maps; with
  Blender installed also a real OBJ export).
- `unit_scale` (optional): files in the wrong units (some FBX exports come in ~0.5 m wide) are
  scaled into metres; the report warns when a model is under 2 m or over 2 km.
- The last output `scene` feeds the two nodes below.

### Architecture passes: windows, columns, left / centre / right, floors

Three passes of kubakub scene render group the facade the way you talk about it (no AI model, from the 3D
position / depth / normal of every pixel; `kubakub/scene3d/architecture.py`):
- **elements**: windows (openings behind the wall, and blind windows a frame closes in; an arcade is split into
  one region per arch), window_frames (what stands out around them), columns (tall narrow protrusions: pilasters,
  quoins), cornices (long horizontal ones), relief (other protrusions), wall, roof (above the top cornice belt where
  the facade narrows: dormers, attic, pediment).
- **sections**: left / centre / right … where the wall steps forward or back (the most common front-facing depth
  over 1.5 m; steps over 15 cm, sections at least 3 m wide), else thirds.
- **floors**: ground_floor, floor_1 … from the rows of windows and the cornice belts, roof above.

Use **elements** as the regions pass of From ID Maps (split "*": every window, column … its own region, grouped
by class) and **sections floors** (plus shelves / facing / layers if wanted) as tag passes. Selectors everywhere
(clip to, holes, glow, plan rules): `group:windows`, `tag:left`, `tag:left group:windows` (space = and),
`tag:floor_1 group:windows`, `group:windows, group:columns` (comma = or). The director's clip to picker lists the
groups, the tags and these combinations. What "shelf_+0.30m" means: a depth slice 30 cm in front of the main wall
(the shelves pass; -0.46 m = 46 cm behind it). 3200x2160 test facade: 27 windows (7 arches, 7 + 7 upper, 2 attic, 4 dormers),
left / centre / right at the avant-corps, ground floor / floor 1 / floor 2 / roof.

## kubakub projector: where the projector stands

For a file without a camera, and for trying another spot or lens on a file that has one. Connect it to
`projector` of scene render; the file's camera is then not used. Set scene render's `width` / `height` to the
projector's resolution.

| input | |
|---|---|
| distance_m | metres from the wall. 0 = as far as the throw ratio needs to cover the model |
| offset_m | metres along the wall from the middle of the facade, + = to the right as the audience sees it |
| height_m | metres above the ground (the lowest point of the model): a tower, a balcony, a roof |
| throw_ratio | distance / picture width, as on a lens data sheet. 0 = the picture just covers the model |
| aim | **lens shift**: square to the wall, the picture shifted onto the model, verticals stay vertical. **tilt**: turned to the middle of the model (keystone). **straight**: square to the wall, no shift |
| side (advanced) | which side of the model is the facade: auto, or front / back / left / right (Blender's axes) |
| turn_deg (advanced) | the projector's direction turned around the building, for a spot at an angle to the facade |
| margin (advanced) | free border around the model when the picture is fitted |

The wall is the depth where most of the surface facing that side lies. The scene report says where the projector
ended up, its picture on the wall in metres, the lens, and how much of the model the picture covers when a given
throw ratio is too long. Everything after scene render (measure, preview, relight, pieces, compensation) uses
that projector. Logic and test: `kubakub/scene3d/autocam.py`, `tests/test_autocam.py`.

## kubakub projector blend: several projectors

One **kubakub projector** and one **kubakub scene render** per projector, all on the same file; connect their
`scene` outputs (2 to 4). Where two or more projectors reach the same surface, each one's share falls off towards
the border of its own picture over `ramp` (a share of the picture's short side), and the shares add up to 1 in
light; `gamma` (2.2) turns that into pixel values. A surface only one projector reaches (the side of a window
reveal) keeps that projector's full picture. Outputs: one mask per projector in its own picture size (multiply
your frames with it, or load it as the blend mask in the media server), a preview per projector (orange = shared)
and a report with the shares. The regions, plan and sampler run once per projector, on its own scene render.
Example: `example_workflows/projector_blend.json`.

## kubakub sample model

In the project menu: the sample facade as a 3D file in metres (an .obj in ComfyUI's temp folder), to try all of
this without a file. Stones in courses, arches with voussoirs, framed windows with glass behind the wall,
pilasters, sill bands, cornices and a door: about 640 loose parts, objects named by element (`wall_F1`,
`window_F2_C03`, `pilaster_04`), four materials. It has no camera. Scene render uses it by itself when its
`file` is empty (starters `07_3d_model_to_regions` and `08_moving_pieces`); the node is for other sizes. Test: `tests/test_sample_model.py`.

## kubakub scene measure: real distances

From the projection view's per-pixel world positions and normals:
| value | meaning |
|---|---|
| distance_m | projector to surface |
| incidence_deg | angle between the projector ray and the surface normal (0 = square on) |
| pixel_mm | size of one matrix pixel on the building (square on) |
| brightness | cos(incidence) / distance^2, relative to the typical square-on facade (1.0) |
| offset_m | offset from the main wall, + towards the audience |
| area_m2 | per region: real surface area |
| viewer_distance_m | per region: distance to the audience spot |

Outputs a colour map of one value, the **regions** with a `scene` entry per region (and the tag
`grazing` above `grazing_deg`, usable in plan rules), a tab-separated **table**, and a **viewer**
built from the projector frame on the main wall (real facade width and bottom edge) and your audience
spot (`viewer_x_m` from the frame centre, `viewer_distance_m`, `eye_height_m`) for frame guide / Frame
Compose. The uniform-scale viewer model is exact for a projector aimed square at the facade (lens shift,
no tilt), as in a festival template.

The test facade (file camera at ground level, 33 m from the wall): frame 33.35 x 22.51 m, ~11 mm per matrix pixel,
incidence 3-83 deg; sills and ledge tops get ~40 % of the wall's brightness; 1048 regions in 6 s.

## kubakub scene preview: previz from the audience

The matrix (one image or a batch of video frames) is projected from the file's camera onto the model
and rendered from audience spots, one per line of `spots`: `x from the frame centre, distance from the
wall, eye height` (m). Each spot is one Blender render (cached); every pixel of that view is projected
into the projector image and depth tested against the projection view, so surfaces the projector cannot
reach stay at `ambient` (projection shadow, also a MASK per spot). `physical` darkens steep / far
surfaces like a real projector. The lookup is computed once per spot, so frames cost one remap each.
Output batch: per spot all frames. 3200x2160 test facade: ~3 s per new spot; from 12-15 m to the side about 10-16 % of
the visible building is in projection shadow. Optional `background` (image or frames, looping) fills
everything behind the building, scaled to cover the view.

`clay_color` (a colour picker, almost white by default) is the colour of the model and its surroundings where no
picture lands. `shader` draws them as `clay`, as `wireframe` (the edges as lines in the clay colour on black, not
dimmed by `ambient`) or as `clay + wireframe`. The lines come from the passes the view already has (a new face, a
turn of the surface, a step in depth), so changing the shader or the colour renders nothing again. Scene
walkthrough has the same two inputs; scene relight and pieces render have `clay_color` (white = the neutral clay).

## kubakub scene walkthrough: previz video

The same previz along a camera path: `path` has one keyframe per line, `x, distance, eye[, look_x,
look_height]` (m, x / look_x along the wall from the frame centre); the keys are spread evenly over
`seconds` and joined by a smooth spline, the camera looks at the frame centre unless a look point is
given. Frame i of the walk shows matrix frame i (looping), so an animated matrix plays in sync; the
optional `background` (still or frames) sits behind the building. Output: IMAGE frames -> core Create
Video -> Save Video. Blender renders 25 camera positions per session (then the chunk is deleted from
temp). The 3200x2160 test facade at 960x540: 0.64 s per frame (150 frames in 96 s). Note: a loader that outputs a *list*
(Load Images From Dir With Names) needs core Rebatch Images before the matrix input, otherwise ComfyUI
runs the walkthrough once per image.

## kubakub scene relight: lamps, HDRI, the lit place

The model lit in Blender Cycles: an HDRI `environment` (Blender's built-in ones or your own `hdri_file`), lamps,
masks that glow, and your picture (input `matrix`) cast as light from the projector. Lamps are one line each in `lights`, in metres
from the facade: `point | area | spot  x  height  distance  watts  [#colour]  [size | angle]` (x along the wall
from the middle, height above the ground, distance from the wall towards the audience) or
`sun  azimuth  elevation  strength  [#colour]`.

- **The plan on the node.** Below its inputs the node shows the place from above, the audience at the bottom
  (after the first run: the real model, projector, audience camera and surroundings; before it a plain grid). Drag
  a lamp to move it along and away from the wall (Shift = whole metres), turn the mouse wheel over it for its
  height, click it for its watts, colour, size or angle; a sun is dragged around the rim to where it shines from.
  `+ point / area / spot / sun` add lamps, `remove` or Delete takes the selected one out. The plan writes the
  `lights` text and follows it when you type there, so the text stays the one truth. The node no longer shows its
  render on itself: connect a preview to `image`.
- **`camera = projector`** (default): the matrix view, the building alone: content, or a base for the director.
- **`camera = previz`**: the very viewpoint of kubakub scene preview: connect the `scene` output of scene preview
  to this node. Same spot, lens and picture size; `previz_spot` picks which line of its `spots`.
- **`camera = audience`**: from a spot in front of the building (`audience_distance_m`, `audience_offset_m`,
  `eye_height_m`, `lens_mm`), with its surroundings when kubakub scene surroundings is connected: the lit place as
  a visitor sees it. A high `eye_height_m` (20-60) gives the view from above. The numbers mean what a scene
  preview spot means (x along the wall, distance, eye height). On the node's plan the camera is the violet ring
  with its angle of view: drag the ring to move it, drag the dot in front of it to turn it (`audience_turn_deg`,
  0 = looking at the middle of the facade). With real ground (`terrain = real heights`) the eye height counts
  from the ground under the camera.
- **`live`** (the pill under the plan): every time you drop a lamp or the camera, or change one of the plan's
  fields, the workflow is queued again, so walking the camera around the square gives one picture after the
  other. Only what changed runs. For quick looks lower `resolution_scale` (0.3) and `samples` (16); set
  `plan_width_m` to keep the plan's scale fixed while you move far.
- **Too bright?** `projector_brightness` 1 lands the picture at about its own brightness, as in the previz;
  lamps add to it. Lower `exposure`, the lamps' watts or `env_strength`; the report says when a large part is
  burnt out to white. A projection only shows where the lamps leave the wall dark.
- **`plan`** (output): the light plan from above, the audience at the bottom. Every lamp in its colour with its
  number (its line in `lights`) and height, a spot with its aim, a sun as an arrow from where it shines; the model
  in orange, the projector as a dot, the audience camera as a ring, the surroundings in grey. Change a lamp's x or
  distance and watch it move on the plan.

## kubakub scene surroundings: the street around the building

A festival file is the building alone, so every previz shows it in the void. This node adds the neighbourhood as
plain boxes: put it between **scene render** and **scene preview / scene walkthrough / pieces render / scene
relight**. Example: `example_workflows/scene_surroundings.json`.

- **`location`**: the middle of your facade on a map, as `latitude, longitude` (Google Maps: right click on the
  wall, click the numbers to copy them; a link with `@lat,lon` works too). The buildings within `radius_m` come from
  OpenStreetMap. **Empty** = a sample square, so the node runs as it is and nothing is downloaded.
- **One download per place.** The node sends the coordinates and the radius to the OpenStreetMap server
  (overpass-api.de) once and keeps the answer in `ComfyUI/user/kubakub_cache/surroundings`; after that the place
  works offline, also for a smaller radius or a spot a few metres away. The map server is busy at times: the
  node asks three times, and if it still gets no answer it uses what is on this computer for that place and
  says so in the report. Downloads are **off by default**: write `surroundings_download = on` under `[settings]` in kubakub.ini to allow them (the node's error says so when a place is not on disk yet). Nothing else in the
  pack uses the network.
- **The fit is found on the map.** The footprint nearest to the location is the building itself: its wall there
  gives the compass direction of the facade, which is turned onto your model's facade, with the location on the
  middle of the model at ground level. `own_building = remove` leaves that footprint out (your model stands
  there); `keep` is for a model that is only the facade. If the fit is a little off: `turn_deg`, `along_m`,
  `out_m`, `lift_m`, or `facing_deg` to set the direction by hand.
- **Heights**: the map's height where it has one, else its level count x 3 m, else `default_height_m` (0 = the
  median of the buildings around that have one). In many towns most heights are unknown and every block gets
  the same height: good for sight lines and shadows, not for a skyline. For true roofs use your own file.
- **`terrain = real heights`**: the real ground of the place instead of a flat disc: hills, slopes, river banks.
  The heights come from the open terrain tiles on AWS (one more download per place, kept like the map), the
  buildings are set down on the ground and the audience cameras of the previz stand on it. The data is coarse
  (about 10-30 m between real measurements, and in a dense town it partly follows the roofs), so it shows the
  lie of the land, not kerbs or steps. That is why the ground stays level within `level_m` (40 m) of the facade
  and blends into the real heights beyond: your model stands on even ground. For a truer ground bring it in your
  own `file`.
- **`file`**: your own surroundings model instead of the map (.glb .fbx .obj .abc .usd ...), in the same space as
  the building: a city model, a Houdini export, a scan. `file_scale` brings it into metres. Not .blend.
- **Outputs**: `scene` (the same scene, now with surroundings), `plan` (top view, the audience at the bottom:
  surroundings grey, your model orange, the left-out footprint as a violet line, the projector as a dot), `view`
  (the clay model in its street, from above or from the street) and a report that warns when the projector
  stands inside a neighbouring building.

The surroundings are context only. They never become regions, IDs, pieces or glow, and the projection never lands
on them. In **scene relight** from the projector they stay out of the picture (it remains the matrix of the
building alone) but shade the building and bounce light; from an audience view (pieces render, `view = audience`)
they are in the picture, in the same clay as the building. A dense old town (about 280 buildings within 250 m): 3.5 s to download, under 1 s per
view more than without.

## kubakub brightness compensation: even light on the building

A projector lights the parts of the building that are near and face it brighter than far or turned-away parts
(light falls off with the square of the distance and with the angle). Put this node last, before the export: it
takes your finished frames and the scene from **kubakub scene render**, and darkens the bright parts so the content
reads equally bright everywhere. Works in linear light, 3200x2160 x 25 frames in about 4 s.

- `match`: **dim areas** = as even as the dimmer parts (most even), **average** = only the brightest parts come down.
- `strength`: 1 = fully even, 0.6 keeps a little of the natural falloff.
- `lift_dim_up_to`: 1 = never brighter than your content. Above 1 the dim parts are lifted as well; the report says
  how many pixels clip.
- `regions` (optional): one gain per region, so a window never gets a gradient inside.
- Side faces of mouldings hit at a grazing angle (`grazing_deg`, 60) are dim by nature and don't set the level;
  otherwise they would darken the whole building.

The `gain_map` output shows what changes. Example: the last group in `example_workflows/scene3d_to_regions.json`.

### Minimum piece size

`min_size_m` on kubakub scene render (default 0.5 m): every connected piece of every ID pass is measured
  across the wall (largest of its width along the wall and its height, depth ignored); pieces below the
  limit take the label of the neighbour they share the longest border with (pieces without a neighbour
  stay). 3200x2160 test facade: planes 3018 -> 1352 at 0.5 m / 535 at 1 m, parts 450 -> 389 / 226; at 1 m the arched
  windows become one surface each while the >1 m rusticated stones stay separate. ~3 s per pass.

## Moving pieces (kubakub/3d/pieces): a small MOPS

The facade model's own parts move: every loose part of the model (a stone, a window frame, a cornice block, a
dormer) is a piece. Build a chain like in MOPS for Houdini:

- **kubakub scene pieces**: the pieces of a scene render. `min_size_m` keeps small parts still, `max_size_m` big ones
  (whole wall panels), `keep_largest` the wall they sit on. The preview shows every piece in its own colour.
  With **regions** connected (regions from id maps, masks, cryptomatte ...) every part belongs to the region most
  of it is in: `regions: one piece each` moves every region as one block (a floor, a bay, all windows),
  `regions: parts inside` moves only the parts in the regions of `selection` (names, `group:windows`, `tag:left`).
- **kubakub pieces falloff**: how strongly a transform acts on each piece over time.
  *wave*: a front runs across the facade (`direction`, `speed` in m/s, a soft edge of `width` m) and the pieces it
  passed stay moved. *pulse*: a band of `width` m runs across; pieces move and come back. *stagger*: one piece after
  the other in an order (left to right, bottom to top, from the centre, random) over `spread` seconds, each taking
  `duration`; `hold` sends them back. *noise*: every piece drifts on its own. `loop` repeats a wave or pulse.
- **kubakub pieces beat falloff**: a kick on the beats, bars or markers of the director's timeline (connect the
  director's `document`): every piece jumps (`attack`) and settles (`decay`); `spread` lets the kick run across the
  facade in an order, like a ripple.
- **kubakub pieces transform**: push (out of the wall: along a flat piece's own face, out of the facade for closed
  stones), across / up / out (metres along the facade), tilt / turn / roll (degrees about each piece's centre),
  random rotate, scale, jitter. Weighted by the falloff; chain several, they add up.
- **kubakub pieces render**: the frames from the projection camera in Cycles (one Blender session, every frame kept:
  a change renders only what changed). *clay* = the model in light: the content to project. *projected matrix* =
  the matrix (one image or its frames) cast from the projector onto the moving pieces. Seen from the projector
  itself the picture stays in place and the movement shows as shadows and gaps; that is what the building does to it.
  `view = audience` renders from a spot in front of the building instead (`audience_distance_m`, `audience_offset_m`,
  `eye_height_m`, `lens_mm`, advanced): the depth of the moving pieces shows, the projector stays where it is.

The motion is computed in numpy (`kubakub/scene3d/pieces.py`, one matrix per piece and frame); Blender only
moves the vertices. On the 12 GB laptop: 451 pieces of a festival facade, 24 frames at 960x648 and 16 samples in
about 10 s (0.4 s per frame).

