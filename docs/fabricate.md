# Relief panels (3d / fabricate)

Turns generated facade elevations into relief panels that can actually be
built, and checks them against the rules of a public art competition before
anything is fabricated.

The point is not another image to 3D node. Image to 3D gives blobs. This keeps
the work as a **heightfield in millimetres**, which is what a CNC milled mould
wants, and it makes the rules of the call into parameters you can turn.

## The chain

```
[your LoRA] -> elevation image
        |
        +-> Depth Anything V2 / Marigold -> depth
                        |
      kubakub relief field (from depth)     panel_width_mm, max_relief_mm, panel_bottom_mm
                        |
      kubakub relief mould prep             min_edge_radius_mm, draft_angle_deg
                        |
      kubakub relief anti perch             sill_angle_deg
                        |
            +-----------+-----------+
            |           |           |
   Safety Check   Mass Estimate   Export Prow (OBJ) / Export Panel (STL)
```

Run Safety Check twice, once before Anti Perch and once after. The two overlays
side by side are a strong page in a submission: here is every surface a person
could stand on, and here is the same facade after the geometry fixed it.

## Nodes

**kubakub relief field (from depth)** normalises any depth map into millimetres. You
give the true panel width and the deepest relief; the pixel size follows. The
field carries its own scale from here on, so you never type a dimension twice.

**kubakub relief mould prep** does two things.
*Minimum edge radius* is a grayscale closing then opening with a spherical
structuring element, so a ball of that radius can touch the surface from both
sides. That is precisely what "no sharp or pointed elements" means to an
inspector, and what a ball nose cutter can reach.
*Draft angle* is an infimal convolution with a cone, which yields the largest
surface below yours that is Lipschitz with constant cot(draft). No undercuts,
one piece mould, clean release. The report tells you how much relief depth the
two operations cost you.

**kubakub relief anti perch** is the one that saves the entry. It applies the same
convolution but only upward, so the relief may grow outward as it descends and
never the other way. Vertical reveals and cornice undersides survive; the flat
top faces a foot needs are gone. On a test facade with protruding window sills
this took 220 flagged footholds to zero while keeping 113.9 of 115 mm relief
depth.

**kubakub relief safety check** flags, in red, every place where the relief steps
outward by more than `foothold_depth_mm` relative to everything within
`clearance_mm` above it, over a continuous horizontal run of at least
`min_width_mm`, below `reach_mm` from the ground. Orange is sitting
(150 mm deep between 300 and 750 mm), yellow is lying. Outputs an overlay
image, a mask, a JSON report and a boolean you can branch on.

**kubakub relief mass estimate** integrates the true relief surface rather than the
flat projection, multiplies by shell thickness and density, adds a frame
allowance, and divides by footprint. That last number is the one a call
usually caps.

**Relief Export Panel** writes a binary STL in millimetres.
**Relief Export Prow** folds one or two panels around a shared vertical edge at
a given interior angle and writes an OBJ. With a single field it mirrors it
onto both wings. Read the angle off the site plan. A `field_b` with another
pixel size (another image resolution or panel width) is resampled to
`field_a`'s, so each wing keeps its own size in millimetres.

**kubakub mesh to relief field** reads a finished mesh (OBJ; GLB, GLTF, PLY and
STL with trimesh) from one direction into the same millimetre field, with a
mask of the undercuts. **kubakub mesh turntable rule check** walks around the
mesh and runs the safety check on every side. Leave `mesh_path` empty on
either and the built-in sample facade runs (an OBJ in metres, Y up); the
report says so in `note`.

## Getting the relief out as a picture

The relief field only travels between the relief nodes. To use the relief
elsewhere (a depth or displacement map, a mask for a region), take `height`:

| Node | Output | What it is | Range in mm |
|---|---|---|---|
| kubakub relief field (from depth) | `height` (MASK) | the relief, 0 = the back, 1 = the highest point | 0 to `max_relief_mm` |
| kubakub relief mould prep | `height` (MASK) | the mould-ready relief, 0 = lowest, 1 = highest | `height_range_mm` in the report |
| kubakub relief anti perch | `height` (MASK) | the relief with sloped sills, 0 = lowest, 1 = highest | printed in the console |
| kubakub mesh to relief field | `height` (MASK) | the mesh seen from the chosen side, 0 = the back, 1 = nearest | `height_range_mm` in the report |

## Turntable check: what was added

| Name | Kind | Default | What it does |
|---|---|---|---|
| `clearance_mm` | input, optional (advanced) | 150 | band above a step, in mm, that it is measured against; the same setting as on relief safety check |
| `flagged` | output (MASK) | | one mask per view, white where a foothold, seat or lying surface was flagged |

## Defaults worth knowing

| Parameter | Default | Why |
|---|---|---|
| `min_edge_radius_mm` | 20 | Safe above the usual 2 mm code minimum, and easy to mill |
| `draft_angle_deg` | 4 | 3 is the practical floor for GFRC on a foam mould |
| `sill_angle_deg` | 45 | The standard anti perch slope |
| `foothold_depth_mm` | 25 | Roughly the smallest edge a shoe can bear on |
| `reach_mm` | 2500 | Above this an unaided person cannot start a climb |
| `density_kg_m3` | 2100 | GFRC. Concrete 2400, aluminium 2700, steel 7850 |

These are reasonable working values, not quoted from a standard. Before a
submission, have them confirmed by the structural engineer doing the technical
review, and treat the mass figure as an order of magnitude check rather than a
calculation: wind ballast is not included.

## Coordinates

Heightfield row 0 is the top of the panel. Local frame is x across, y up, z
outward. In the prow export the fold edge is the world Z axis, the tip points
towards -Y and the two wings open towards +Y.
