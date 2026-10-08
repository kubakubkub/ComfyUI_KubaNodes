# Sketch tools: from paper to regions (2d / sketch)

**kubakub scan to line** takes a phone photo of a drawing. It finds the paper (or takes 4 corners, `tl tr br bl` in
photo pixels, when the table is lighter than the paper), straightens it to the matrix size (link width / height from
project settings; 0 = the paper's own size) and divides out paper colour, the lamp's gradient, shadows and grain.
`pencil` keeps the soft greys of the stroke, `ink` gives clean black. `clean` removes more grain and faint marks,
`boost` darkens light pencil. Outputs: the drawing on white (coloured strokes kept), the line as a mask, white lines on
black to project directly, the photo with the paper outline used, and the corners. A batch of photos (drawn
animation) is scanned frame by frame. Draw on a print of the festival template and the scan lands on it.
`report` has one line per photo: how the paper was found (found, given corners, or the whole photo), its size in
the photo, the size it was straightened to and the corners used, so a frame that went wrong is easy to spot.

**kubakub regions from sketch**: every closed shape becomes a region. Hand strokes rarely meet, so each stroke end
is extended in its own direction to the next line within `gap_px` (or joined to the nearest stroke end in front
of it at corners). A **dot of coloured marker** inside a shape names it: red, orange, yellow, green, cyan, blue,
purple or pink become its tag and name (`red_1`), so a rule `[tag:red]` picks it, and the dot is not a wall.
Shapes drawn alike land in one group. `outside = drop` leaves the paper around a drawn building out;
`min_region_area` merges hatching and thin strips.

| optional input | what it does |
|---|---|
| `matrix` | Your matrix: the preview background and the size of the regions. A drawing of another size is fitted to it (same proportions only; another aspect is refused: set width / height of scan to line to the matrix). `gap_px` stays in pixels of the drawing. |
| `scope` | Pixels outside this mask are never part of a region (the size of the matrix, or of the drawing without one). Works together with `outside = drop`. |

| output (after `regions`, `region_masks`, `regions_json`, `preview`) | what it is |
|---|---|
| `report` | Regions and groups found, shapes named by colour, gaps bridged, small shapes merged or dropped. |
| `closed_lines` | The lines the shapes were cut with: your strokes plus the bridges over the gaps (white = wall). Lay it over the drawing to see where `gap_px` closed a shape, or where one is still open. |

**kubakub line overlay** puts your line back on top of the render: `amount` 0.3 is a light trace of the hand, 1
exactly your stroke; `multiply` = dark line, `screen` = light line on dark projection content, `colour` paints it.
The `mask` output is the line as it was laid on (after `grow_px` and `soften_px`, before `amount`; one per frame),
for a glow, a cut-out or any other node that takes a mask.

Starter **05 sketch to render** runs the whole chain on the sample facade's `sketch_photo` (a synthetic phone photo
of a pencil sketch) with Klein 4B.
