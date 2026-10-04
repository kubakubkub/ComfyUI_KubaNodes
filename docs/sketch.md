# Sketch tools: from paper to regions (2d / sketch)

**kubakub scan to line** takes a phone photo of a drawing. It finds the paper (or takes 4 corners, `tl tr br bl` in
photo pixels, when the table is lighter than the paper), straightens it to the matrix size (link width / height from
project settings; 0 = the paper's own size) and divides out paper colour, the lamp's gradient, shadows and grain.
`pencil` keeps the soft greys of the stroke, `ink` gives clean black. `clean` removes more grain and faint marks,
`boost` darkens light pencil. Outputs: the drawing on white (coloured strokes kept), the line as a mask, white lines on
black to project directly, the photo with the paper outline used, and the corners. A batch of photos (drawn
animation) is scanned frame by frame. Draw on a print of the festival template and the scan lands on it.

**kubakub regions from sketch**: every closed shape becomes a region. Hand strokes rarely meet, so each stroke end
is extended in its own direction to the next line within `gap_px` (or joined to the nearest stroke end in front
of it at corners). A **dot of coloured marker** inside a shape names it: red, orange, yellow, green, cyan, blue,
purple or pink become its tag and name (`red_1`), so a rule `[tag:red]` picks it, and the dot is not a wall.
Shapes drawn alike land in one group. `outside = drop` leaves the paper around a drawn building out;
`min_region_area` merges hatching and thin strips.

**kubakub line overlay** puts your line back on top of the render: `amount` 0.3 is a light trace of the hand, 1
exactly your stroke; `multiply` = dark line, `screen` = light line on dark projection content, `colour` paints it.

Starter **05 sketch to render** runs the whole chain on the sample facade's `sketch_photo` (a synthetic phone photo
of a pencil sketch) with Klein 4B.
