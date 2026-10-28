# kubakub timeline (the master timeline)

The whole show on one line, always there: a tab of ComfyUI's bottom panel (**kubakub timeline**; open the bottom
panel with the panel button or its shortcut). It sits above the directors: every director of the workflow is a block
on it, and rendered clips and your own files go around and over them. The timeline is saved in the workflow.

## Project: length, fps, size

The bar at the top: **length** (seconds), **fps** and **size**. With **kubakub project settings** in the workflow,
fps and size are that node's (change them in either place); without it they are the timeline's own. fps is a whole
number, as in the director, and every director follows it: change it here or on project settings and the directors
take it over (one whose window is open says so; close it and press its *fps* pill in the list). A director block
with another fps shows it on the block.

## Directors and clips

The list next to the preview has three tabs:
- **directors**: the directors of the workflow, with their length and fps. Drag one onto the timeline; double click a
  block (or **open**) to open that director. The first time the panel sees a workflow it lays its directors one after
  another on v1.
- **renders**: videos, pictures and sounds in ComfyUI's output folder, newest first (refreshed after every run).
- **in this show**: the files the timeline uses. **+** imports files; dropping videos, pictures or sounds (wav, mp3, ogg, flac, m4a) from Explorer onto
  the list or the timeline works too. They are copied into input/kuba_director (the director's import folder), at
  any size; the same file dropped again is used again, not copied twice.

A preview dragged out of a node (a picture or video in ComfyUI's output) can be dropped onto the timeline as well.

## Editing

- **drag** a piece to move it, also to another track; it overwrites what it lands on (that piece is shortened or cut).
- **left grip** = trim in (a picture: its start), **right grip** = trim out, **Alt** + right grip = speed, **Shift** + right grip = length
  (longer than the clip = it holds its last frame or loops). The same as a video layer in the director.
- **S** or **Ctrl K** (or *cut*) cuts the selected pieces at the playhead (nothing selected: everything under it).
- **Del** removes, **Ctrl Z / Ctrl Y** undo / redo, **Ctrl A** selects everything, **Esc** cancels a drag.
- **snap**: edges snap to other pieces, the playhead, 0 and the end; hold **Ctrl** to move freely.
- **P** or **space** plays, **,** / **.** (or the arrows) one frame back / forward (Shift: 10), **Home / End**,
  **F** shows everything. Wheel = scroll, **Ctrl** + wheel = zoom, **Shift** + wheel = more tracks, middle drag = pan.
- Track heads: **m** mutes a track's sound, **◉** hides it; double click the name to rename, right click to remove.
  **+ track** adds one on top; upper tracks are in front.

The inspector on the right edits the selected piece: **at**, **in**, **out**, **speed**, **length**, **then**
(hold / loop), **backwards**, **sound**; and its **picture**: opacity, x / y (pixels of the show size), size and fit
(fit / fill / stretch). A director block has *at*, *in* (seconds into the director's timeline) and *length*.

## Sound

A video's own sound and sound files (their own pieces, blue) play with the preview, at the clip's speed. Clips with sound
show their waveform, following in / out, speed, hold and loop. **sound** in the inspector (or **m** on a track) turns
it off; a hidden track (◉) is not seen but still heard.

## The preview

Bottom left. The separators between preview, list, timeline and inspector can be dragged (double click = back to the
default); the widths are kept in this browser. Clips play from small preview copies (made once per file, the same ones the director uses), in time with
the playhead; it is a preview, close but not frame exact. A director block shows the director's last picture (run the
workflow once). A render node that writes the whole show at the project size and fps comes next.
