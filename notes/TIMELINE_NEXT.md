# kubakub timeline: what is open (working note, 2026-10-28)

Delete this file before dev goes into main. Done so far on dev: the panel (0ddc519), the render node and keys
(b88a623); see docs/master.md. Tests: tests/test_master.py, tests/test_master_render.py, tests/test_media.py.

## 1. Click through the panel in a browser (never done yet)

Only the model is tested (node + Python); the UI was never clicked. Open the bottom panel, tab *kubakub timeline*:
- drag a director, a render and a file from Explorer onto it; move, trim (both grips), Alt = speed (not on
  pictures), Shift = length, S / Ctrl K cut, Del, Ctrl Z / Y
- keys: ◆ next to opacity / x / y / size, edit at another time = a second key, diamonds on the piece (click = go
  there), keys ‹ ›, ease pills; the preview plays the keys; a cut keeps the keys in time
- the inspector does not rebuild while playing and shows the keyed values after pause
- a *kubakub timeline render* node: queue, edit the timeline, queue again: it must render again (the hidden
  'timeline' input is filled at queue time by serializeValue in web/kubakub_master.js)
- connect a director's 'director' output to the render node: its block is drawn; unconnected = in the report

## 2. probe gives the average fps for a constant-rate file with a dropped frame

kubakub/director/media.py probe: a 24 fps file with one dropped frame reports 23.83 (and is not flagged vfr), so
decode lays its grid at 23.83 and drifts. Affects the director and the timeline. Idea: when the stream's r_frame_rate
(tbr) is a clean rate and avg differs only by the missing frames, use tbr. A test file: make 24 fps frames with
ffmpeg, drop frame 40 with select='not(eq(n\,40))' and -fps_mode passthrough.
(The seek fix of b88a623, start a quarter frame early, is right for these files once the fps is right.)

## 3. Long shows

- The render node holds every frame in RAM: frames x width x height x 12 bytes (1 min full HD at 25 fps = 37 GB at
  scale 1). Idea: render straight into a video file in pieces (export.py), for delivery without the IMAGE output.
- Video clips decode in ~512 MB batches (kubakub/master_render.py CHUNK_BYTES), each an ffmpeg start with ~4 s
  pre-roll. One streaming ffmpeg per clip (Popen, read frame by frame) in media.py would cut that.

## 4. Smaller things

- Sound at a speed other than 1 is left out (director, render and now the preview too). ffmpeg atempo would let it
  play (director: media.layer_audio; timeline: master_render._add_sound).
- Director blocks inside subgraphs: the panel lists root-graph directors only, the render matches blocks by the
  node's unique_id (DirectorState.uid). Subgraph support needs the execution-path id stored on the block.
- Keys could also take rotation; director blocks already take the picture keys.
- tests/test_tracking.py fails on the desktop PC (with or without the timeline): "yolo: none of ['person'] is a
  kind this model knows" - the YOLO model found there is not the person one. Check it passes on the laptop.

## Ideas not started

- wire cutter: Y + drag across links cuts them (KJNodes already has shake to disconnect, off by default).
