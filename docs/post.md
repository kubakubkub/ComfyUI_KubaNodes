# Post (2d / post)

Everything works on one still or a batch of frames, before **export video / frames**. No models.

- **kubakub align to source**: puts an edited picture back onto the picture it was made from. An image model that
  repaints a whole facade (versions with *whole picture*, a Qwen or Klein edit in any workflow) returns it a little
  moved and scaled: measured here, 3 to 30 px at the corners of a 1.5 k picture, one to two percent in scale. On a
  building that is a visible offset. `images` = the edit, `source` = the clay, matrix or render it came from; the
  result has the source's size. The fit starts from "nothing moved" and looks only for a small shift, scale and
  shear on the edges both pictures share, so rows of identical windows cannot send it to the wrong window. It is
  refused, and the picture returned as it is, when the pictures have too little in common, when the fit asks for
  more than a drift (4 % of the long edge, 5 % scale), or when the same fit from other starting points ends
  somewhere else; the report says which. `fit = move only` for a plain shift; for a clip, `frames = one fit for
  all` (advanced) gives every frame the first frame's correction. About 3 s per picture. It corrects the picture
  as a whole: parts that moved differently from each other stay as they are. Not needed after the region sampler,
  which pastes inside masks. kubakub versions does this by itself for `generate = whole picture`.
  Test: `tests/test_align.py`.
- **kubakub colour match**: your frames get the colours of a `reference` picture (any size, any content). The
  transform is fitted once (first, middle and last frame) and used for every frame, so nothing flickers. `mkl`
  moves colours and their mix, `mean_std` only brightness and contrast per channel. `save_lut` writes the look as
  `output/kubakub_luts/<name>.cube` for After Effects, Resolume, MadMapper, Resolve. `strength` 1 = the whole
  look, 0 = the frames as they are.
- **kubakub apply lut**: a `.cube` (1D or 3D) from `input/luts`, `models/luts` or `output/kubakub_luts`, or any
  path, with `strength` (1 = the whole look, 0 = the frames as they are).
- **kubakub deflicker**: brightness and colour jumps between frames (diffusion video, timelapse) are evened out:
  the frame is split into `grid` x `grid` zones and each zone's colour is moved to its average over `window`
  frames. Slow fades and movement survive. `strength` 1 takes all of the jumps out. It needs at least 3 frames;
  fewer come back unchanged, with a line in the console.
- **mask** (optional, on colour match, apply lut and deflicker): the change happens only inside the mask, white =
  full, grey = partly; outside the frames stay as they are. One mask serves every frame, a batch of masks goes
  frame by frame, and a mask of another size is resized to the frames. The `.cube` of `save_lut` is always the
  whole look.
- **kubakub retime**: `speed` 0.5 = half speed with in-between frames from optical flow (OpenCV DIS), 2 = double
  speed; or an exact `frames` count. `blend` cross-fades, `nearest` repeats / drops frames. Give it the `fps` of
  your frames (optional, 0 = unknown) and the outputs `fps` and `seconds` say how to play the result and how long
  it is. The rule: by `speed` the fps stays the same, the slow or fast motion is in the frames; by an exact
  `frames` count the fps is fps x frames out / frames in, so the clip lasts as long as before. Without an fps both
  are 0.
- **kubakub burn in**: `{name} {frame} {timecode} {total}` on every frame for review copies (link `name` from
  project settings). `start_frame` is the number of the first frame, `fps` is for the timecode, `position`,
  `size` (share of the frame height) and `opacity` place the text box.
