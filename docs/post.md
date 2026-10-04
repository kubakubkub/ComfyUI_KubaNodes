# Post (2d / post)

Everything works on one still or a batch of frames, before **export video / frames**. No models.

- **kubakub colour match**: your frames get the colours of a `reference` picture (any size, any content). The
  transform is fitted once (first, middle and last frame) and used for every frame, so nothing flickers. `mkl`
  moves colours and their mix, `mean_std` only brightness and contrast per channel. `save_lut` writes the look as
  `output/kubakub_luts/<name>.cube` for After Effects, Resolume, MadMapper, Resolve.
- **kubakub apply lut**: a `.cube` (1D or 3D) from `input/luts`, `models/luts` or `output/kubakub_luts`, or any
  path, with `strength`.
- **kubakub deflicker**: brightness and colour jumps between frames (diffusion video, timelapse) are evened out:
  the frame is split into `grid` x `grid` zones and each zone's colour is moved to its average over `window`
  frames. Slow fades and movement survive.
- **kubakub retime**: `speed` 0.5 = half speed with in-between frames from optical flow (OpenCV DIS), 2 = double
  speed; or an exact `frames` count. `blend` cross-fades, `nearest` repeats / drops frames.
- **kubakub burn in**: `{name} {frame} {timecode} {total}` on every frame for review copies (link `name` from
  project settings).
