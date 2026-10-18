# Live link: Blender and Houdini

With the live link, **kubakub scene render** takes the scene that is open in Blender or Houdini right now instead of
a file: you model, move the view, press **refresh** on the node and see the new state. How the node uses it
(cameras, frame size, caching) is in [3d scene](scene3d.md#live-from-blender-or-houdini-model-press-refresh).

The link is files in one local folder (default `kubakub_live` in your user folder): no network, no port, nothing
leaves your computer. Both add-ons are in this pack, in the `tools` folder:

| app | file | tested with |
|---|---|---|
| Blender 4.2 or newer | `tools/blender/kubakub_live.py` | Blender 4.5 |
| Houdini | `tools/houdini/` (installed with `tools/houdini/install_houdini_link.py`) | Houdini 21 |

Below, `<pack>` is the folder of this pack, e.g. `ComfyUI\custom_nodes\ComfyUI_KubaNodes` (the portable build:
`ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_KubaNodes`).

## Blender

**Install (once)**
1. Open Blender. **Edit > Preferences > Add-ons**.
2. Top right: the **v** menu > **Install from Disk...** and pick `<pack>\tools\blender\kubakub_live.py`.
3. Tick **kubakub live** in the list (search for *kubakub* if you don't see it). Preferences save by themselves
   (or **Save Preferences** in the menu at the bottom left).

**Check it works**
- In a 3D view press **N**: the sidebar has a **kubakub** tab with the link state (*waiting for ComfyUI*, later
  *sent 14:05:12*), the live folder and a **Send to ComfyUI now** button.
- In ComfyUI, on **kubakub scene render**: `source` = **live blender**. The `file` field hides, a **refresh from
  blender** button appears. Press it: the preview shows your scene, the report's first line starts with *live from Blender*.

**Use**
- Model, then press **refresh from blender**. Blender saves a *copy* of the scene into the live folder; your own
  .blend is not saved, renamed or changed.
- **camera: ... ▾** on the node lists the cameras of the open scene, or *what you look through* (the viewport view).

**Update** after updating this pack: install the add-on again the same way (when a picked camera needs a newer
add-on, the node's report says so). **Remove**: Preferences > Add-ons > kubakub live > **Uninstall** (or untick it to
switch it off for now).

## Houdini

**Install (once)**
1. Close Houdini.
2. Run the installer with the Python of your ComfyUI. Portable build, from the `ComfyUI_windows_portable` folder:
   ```
   python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_KubaNodes\tools\houdini\install_houdini_link.py
   ```
   Any Python 3 works (it uses the standard library only): `python <pack>\tools\houdini\install_houdini_link.py`.
3. It lists the Houdini versions found in your Documents folder (newest first) and asks for each one. It writes
   one file, `Documents\houdiniXX.Y\packages\kubakub_live.json`, which points Houdini at `<pack>\tools\houdini`.
   Nothing else changes.
4. Start Houdini. Shelf area: the **+** tab > **Shelves** > tick **kubakub**. The shelf has **comfy link** and
   **send now**.

**Check it works**
- Press **comfy link** on the kubakub shelf. The status bar at the bottom says *kubakub live: linked to ComfyUI through ...*.
- In ComfyUI, on **kubakub scene render**: `source` = **live houdini**, press **refresh from houdini**. The preview
  shows what /obj displays at the current frame, the report's first line starts with *live from Houdini*.

**Use**
- Press **comfy link** once per Houdini session (press it again to stop). Then refresh from the node whenever you
  want. Houdini writes the displayed geometry as Alembic into the live folder (hidden objects are left out; your
  .hip is not saved). The export uses a temporary Alembic ROP in /out that is removed again; Houdini may still mark
  the .hip as changed.

**Update** after updating this pack: just restart Houdini (the package points at the pack folder, so it always
loads the current version). Run the installer again only if you moved the pack.
**Remove**: run the installer with `--remove`; it deletes the package file again.
**Several Houdini versions**: the installer asks for each; `--yes` installs into all of them without asking.

## Live folder

Default `~/kubakub_live` (your user folder). To use another folder, set it in all three places to the same path:
- ComfyUI: `live_folder = ~/projects/kubakub_live` under `[settings]` in `kubakub.ini` (in the pack folder; copy
  `kubakub.ini.example` if you have none),
- Blender: Preferences > Add-ons > kubakub live > **Live folder**,
- Houdini: run the installer again after changing `kubakub.ini`; it copies the setting into the package.

## When it does not work

| the report says / you see | what to do |
|---|---|
| *Blender (Houdini) is not linked right now* | Blender: is the add-on ticked? Houdini: press **comfy link**. Both: same live folder in every place (above). |
| no **kubakub** shelf in Houdini | **+** tab > Shelves > kubakub. Not in the list: run the installer again and restart Houdini; check that `Documents\houdiniXX.Y\packages\kubakub_live.json` exists for the version you start. |
| *did not answer in 60 s (busy rendering?)* | Blender or Houdini was rendering or cooking. The last copy is used; press refresh again when it is free. The wait is `live_wait` in `kubakub.ini` (seconds). |
| *this ... link is older and cannot pick a camera* | Blender: install the add-on again. Houdini: restart it. |
| the picture looks wrong (scale) | `unit_scale` 0 (auto) on the node finds centimetres / millimetres scenes; the report says what it used. |
| the scene is the sample building | nothing was sent yet: the app was closed or not linked when you pressed refresh. |

The model-free test of the link is `tests/test_live.py` (fake apps answer through a temporary folder).

Cinema 4D: not yet.
