"""Kub custom nodes for ComfyUI.

Each submodule is imported on its own. If one fails, the others still load and
the reason is printed once, instead of the whole pack disappearing from the
node menu with a single traceback in the console.
"""

import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"          # frontend of the kubakub director (served at /extensions/ComfyUI_KubaNodes/)

_MODULES = [                        # nodes/<menu group>/<module>
    "nodes.project.nodes_project",
    "nodes.project.nodes_sample",
    "nodes.regions.nodes_sources",
    "nodes.regions.nodes_vector",
    "nodes.regions.nodes_canvas",
    "nodes.regions.facade_mask_atlas_Kub",
    "nodes.sketch.nodes_sketch",
    "nodes.generate.nodes_plan",
    "nodes.generate.nodes_sampler",
    "nodes.generate.nodes_versions",
    "nodes.generate.nodes_frames",
    "nodes.director.nodes_director",
    "nodes.motion.nodes_video",
    "nodes.motion.nodes_keyframes",
    "nodes.motion.ltxv_trim_exact_Kub",
    "nodes.post.nodes_post",
    "nodes.scene3d.nodes_scene3d",
    "nodes.scene3d.nodes_pieces",
    "nodes.scene3d.nodes_compensate",
    "nodes.scene3d.nodes_surroundings",
    "nodes.from_renders.nodes_cryptomatte",
    "nodes.from_renders.nodes_passes",
    "nodes.fabricate.relief_forge_Kub",
    "nodes.fabricate.mesh_to_field_Kub",
    "nodes.fabricate.save_trimesh_Kub",
    "nodes.lab.mosaic_illusion",
    "nodes.lab.onnx_style_transfer_Kub",
    "nodes.utils.batch_image_loader_Kub",
    "nodes.utils.timecode_filename_Kub",
]

_loaded, _failed = [], []

for _name in _MODULES:
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
        _classes = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
        _names = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

        for _key in _classes:
            if _key in NODE_CLASS_MAPPINGS:
                print(f"[Kub] duplicate node id '{_key}' from {_name}, keeping the first")
        NODE_CLASS_MAPPINGS.update(_classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(_names)
        _loaded.append(_name)
    except Exception:
        _failed.append(_name)
        print(f"[Kub] failed to load {_name.split('nodes.', 1)[-1]}:")
        traceback.print_exc()

try:                              # kubakub.ini: switch whole menu groups off (e.g. "3d = off")
    from .menu_switches import apply as _apply_switches
    _apply_switches(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _e:
    print(f"[Kub] kubakub.ini ignored: {_e}")

print(f"[Kub] {len(NODE_CLASS_MAPPINGS)} nodes from {len(_loaded)} modules"
      + (f", {len(_failed)} failed: {', '.join(f.split('nodes.', 1)[-1] for f in _failed)}" if _failed else ""))

try:                              # what this install lacks (silent when everything is there)
    from .kub_env import report as _env_report
    _env_report()
except Exception as _e:           # the check must never cost the pack
    print(f"[Kub] environment check skipped: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
