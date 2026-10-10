"""
nodes_align3d.py

kubakub align 3d model: a model from an image-to-3D workflow (Trellis, Hunyuan3D ...) is moved, turned and scaled onto
the real model of the building, so it sits where the projector and the camera expect it when it is opened in Blender
or Houdini. A scale per axis corrects a length the generator misread (mostly the depth). The fit is in
kubakub/meshalign.py (numpy + scipy); files are read and written with trimesh, which ComfyUI's own 3D nodes use.
V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
import torch

import folder_paths
from comfy_api.latest import io

from ...kubakub import meshalign as ma
from ...kubakub import optional
from ...kubakub import save_paths as sp
from ...kubakub.scene3d.bridge import clean_path

log = logging.getLogger("KUBA.scene3d")
NAME = "kubakub align 3d model"
READS = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")


def _load(path, what):
    """A model file as (trimesh scene, vertices [V, 3], faces [F, 3]) with every part where the file puts it."""
    trimesh = optional.need("trimesh", "align 3d model")
    if not os.path.isfile(path):
        raise ValueError(f"{NAME}: {what} '{path}' is not a file.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in READS:
        raise ValueError(f"{NAME}: {what} is a {ext or 'file without an ending'}; this node reads {', '.join(READS)}. "
                         "Export it from Blender or Houdini as .glb (or .obj) and give that file.")
    scene = trimesh.load(path, force="scene", process=False)
    parts = [g for g in scene.dump(concatenate=False) if hasattr(g, "faces") and len(g.vertices)]
    if not parts:
        raise ValueError(f"{NAME}: {what} '{os.path.basename(path)}' holds no surface (no triangles).")
    v, f, at = [], [], 0
    for g in parts:
        v.append(np.asarray(g.vertices, np.float64))
        f.append(np.asarray(g.faces, np.int64) + at)
        at += len(g.vertices)
    return scene, np.concatenate(v), np.concatenate(f)


def _from_mesh(mesh):
    """The first model of a MESH input as (vertices, faces)."""
    v, f = mesh.vertices, mesh.faces
    v = v[0] if v.dim() == 3 else v
    f = f[0] if f.dim() == 3 else f
    counts = getattr(mesh, "vertex_counts", None), getattr(mesh, "face_counts", None)
    if counts[0] is not None and counts[1] is not None:       # a padded batch: only what belongs to the first model
        v, f = v[:int(counts[0][0])], f[:int(counts[1][0])]
    return v.detach().cpu().numpy().astype(np.float64), f.detach().cpu().numpy().astype(np.int64)


def _sample():
    """Nothing connected: the sample facade as the template, and the same building turned, moved, shrunk and a
    quarter too deep as the 'generated' model."""
    from ...kubakub import sample_model as sm
    m, _info = sm.build()
    v = np.asarray(m.verts, np.float64)
    f = np.asarray([[face[0], face[k], face[k + 1]] for _n, _mat, faces in m.objects for face in faces
                    for k in range(1, len(face) - 1)], np.int64)
    a = np.radians(37.0)
    turn = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    lost = (v * np.array([1.0, 1.0, 1.25])) @ turn.T * 0.031 + np.array([4.0, -2.5, 7.0])
    return (v, f), (lost, f)


class KUBA_AlignModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KUBA_AlignModel",
            display_name=NAME,
            category="kubakub/3d/scene",
            search_aliases=["trellis", "hunyuan3d", "image to 3d", "register", "icp", "fit mesh", "glb", "align mesh"],
            description=(
                "Puts a generated 3D model onto the real one. A model from an image-to-3D workflow comes back "
                "somewhere in space, at some size, often turned. Give the real model of the building as the "
                "template and the generated one is moved, turned and scaled to lie on it, so it opens in Blender "
                "or Houdini exactly where the projector and the camera expect it. 'per axis' also corrects a "
                "length the generator misread (mostly the depth). Writes the aligned model as .glb and gives the "
                "matrix. Nothing connected = a sample."),
            inputs=[
                io.Mesh.Input("model", optional=True,
                              tooltip="The generated model, from an image-to-3D node. Or give a file in model_file."),
                io.Mesh.Input("template", optional=True,
                              tooltip="The real model it should lie on. Or give a file in template_file."),
                io.String.Input("model_file", default="", optional=True,
                                tooltip="The generated model as a file (.glb, .gltf, .obj, .ply, .stl). Its materials "
                                        "and textures are kept in the aligned file."),
                io.String.Input("template_file", default="", optional=True,
                                tooltip="The real model as a file (.glb, .gltf, .obj, .ply, .stl), in the place and at "
                                        "the size your scene uses. From Blender or Houdini: export as .glb."),
                io.Combo.Input("scale", options=list(ma.SCALES), default=ma.SCALES[0],
                               tooltip="per axis: a scale for each of the template's axes, so a misread depth or "
                                       "height is corrected. one scale: the model keeps its proportions. keep the "
                                       "size: only moved and turned."),
                io.Boolean.Input("mirror", default=False,
                                 tooltip="Also try mirrored fits. Turn it on when the result sits right but left and "
                                         "right are swapped."),
                io.String.Input("save_name", default="aligned_model", optional=True,
                                tooltip="Name of the .glb written to the output folder (3d/<name>_00001.glb)."),
                io.Int.Input("points", default=40000, min=2000, max=200000, step=1000, optional=True, advanced=True,
                             tooltip="Points taken from the generated model for the fit (five times as many from the "
                                     "template). More = slower and more exact; 40000 takes a few seconds."),
                io.Int.Input("seed", default=0, min=0, max=2 ** 31 - 1, optional=True, advanced=True,
                             control_after_generate=False, tooltip="Which points are taken."),
            ],
            outputs=[
                io.Mesh.Output("model", tooltip="The generated model, lying on the template."),
                io.String.Output("file", tooltip="The aligned .glb: open it in Blender or Houdini, it is in place."),
                io.String.Output("matrix", tooltip="The 4x4 matrix that takes the generated model to the template, row "
                                                   "by row, for a transform node of your own."),
                io.String.Output("report", tooltip="How far off it was, what was done, how close it is now, and which "
                                                   "length had been misread."),
            ],
        )

    @classmethod
    def execute(cls, model=None, template=None, model_file="", template_file="", scale=ma.SCALES[0], mirror=False,
                save_name="aligned_model", points=40000, seed=0) -> io.NodeOutput:
        notes, scene = [], None
        mf_, tf_ = clean_path(model_file), clean_path(template_file)
        if model is None and not mf_ and template is None and not tf_:
            (tv, tfc), (mv, mfc) = _sample()
            notes.append("nothing connected: a sample is used (the sample facade as the template, and the same building "
                         "turned 37°, moved, shrunk and a quarter too deep as the generated model)")
        else:
            if model is not None:
                mv, mfc = _from_mesh(model)
                if mf_:
                    notes.append("model and model_file are both given: the connected model is used")
            elif mf_:
                scene, mv, mfc = _load(mf_, "model_file")
            else:
                raise ValueError(f"{NAME}: connect the generated model to 'model' or give its file in model_file.")
            if template is not None:
                tv, tfc = _from_mesh(template)
                if tf_:
                    notes.append("template and template_file are both given: the connected template is used")
            elif tf_:
                _s, tv, tfc = _load(tf_, "template_file")
            else:
                raise ValueError(f"{NAME}: the real model is missing. Connect it to 'template' or give its file in "
                                 "template_file (export it from Blender or Houdini as .glb).")
        scale = scale if scale in ma.SCALES else ma.SCALES[0]
        try:
            res = ma.align(*ma.sample_pair((mv, mfc), (tv, tfc), points, seed), scale, bool(mirror), seed=int(seed))
        except ValueError as e:
            raise ValueError(f"{NAME}: {e}") from e
        M = res["matrix"]
        moved = ma.apply(M, mv)

        trimesh = optional.need("trimesh", "align 3d model")
        folder = os.path.join(folder_paths.get_output_directory(), "3d")
        os.makedirs(folder, exist_ok=True)
        stem = sp.file_stem(save_name, "aligned_model")
        n = 1
        while os.path.exists(os.path.join(folder, f"{stem}_{n:05d}.glb")):
            n += 1
        out_path = os.path.join(folder, f"{stem}_{n:05d}.glb")
        if scene is not None:
            # the file's own parts, materials and textures, each with the fit written into its points: a scale per axis
            # after a turn is a shear, which a .glb's part cannot carry as its own move / turn / scale
            baked = trimesh.Scene()
            for name in scene.graph.nodes_geometry:
                world, key = scene.graph[name]
                part = scene.geometry[key].copy()
                part.apply_transform(M @ world)
                baked.add_geometry(part, node_name=name, geom_name=key if key not in baked.geometry else None)
            baked.export(out_path)
        else:
            trimesh.Trimesh(vertices=moved, faces=mfc, process=False).export(out_path)

        if model is not None:                          # every model of a batch, with whatever else the MESH carries
            out = copy.copy(model)
            v = model.vertices
            flat = v.reshape(-1, 3).detach().cpu().numpy().astype(np.float64)
            out.vertices = torch.from_numpy(ma.apply(M, flat).astype(np.float32)).reshape(v.shape).to(v.device)
        else:
            out = io.Mesh.Type(torch.from_numpy(moved.astype(np.float32))[None], torch.from_numpy(mfc.astype(np.int64))[None])
        rows = "\n".join(", ".join(f"{x:.8g}" for x in row) for row in M)
        report = "\n".join([f"model {len(mv)} vertices, template {len(tv)} vertices, scale = {scale}"] + ma.describe(res)
                           + [f"written: {out_path}"] + notes)
        log.info("[KUBA align 3d] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(out, out_path, rows, report)


NODE_CLASS_MAPPINGS = {"KUBA_AlignModel": KUBA_AlignModel}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_AlignModel": NAME}
