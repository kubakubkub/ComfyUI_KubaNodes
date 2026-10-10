"""
nodes_align3d.py

kubakub align 3d model: a model from an image-to-3D workflow (Trellis, Hunyuan3D ...) is moved, turned and scaled onto
the real model of the building, so it sits where the projector and the camera expect it when it is opened in Blender
or Houdini. It is written as .glb or as .obj with its textures, in the template's coordinates and size. A scale per axis corrects a length the generator misread (mostly the depth). A model made from a picture
of a facade is a whole invented building; it is fitted by its front alone and the invented back can be cut away. The
fit is in kubakub/meshalign.py (numpy + scipy); files are read and written with trimesh, which ComfyUI's own 3D nodes
use.
V3 node schema, registered via ../__init__.py.
"""

from __future__ import annotations

import copy
import logging
import os
from io import BytesIO

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
FORMATS = (".glb", ".obj")
MAPS = (("metallicRoughnessTexture", "roughness_metal"), ("normalTexture", "normal"), ("emissiveTexture", "emissive"))


def _load(path, what, name=NAME):
    """A model file as (trimesh scene, vertices [V, 3], faces [F, 3]) with every part where the file puts it. name: the
    node that asks (kubakub form height reads its model here too)."""
    trimesh = optional.need("trimesh", name.replace("kubakub ", ""))
    if not os.path.isfile(path):
        raise ValueError(f"{name}: {what} '{path}' is not a file.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in READS:
        raise ValueError(f"{name}: {what} is a {ext or 'file without an ending'}; this node reads {', '.join(READS)}. "
                         "Export it from Blender or Houdini as .glb (or .obj) and give that file.")
    scene = trimesh.load(path, force="scene", process=False)
    parts = [g for g in scene.dump(concatenate=False) if hasattr(g, "faces") and len(g.vertices)]
    if not parts:
        raise ValueError(f"{name}: {what} '{os.path.basename(path)}' holds no surface (no triangles).")
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


def _carries(mesh):
    """Does a MESH hold more than its shape (UVs with a texture, or colours on its points)?"""
    return mesh.vertices.dim() == 3 and (getattr(mesh, "vertex_colors", None) is not None
                                         or (getattr(mesh, "uvs", None) is not None and getattr(mesh, "texture", None) is not None))


def _turn_normals(mesh, M):
    """The normals and tangents of a MESH follow the fit (they are directions: no move, and a scale per axis bends them)."""
    lin = torch.from_numpy(M[:3, :3].astype(np.float64))
    n, t = getattr(mesh, "normals", None), getattr(mesh, "tangents", None)
    if n is not None:
        mesh.normals = torch.nn.functional.normalize(n.cpu().double() @ torch.linalg.inv(lin), dim=-1).to(n.dtype).to(n.device)
    if t is not None:
        xyz = torch.nn.functional.normalize(t[..., :3].cpu().double() @ lin.T, dim=-1)
        side = t[..., 3:].cpu().double() * (1.0 if torch.linalg.det(lin) > 0 else -1.0)
        mesh.tangents = torch.cat([xyz, side], -1).to(t.dtype).to(t.device)


def _cut_faces(mesh, faces, cut):
    """mesh keeps only the faces of `faces` ([F, 3] or [1, F, 3]) that `cut` marks; its points stay (colours and UVs
    hang on them)."""
    keep = torch.from_numpy(cut).to(faces.device)
    mesh.faces = faces[..., :len(cut), :][..., keep, :]
    if getattr(mesh, "face_counts", None) is not None:
        mesh.face_counts = torch.tensor([int(cut.sum())], dtype=mesh.face_counts.dtype)


def _glb_of(mesh, notes):
    """The first model of a MESH as the bytes of a .glb with all it carries (UVs, colours, the texture and its maps),
    from ComfyUI's own writer. None when that writer cannot be had; `notes` then says so."""
    try:
        from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
        return mesh_item_to_glb_bytes(mesh, 0)
    except Exception as e:                                # the writer is ComfyUI's and may change: the shape is still written
        notes.append(f"the model's texture and colours could not be read ({e}): the file holds the shape only")
        return None


def _write(model, path):
    """model (a trimesh mesh or scene) -> path. A .glb is one file. An .obj gets its .mtl and its colour texture beside
    it, named after it; the maps an .obj material cannot name (roughness / metal, normal, glow) are written beside it as
    PNGs. Returns every file written."""
    if not path.lower().endswith(".obj"):
        model.export(path)
        return [path]
    folder, base = os.path.split(path[:-4])
    before = set(os.listdir(folder))
    parts = list(model.geometry.values()) if hasattr(model, "geometry") else [model]
    seen = {}
    for part in parts:
        mat = getattr(getattr(part, "visual", None), "material", None)
        if mat is None or getattr(part.visual, "uv", None) is None:
            continue
        try:
            key = hash(mat)                               # parts are copies: the same material comes as several objects
        except Exception:  # noqa: BLE001
            key = id(mat)
        if key not in seen:
            name = base if not seen else f"{base}_{len(seen) + 1}"
            for attr, word in MAPS:
                img = getattr(mat, attr, None)
                if img is not None:
                    img.save(os.path.join(folder, f"{name}_{word}.png"))
            plain = mat.to_simple() if hasattr(mat, "to_simple") else mat      # what an .mtl can say; its name names the texture
            plain.name = name
            seen[key] = plain
        part.visual.material = seen[key]
    model.export(path, mtl_name=base + ".mtl")
    return [path] + sorted(os.path.join(folder, f) for f in set(os.listdir(folder)) - before if f != os.path.basename(path))


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
                "length the generator misread (mostly the depth). A model made from a picture of a facade comes "
                "back as a whole invented building: it is fitted by its front, and the invented back, sides and "
                "roof are cut away. Writes the aligned model as .glb or as .obj with its textures, in the "
                "template's coordinates and size, and gives the matrix. Nothing connected = a sample."),
            inputs=[
                io.Mesh.Input("model", optional=True,
                              tooltip="The generated model, from an image-to-3D node, with its texture or colours. Or give "
                                      "a file in model_file."),
                io.Mesh.Input("template", optional=True,
                              tooltip="The real model it should lie on. Or give a file in template_file."),
                io.String.Input("model_file", default="", optional=True,
                                tooltip="The generated model as a file (.glb, .gltf, .obj, .ply, .stl). Its materials "
                                        "and textures are kept in the aligned file."),
                io.String.Input("template_file", default="", optional=True,
                                tooltip="The real model as a file (.glb, .gltf, .obj, .ply, .stl), in the place and at "
                                        "the size your scene uses. From Blender or Houdini: export as .glb."),
                io.Combo.Input("fit", options=list(ma.FITS), default=ma.FITS[0],
                               tooltip="the facade front: for a model made from a picture of a facade. Only what is "
                                       "seen from the front is laid onto the template (the outline, then the wall), "
                                       "so an invented back wall and roof do not pull, and what stands in front of "
                                       "the wall keeps its depth. The model must be upright and seen from the front, "
                                       "as image to 3D writes it. the whole model: surface onto surface, for a model "
                                       "that shows the same as the template, turned any way. auto: the facade front "
                                       "when the template is a thin relief and the model is far deeper."),
                io.Combo.Input("scale", options=list(ma.SCALES), default=ma.SCALES[0],
                               tooltip="per axis: a scale for each of the template's axes, so a misread depth or "
                                       "height is corrected. one scale: the model keeps its proportions. keep the "
                                       "size: only moved and turned."),
                io.Boolean.Input("mirror", default=False,
                                 tooltip="Also try mirrored fits. Turn it on when the result sits right but left and "
                                         "right are swapped. Not used with the facade front."),
                io.Boolean.Input("cut_the_back", default=True,
                                 tooltip="With the facade front: everything behind the template's back is cut away (the "
                                         "invented back wall, sides and roof). The facade and what stands in front of "
                                         "it stay. Off = the whole generated model is kept."),
                io.String.Input("save_name", default="aligned_model", optional=True,
                                tooltip="Name of the file written to the output folder (3d/<name>_00001.glb or .obj)."),
                io.Int.Input("points", default=40000, min=2000, max=200000, step=1000, optional=True, advanced=True,
                             tooltip="Points taken from the generated model for the fit (five times as many from the "
                                     "template). More = slower and more exact; 40000 takes a few seconds."),
                io.Int.Input("seed", default=0, min=0, max=2 ** 31 - 1, optional=True, advanced=True,
                             control_after_generate=False, tooltip="Which points are taken."),
                io.Combo.Input("file_format", options=list(FORMATS), default=FORMATS[0], optional=True,
                               tooltip=".glb: one file with the textures inside, for Blender (File > Import > glTF) and "
                                       "Houdini (File > Import > glTF Scene, or a glTF Hierarchy node). .obj: the .obj, "
                                       "its .mtl and the colour texture as a PNG beside it, for any 3D software; the "
                                       "roughness / metal and normal maps are written beside it too. Both hold the model "
                                       "in the template's coordinates and size: import it without a transform."),
            ],
            outputs=[
                io.Mesh.Output("model", tooltip="The generated model, lying on the template."),
                io.String.Output("file", tooltip="The aligned file (.glb or .obj): import it in Blender or Houdini, it is "
                                                 "in place."),
                io.String.Output("matrix", tooltip="The 4x4 matrix that takes the generated model to the template, row "
                                                   "by row, for a transform node of your own."),
                io.String.Output("report", tooltip="How far off it was, what was done, how close it is now, and which "
                                                   "length had been misread."),
            ],
        )

    @classmethod
    def execute(cls, model=None, template=None, model_file="", template_file="", fit=ma.FITS[0], scale=ma.SCALES[0],
                mirror=False, cut_the_back=True, save_name="aligned_model", points=40000, seed=0,
                file_format=FORMATS[0]) -> io.NodeOutput:
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
        ext = file_format if file_format in FORMATS else FORMATS[0]
        fit = fit if fit in ma.FITS else ma.FITS[0]
        try:
            pair = ma.sample_pair((mv, mfc), (tv, tfc), points, seed)
            front = fit == ma.FITS[1] or (fit == ma.FITS[0] and ma.looks_like_a_front(*pair))
            if front:
                if fit == ma.FITS[0]:
                    notes.append("fit = auto: the template is a thin relief and the model is far deeper, so the facade "
                                 "front is fitted")
                if mirror:
                    notes.append("mirror is not used with the facade front")
                dense = ma.FRONT_DENSITY * int(points)
                res = ma.align_front(ma.sample_surface(mv, mfc, dense, int(seed)),
                                     ma.sample_surface(tv, tfc, dense * 3 // 2, int(seed) + 1), scale, seed=int(seed))
            else:
                res = ma.align(*pair, scale, bool(mirror), seed=int(seed))
        except ValueError as e:
            raise ValueError(f"{NAME}: {e}") from e
        M = res["matrix"]
        moved = ma.apply(M, mv)
        cut = ma.front_faces(moved, mfc, res["front"]) if front and cut_the_back else None
        if cut is not None and not cut.any():                 # nothing would be left: the fit is off, keep the model
            cut = None
            notes.append("cut_the_back would have cut everything away, so nothing was cut: check the fit")
        elif cut is not None:
            notes.append(f"cut_the_back: {int((~cut).sum())} of {len(cut)} faces behind the template's back were cut away")
        kept = mfc if cut is None else mfc[cut]
        mirrored = float(np.linalg.det(M[:3, :3])) < 0     # a mirrored fit turns every face inside out: turn them back
        if mirrored:
            kept = kept[:, ::-1]

        first = None                                   # the model the file is written from, when the MESH carries colours
        if model is not None:                          # every model of a batch, with whatever else the MESH carries
            out = copy.copy(model)
            v = model.vertices
            several = v.dim() == 3 and v.shape[0] > 1
            flat = v.reshape(-1, 3).detach().cpu().numpy().astype(np.float64)
            out.vertices = torch.from_numpy(ma.apply(M, flat).astype(np.float32)).reshape(v.shape).to(v.device)
            _turn_normals(out, M)
            if cut is not None and not several:
                _cut_faces(out, model.faces, cut)
            elif cut is not None:
                notes.append("the MESH output holds several models, so it is not cut; the written file is")
            if mirrored:
                out.faces = out.faces.flip(-1)
            if v.dim() != 3 and (getattr(model, "texture", None) is not None or getattr(model, "vertex_colors", None) is not None):
                notes.append("the MESH is not a batch as ComfyUI's 3D nodes make it: the file holds the shape only")
            if _carries(model):
                first = out
                if cut is not None and several:
                    first = copy.copy(out)
                    _cut_faces(first, model.faces[:1], cut)
                    if mirrored:
                        first.faces = first.faces.flip(-1)
        else:
            out = io.Mesh.Type(torch.from_numpy(moved.astype(np.float32))[None], torch.from_numpy(kept.astype(np.int64))[None])

        trimesh = optional.need("trimesh", "align 3d model")
        folder = os.path.join(folder_paths.get_output_directory(), "3d")
        os.makedirs(folder, exist_ok=True)
        stem = sp.file_stem(save_name, "aligned_model")
        taken, n = os.listdir(folder), 1                  # a number is free when no file of any kind carries it (an
        while any(f.startswith((f"{stem}_{n:05d}.", f"{stem}_{n:05d}_")) for f in taken):      # .obj writes several)
            n += 1
        out_path = os.path.join(folder, f"{stem}_{n:05d}{ext}")
        baked = files = None
        if scene is not None:
            # the file's own parts, materials and textures, each with the fit written into its points: a scale per axis
            # after a turn is a shear, which a .glb's part cannot carry as its own move / turn / scale
            baked = trimesh.Scene()
            for name in scene.graph.nodes_geometry:
                world, key = scene.graph[name]
                part = scene.geometry[key].copy()
                part.apply_transform(M @ world)
                if cut is not None and hasattr(part, "faces"):
                    part.update_faces(ma.front_faces(part.vertices, part.faces, res["front"]))
                    part.remove_unreferenced_vertices()
                    if not len(part.faces):
                        continue
                baked.add_geometry(part, node_name=name, geom_name=key if key not in baked.geometry else None)
        elif first is not None:
            data = _glb_of(first, notes)
            if data is not None and ext == ".glb" and cut is None:      # ComfyUI's .glb is the file as it is
                with open(out_path, "wb") as fh_:
                    fh_.write(data)
                files = [out_path]
            elif data is not None:
                baked = trimesh.load(BytesIO(data), file_type="glb", force="scene", process=False)
                for part in baked.geometry.values():      # the points of cut faces are still in the MESH
                    if hasattr(part, "faces"):
                        part.remove_unreferenced_vertices()
        if files is None:
            if baked is None:
                baked = trimesh.Trimesh(vertices=moved, faces=kept, process=False)
                if cut is not None:
                    baked.remove_unreferenced_vertices()
            files = _write(baked, out_path)

        rows = "\n".join(", ".join(f"{x:.8g}" for x in row) for row in M)
        report = "\n".join([f"model {len(mv)} vertices, template {len(tv)} vertices, scale = {scale}"] + ma.describe(res)
                           + [f"written: {out_path}"] + [f"   with: {os.path.basename(f)}" for f in files[1:]] + notes)
        log.info("[KUBA align 3d] %s", report.replace("\n", "\n    "))
        return io.NodeOutput(out, out_path, rows, report)


NODE_CLASS_MAPPINGS = {"KUBA_AlignModel": KUBA_AlignModel}
NODE_DISPLAY_NAME_MAPPINGS = {"KUBA_AlignModel": NAME}
