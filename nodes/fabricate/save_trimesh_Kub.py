import os
import folder_paths

class SaveTrimeshGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "trimesh": ("TRIMESH", {"tooltip": "The 3D mesh to save (e.g. from an image to 3D node)."}),
            "filename": ("STRING", {"default": "trellis_output",
                                    "tooltip": "Name inside the output folder; 'sub/name' makes a subfolder. "
                                               "A number is added (_0001, _0002 ...) so nothing is overwritten."})
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_TOOLTIPS = ("The full path of the saved .glb file.",)
    DESCRIPTION = ("Saves a 3D mesh as a .glb file in the output folder, to open in Blender or any 3D tool. "
                   "Never overwrites: each save gets the next number.")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "kubakub/3d/fabricate"

    def save(self, trimesh, filename):
        output_dir = os.path.realpath(folder_paths.get_output_directory())
        base = os.path.realpath(os.path.join(output_dir, filename.strip().strip("/\\") or "trellis_output"))
        if not base.startswith(output_dir + os.sep):
            raise ValueError(f"filename must stay inside the output folder: {filename}")
        os.makedirs(os.path.dirname(base), exist_ok=True)   # 'sub/name' works
        i = 1
        while os.path.exists(f"{base}_{i:04d}.glb"):
            i += 1
        path = f"{base}_{i:04d}.glb"
        trimesh.export(path)
        print(f"[SaveTrimeshGLB] Saved {len(trimesh.vertices)} verts to {path}")
        return (path,)

NODE_CLASS_MAPPINGS = {"SaveTrimeshGLB": SaveTrimeshGLB}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveTrimeshGLB": "kubakub save trimesh glb"}