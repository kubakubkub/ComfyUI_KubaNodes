"""
Model free test of the shelf's library (kubakub/library.py): what it reads from ComfyUI PNG / WEBP / video files and
A1111 pictures (prompts followed from the sampler, seed, steps, models, LoRAs), the search, the filters, tags,
indexed folders and the path guard. No ComfyUI server, no GPU.

Run from the ComfyUI_windows_portable folder:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI_KubaNodes\\tests\\test_library.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kubakub import library as lib  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# a ComfyUI API graph as SaveImage writes it: checkpoint -> lora -> two text encoders -> KSampler
CLASSIC = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/juggernaut_v9.safetensors"}},
    "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "kubakub_style.safetensors", "strength_model": 0.8,
                                                 "strength_clip": 1.0, "model": ["4", 0], "clip": ["4", 1]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "baroque facade at night, warm windows", "clip": ["10", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, text", "clip": ["10", 1]}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 576, "batch_size": 1}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 123456, "steps": 28, "cfg": 5.5, "sampler_name": "dpmpp_2m",
                                              "scheduler": "karras", "denoise": 1.0, "model": ["10", 0],
                                              "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "facade", "images": ["8", 0]}},
}
# Flux 2 style: a string primitive -> text encode -> reference latent -> guidance -> CFGGuider -> SamplerCustomAdvanced
FLUX = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
    "20": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a palace wall made of moss"}},
    "21": {"class_type": "CLIPTextEncode", "inputs": {"text": ["20", 0], "clip": ["2", 0]}},
    "22": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["21", 0], "latent": ["30", 0]}},
    "30": {"class_type": "VAEEncode", "inputs": {"pixels": ["31", 0], "vae": ["32", 0]}},
    "31": {"class_type": "LoadImage", "inputs": {"image": "matrix.png"}},
    "32": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
    "23": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["22", 0], "guidance": 4.0}},
    "24": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["21", 0]}},
    "25": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["23", 0], "negative": ["24", 0], "cfg": 1.0}},
    "26": {"class_type": "RandomNoise", "inputs": {"noise_seed": 777}},
    "27": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "28": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["26", 0], "guider": ["25", 0], "sampler": ["27", 0],
                                                            "sigmas": ["29", 0], "latent_image": ["30", 0]}},
    "29": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 4, "denoise": 1.0, "model": ["1", 0]}},
}
WORKFLOW = {"last_node_id": 9, "nodes": [{"id": 3, "type": "KSampler"}], "links": [], "version": 0.4}
A1111 = ("masterpiece, palace facade <lora:gold_leaf:0.6>\nNegative prompt: lowres, watermark\n"
         "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 42, Size: 768x512, Model hash: abc, Model: realisticVision")

with tempfile.TemporaryDirectory() as tmp:
    inp, out, tmpd, extra, outside = (os.path.join(tmp, n) for n in ("input", "output", "temp", "renders", "elsewhere"))
    for d in (inp, out, tmpd, extra, outside, os.path.join(out, "show")):
        os.makedirs(d, exist_ok=True)
    pic = Image.fromarray(np.full((64, 96, 3), 120, np.uint8))

    def png(path, **text):
        pi = PngInfo()
        for k, v in text.items():
            pi.add_text(k, v if isinstance(v, str) else json.dumps(v))
        pic.save(path, pnginfo=pi)
        return path

    p_classic = png(os.path.join(out, "show", "facade_00001_.png"), prompt=CLASSIC, workflow=WORKFLOW)
    p_flux = png(os.path.join(out, "klein_00001_.png"), prompt=FLUX)
    p_a1111 = png(os.path.join(inp, "a1111.png"), parameters=A1111)
    p_plain = png(os.path.join(inp, "plain.png"))
    # WEBP as core SaveAnimatedWEBP writes it: exif Model = "prompt:{...}", Make = "workflow:{...}"
    p_webp = os.path.join(out, "anim.webp")
    ex = Image.Exif()
    ex[0x0110] = "prompt:" + json.dumps(CLASSIC)
    ex[0x010F] = "workflow:" + json.dumps(WORKFLOW)
    pic.save(p_webp, exif=ex)

    # ---- reading files
    d = lib.describe(p_classic)
    check("png: positive from the sampler's positive", d["positive"] == "baroque facade at night, warm windows", d["positive"])
    check("png: negative", d["negative"] == "blurry, text", d["negative"])
    check("png: seed, steps, cfg, sampler, scheduler",
          (d["seed"], d["steps"], d["cfg"], d["sampler"], d["scheduler"]) == (123456, 28, 5.5, "dpmpp_2m", "karras"), d)
    check("png: model and lora with its strength", d["models"] == ["sdxl/juggernaut_v9.safetensors"]
          and d["loras"] == [{"name": "kubakub_style.safetensors", "strength": 0.8}], (d["models"], d["loras"]))
    check("png: size from the file", (d["w"], d["h"]) == (96, 64) and d["has_workflow"])
    f = lib.describe(p_flux)
    check("flux: prompt through primitive, reference latent, guidance and guider", f["positive"] == "a palace wall made of moss", f["positive"])
    check("flux: zero-out negative is no text, and the load image name is not a prompt", f["negative"] == "", f["negative"])
    check("flux: seed from RandomNoise, steps from the scheduler, sampler from the select",
          (f["seed"], f["steps"], f["sampler"], f["scheduler"]) == (777, 4, "euler", "simple"), f)
    check("flux: unet, text encoder, vae listed", set(f["models"]) == {"flux-2-klein-4b-fp8.safetensors", "qwen_3_4b.safetensors", "flux2-vae.safetensors"}, f["models"])
    a = lib.describe(p_a1111)
    check("a1111: prompt, negative", a["positive"].startswith("masterpiece, palace facade") and a["negative"] == "lowres, watermark", a)
    check("a1111: settings", (a["steps"], a["cfg"], a["seed"], a["sampler"], a["width"], a["height"]) == (30, 7.0, 42, "DPM++ 2M Karras", 768, 512), a)
    check("a1111: model and lora", a["models"] == ["realisticVision"] and a["loras"] == [{"name": "gold_leaf", "strength": 0.6}], a)
    w = lib.describe(p_webp)
    check("webp: exif prompt read", w["positive"] == "baroque facade at night, warm windows" and w["has_workflow"], w["positive"])
    check("webp: workflow handed out", lib.workflow_of(p_webp) == {"workflow": WORKFLOW})
    check("png: workflow handed out, api prompt when there is no workflow",
          lib.workflow_of(p_classic) == {"workflow": WORKFLOW} and lib.workflow_of(p_flux) == {"prompt": FLUX})
    check("a plain picture: nothing, but no error", lib.describe(p_plain)["positive"] == "" and lib.workflow_of(p_plain) is None)
    check("no sampler: every text of the graph", lib.parse_prompt({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "only words"}}})["positive"] == "only words")
    check("a broken graph never raises", lib.parse_prompt({"1": "x", "2": {"inputs": {"positive": ["99", 0]}, "class_type": "KSampler"}})["positive"] == "")

    # a video as core SaveVideo writes it (container metadata prompt / workflow)
    p_mp4 = os.path.join(out, "clip.mp4")
    try:
        import av
        with av.open(p_mp4, "w", options={"movflags": "use_metadata_tags"}) as c:
            c.metadata["prompt"] = json.dumps(FLUX)
            c.metadata["workflow"] = json.dumps(WORKFLOW)
            s = c.add_stream("libx264", rate=8)
            s.width, s.height, s.pix_fmt = 96, 64, "yuv420p"
            for i in range(8):
                for pkt in s.encode(av.VideoFrame.from_ndarray(np.full((64, 96, 3), i * 20, np.uint8), format="rgb24")):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        v = lib.describe(p_mp4)
        check("video: prompt from the container metadata, size, length",
              v["positive"] == "a palace wall made of moss" and (v["w"], v["h"]) == (96, 64) and 0.5 < (v["duration"] or 0) < 2, v)
    except Exception as e:  # noqa: BLE001
        print("skip the video check:", e)

    # ---- the database
    L = lib.Library(os.path.join(tmp, "user", "library.db"), {"input": inp, "output": out, "temp": tmpd})
    ids = [L.add(p) for p in (p_classic, p_flux, p_a1111, p_plain, p_webp)]
    check("files are indexed", all(i is not None for i in ids), ids)
    check("the same file again keeps its id", L.add(p_classic) == ids[0])
    r = L.row(ids[0])
    check("a row: source the loaders take, lists back as lists", r["source"] == "output:show/facade_00001_.png"
          and r["loras"][0]["name"] == "kubakub_style.safetensors" and r["tags"] == [], r)
    check("input files get file: sources", L.row(ids[2])["source"] == "file:a1111.png")
    s = L.search("warm windows")
    check("search finds words of a prompt (png + webp)", s["total"] == 2, [i["name"] for i in s["items"]])
    check("search by the start of a word", L.search("baroq")["total"] == 2 and L.search("pala")["total"] >= 2)
    check("search by model name parts", L.search("juggernaut")["total"] == 2 and L.search("klein")["total"] == 1)
    check("search by lora", L.search("gold_leaf")["total"] == 1 and L.search("kubakub style")["total"] == 2)
    check("several words = all of them", L.search("palace moss")["total"] == 1)
    check("odd characters in a search never raise", isinstance(L.search('"(*) OR AND -x')["total"], int))
    check("filter by kind", L.search(kind="image")["total"] == 5 and L.search(kind="video")["total"] == 0)
    check("filter by model (exact)", L.search(model="flux-2-klein-4b-fp8.safetensors")["total"] == 1 and L.search(model="flux")["total"] == 0)
    check("filter by lora", L.search(lora="kubakub_style.safetensors")["total"] == 2)
    check("sort by name", [i["name"] for i in L.search(sort="name")["items"]][0] == "a1111.png")
    L.tag(ids[3], "keep")
    check("tags: added, searched, filtered", L.row(ids[3])["tags"] == ["keep"] and L.search(tag="keep")["total"] == 1 and L.search("keep")["total"] == 1)
    L.tag(ids[3], "keep", on=False)
    check("tags: removed", L.search(tag="keep")["total"] == 0)
    fc = L.facets()
    check("facets: models, loras, counts", fc["count"] == 5 and fc["loras"][0] == {"name": "kubakub_style.safetensors", "count": 2}
          and any(m["name"] == "realisticVision" for m in fc["models"]), fc)

    # the path guard
    p_out = png(os.path.join(outside, "secret.png"), prompt=CLASSIC)
    check("a file outside the allowed folders is not indexed", L.add(p_out) is None)
    check("no escape with ..", L.add(os.path.join(out, "..", "elsewhere", "secret.png")) is None)
    check("not a picture, video or sound", L.add(__file__) is None)
    try:
        L.add_root(os.path.join(tmp, "nope"))
        check("a missing folder is refused", False)
    except ValueError:
        check("a missing folder is refused", True)
    try:
        L.add_root(os.path.abspath(os.sep))
        check("a whole drive is refused", False)
    except ValueError:
        check("a whole drive is refused", True)

    # an indexed folder: read in place, named by its tag, again = new and gone files
    for i in range(3):
        png(os.path.join(extra, f"render_{i}.png"), prompt=CLASSIC)
    os.makedirs(os.path.join(extra, "sub"))
    png(os.path.join(extra, "sub", "deep.png"))
    open(os.path.join(extra, "notes.txt"), "w").write("x")
    root = L.add_root(extra)
    res = L.add_folder(root, ["renders"])
    check("folder: every picture, deep, not the text", res["total"] == 4 and L.search(tag="renders")["total"] == 4, res)
    check("folder files have no loader source (outside input/output)", L.search(tag="renders")["items"][0]["source"] is None)
    check("folder files are allowed now", L.allowed(os.path.join(extra, "render_0.png")))
    os.remove(os.path.join(extra, "render_0.png"))
    png(os.path.join(extra, "render_9.png"))
    res = L.add_folder(root, ["renders"])
    check("folder again: gone ones dropped, new ones in", L.search(tag="renders")["total"] == 4
          and not L.search("render_0")["total"] and L.search("render_9")["total"] == 1)
    check("cap on a big folder", L.add_folder(root, ["renders"], limit=2)["capped"])
    check("forgetting a folder drops its files, keeps the rest", L.remove_root(root) == 4 and L.search()["total"] == 5)
    check("after forgetting, the folder is not allowed", not L.allowed(os.path.join(extra, "render_1.png")))
    os.remove(p_plain)
    check("prune drops gone files", L.prune() == 1 and L.search()["total"] == 4)
    L.close()
    L2 = lib.Library(os.path.join(tmp, "user", "library.db"), {"input": inp, "output": out, "temp": tmpd})
    check("the database keeps everything after a restart", L2.search("moss")["total"] == 1)
    L2.close()

print()
print("all library tests passed" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
