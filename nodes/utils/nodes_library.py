"""
nodes_library.py

The routes of the shelf's library (kubakub/library.py; the view is in web/kubakub_library.js). No nodes.
The database is user/default/kubakub/library.db; small previews are kept next to it in library_thumbs/.

    GET  /kubakub/library/search    ?q=&kind=&model=&lora=&tag=&sort=newest|oldest|name&offset=&limit=
    GET  /kubakub/library/facets    the pick lists (models, LoRAs, tags with counts), the added folders, the count
    GET  /kubakub/library/item      ?id= -> everything stored about one file
    GET  /kubakub/library/workflow  ?id= -> the workflow embedded in the file ({workflow} or {prompt})
    GET  /kubakub/library/thumb     ?id= -> a small preview (webp; videos: their first frame)
    GET  /kubakub/library/file      ?id= -> the file itself (only files in the library)
    POST /kubakub/library/add       {sources: ['file:..', 'output:..'], files: [{filename, subfolder, type}], tags}
    POST /kubakub/library/folder    {path} -> index a folder of this computer in place (local requests only); an
                                    empty path indexes ComfyUI's output folder (tag "output")
    POST /kubakub/library/forget    {path} -> stop showing an added folder (the files stay on disk)
    POST /kubakub/library/to_input  {id} -> a source the loaders take (a file outside input/ and output/ is copied
                                    into input/kuba_director)
    POST /kubakub/library/reveal    {id} -> show the file in the file browser (local requests only)
    POST /kubakub/library/tag       {id, tag, on}
    POST /kubakub/library/prune     drop entries whose file is gone

Only files under ComfyUI's input / output / temp and the added folders are ever indexed or handed out.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys

import folder_paths

from ...kubakub import library as lib
from ...kubakub import master as ms

log = logging.getLogger("KUBA.library")

UPLOAD_SUB = "kuba_director"
_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        base = os.path.join(folder_paths.get_user_directory(), "default", "kubakub")
        _LIB = lib.Library(os.path.join(base, "library.db"), {"input": folder_paths.get_input_directory(),
                                                              "output": folder_paths.get_output_directory(),
                                                              "temp": folder_paths.get_temp_directory()})
    return _LIB


def _thumb_dir():
    d = os.path.join(folder_paths.get_user_directory(), "default", "kubakub", "library_thumbs")
    os.makedirs(d, exist_ok=True)
    return d


def thumb(path: str, out: str, size: int = 320) -> bool:
    """A small webp of a picture or of a video's first frame."""
    from PIL import Image
    kind = lib.kind_of(path)
    try:
        if kind == "image":
            with Image.open(path) as im:
                im.seek(0)
                im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
                im.thumbnail((size, size))
                im.save(out, "WEBP", quality=80)
            return True
        if kind == "video":
            import av
            with av.open(path) as c:
                for frame in c.decode(video=0):
                    im = frame.to_image()
                    im.thumbnail((size, size))
                    im.save(out, "WEBP", quality=80)
                    return True
    except Exception:  # noqa: BLE001
        log.debug("[KUBA library] no preview for %s", path, exc_info=True)
    return False


def _local(request) -> bool:
    return (request.remote or "") in ("127.0.0.1", "::1", "localhost")


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # noqa: BLE001  (tests / no server)
        return

    def run(fn, *a):
        return asyncio.get_running_loop().run_in_executor(None, fn, *a)

    def fail(e, status=500):
        return web.json_response({"error": str(e).splitlines()[0] if str(e) else type(e).__name__}, status=status)

    def ident(request):
        try:
            return int(request.rel_url.query.get("id", ""))
        except ValueError:
            return -1

    async def search(request):
        q = request.rel_url.query
        try:
            res = await run(lambda: _library().search(q.get("q", ""), q.get("kind", ""), q.get("model", ""),
                                                      q.get("lora", ""), q.get("tag", ""), q.get("sort", "newest"),
                                                      min(int(q.get("limit", 120)), 500), int(q.get("offset", 0))))
            return web.json_response(res)
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA library] search failed")
            return fail(e)

    async def facets(request):
        try:
            return web.json_response(await run(_library().facets))
        except Exception as e:  # noqa: BLE001
            return fail(e)

    async def item(request):
        r = await run(_library().row, ident(request))
        return web.json_response(r) if r else fail("not in the library", 404)

    async def workflow(request):
        r = await run(_library().row, ident(request))
        if not r or not os.path.isfile(r["path"]):
            return fail("not in the library", 404)
        wf = await run(lib.workflow_of, r["path"])
        return web.json_response(wf) if wf else fail("this file carries no workflow", 404)

    async def thumbnail(request):
        L = _library()
        r = await run(L.row, ident(request))
        if not r or not os.path.isfile(r["path"]) or not L.allowed(r["path"]):
            return fail("not in the library", 404)
        out = os.path.join(_thumb_dir(), f"{r['id']}_{int(r['mtime'] or 0)}_{r['size']}.webp")
        if not os.path.isfile(out) and not await run(thumb, r["path"], out):
            return fail("no preview", 404)
        return web.FileResponse(out, headers={"Cache-Control": "max-age=86400"})

    async def file(request):
        L = _library()
        r = await run(L.row, ident(request))
        if not r or not os.path.isfile(r["path"]) or not L.allowed(r["path"]):
            return fail("not in the library", 404)
        return web.FileResponse(r["path"])

    async def add(request):
        try:
            body = await request.json()
            L = _library()
            tags = [str(t) for t in body.get("tags") or [] if t][:8]
            paths = []
            for s in body.get("sources") or []:
                p = ms.resolve(str(s), folder_paths.get_input_directory(), folder_paths.get_output_directory())
                if p:
                    paths.append(p)
            for f in body.get("files") or []:              # /history outputs
                base = folder_paths.get_directory_by_type(str(f.get("type") or "output"))
                if not base:
                    continue
                p = os.path.realpath(os.path.join(base, str(f.get("subfolder") or ""), str(f.get("filename") or "")))
                if lib.inside(p, base) and os.path.isfile(p):
                    paths.append(p)

            def work():
                with L.lock:
                    known = {r["path"] for r in L.con.execute("SELECT path FROM files")}
                ids, new = [], 0
                for p in paths:
                    new += os.path.realpath(p) not in known
                    try:
                        i = L.add(p, tags)
                    except Exception:  # noqa: BLE001
                        log.debug("[KUBA library] could not index %s", p, exc_info=True)
                        i = None
                    if i is not None:
                        ids.append(i)
                return {"ids": ids, "new": new, "items": [L.row(i) for i in ids]}
            return web.json_response(await run(work))
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA library] add failed")
            return fail(e)

    async def folder(request):
        if not _local(request):
            return fail("folders can be added only from this computer", 403)
        try:
            body = await request.json()
            L = _library()
            given = str(body.get("path") or "").strip()
            if given:
                path = await run(L.add_root, given)
                tag = os.path.basename(path)
            else:                                       # an empty field: ComfyUI's own output folder (always allowed)
                path, tag = os.path.realpath(folder_paths.get_output_directory()), "output"
            res = await run(lambda: L.add_folder(path, [tag]))
            return web.json_response(dict(res, path=path, tag=tag))
        except ValueError as e:
            return fail(e, 400)
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA library] index folder failed")
            return fail(e)

    async def forget(request):
        try:
            body = await request.json()
            return web.json_response({"removed": await run(_library().remove_root, str(body.get("path") or ""))})
        except Exception as e:  # noqa: BLE001
            return fail(e)

    async def to_input(request):
        try:
            body = await request.json()
            L = _library()
            r = await run(L.row, int(body.get("id", -1)))
            if not r or not os.path.isfile(r["path"]):
                return fail("not in the library", 404)
            if r["source"]:
                return web.json_response({"source": r["source"]})
            name = ms.safe_name(r["name"])
            if not name:
                return fail("not a picture, video or sound", 400)
            folder = os.path.join(folder_paths.get_input_directory(), UPLOAD_SUB)
            os.makedirs(folder, exist_ok=True)

            def copy():
                target = os.path.join(folder, name)
                if os.path.isfile(target) and os.path.getsize(target) == r["size"]:
                    return name
                n = ms.free_name(folder, name)
                shutil.copy2(r["path"], os.path.join(folder, n))
                return n
            n = await run(copy)
            src = f"file:{UPLOAD_SUB}/{n}"
            await run(lambda: L.add(os.path.join(folder, n), r["tags"]))
            return web.json_response({"source": src})
        except Exception as e:  # noqa: BLE001
            log.exception("[KUBA library] to input failed")
            return fail(e)

    async def reveal(request):
        if not _local(request):
            return fail("only from this computer", 403)
        try:
            body = await request.json()
            r = await run(_library().row, int(body.get("id", -1)))
            if not r or not os.path.isfile(r["path"]):
                return fail("not in the library", 404)
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", r["path"]])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", r["path"]])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(r["path"])])
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            return fail(e)

    async def tag(request):
        try:
            body = await request.json()
            await run(_library().tag, int(body.get("id", -1)), str(body.get("tag") or "").strip(), bool(body.get("on", True)))
            return web.json_response(await run(_library().row, int(body.get("id", -1))) or {})
        except Exception as e:  # noqa: BLE001
            return fail(e)

    async def prune(request):
        return web.json_response({"removed": await run(_library().prune)})

    try:
        R = PromptServer.instance.routes
        R.get("/kubakub/library/search")(search)
        R.get("/kubakub/library/facets")(facets)
        R.get("/kubakub/library/item")(item)
        R.get("/kubakub/library/workflow")(workflow)
        R.get("/kubakub/library/thumb")(thumbnail)
        R.get("/kubakub/library/file")(file)
        R.post("/kubakub/library/add")(add)
        R.post("/kubakub/library/folder")(folder)
        R.post("/kubakub/library/forget")(forget)
        R.post("/kubakub/library/to_input")(to_input)
        R.post("/kubakub/library/reveal")(reveal)
        R.post("/kubakub/library/tag")(tag)
        R.post("/kubakub/library/prune")(prune)
    except (AttributeError, RuntimeError) as e:
        log.warning("[KUBA library] routes not registered: %s", e)


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
