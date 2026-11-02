"""A web module made runnable by node for the JavaScript checks: ComfyUI's app / api replaced by stubs, and the
shared kubakub_media.js pasted in where it is imported (so the checks need no browser and no import map)."""

import os
import re

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
API_STUB = 'const api = { apiURL: p => "/api" + p, addEventListener() {} };'


def _read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as f:
        return f.read()


def module_source(name):
    src = _read(name)
    src = re.sub(r"(?m)^import \{ app \} from .*$", "const app = { registerExtension() {} };", src)
    src = re.sub(r"(?m)^import \{ api \} from .*$", API_STUB, src)
    media = re.sub(r"(?m)^import \{ api \} from .*$", "", _read("kubakub_media.js"))
    media = re.sub(r"(?m)^export (const|function|async function) ", r"\1 ", media)
    src = re.sub(r"(?m)^import \{[^}]*\} from \"\./kubakub_media\.js\";$", lambda m: media, src)
    return "globalThis.location = globalThis.location || { href: 'http://127.0.0.1:8188/', origin: 'http://127.0.0.1:8188' };\n" + src
