"""
Kuba Regions: many diffusion stories inside one image or video.

Modules in this package that hold pure
logic (geometry.py, ...) import no ComfyUI code, so the model free tests in
tests/ can load them without starting ComfyUI. Node wrappers live in the
nodes_*.py modules and are registered by the pack's loader in ../__init__.py.
"""
