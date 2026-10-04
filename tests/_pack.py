"""Lets a test import the node modules without starting ComfyUI: `from kubapack.nodes.motion import nodes_video`.

The node files import the logic relative to the pack (`from ...kubakub import plan`), so they need the pack as their
parent package, and a top level `nodes` would hide ComfyUI's own `nodes` module. `kubapack` is the pack folder as a
package whose `__init__.py` is NOT run (that one registers every node and needs ComfyUI). Everything in it except
`nodes` is the same module object as the top level one (`kubapack.kubakub.plan is kubakub.plan`), so a test that
patches `kubakub.plan` patches what the node sees.
"""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types

ALIAS = "kubapack"
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACK not in sys.path:
    sys.path.insert(0, PACK)


class _SameModule(importlib.abc.Loader):
    def create_module(self, spec):
        return importlib.import_module(spec.name[len(ALIAS) + 1:])

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if not name.startswith(ALIAS + ".") or name.split(".")[1] == "nodes":
            return None
        return importlib.machinery.ModuleSpec(name, _SameModule())


if ALIAS not in sys.modules:
    pack = types.ModuleType(ALIAS)
    pack.__path__ = [PACK]
    pack.__package__ = ALIAS
    sys.modules[ALIAS] = pack
    sys.meta_path.insert(0, _Finder())
