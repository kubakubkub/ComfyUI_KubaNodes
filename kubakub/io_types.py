"""ComfyUI socket types of Kuba Regions, defined once so every node uses the same class."""

from comfy_api.latest import io

CanvasPlanType = io.Custom("KUBA_CANVAS_PLAN")   # geometry.CanvasPlan
RegionsType = io.Custom("KUBA_REGIONS")          # types.Regions
PlanType = io.Custom("KUBA_PLAN")                # types.RegionPlan
ViewerType = io.Custom("KUBA_VIEWER")            # viewer.Viewer
SceneType = io.Custom("KUBA_SCENE")              # dict: export folder, ids folder, file, settings
DirectorType = io.Custom("KUBA_DIRECTOR")        # nodes_director.DirectorState
PiecesType = io.Custom("KUBA_PIECES")            # dict: scene, piece labels file, piece stats, operators (scene3d/pieces.py)
FalloffType = io.Custom("KUBA_FALLOFF")          # dict: a falloff for the pieces operators
