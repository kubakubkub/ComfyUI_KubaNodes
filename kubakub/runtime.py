"""Small runtime helpers for nodes that run a model several times inside ONE node execution."""


def between_runs():
    """The cleanup ComfyUI runs after every node (execution.py, when dynamic VRAM / aimdo is on).

    A node that calls core Generate Text several times (one prompt enhance per clip) skips it between the calls; the
    second call then reuses the first run's recorded memory graph and aborts with a CUDA 'scatter gather index out of
    bounds' assert (found 2026-09-29). Calling this after each run does what the node boundary would do. Harmless when
    dynamic VRAM is off or the core functions are missing."""
    try:
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_prefetch
        if not getattr(comfy.memory_management, "aimdo_enabled", False):
            return
        comfy.model_prefetch.cleanup_prefetch_queues()
        comfy.model_management.reset_cast_buffers()
        import comfy_aimdo.model_vbar
        comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
    except Exception as e:  # noqa: BLE001  older / newer core: nothing to clean
        import logging
        logging.getLogger("KUBA.regions").debug("between_runs: %s", e)
