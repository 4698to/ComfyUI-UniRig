import sys
import logging

log = logging.getLogger("unirig")

log.info("loading...")
from comfy_env import register_nodes
log.info("calling register_nodes")
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

try:
    from .nodes import on_custom_loaded as on_custom_loaded
except Exception as e:
    log.warning("on_custom_loaded hook unavailable: %s", e)
    def on_custom_loaded(app):
        return None

try:
    from server import PromptServer  # type: ignore[import-not-found]
    _ps = getattr(PromptServer, "instance", None)
    if _ps is not None:
        log.info("PromptServer instance ready; calling on_custom_loaded fallback")
        on_custom_loaded(_ps)
except Exception as e:
    log.debug("PromptServer fallback registration skipped: %s", e)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "on_custom_loaded"]
