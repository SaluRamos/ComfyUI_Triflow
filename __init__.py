"""ComfyUI entry point. CUDA dependencies load only when executing the node."""
from .nodes import TriFlowRemesh

NODE_CLASS_MAPPINGS = {"TriFlowRemesh": TriFlowRemesh}
NODE_DISPLAY_NAME_MAPPINGS = {"TriFlowRemesh": "TriFlow Remesh"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
