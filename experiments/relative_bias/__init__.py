"""Experimental, correctness-gated fusion of Inkling2 relative-bias un-shearing.

Importing this package does not import Torch/Triton or initialize a GPU.
"""

from .kernel import NATIVE_SOURCE_SHA256, unshear_rel_logits

__all__ = ["NATIVE_SOURCE_SHA256", "unshear_rel_logits"]
