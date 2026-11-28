"""
vLLM plugin: smaller attention tiles for fp32, so the Kaggle T4 route fits the GPU's shared memory.

vLLM's Triton attention kernel holds one Q tile and one K and one V tile in shared memory. For Gemma 3
(head size 256) it uses 32-token KV tiles. In fp32 that is 16 + 32 + 32 KB = 80 KB, but a T4 has 64 KB, so
the engine fails at warm-up with "out of resource: shared memory, Required: 81920, Hardware limit: 65536".
16-token tiles need 48 KB.

The tile size only changes how the attention sum is blocked, not what is computed. Only fp32 inputs
(4 bytes per element) are touched, so bf16 and FP8 runs (the RTX 5060 setup) are unchanged.

Registered in pyproject.toml under the "vllm.general_plugins" entry point group, which vLLM loads in
every process, including the tensor-parallel workers.
"""

import logging
from collections.abc import Callable

FP32_ATTENTION_TILE = 16
FP32_ELEMENT_SIZE = 4

logger = logging.getLogger(__name__)

TileFn = Callable[[int, int, int, bool], int]


def small_fp32_tiles(original: TileFn) -> TileFn:
    """Wrap vLLM's tile-size choice: fp32 gets FP32_ATTENTION_TILE, everything else is unchanged."""

    def tile_size(head_size: int, sliding_window: int, element_size: int, is_prefill: bool) -> int:
        chosen = original(head_size, sliding_window, element_size, is_prefill)
        return min(chosen, FP32_ATTENTION_TILE) if element_size >= FP32_ELEMENT_SIZE else chosen

    tile_size.payo_patched = True  # type: ignore[attr-defined]
    return tile_size


def register() -> None:
    """vLLM general-plugin hook. Runs in every vLLM process."""
    try:
        from vllm.v1.attention.ops import triton_unified_attention as tua  # noqa: PLC0415
    except ImportError:
        logger.warning("payo fp32 attention plugin: vLLM's Triton attention module not found; not patched")
        return
    current = getattr(tua, "_get_tile_size", None)
    if current is None:
        logger.warning("payo fp32 attention plugin: _get_tile_size not found in this vLLM; not patched")
        return
    if not getattr(current, "payo_patched", False):
        tua._get_tile_size = small_fp32_tiles(current)
