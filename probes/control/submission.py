# okc: problem=muse_bf16_proj
"""Starter: the projections in plain PyTorch, correct but slow.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


def prepare(weights: Weights) -> Weights:
    """Called once, untimed, before any case. Repack the weights as you like."""
    return weights


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    """Return x @ W[:n].T, where W is weights[config["layer"]]: bfloat16, or float32 for the router."""
    w = weights[data["config"]["layer"]][: data["config"]["n"]]
    if w.dtype == torch.float32:
        return data["x"].float() @ w.T
    return data["x"] @ w.T
