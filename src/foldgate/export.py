"""Export to ONNX for serving, with a parity check against PyTorch.

Two graphs per model version:

1. ``regressor.onnx``: the fine-tuned network with LoRA merged into the weights; inputs
   ``input_ids``, ``attention_mask``, ``position``, ``zero_shot``; outputs the predicted
   score and the mutated position's embedding (for the difficulty model and drift).
2. ``mlm.onnx``: the base language model's logits, for the zero-shot score of proteins the
   router has not seen before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _Serving(nn.Module):
    def __init__(self, encoder: Any, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        zero_shot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = position.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.shape[-1]) + 1
        at = torch.gather(h, 1, idx).squeeze(1)
        score = self.head(torch.cat([at, zero_shot.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return score, at


class _Logits(nn.Module):
    def __init__(self, mlm: Any) -> None:
        super().__init__()
        self.mlm = mlm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits, -1)


def export_regressor(module: Any, out: Path, sample: dict[str, torch.Tensor]) -> Path:
    net = module.net
    merged = net.encoder.merge_and_unload()
    serving = _Serving(merged, net.head).cpu().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    args = (sample["input_ids"], sample["attention_mask"], sample["position"], sample["zero_shot"])
    torch.onnx.export(
        serving,
        args,
        str(out),
        input_names=["input_ids", "attention_mask", "position", "zero_shot"],
        output_names=["score", "embedding"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "length"},
            "attention_mask": {0: "batch", 1: "length"},
            "position": {0: "batch"},
            "zero_shot": {0: "batch"},
            "score": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    return out


def export_mlm(backbone: str, out: Path, sample_ids: torch.Tensor) -> Path:
    from transformers import EsmForMaskedLM

    model = _Logits(EsmForMaskedLM.from_pretrained(backbone)).cpu().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample_ids, torch.ones_like(sample_ids)),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["log_probs"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "length"},
            "attention_mask": {0: "batch", 1: "length"},
            "log_probs": {0: "batch", 1: "length"},
        },
    )
    return out


def session(path: Path, threads: int = 0) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def reference_scores(module: Any, batch: dict[str, torch.Tensor]) -> np.ndarray:
    """PyTorch scores for one batch. Take these before exporting: merging LoRA is in place."""
    module = module.cpu().eval()
    with torch.no_grad():
        return np.asarray(module({k: v.cpu() for k, v in batch.items()}).numpy())


def parity(reference: np.ndarray, onnx_path: Path, batch: dict[str, torch.Tensor]) -> float:
    """Largest absolute difference between ONNX and PyTorch scores on one batch."""
    sess = session(onnx_path)
    feeds = {k: batch[k].cpu().numpy() for k in ("input_ids", "attention_mask", "position", "zero_shot")}
    feeds["zero_shot"] = feeds["zero_shot"].astype(np.float32)
    got = sess.run(["score"], feeds)[0]
    return float(np.max(np.abs(got - reference)))
