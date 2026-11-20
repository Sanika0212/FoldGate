"""ESM-2 for variant effects: a zero-shot score and a LoRA-fine-tuned regressor.

Zero-shot (no labels): the wild-type marginal. One forward pass over the wild-type
sequence gives, at every position, the model's log-probability of each amino acid; a
variant's score is log p(mutant) minus log p(wild type) at the mutated position.

Supervised: the same network with LoRA adapters on the attention query and value
projections, and a small head that reads the representation at the mutated position plus
the zero-shot score and predicts the standardized assay score. Only the adapters and the
head train (about 2% of the weights for ESM-2 35M).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

DEFAULT_BACKBONE = "facebook/esm2_t12_35M_UR50D"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenizer(backbone: str = DEFAULT_BACKBONE) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(backbone)


@torch.no_grad()
def zero_shot(
    wild_type: str,
    positions: np.ndarray,
    wt: np.ndarray,
    mt: np.ndarray,
    backbone: str = DEFAULT_BACKBONE,
    dev: torch.device | None = None,
) -> np.ndarray:
    """Wild-type marginal scores for a batch of single mutants of one protein."""
    from transformers import EsmForMaskedLM

    dev = dev or device()
    tok = tokenizer(backbone)
    model: Any = EsmForMaskedLM.from_pretrained(backbone)
    model = model.to(dev)
    model.eval()
    enc = tok(wild_type, return_tensors="pt").to(dev)
    logp = torch.log_softmax(model(**enc).logits[0], dim=-1).cpu().numpy()
    vocab = tok.get_vocab()
    rows = positions + 1  # position 0 of the sequence is token 1, after <cls>
    wt_ids = np.array([vocab[a] for a in wt])
    mt_ids = np.array([vocab[a] for a in mt])
    return np.asarray(logp[rows, mt_ids] - logp[rows, wt_ids], dtype=np.float32)


class VariantRegressor(nn.Module):
    """ESM-2 with LoRA adapters and a head on the mutated position's representation."""

    def __init__(
        self, backbone: str = DEFAULT_BACKBONE, rank: int = 8, alpha: int = 16, dropout: float = 0.1
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import EsmModel

        base = EsmModel.from_pretrained(backbone, add_pooling_layer=False)
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=["query", "value"])
        self.encoder = get_peft_model(base, config)
        hidden = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        zero_shot: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        at = h[torch.arange(h.shape[0], device=h.device), position + 1]
        out: torch.Tensor = self.head(torch.cat([at, zero_shot.unsqueeze(-1)], dim=-1))
        return out.squeeze(-1)

    def embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """The mutated position's representation, used for drift monitoring."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        at: torch.Tensor = h[torch.arange(h.shape[0], device=h.device), position + 1]
        return at

    def trainable_share(self) -> float:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable / total
