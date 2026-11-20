"""Fine-tune the LoRA regressor with PyTorch Lightning; predict and embed.

The validation set for early stopping is carved out of the training split (10%), so the
calibration split is never seen during training: conformal guarantees depend on it.
Runs on Apple's GPU (MPS) or CUDA when present; ``strategy="ddp"`` in the trainer config
spreads the same code over several GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from foldgate.model import DEFAULT_BACKBONE, VariantRegressor, tokenizer


class Variants(Dataset[dict[str, Any]]):
    def __init__(self, df: pd.DataFrame) -> None:
        self.seq = df["sequence"].tolist()
        self.pos = df["position"].to_numpy()
        self.zs = df["zero_shot"].to_numpy(dtype=np.float32)
        self.y = df["z"].to_numpy(dtype=np.float32) if "z" in df else np.zeros(len(df), np.float32)

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"seq": self.seq[i], "pos": int(self.pos[i]), "zs": self.zs[i], "y": self.y[i]}


class Collate:
    def __init__(self, backbone: str) -> None:
        self.tok = tokenizer(backbone)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        enc = self.tok([b["seq"] for b in batch], return_tensors="pt", padding=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "position": torch.tensor([b["pos"] for b in batch]),
            "zero_shot": torch.tensor([b["zs"] for b in batch]),
            "y": torch.tensor([b["y"] for b in batch]),
        }


class Module(L.LightningModule):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, rank: int = 8, lr: float = 3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = VariantRegressor(backbone, rank=rank)
        self.lr = lr

    def on_fit_start(self) -> None:
        self.net.train()  # from_pretrained hands back an eval-mode module; dropout must be on

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out: torch.Tensor = self.net(
            batch["input_ids"], batch["attention_mask"], batch["position"], batch["zero_shot"]
        )
        return out

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(self(batch), batch["y"])
        self.log("train_mse", loss, prog_bar=True, batch_size=len(batch["y"]))
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> None:
        loss = torch.nn.functional.mse_loss(self(batch), batch["y"])
        self.log("val_mse", loss, prog_bar=True, batch_size=len(batch["y"]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)


@dataclass
class TrainConfig:
    backbone: str = DEFAULT_BACKBONE
    rank: int = 8
    lr: float = 3e-4
    epochs: int = 6
    batch_size: int = 32
    seed: int = 0
    strategy: str = "auto"
    devices: int | str = 1
    accelerator: str = "auto"


def fit(train: pd.DataFrame, cfg: TrainConfig, logger: Any = None) -> Module:
    L.seed_everything(cfg.seed, workers=True)
    rng = np.random.default_rng(cfg.seed)
    val_mask = rng.random(len(train)) < 0.1
    collate = Collate(cfg.backbone)
    tr = DataLoader(Variants(train[~val_mask]), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(Variants(train[val_mask]), batch_size=cfg.batch_size * 2, collate_fn=collate)
    module = Module(cfg.backbone, cfg.rank, cfg.lr)
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        logger=logger or False,
        enable_checkpointing=False,
        callbacks=[L.pytorch.callbacks.EarlyStopping("val_mse", patience=2)],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=20,
    )
    trainer.fit(module, tr, va)
    return module


@torch.no_grad()
def predict(
    module: Module, df: pd.DataFrame, batch_size: int = 64, embeddings: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    from foldgate.model import device

    dev = device()
    module = module.to(dev).eval()
    loader = DataLoader(Variants(df), batch_size=batch_size, collate_fn=Collate(module.hparams["backbone"]))
    preds, embs = [], []
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        preds.append(module(batch).float().cpu().numpy())
        if embeddings:
            e = module.net.embed(batch["input_ids"], batch["attention_mask"], batch["position"])
            embs.append(e.float().cpu().numpy())
    return np.concatenate(preds), (np.concatenate(embs) if embeddings else None)
