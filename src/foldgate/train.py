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
from torch.utils.data import DataLoader, Dataset, Sampler

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


class LengthGrouped(Sampler[list[int]]):
    """Batches of similar-length proteins.

    The pooled assays run from 86 to 286 residues, and padding is to the longest sequence in
    the batch, so shuffling uniformly makes almost every batch as expensive as the longest
    protein in the set. Shuffling, sorting inside large chunks and then shuffling the batch
    order keeps the randomness that training needs and pays for the padding only once.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        chunk_batches: int = 50,
        seed: int = 0,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk = batch_size * chunk_batches
        self.epoch = 0
        self.seed = seed

    def __len__(self) -> int:
        return int(np.ceil(len(self.lengths) / self.batch_size))

    def __iter__(self) -> Any:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = order[start : start + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [
                chunk[i : i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)
            ]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


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
    name: str = "train"
    # Measured on this Mac's GPU, batch 32: uniform batches 5.82 s a step, length-grouped
    # 0.97, length-grouped in mixed precision 0.95. The grouping is the whole win, so the
    # default stays at full precision and there is no numerical question to answer.
    precision: str = "32-true"


def fit(train: pd.DataFrame, cfg: TrainConfig, logger: Any = None) -> Module:
    L.seed_everything(cfg.seed, workers=True)
    rng = np.random.default_rng(cfg.seed)
    val_mask = rng.random(len(train)) < 0.1
    collate = Collate(cfg.backbone)
    fit_rows, val_rows = train[~val_mask], train[val_mask]
    tr = DataLoader(
        Variants(fit_rows),
        batch_sampler=LengthGrouped(fit_rows["sequence"].str.len().to_numpy(), cfg.batch_size, seed=cfg.seed),
        collate_fn=collate,
    )
    va = DataLoader(
        Variants(val_rows),
        batch_sampler=LengthGrouped(
            val_rows["sequence"].str.len().to_numpy(), cfg.batch_size * 2, shuffle=False
        ),
        collate_fn=collate,
    )
    module = Module(cfg.backbone, cfg.rank, cfg.lr)
    if logger is None:
        # A run takes tens of minutes with the progress bar off, so metrics land in a CSV
        # where they can be watched: runs/lightning/<name>/version_*/metrics.csv.
        logger = L.pytorch.loggers.CSVLogger("runs/lightning", name=cfg.name, flush_logs_every_n_steps=20)
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        precision=cfg.precision,  # type: ignore[arg-type]
        logger=logger,
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
    order = np.argsort(df["sequence"].str.len().to_numpy(), kind="stable")
    loader = DataLoader(
        Variants(df.iloc[order]),
        batch_size=batch_size,
        collate_fn=Collate(module.hparams["backbone"]),
    )
    preds, embs = [], []
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        preds.append(module(batch).float().cpu().numpy())
        if embeddings:
            e = module.net.embed(batch["input_ids"], batch["attention_mask"], batch["position"])
            embs.append(e.float().cpu().numpy())
    # Scoring went in length order; the caller gets its own order back.
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    out = np.concatenate(preds)[inverse]
    return out, (np.concatenate(embs)[inverse] if embeddings else None)
