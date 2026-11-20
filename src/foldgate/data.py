"""ProteinGym deep mutational scanning assays: download, select, split.

Twelve substitution assays from different protein families (proteins of at most 300
residues with 1,500 to 6,000 single mutants measured), including calmodulin, beta-lactamase,
DHFR and alpha-synuclein. Scores are standardized within each assay so that one model can
learn across proteins.

Two splits per assay, as in ProteinGym's supervised setting:

1. random: variants shuffled into train (60%), calibration (20%) and test (20%);
2. contiguous: the protein cut into five stretches of positions; train and calibration
   come from three of them, test from the held-out stretches, which is harder because the
   model never saw a mutation at those positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "OATML-Markslab/ProteinGym_v1"
SHARDS = [f"DMS_substitutions/train-{i:05d}-of-00005.parquet" for i in range(5)]
ROOT = Path("data/proteingym")

ASSAYS = [
    "CCDB_ECOLI_Tripathi_2016",
    "SUMO1_HUMAN_Weile_2017",
    "CALM1_HUMAN_Weile_2017",
    "TAT_HV1BR_Fernandes_2016",
    "ESTA_BACSU_Nutschel_2020",
    "DYR_ECOLI_Thompson_2019",
    "SYUA_HUMAN_Newberry_2020",
    "UBC9_HUMAN_Weile_2017",
    "NUD15_HUMAN_Suiter_2020",
    "RASH_HUMAN_Bandaru_2017",
    "TPMT_HUMAN_Matreyek_2018",
    "BLAT_ECOLX_Firnberg_2014",
]
# Never trained on: the shift test asks what happens to a model's intervals on new proteins.
HELD_OUT_FAMILIES = ["SYUA_HUMAN_Newberry_2020", "TPMT_HUMAN_Matreyek_2018", "UBC9_HUMAN_Weile_2017"]
MUTANT = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass
class Assay:
    dms_id: str
    wild_type: str
    variants: pd.DataFrame  # mutant, position (0-based), wt, mt, sequence, score, z


def download(root: Path = ROOT) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(REPO, s, repo_type="dataset", local_dir=str(root))) for s in SHARDS]


def load(assays: list[str] | None = None, root: Path = ROOT) -> dict[str, Assay]:
    wanted = set(assays or ASSAYS)
    paths = sorted(root.glob("DMS_substitutions/*.parquet")) or download(root)
    cols = ["mutant", "DMS_score", "DMS_id", "mutated_sequence"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in paths])
    df = df[df["DMS_id"].isin(wanted) & ~df["mutant"].str.contains(":")]
    out = {}
    for dms_id, g in df.groupby("DMS_id"):
        parts = g["mutant"].str.extract(MUTANT)
        g = g.assign(wt=parts[0], position=parts[1].astype(int) - 1, mt=parts[2])
        seq = list(g["mutated_sequence"].iloc[0])
        first = g.iloc[0]
        seq[first["position"]] = first["wt"]
        wild_type = "".join(seq)
        z = (g["DMS_score"] - g["DMS_score"].mean()) / g["DMS_score"].std()
        variants = pd.DataFrame(
            {
                "mutant": g["mutant"].to_numpy(),
                "position": g["position"].to_numpy(),
                "wt": g["wt"].to_numpy(),
                "mt": g["mt"].to_numpy(),
                "sequence": g["mutated_sequence"].to_numpy(),
                "score": g["DMS_score"].to_numpy(),
                "z": z.to_numpy(),
            }
        ).reset_index(drop=True)
        out[str(dms_id)] = Assay(str(dms_id), wild_type, variants)
    return out


def split(assay: Assay, scheme: str = "random", seed: int = 0) -> pd.Series:
    """'train', 'calib' or 'test' for each variant."""
    v = assay.variants
    rng = np.random.default_rng(seed)
    if scheme == "random":
        u = rng.random(len(v))
        return pd.Series(np.where(u < 0.6, "train", np.where(u < 0.8, "calib", "test")), index=v.index)
    if scheme == "contiguous":
        positions = np.sort(v["position"].unique())
        blocks = np.array_split(positions, 5)
        test_blocks = {int(b) for b in rng.choice(5, size=2, replace=False)}
        block_of = {int(p): i for i, b in enumerate(blocks) for p in b}
        in_test = v["position"].map(lambda p: block_of[int(p)] in test_blocks).to_numpy()
        u = rng.random(len(v))
        return pd.Series(np.where(in_test, "test", np.where(u < 0.75, "train", "calib")), index=v.index)
    raise ValueError(scheme)
