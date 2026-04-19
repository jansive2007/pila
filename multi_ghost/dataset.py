from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ACTION_COLUMNS = ["w", "a", "s", "d", "q", "e", "space", "shift_ctrl"]


@dataclass
class SampleRef:
    frame_paths: List[Path]
    action: np.ndarray
    ghost_id: int
    run_id: str


class MultiGhostDataset(Dataset):
    """
    Loads samples from dataset/ghost_*/runs/run_*/actions.csv
    and returns stacked frames + keyboard action vector.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        stack_size: int = 4,
        image_size: Tuple[int, int] = (180, 320),
        ghost_ids: Optional[Sequence[int]] = None,
        randomize: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.stack_size = stack_size
        self.image_size = image_size
        self.ghost_ids = set(ghost_ids) if ghost_ids else None

        self.samples = self._build_samples()
        if randomize:
            random.shuffle(self.samples)

    def _build_samples(self) -> List[SampleRef]:
        samples: List[SampleRef] = []

        action_files = sorted(self.root.glob("ghost_*/runs/run_*/actions.csv"))
        for action_file in action_files:
            rows = self._read_rows(action_file)
            if not rows:
                continue

            ghost_id = int(rows[0]["ghost_id"])
            run_id = rows[0]["run_id"]
            if self.ghost_ids is not None and ghost_id not in self.ghost_ids:
                continue

            frame_paths = [self.root / row["frame_path"] for row in rows]
            actions = [np.array([float(row[c]) for c in ACTION_COLUMNS], dtype=np.float32) for row in rows]

            for i in range(self.stack_size - 1, len(rows)):
                stack = frame_paths[i - self.stack_size + 1 : i + 1]
                samples.append(SampleRef(frame_paths=stack, action=actions[i], ghost_id=ghost_id, run_id=run_id))

        return samples

    @staticmethod
    def _read_rows(action_file: Path) -> List[Dict[str, str]]:
        with action_file.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames = [self._load_frame(p) for p in sample.frame_paths]

        stacked = np.concatenate(frames, axis=0)  # (stack*C, H, W)
        frame_tensor = torch.from_numpy(stacked).float() / 255.0
        action_tensor = torch.from_numpy(sample.action)

        meta = {
            "ghost_id": sample.ghost_id,
            "run_id": sample.run_id,
            "last_frame": str(sample.frame_paths[-1]),
        }
        return frame_tensor, action_tensor, meta

    def _load_frame(self, frame_path: Path) -> np.ndarray:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Frame missing: {frame_path}")
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        return image


def build_dataloader(
    dataset_root: str | Path,
    batch_size: int = 32,
    stack_size: int = 4,
    image_size: Tuple[int, int] = (180, 320),
    ghost_ids: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MultiGhostDataset(
        dataset_root=dataset_root,
        stack_size=stack_size,
        image_size=image_size,
        ghost_ids=ghost_ids,
        randomize=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
