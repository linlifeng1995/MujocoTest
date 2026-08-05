"""Minimal PyTorch consumption example; install torch in your training environment."""

from pathlib import Path

import h5py
import torch


def load_episode(path: str | Path) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as episode:
        return {
            "qpos": torch.from_numpy(episode["observations/qpos"][...]),
            "qvel": torch.from_numpy(episode["observations/qvel"][...]),
            "action": torch.from_numpy(episode["actions"][...]),
            "reward": torch.from_numpy(episode["rewards"][...]),
            "rgb": torch.from_numpy(episode["images/rgb"][...]).permute(0, 3, 1, 2),
            "depth_m": torch.from_numpy(episode["images/depth_m"][...]).unsqueeze(1),
            "instance_id": torch.from_numpy(episode["images/instance_id"][...]),
        }
