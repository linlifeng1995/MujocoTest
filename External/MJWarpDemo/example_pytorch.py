"""Minimal PyTorch consumption example; install torch in your training environment."""

from pathlib import Path

import h5py
import torch


def load_episode(path: str | Path) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as episode:
        qpos = torch.from_numpy(episode["observations/qpos"][...])
        frame_count = qpos.shape[0]
        return {
            "qpos": qpos,
            "qvel": torch.from_numpy(episode["observations/qvel"][...]),
            "goal_position": (
                torch.from_numpy(episode["observations/goal_position"][...])
                if "observations/goal_position" in episode
                else torch.zeros((frame_count, 3), dtype=torch.float32)
            ),
            "task_stage": (
                torch.from_numpy(episode["observations/task_stage"][...])
                if "observations/task_stage" in episode
                else torch.zeros(frame_count, dtype=torch.int16)
            ),
            "action": torch.from_numpy(episode["actions"][...]),
            "reward": torch.from_numpy(episode["rewards"][...]),
            "rgb": torch.from_numpy(episode["images/rgb"][...]).permute(0, 3, 1, 2),
            "depth_m": torch.from_numpy(episode["images/depth_m"][...]).unsqueeze(1),
            "instance_id": torch.from_numpy(episode["images/instance_id"][...]),
        }
