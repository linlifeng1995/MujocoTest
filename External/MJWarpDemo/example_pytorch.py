"""可直接复用的 PyTorch 数据配对示例；训练环境需安装 training 依赖组。"""

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
            "action": torch.from_numpy(
                episode["actions/normalized"][...] if "actions/normalized" in episode else episode["actions"][...]
            ),
            "reward": torch.from_numpy(episode["rewards"][...]),
            "rgb": torch.from_numpy(
                episode["images/front_rgb"][...] if "images/front_rgb" in episode else episode["images/rgb"][...]
            ).permute(0, 3, 1, 2),
            "depth_m": torch.from_numpy(
                episode["images/front_depth_m"][...] if "images/front_depth_m" in episode else episode["images/depth_m"][...]
            ).unsqueeze(1),
            "instance_id": torch.from_numpy(
                episode["images/front_instance_id"][...] if "images/front_instance_id" in episode else episode["images/instance_id"][...]
            ),
            "schema_v2": torch.tensor("actions/normalized" in episode),
        }


def behavior_cloning_pairs(episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """当前状态预测 action_t；Schema 2.0 无需动作错位补偿。"""
    observations = torch.cat(
        (
            episode["qpos"][:-1],
            episode["qvel"][:-1],
            episode["goal_position"][:-1],
            episode["task_stage"][:-1].float().unsqueeze(1),
        ),
        dim=1,
    )
    return observations, episode["action"] if bool(episode["schema_v2"]) else episode["action"][1:]


def segmentation_pairs(episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB 图像与同帧实例 ID 标签严格对齐，无需时序偏移。"""
    return episode["rgb"].float() / 255.0, episode["instance_id"].long()


def dynamics_transitions(
    episode: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 state、action、next_state、reward，用于单步动力学学习。"""
    state = torch.cat(
        (
            episode["qpos"],
            episode["qvel"],
            episode["goal_position"],
            episode["task_stage"].float().unsqueeze(1),
        ),
        dim=1,
    )
    if bool(episode["schema_v2"]):
        return state[:-1], episode["action"], state[1:], episode["reward"]
    return state[:-1], episode["action"][1:], state[1:], episode["reward"][1:]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="查看一个 MJWarp HDF5 回合的三类训练配对")
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    loaded = load_episode(args.episode)
    observation, action = behavior_cloning_pairs(loaded)
    rgb, instance = segmentation_pairs(loaded)
    state, dynamics_action, next_state, reward = dynamics_transitions(loaded)
    print(f"行为克隆：{tuple(observation.shape)} -> {tuple(action.shape)}")
    print(f"实例分割：{tuple(rgb.shape)} -> {tuple(instance.shape)}")
    print(
        f"动力学：state={tuple(state.shape)}, action={tuple(dynamics_action.shape)}, "
        f"next_state={tuple(next_state.shape)}, reward={tuple(reward.shape)}"
    )
