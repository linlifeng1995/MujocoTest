from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import default_artifacts_root, new_run_directory, select_device, set_deterministic, write_json
from .data import episode_path, load_manifest, records_for
from .models import TinyUNet


def frame_index(manifest: dict[str, Any], split: str, stride: int, max_frames: int) -> list[tuple[Path, int]]:
    result: list[tuple[Path, int]] = []
    for record in records_for(manifest, split=split):
        path = episode_path(manifest, record)
        result.extend((path, frame) for frame in range(0, int(record["frames"]), max(1, stride)))
    return result[:max_frames] if max_frames > 0 else result


def discover_class_ids(samples: list[tuple[Path, int]]) -> list[int]:
    identifiers: set[int] = {0}
    for path, frame in samples:
        with h5py.File(path, "r") as episode:
            instance_key = "images/front_instance_id" if "images/front_instance_id" in episode else "images/instance_id"
            identifiers.update(int(value) for value in np.unique(episode[instance_key][frame]))
    return sorted(identifiers)


class InstanceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[tuple[Path, int]], class_ids: list[int]) -> None:
        self.samples = samples
        self.lookup = np.zeros(65536, dtype=np.int64)
        for class_index, instance_id in enumerate(class_ids):
            self.lookup[instance_id] = class_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, frame = self.samples[index]
        with h5py.File(path, "r") as episode:
            image_key = "images/front_rgb" if "images/front_rgb" in episode else "images/rgb"
            rgb = np.asarray(episode[image_key][frame], dtype=np.float32) / 255.0
            instance_key = "images/front_instance_id" if "images/front_instance_id" in episode else "images/instance_id"
            instance = np.asarray(episode[instance_key][frame], dtype=np.uint16)
        image = torch.from_numpy(rgb).permute(2, 0, 1)
        target = torch.from_numpy(self.lookup[instance])
        return image, target


def evaluate(
    model: TinyUNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    class_count: int,
    device: torch.device,
) -> dict[str, Any]:
    intersections = np.zeros(class_count, dtype=np.float64)
    unions = np.zeros(class_count, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for image, target in loader:
            prediction = model(image.to(device)).argmax(dim=1).cpu().numpy()
            truth = target.numpy()
            for class_id in range(class_count):
                predicted_mask = prediction == class_id
                truth_mask = truth == class_id
                intersections[class_id] += np.count_nonzero(predicted_mask & truth_mask)
                unions[class_id] += np.count_nonzero(predicted_mask | truth_mask)
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    present = unions > 0
    return {
        "miou": float(np.mean(iou[present])) if np.any(present) else 0.0,
        "class_iou": iou.tolist(),
        "class_pixels_union": unions.astype(np.int64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 RGB 与实例 ID 训练轻量实例分割模型")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-train-frames", type=int, default=5000)
    parser.add_argument("--max-validation-frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = select_device(args.device)
    manifest = load_manifest(args.manifest)
    train_samples = frame_index(manifest, "train", args.frame_stride, args.max_train_frames)
    validation_samples = frame_index(
        manifest, "validation", args.frame_stride, args.max_validation_frames
    )
    if not train_samples:
        raise ValueError("training split contains no RGB/instance frames")
    if not validation_samples:
        validation_samples = train_samples[: min(len(train_samples), args.max_validation_frames)]
    class_ids = discover_class_ids(train_samples)
    train_dataset = InstanceDataset(train_samples, class_ids)
    validation_dataset = InstanceDataset(validation_samples, class_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyUNet(len(class_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for image, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
    validation_metrics = evaluate(model, validation_loader, len(class_ids), device)
    metrics = {
        "model_type": "instance_segmentation",
        "scenario": manifest["scenario"],
        "validation": validation_metrics,
        "class_ids": class_ids,
        "unknown_instance_policy": "map to background class 0",
    }
    run_id, directory = new_run_directory(args.artifacts_dir, manifest["scenario"], "segmentation")
    torch.save(model.state_dict(), directory / "model.pt")
    shutil.copy2(args.manifest, directory / "dataset_manifest.json")
    first_record = manifest["records"][0]
    write_json(
        directory / "model_spec.json",
        {
            "artifact_id": f"{manifest['scenario']}/{run_id}",
            "model_type": "instance_segmentation",
            "scenario": manifest["scenario"],
            "class_ids": class_ids,
            "class_count": len(class_ids),
            "image_width": first_record["image_width"],
            "image_height": first_record["image_height"],
            "base_channels": 16,
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    write_json(directory / "metrics.json", metrics)
    print(f"实例分割模型：{directory.resolve()}")
    print(f"验证 mIoU：{validation_metrics['miou']:.4f}")


if __name__ == "__main__":
    main()
