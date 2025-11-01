"""Cross-domain evaluation toolkit for feature extractor failure analysis.

This script implements the evaluation procedures that quantify and visualize
how a feature extractor that is trained only on the source domain fails when
directly applied to a disjoint target domain.  The pipeline follows the
multi-stage protocol described in the experiment plan and produces:

* Maximum Mean Discrepancy (MMD) between source/target deep features.
* t-SNE visualization for the mixed feature distribution.
* Grad-CAM visual evidence together with IoU / Precision / Recall metrics.
* Silhouette score and cosine-similarity heatmap for target features.
* Mean activation magnitude comparison for shallow features (layer1).

Example usage:

```
python domain_failure_analysis.py
```

The script assumes that the datasets are already downloaded and organised in
the same directory structure that is required by the training/testing scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from common import utils
from data.dataset import FSSDataset
from model.patnet import PATNetwork


# Normalisation statistics that are used during training/testing.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-domain feature extractor evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='./logs/pascal_resnet50_run_none',
        help="训练好的模型权重路径。如果未提供，将尝试从配置文件中读取。",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default='./data',
        help="数据集根目录（覆盖配置文件中的设置）",
    )
    parser.add_argument("--source-benchmark", type=str, default="pascal", help="源域基准名称")
    parser.add_argument("--target-benchmark", type=str, default="lung", help="目标域基准名称")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vgg16"],
                        help="待评估的特征提取器骨干网络，目前run_backbone仅支持ResNet-50")
    parser.add_argument("--img-size", type=int, default=400, help="输入图片的尺寸")
    parser.add_argument("--batch-size", type=int, default=8, help="特征抽取时的数据批大小")
    parser.add_argument("--nshot", type=int, default=1, help="few-shot episode中的shot数量")
    parser.add_argument("--num-source-samples", type=int, default=256, help="用于统计的源域样本数")
    parser.add_argument("--num-target-samples", type=int, default=256, help="用于统计的目标域样本数")
    parser.add_argument(
        "--cam-samples",
        type=int,
        default=12,
        help="用于Grad-CAM可视化和掩码指标的样本数量",
    )
    parser.add_argument(
        "--cam-threshold",
        type=str,
        default="mean",
        choices=["mean", "median", "otsu"],
        help="Grad-CAM热力图二值化阈值的选择方式",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE可视化的perplexity参数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备，自动检测GPU/CPU",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./log/analysis_outputs",
        help="结果保存目录",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="与测试脚本保持一致的fold设置（Pascal默认值为4）",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def load_config_datapath(config_path: Path, override_datapath: Optional[str]) -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if override_datapath:
        return override_datapath

    datapath = None
    if "test" in config and isinstance(config["test"], dict):
        datapath = config["test"].get("datapath")
    if datapath is None and "train" in config and isinstance(config["train"], dict):
        datapath = config["train"].get("datapath")
    if datapath is None:
        raise ValueError("无法从配置文件中解析到数据路径，请通过 --datapath 指定。")
    return datapath


def resolve_checkpoint(config_path: Path, override_checkpoint: Optional[str]) -> str:
    if override_checkpoint:
        return override_checkpoint

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint = None
    if "test" in config and isinstance(config["test"], dict):
        checkpoint = config["test"].get("load_model_path")

    if not checkpoint:
        raise ValueError("请通过 --checkpoint 指定模型权重路径，或在配置文件test段落中提供 load_model_path。")

    return checkpoint


def build_model(checkpoint_path: str, device: torch.device, backbone: str) -> PATNetwork:
    logging.info("加载模型权重: %s", checkpoint_path)
    model = PATNetwork(backbone=backbone)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_backbone(backbone: torch.nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """前向传播得到 layer1 特征、layer4 特征和 GAP 后的全局特征。"""
    x = backbone.conv1(images)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    layer1 = backbone.layer1(x)
    layer2 = backbone.layer2(layer1)
    layer3 = backbone.layer3(layer2)
    layer4 = backbone.layer4(layer3)

    pooled = backbone.avgpool(layer4)
    flattened = torch.flatten(pooled, 1)

    return layer1, layer4, flattened


@torch.no_grad()
def collect_features(
    dataloader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    device: torch.device,
    max_samples: int,
) -> Dict[str, torch.Tensor]:
    """收集指定数量样本的深层特征和浅层激活统计。"""

    features: List[torch.Tensor] = []
    activation_magnitudes: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    processed = 0
    for batch in dataloader:
        imgs = batch["query_img"].to(device)
        layer1, layer4, gap_feats = run_backbone(backbone, imgs)

        features.append(gap_feats.cpu())
        activation_magnitudes.append(layer1.detach().abs().mean(dim=(1, 2, 3)).cpu())
        labels.append(batch["class_id"].cpu())

        processed += imgs.size(0)
        if processed >= max_samples:
            break

    feature_tensor = torch.cat(features, dim=0)[:max_samples]
    activation_tensor = torch.cat(activation_magnitudes, dim=0)[:max_samples]
    label_tensor = torch.cat(labels, dim=0)[:max_samples]

    return {
        "features": feature_tensor,
        "activation": activation_tensor,
        "labels": label_tensor,
    }


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    squared_dist = x_norm + y_norm.T - 2 * x @ y.T
    return squared_dist.clamp_min(0.0)


def gaussian_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.size(0) < 2 or y.size(0) < 2:
        return float("nan")

    device = x.device
    combined = torch.cat([x, y], dim=0)
    distance_matrix = pairwise_squared_distance(combined, combined)
    mask = ~torch.eye(distance_matrix.size(0), dtype=torch.bool, device=device)
    valid_distances = distance_matrix[mask]
    median = valid_distances.median().item()
    if median <= 0:
        median = distance_matrix.mean().item()
    if median <= 0:
        median = 1.0
    gamma = 1.0 / (2 * median)

    k_xx = torch.exp(-gamma * pairwise_squared_distance(x, x))
    k_yy = torch.exp(-gamma * pairwise_squared_distance(y, y))
    k_xy = torch.exp(-gamma * pairwise_squared_distance(x, y))

    mmd_value = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd_value.item()


def run_tsne(features: np.ndarray, labels: np.ndarray, perplexity: float) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42)
    embedding = tsne.fit_transform(features)
    return embedding


def plot_tsne(embedding: np.ndarray, domain_labels: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    markers = {0: "o", 1: "s"}
    colors = {0: "#1f77b4", 1: "#d62728"}
    for domain in np.unique(domain_labels):
        idx = domain_labels == domain
        plt.scatter(embedding[idx, 0], embedding[idx, 1], s=20, marker=markers.get(domain, "o"),
                    c=colors.get(domain, "#333333"), label="Source" if domain == 0 else "Target", alpha=0.7)
    plt.legend()
    plt.title("t-SNE of Source vs Target Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cosine_similarity(features: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    order = np.argsort(labels)
    sorted_features = features[order]
    sorted_labels = labels[order]
    similarity = cosine_similarity(sorted_features)

    plt.figure(figsize=(8, 8))
    plt.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Cosine Similarity")
    plt.title("Target Feature Cosine Similarity (Sorted by Class)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    np.save(output_path.with_suffix(".npy"), similarity)
    np.save(output_path.with_name(output_path.stem + "_labels.npy"), sorted_labels)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    img = tensor * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    return img


def resize_cam(cam: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    return cam


def normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)
    normalised = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return normalised


def otsu_threshold(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = values.size
    sum_total = np.dot(hist, (bin_edges[:-1] + bin_edges[1:]) / 2.0)
    sum_bg = 0.0
    weight_bg = 0.0
    max_var = -1.0
    threshold = 0.5

    for idx in range(256):
        weight_bg += hist[idx]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[idx] * ((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > max_var:
            max_var = between_var
            threshold = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0

    return float(threshold)


def threshold_cam(cam: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(cam.mean())
    if mode == "median":
        return float(np.median(cam))
    return otsu_threshold(cam)


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred_binary = pred_mask.astype(bool)
    gt_binary = gt_mask.astype(bool)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_area = pred_binary.sum()
    gt_area = gt_binary.sum()

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0

    return {"iou": float(iou), "precision": float(precision), "recall": float(recall)}


def save_gradcam_visualisation(
    image: torch.Tensor,
    cam: torch.Tensor,
    mask: Optional[torch.Tensor],
    save_path: Path,
) -> None:
    img_np = tensor_to_image(image)
    cam_np = cam.detach().cpu().numpy()
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(img_np)
    plt.imshow(cam_np, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")
    else:
        plt.imshow(np.zeros_like(cam_np), cmap="gray")
        plt.title("No GT")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_layer1_grid(layer1: torch.Tensor, save_path: Path, max_channels: int = 8) -> None:
    channels = min(layer1.size(1), max_channels)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()
    for idx in range(rows * cols):
        axes[idx].axis("off")

    for idx in range(channels):
        channel_map = layer1[0, idx].detach().cpu().numpy()
        channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-6)
        axes[idx].imshow(channel_map, cmap="viridis")
        axes[idx].set_title(f"Ch {idx}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_gradcam_results(
    dataloader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    threshold_mode: str,
    prefix: str,
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    saved = 0
    iterator = iter(dataloader)

    while saved < num_samples:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        imgs = batch["query_img"].to(device)
        masks = batch.get("query_mask")

        for idx in range(imgs.size(0)):
            image = imgs[idx: idx + 1]
            mask = masks[idx] if masks is not None else None

            layer1, layer4, _ = run_backbone(backbone, image)

            if saved == 0:
                save_layer1_grid(layer1, output_dir / f"{prefix}_layer1_channels.png")

            layer4 = layer4.requires_grad_()
            score = layer4.mean(dim=(1, 2, 3))
            backbone.zero_grad(set_to_none=True)
            score.backward()
            gradients = layer4.grad

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * layer4).sum(dim=1))
            cam = resize_cam(cam, image.shape[-2:])
            cam = normalise_cam(cam)[0]

            cam_np = cam.detach().cpu().numpy()
            threshold = threshold_cam(cam_np, threshold_mode)
            pred_mask = (cam_np >= threshold).astype(np.uint8)
            if mask is not None:
                mask_cpu = mask.detach().cpu()
                gt_mask = mask_cpu.squeeze().numpy()
                metrics.append(compute_mask_metrics(pred_mask, gt_mask))
            else:
                metrics.append({"iou": float("nan"), "precision": float("nan"), "recall": float("nan")})

            save_gradcam_visualisation(image.squeeze(0), cam, mask_cpu if mask is not None else None,
                                       output_dir / f"{prefix}_cam_{saved:03d}.png")

            saved += 1
            if saved >= num_samples:
                break

    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    utils.fix_randseed(0)

    config_path = Path(args.config)
    datapath = load_config_datapath(config_path, args.datapath)
    checkpoint_path = resolve_checkpoint(config_path, args.checkpoint)

    if args.backbone != "resnet50":
        raise NotImplementedError("当前run_backbone实现仅支持ResNet-50骨干。")

    device = torch.device(args.device)
    model = build_model(checkpoint_path, device, args.backbone)
    backbone = model.backbone

    FSSDataset.initialize(img_size=args.img_size, datapath=datapath)
    source_loader = FSSDataset.build_dataloader(
        args.source_benchmark, args.batch_size, nworker=0, fold=args.fold, split="test", shot=args.nshot
    )
    target_loader = FSSDataset.build_dataloader(
        args.target_benchmark, args.batch_size, nworker=0, fold=args.fold, split="test", shot=args.nshot
    )

    logging.info("收集源域特征 (%s)...", args.source_benchmark)
    source_stats = collect_features(source_loader, backbone, device, args.num_source_samples)
    logging.info("收集目标域特征 (%s)...", args.target_benchmark)
    target_stats = collect_features(target_loader, backbone, device, args.num_target_samples)

    source_features = source_stats["features"].double()
    target_features = target_stats["features"].double()

    mmd_value = gaussian_mmd(source_features, target_features)
    logging.info("MMD (Source vs Target) = %.6f", mmd_value)

    mean_activation_source = source_stats["activation"].mean().item()
    mean_activation_target = target_stats["activation"].mean().item()

    logging.info("平均浅层激活 (源域) = %.6f", mean_activation_source)
    logging.info("平均浅层激活 (目标域) = %.6f", mean_activation_target)

    combined_features = torch.cat([source_features, target_features], dim=0).numpy()
    domain_labels = np.concatenate([
        np.zeros(source_features.size(0), dtype=np.int32),
        np.ones(target_features.size(0), dtype=np.int32),
    ])

    logging.info("进行t-SNE降维并绘图...")
    embedding = run_tsne(combined_features, domain_labels, args.tsne_perplexity)
    plot_tsne(embedding, domain_labels, output_dir / "tsne_source_target.png")

    target_np = target_features.numpy()
    target_labels_np = target_stats["labels"].numpy()

    logging.info("计算目标域特征的轮廓系数...")
    try:
        silhouette = silhouette_score(target_np, target_labels_np)
    except Exception as exc:  # sklearn可能在类别数不足时抛出异常
        logging.warning("无法计算轮廓系数: %s", exc)
        silhouette = float("nan")

    logging.info("Silhouette Score = %s", silhouette)
    plot_cosine_similarity(target_np, target_labels_np, output_dir / "target_feature_similarity.png")

    logging.info("生成源域Grad-CAM分析...")
    source_cam_metrics = generate_gradcam_results(
        source_loader, backbone, device, output_dir, args.cam_samples, args.cam_threshold, prefix="source"
    )
    logging.info("生成目标域Grad-CAM分析...")
    target_cam_metrics = generate_gradcam_results(
        target_loader, backbone, device, output_dir, args.cam_samples, args.cam_threshold, prefix="target"
    )

    def summarise_cam(metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {"iou": float("nan"), "precision": float("nan"), "recall": float("nan")}
        keys = metrics[0].keys()
        return {key: float(np.nanmean([m[key] for m in metrics])) for key in keys}

    summary = {
        "mmd": mmd_value,
        "mean_activation": {
            "source": mean_activation_source,
            "target": mean_activation_target,
        },
        "silhouette": silhouette,
        "gradcam_metrics": {
            "source": summarise_cam(source_cam_metrics),
            "target": summarise_cam(target_cam_metrics),
        },
        "num_source_samples": int(source_features.size(0)),
        "num_target_samples": int(target_features.size(0)),
        "cam_samples": args.cam_samples,
        "cam_threshold": args.cam_threshold,
        "source_benchmark": args.source_benchmark,
        "target_benchmark": args.target_benchmark,
    }

    metrics_path = output_dir / "metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("指标已保存至 %s", metrics_path)


if __name__ == "__main__":
    main()

