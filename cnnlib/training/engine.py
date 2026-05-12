"""
训练引擎

单轮训练和验证的核心循环,不涉及调度、日志、checkpoint 等外部关注点.
Trainer 在每轮开始时调用这些函数,拿到指标后自行决策.

用法:
    from cnnlib.training.engine import trainOneEpoch, validate

    trainMetrics = trainOneEpoch(model, loader, lossFn, optimizer, device, epoch)
    valMetrics = validate(model, valLoader, lossFn, device)
"""

from typing import Dict, Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnnlib.training.logger import TrainingLogger


def trainOneEpoch(
    model: nn.Module,
    loader: DataLoader,
    lossFn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger | None = None,
    gradClip: float = 0.0,
) -> Dict[str, float]:
    """
    训练一个 epoch

    Args:
        model:     模型
        loader:    训练集 DataLoader
        lossFn:    损失函数
        optimizer: 优化器
        device:    计算设备
        epoch:     当前轮次(仅用于进度条显示)
        logger:    日志器(可选,用于记录 batch-level 指标)
        gradClip:  梯度裁剪阈值(0=不裁剪)

    Returns:
        {"loss": 平均损失, "accuracy": 准确率%}
    """
    model.train()
    totalLoss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = lossFn(outputs, labels)
        loss.backward()

        if gradClip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradClip)

        optimizer.step()

        totalLoss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}", acc=f"{100.0 * correct / total:.1f}%"
        )

    pbar.close()

    return {
        "loss": totalLoss / total,
        "accuracy": 100.0 * correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    lossFn: nn.Module,
    device: torch.device,
    desc: Literal["Test", "Val"] = "Val",
) -> Dict[str, float]:
    """
    验证/测试

    Args:
        model:  模型
        loader: 验证/测试集 DataLoader
        lossFn: 损失函数
        device: 计算设备
        desc:   进度条描述

    Returns:
        {"loss": 平均损失, "accuracy": 准确率%}
    """
    model.eval()
    totalLoss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = lossFn(outputs, labels)

        totalLoss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}", acc=f"{100.0 * correct / total:.1f}%"
        )

    pbar.close()

    return {
        "loss": totalLoss / total,
        "accuracy": 100.0 * correct / total,
    }
