"""
评估器

对模型在给定数据加载器上运行完整评估,聚合所有指标.
支持总体评估 + 逐类细粒度分析.

用法:
    from cnnlib.evaluation.evaluator import Evaluator

    evaluator = Evaluator(model, testLoader, lossFn, device, numClasses=10)
    result = evaluator.evaluate()
    # → {"loss": 0.23, "accuracy": 0.93, "confusion_matrix": ..., ...}
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnnlib.evaluation.metrics import computeAllMetrics


class Evaluator:
    """
    模型评估器

    在给定 DataLoader 上运行完整评估,收集:
        - 损失值
        - Top-1 / Top-K 准确率
        - 混淆矩阵
        - 每类准确率
        - 精确率 / 召回率 / F1(宏观 & 微观)
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        lossFn: nn.Module,
        device: torch.device,
        numClasses: int,
        topK: int = 5,
    ):
        """
        Args:
            model:      模型实例
            loader:     数据加载器
            lossFn:     损失函数
            device:     计算设备
            numClasses: 类别总数
            topK:       Top-K 准确率的 K 值
        """
        self.model = model
        self.loader = loader
        self.lossFn = lossFn
        self.device = device
        self.numClasses = numClasses
        self.topK = topK

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        运行评估

        Returns:
            完整指标字典
        """
        self.model.eval()

        allOutputs = []
        allLabels = []
        totalLoss = 0.0

        for images, labels in self.loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.lossFn(outputs, labels)

            totalLoss += loss.item() * images.size(0)
            allOutputs.append(outputs.cpu())
            allLabels.append(labels.cpu())

        allOutputs = torch.cat(allOutputs)
        allLabels = torch.cat(allLabels)

        metrics = computeAllMetrics(allOutputs, allLabels, self.numClasses, self.topK)
        metrics["loss"] = totalLoss / allLabels.size(0)
        metrics["num_samples"] = allLabels.size(0)

        return metrics

    @torch.no_grad()
    def evaluatePerClass(self) -> Dict[int, Dict[str, float]]:
        """
        逐类评估

        Returns:
            {classIdx: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "support": ...}, ...}
        """
        self.model.eval()

        allOutputs = []
        allLabels = []
        for images, labels in self.loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            allOutputs.append(outputs.cpu())
            allLabels.append(labels.cpu())

        allOutputs = torch.cat(allOutputs)
        allLabels = torch.cat(allLabels)
        _, predicted = allOutputs.max(1)

        result = {}
        eps = 1e-8
        for clsIdx in range(self.numClasses):
            # TP: 预测=clsIdx 且 真实=clsIdx
            tp = ((predicted == clsIdx) & (allLabels == clsIdx)).sum().item()
            # FP: 预测=clsIdx 且 真实≠clsIdx
            fp = ((predicted == clsIdx) & (allLabels != clsIdx)).sum().item()
            # FN: 预测≠clsIdx 且 真实=clsIdx
            fn = ((predicted != clsIdx) & (allLabels == clsIdx)).sum().item()
            # 支持数(真实为该类的样本数)
            support = (allLabels == clsIdx).sum().item()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            result[clsIdx] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": tp / (support + eps) if support > 0 else 0.0,
                "support": support,
            }

        return result
