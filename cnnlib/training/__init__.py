"""
训练模块

提供完整的模型训练基础设施:
    - 损失函数工厂 (loss)
    - 优化器工厂 (optimizer)
    - 学习率调度器工厂 (scheduler)
    - 早停机制 (earlyStopping)
    - Checkpoint 管理 (checkpoint)
    - 日志 + TensorBoard (logger)
    - 训练/验证引擎 (engine)
    - 训练编排器 (trainer)

用法:
    from cnnlib.training import Trainer, createLoss, createOptimizer
"""

from cnnlib.training.checkpoint import loadCheckpoint, saveCheckpoint
from cnnlib.training.earlyStopping import EarlyStopping
from cnnlib.training.engine import trainOneEpoch, validate
from cnnlib.training.logger import TrainingLogger
from cnnlib.training.loss import createLoss
from cnnlib.training.optimizer import createOptimizer
from cnnlib.training.scheduler import createScheduler
from cnnlib.training.trainer import Trainer

__all__ = [
    "Trainer",
    "trainOneEpoch",
    "validate",
    "createLoss",
    "createOptimizer",
    "createScheduler",
    "EarlyStopping",
    "saveCheckpoint",
    "loadCheckpoint",
    "TrainingLogger",
]
