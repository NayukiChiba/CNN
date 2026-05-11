"""
训练超参数默认值：优化器、学习率调度、正则化

用法:
    from config.training import TrainingParams
    from config import TrainingParams
"""


class TrainingParams:
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4

    # ReduceLROnPlateau 调度器
    LR_FACTOR: float = 0.5
    LR_PATIENCE: int = 3
    LR_MIN: float = 1e-6
