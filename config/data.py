"""
数据加载与预处理默认超参数

用法:
    from config.data import DataParams
    from config import DataParams
"""


class DataParams:
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    VAL_SPLIT: float = 0.1

    # 数据增强
    ENABLE_DATA_AUGMENTATION: bool = True
    ROTATION_DEGREES: int = 10
    TRANSLATION_RATIO: float = 0.1
