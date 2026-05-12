# cnnlib/registry/datasets.py

"""
数据集注册表

只做三件事:
    1. 登记:每个数据集一行字典条目
    2. 查询:列出所有数据集、获取单个数据集元信息
    3. 告诉 transform 和 loader 该怎么处理数据

用法:
    from cnnlib.registry.datasets import list_datasets, get_dataset_info

    info = get_dataset_info("cifar10")
    # info["channels"] → 3
    # info["num_classes"] → 10
"""

from typing import Any, Dict, List

# train_kwargs / test_kwargs:
#   透传给 torchvision 数据集类的构造参数（root, download, transform 由 loader 统一注入）
#   默认 {"train": True} / {"train": False}；SVHN、EMNIST 等需要特殊构造时在此覆盖

datasets: Dict[str, Dict[str, Any]] = {
    # ── 灰度 28×28 ──────────────────────────────────────
    "mnist": {
        "channels": 1,
        "num_classes": 10,
        "image_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
        "description": "MNIST handwritten digits (10 classes)",
    },
    "fashionmnist": {
        "channels": 1,
        "num_classes": 10,
        "image_size": 28,
        "mean": (0.2860,),
        "std": (0.3530,),
        "description": "Fashion-MNIST clothing (10 classes)",
    },
    "emnist": {
        "channels": 1,
        "num_classes": 47,
        "image_size": 28,
        "mean": (0.1736,),
        "std": (0.3317,),
        "description": "EMNIST letters + digits (47 classes)",
        # EMNIST 需要指定 split，且同时传 train
        "train_kwargs": {"split": "balanced", "train": True},
        "test_kwargs": {"split": "balanced", "train": False},
    },
    # ── RGB 32×32 ───────────────────────────────────────
    "cifar10": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "description": "CIFAR-10 natural images (10 classes)",
    },
    "cifar100": {
        "channels": 3,
        "num_classes": 100,
        "image_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "description": "CIFAR-100 natural images (100 classes)",
    },
    "svhn": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 32,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "description": "SVHN street view house numbers (10 classes)",
        # SVHN 用 split= 而非 train=
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "test"},
    },
    # ── RGB 96×96 ───────────────────────────────────────
    "stl10": {
        "channels": 3,
        "num_classes": 10,
        "image_size": 96,
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "description": "STL-10 natural images (10 classes)",
        # STL10 用 split= 而非 train=
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "test"},
    },
    # ── RGB 尺寸不固定 ──────────────────────────────────
    "caltech101": {
        "channels": 3,
        "num_classes": 101,
        "image_size": None,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Caltech-101 objects (101 classes)",
        # Caltech101 无内置 train/test 分集，全量加载后自行切分
        "train_kwargs": {},
        "test_kwargs": {},
    },
    "gtsrb": {
        "channels": 3,
        "num_classes": 43,
        "image_size": None,
        "mean": (0.3403, 0.3121, 0.3214),
        "std": (0.2724, 0.2608, 0.2669),
        "description": "GTSRB traffic signs (43 classes)",
        # GTSRB 用 split= 而非 train=
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "test"},
    },
    "flowers102": {
        "channels": 3,
        "num_classes": 102,
        "image_size": None,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Oxford Flowers-102 (102 classes)",
        # Flowers102 用 split= 而非 train=（split 取值: train/val/test）
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "test"},
    },
}


def list_datasets() -> List[str]:
    """返回所有已注册数据集的名称列表"""
    return list(datasets.keys())


def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    获取数据集元信息

    Args:
        name: 数据集名称(大小写不敏感),如 "cifar10", "mnist"

    Returns:
        {"channels": int, "num_classes": int, "image_size": int|None,
         "mean": tuple, "std": tuple, "description": str,
         "train_kwargs": dict|None, "test_kwargs": dict|None}
    """
    key = name.lower()
    if key not in datasets:
        available = ", ".join(datasets.keys())
        raise KeyError(f"Unknown dataset: '{name}'. Available: {available}")
    return datasets[key]


def print_datasets():
    """打印所有已注册数据集(调试用)"""
    print(f"{'Name':<14} {'Ch':<4} {'Classes':<8} {'Size':<10} Description")
    print("-" * 75)
    for name, info in datasets.items():
        size_str = (
            f"{info['image_size']}x{info['image_size']}"
            if info["image_size"]
            else "variable"
        )
        extra = ""
        if "train_kwargs" in info:
            extra = f"  [kwargs: {list(info['train_kwargs'].keys())}]"
        print(
            f"  {name:<12} {info['channels']:<4} "
            f"{info['num_classes']:<8} {size_str:<10} {info['description']}{extra}"
        )


if __name__ == "__main__":
    print_datasets()
