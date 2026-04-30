# MNIST-CNN

基于 PyTorch 实现的 MNIST 手写数字识别 CNN 模型，完整工程化项目。

## 项目结构

```
MNIST-CNN/
├── main.py
│
├── config/
│   ├── __init__.py
│   ├── default.py
│   ├── paths.py
│   └── settings.py
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transform.py
│   │   └── loader.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── cnn.py
│   │   └── factory.py
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── checkpoint.py
│   │   └── logger.py
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualize.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
│
├── docs/
│   ├── architecture.md
│   ├── usage.md
│   └── api.md
│
├── checkpoints/
├── outputs/
├── datasets/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境要求

- Python >= 3.11
- CUDA >= 12.8（可选，支持 CPU 训练）

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
python main.py train
```

### 评估

```bash
python main.py eval --checkpoint checkpoints/best.pth
```

### 推理

```bash
python main.py infer --image path/to/digit.png
```

## 配置

所有超参数和路径通过 `config/` 模块集中管理，支持命令行 `--config` 覆盖。

主要可配置项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批次大小 |
| `lr` | 1e-3 | 初始学习率 |
| `epochs` | 20 | 训练轮数 |
| `device` | auto | 训练设备（cuda/cpu） |

## 模型

CNN 结构：ConvBlock × 2 → Flatten → LinearBlock × 2 → Linear(10)

目标：测试集准确率 > 99%

## 开发

```bash
# 代码格式化与检查
ruff format .
ruff check .

# 运行测试
pytest
```
