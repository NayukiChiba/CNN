"""
基准测试入口脚本

编排从 CLI 参数到基准测试完成的完整流程.
支持单组和全量模式，训练后自动出图.

用法:
    python main.py --model vgg16 --dataset cifar10 benchmark --epochs 5
    python main.py --model all --dataset cifar10 benchmark --epochs 3
    python main.py --model all --dataset all benchmark --epochs 1
"""

import argparse

from cnnlib.experiments.benchmark import runAllBenchmarks, runBenchmark
from cnnlib.registry.datasets import list_datasets
from cnnlib.registry.models import list_models


def run(args: argparse.Namespace) -> None:
    """基准测试入口"""

    models = list_models() if args.model == "all" else [args.model]
    datasets = list_datasets() if args.dataset == "all" else [args.dataset]

    isFull = args.model == "all" or args.dataset == "all"

    print(f"模型: {', '.join(models)}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"设备: {args.device}")
    print(f"Epochs: {args.epochs}")
    print()

    if isFull:
        runAllBenchmarks(
            models=models,
            datasets=datasets,
            device=args.device,
            epochs=args.epochs,
            batchSize=args.batch_size,
            dataDir=args.data_dir,
            outputDir=args.output_dir,
        )
    else:
        runBenchmark(
            modelName=models[0],
            datasetName=datasets[0],
            device=args.device,
            epochs=args.epochs,
            batchSize=args.batch_size,
            numWorkers=args.num_workers,
            dataDir=args.data_dir,
            outputDir=args.output_dir,
            seed=args.seed,
        )
