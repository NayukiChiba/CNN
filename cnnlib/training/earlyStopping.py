"""
早停机制

监控验证指标,在连续 patience 轮无改善时触发停止信号.

用法:
    from cnnlib.training.earlyStopping import EarlyStopping

    es = EarlyStopping(patience=10, minDelta=0.001, mode="max")
    for epoch in range(epochs):
        valMetric = validate(model, valLoader, ...)
        if es.step(valMetric):
            print(f"早停触发于 epoch {epoch}, 最佳指标: {es.bestMetric:.4f}")
            break
"""


class EarlyStopping:
    """
    早停控制器

    mode="max" 时指标越大越好(如准确率)
    mode="min" 时指标越小越好(如损失值)
    """

    def __init__(self, patience: int = 10, minDelta: float = 0.001, mode: str = "max"):
        """
        Args:
            patience:  容忍轮数,连续无改善达此值即停止
            minDelta:  最小改善阈值,超过此值才算有效改善
            mode:      "max" 或 "min"
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode 必须是 'max' 或 'min',收到: '{mode}'")

        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode
        self.bestMetric: float | None = None
        self.counter: int = 0
        self._shouldStop: bool = False

    def step(self, metric: float) -> bool:
        """
        记录一轮指标,返回是否应该停止

        Args:
            metric: 当前轮的验证指标

        Returns:
            True 表示应该停止训练
        """
        if self.bestMetric is None:
            self.bestMetric = metric
            self.counter = 0
            return False

        if self.mode == "max":
            improved = metric > self.bestMetric + self.minDelta
        else:
            improved = metric < self.bestMetric - self.minDelta

        if improved:
            self.bestMetric = metric
            self.counter = 0
        else:
            self.counter += 1

        self._shouldStop = self.counter >= self.patience
        return self._shouldStop

    @property
    def shouldStop(self) -> bool:
        """是否应该停止训练"""
        return self._shouldStop

    def reset(self) -> None:
        """重置早停状态(用于重新训练)"""
        self.bestMetric = None
        self.counter = 0
        self._shouldStop = False
