"""
InteractiveCLI 单元测试

覆盖: 输入辅助函数、模型选择、数据集选择、参数设置、工作流分支、边界条件
"""

from unittest.mock import patch

# 触发模型 & 数据集注册
import cnnlib.models  # noqa: F401
from cnnlib.registry.datasets import list_datasets
from cnnlib.registry.models import list_models

# ── 输入辅助函数 ────────────────────────────────────────────────


class TestPromptHelpers:
    """_prompt / _promptChoice / _promptInt / _promptFloat"""

    def test_prompt_returns_input_when_not_empty(self, monkeypatch):
        from cnnlib.cli.interactive import _prompt

        monkeypatch.setattr("builtins.input", lambda _: "  hello  ")
        result = _prompt("名称")
        assert result == "hello"

    def test_prompt_returns_default_when_empty(self, monkeypatch):
        from cnnlib.cli.interactive import _prompt

        monkeypatch.setattr("builtins.input", lambda _: "")
        result = _prompt("名称", default="defaultVal")
        assert result == "defaultVal"

    def test_prompt_returns_empty_when_no_default(self, monkeypatch):
        from cnnlib.cli.interactive import _prompt

        monkeypatch.setattr("builtins.input", lambda _: "")
        result = _prompt("名称")
        assert result == ""

    def test_prompt_choice_by_index(self, monkeypatch):
        from cnnlib.cli.interactive import _promptChoice

        inputs = iter(["2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = _promptChoice("选择", ["a", "b", "c"])
        assert result == "b"

    def test_prompt_choice_default_first(self, monkeypatch):
        from cnnlib.cli.interactive import _promptChoice

        monkeypatch.setattr("builtins.input", lambda _: "")
        result = _promptChoice("选择", ["x", "y", "z"], default="1")
        assert result == "x"  # 序号 1 → 第一项

    def test_prompt_choice_invalid_index_returns_first(self, monkeypatch):
        from cnnlib.cli.interactive import _promptChoice

        inputs = iter(["99", "abc"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result1 = _promptChoice("选择", ["a", "b"])
        result2 = _promptChoice("选择", ["a", "b"])
        # 超出范围或非数字 → 返回第一项
        assert result1 == "a"
        assert result2 == "a"

    def test_prompt_int_normal(self, monkeypatch):
        from cnnlib.cli.interactive import _promptInt

        monkeypatch.setattr("builtins.input", lambda _: "128")
        assert _promptInt("数量", default=64) == 128

    def test_prompt_int_returns_default_on_garbage(self, monkeypatch):
        from cnnlib.cli.interactive import _promptInt

        monkeypatch.setattr("builtins.input", lambda _: "xyz")
        assert _promptInt("数量", default=32) == 32

    def test_prompt_int_clamped(self, monkeypatch):
        from cnnlib.cli.interactive import _promptInt

        monkeypatch.setattr("builtins.input", lambda _: "0")
        assert _promptInt("数量", default=10, minV=1) == 1

        monkeypatch.setattr("builtins.input", lambda _: "99999")
        assert _promptInt("数量", default=10, maxV=100) == 100

    def test_prompt_float_normal(self, monkeypatch):
        from cnnlib.cli.interactive import _promptFloat

        monkeypatch.setattr("builtins.input", lambda _: "0.005")
        assert _promptFloat("学习率", default=0.001) == 0.005

    def test_prompt_float_returns_default_on_garbage(self, monkeypatch):
        from cnnlib.cli.interactive import _promptFloat

        monkeypatch.setattr("builtins.input", lambda _: "not a number")
        assert _promptFloat("学习率", default=0.001) == 0.001

    def test_prompt_float_clamped(self, monkeypatch):
        from cnnlib.cli.interactive import _promptFloat

        monkeypatch.setattr("builtins.input", lambda _: "-1")
        assert _promptFloat("学习率", default=0.001, minV=0.0) == 0.0

        monkeypatch.setattr("builtins.input", lambda _: "200")
        assert _promptFloat("学习率", default=0.001, maxV=100.0) == 100.0


# ── 颜色常量 ──────────────────────────────────────────────────────


class TestColors:
    """ANSI 颜色码存在性"""

    def test_all_colors_defined(self):
        from cnnlib.cli.interactive import _C

        assert _C.BOLD == "\033[1m"
        assert _C.END == "\033[0m"
        for attr in ["HEADER", "BLUE", "CYAN", "GREEN", "YELLOW", "RED", "BOLD", "DIM"]:
            assert hasattr(_C, attr)


# ── 注册表显示函数 ───────────────────────────────────────────────


class TestShowRegistered:
    """_showRegisteredModels / _showRegisteredDatasets"""

    def test_show_models_output(self, capsys):
        from cnnlib.cli.interactive import _showRegisteredModels

        _showRegisteredModels()
        captured = capsys.readouterr().out
        models = list_models()
        assert f"({len(models)}" in captured
        for name in sorted(models):
            assert name in captured

    def test_show_datasets_output(self, capsys):
        from cnnlib.cli.interactive import _showRegisteredDatasets

        _showRegisteredDatasets()
        captured = capsys.readouterr().out
        datasets = list_datasets()
        assert f"({len(datasets)}" in captured
        for name in sorted(datasets):
            assert name in captured


# ── InteractiveCLI 初始化 & 默认值 ───────────────────────────────


class TestCLIInit:
    """InteractiveCLI 构造 & 默认值"""

    def test_defaults(self):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        assert cli.model == "lenet"
        assert cli.dataset == "mnist"
        assert cli.device in ("cpu", "cuda")
        assert cli.batchSize == 64
        assert cli.epochs == 20
        assert cli.lr == 0.001
        assert cli.optimizer == "adam"
        assert cli.valSplit == 0.1
        assert cli.gradClip == 0.0

    def test_banner(self, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        cli._printBanner()
        captured = capsys.readouterr().out
        assert "CNN Image Classification CLI" in captured
        assert cli.device in captured


# ── 模型选择 ─────────────────────────────────────────────────────


class TestSelectModel:
    """_selectModel"""

    def test_select_valid_index(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        models = sorted(list_models())
        # 选择第2个模型
        monkeypatch.setattr("builtins.input", lambda _: "2")
        cli._selectModel()
        assert cli.model == models[1]
        captured = capsys.readouterr().out
        assert "已选择" in captured

    def test_select_invalid_index_keeps_current(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        original = cli.model
        monkeypatch.setattr("builtins.input", lambda _: "999")
        cli._selectModel()
        assert cli.model == original

    def test_select_empty_selects_first(self, monkeypatch):
        """空输入 → 默认 "1" → 选中第一个模型"""
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        models = sorted(list_models())
        cli.model = "vgg19"  # 刻意设为非第一个
        monkeypatch.setattr("builtins.input", lambda _: "")
        cli._selectModel()
        assert cli.model == models[0]


# ── 数据集选择 ───────────────────────────────────────────────────


class TestSelectDataset:
    """_selectDataset"""

    def test_select_valid_index(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        datasets = sorted(list_datasets())
        # 选择第1个
        monkeypatch.setattr("builtins.input", lambda _: "1")
        cli._selectDataset()
        assert cli.dataset == datasets[0]
        captured = capsys.readouterr().out
        assert "已选择" in captured

    def test_select_invalid_keeps_current(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        original = cli.dataset
        monkeypatch.setattr("builtins.input", lambda _: "abc")
        cli._selectDataset()
        assert cli.dataset == original


# ── 参数设置 ─────────────────────────────────────────────────────


class TestSettingsMenu:
    """_settingsMenu"""

    def test_update_all_settings(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        # 依次输入: epochs, batchSize, lr, valSplit, gradClip, optimizer
        inputs = iter(["50", "128", "0.01", "0.2", "1.0", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._settingsMenu()

        assert cli.epochs == 50
        assert cli.batchSize == 128
        assert cli.lr == 0.01
        assert cli.valSplit == 0.2
        assert cli.gradClip == 1.0
        assert cli.optimizer == "adamw"
        captured = capsys.readouterr().out
        assert "参数已更新" in captured

    def test_defaults_kept_on_empty_input(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        originalEpochs = cli.epochs
        originalLr = cli.lr
        # 全部回车 → 保持默认
        inputs = iter(["", "", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._settingsMenu()
        assert cli.epochs == originalEpochs
        assert cli.lr == originalLr


# ── 主循环 ───────────────────────────────────────────────────────


class TestMainLoop:
    """run() 主循环"""

    def test_exit_immediately(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        monkeypatch.setattr("builtins.input", lambda _: "0")
        cli.run()
        captured = capsys.readouterr().out
        assert "再见" in captured

    def test_view_models_then_exit(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["7", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        captured = capsys.readouterr().out
        assert "已注册模型" in captured

    def test_view_datasets_then_exit(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["8", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        captured = capsys.readouterr().out
        assert "已注册数据集" in captured

    def test_empty_input_continues(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        captured = capsys.readouterr().out
        assert "再见" in captured

    def test_invalid_choice_shows_warning(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["invalid", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        captured = capsys.readouterr().out
        assert "无效选项" in captured

    def test_select_model_then_settings_then_exit(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["1", "1", "9", "50", "128", "0.01", "0.2", "0.0", "3", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run()
        # 选中了第一个模型
        models = sorted(list_models())
        assert cli.model == models[0]
        assert cli.epochs == 50
        assert cli.batchSize == 128
        assert cli.optimizer == "sgd"


# ── 工作流分支 ───────────────────────────────────────────────────


class TestWorkflows:
    """训练/评估/推理/基准测试 流程入口"""

    def test_train_cancel(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cli._trainWorkflow()
        captured = capsys.readouterr().out
        assert "已取消" in captured

    def test_train_confirm(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        # 确认训练 → 但会调用 _executeTrain，里面会下载数据
        # 这里只测到确认阶段
        inputs = iter(["y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        # monkeypatch _executeTrain 防止实际执行
        with patch.object(cli, "_executeTrain"):
            cli._trainWorkflow()
        captured = capsys.readouterr().out
        assert "启动训练" in captured

    def test_train_yes_full_word(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        with patch.object(cli, "_executeTrain"):
            cli._trainWorkflow()

    def test_eval_missing_checkpoint(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["nonexistent.pth"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._evalWorkflow()
        captured = capsys.readouterr().out
        assert "不存在" in captured

    def test_eval_cancel(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter([__file__, "n"])  # 用自身文件假装 checkpoint
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._evalWorkflow()
        captured = capsys.readouterr().out
        assert "已取消" in captured

    def test_infer_missing_image_path(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter([__file__, ""])  # 空图片路径
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._inferWorkflow()
        captured = capsys.readouterr().out
        assert "请输入图片路径" in captured

    def test_infer_missing_checkpoint(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["nonexistent.pth"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._inferWorkflow()
        captured = capsys.readouterr().out
        assert "不存在" in captured

    def test_benchmark_cancel(self, monkeypatch, capsys):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        # 输入: mode=单组, epochs=3, confirm=n
        inputs = iter(["1", "3", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._benchmarkWorkflow()
        captured = capsys.readouterr().out
        assert "已取消" in captured

    def test_benchmark_confirm(self, monkeypatch):
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        # 输入: mode=单组, epochs=2, confirm=y
        inputs = iter(["1", "2", "y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with patch.object(cli, "_executeBenchmark"):
            cli._benchmarkWorkflow()

    def test_benchmark_full_cancel(self, monkeypatch, capsys):
        """全量模式取消"""
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()
        inputs = iter(["2", "3", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli._benchmarkWorkflow()
        captured = capsys.readouterr().out
        assert "已取消" in captured


# ── 边界条件 & Windows 编码兼容性 ────────────────────────────────


class TestEdgeCases:
    """边界条件"""

    def test_unicode_encoding(self, capsys):
        """验证 Unicode 字符可正常输出（不抛 UnicodeEncodeError）"""
        from cnnlib.cli.interactive import _error, _info, _section, _success, _warn

        _section("测试章节")
        _success("操作成功")
        _warn("警告信息")
        _error("错误信息")
        _info("键", "值")

        captured = capsys.readouterr().out
        # 即使 GBK 编码也应可输出（字符已在 BMP 范围内）
        assert "测试章节" in captured
        assert "操作成功" in captured
        assert "警告信息" in captured
        assert "错误信息" in captured
        assert "键" in captured
        assert "值" in captured

    def test_print_box(self, capsys):
        from cnnlib.cli.interactive import _printBox

        _printBox("Hello Test")
        captured = capsys.readouterr().out
        assert "Hello Test" in captured

    def test_multiple_rounds_no_state_leak(self, monkeypatch):
        """多轮交互后状态正确切换"""
        from cnnlib.cli.interactive import InteractiveCLI

        cli = InteractiveCLI()

        # 第一轮: 选模型
        monkeypatch.setattr("builtins.input", lambda _: "2")
        cli._selectModel()
        modelAfterRound1 = cli.model

        # 第二轮: 选数据集
        datasets = sorted(list_datasets())
        monkeypatch.setattr("builtins.input", lambda _: "1")
        cli._selectDataset()
        assert cli.model == modelAfterRound1  # 模型不变
        assert cli.dataset == datasets[0]
