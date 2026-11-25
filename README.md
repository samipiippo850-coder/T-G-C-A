# T-G-C-A

## 中文

本仓库整理 T-G-C-A 长期多变量时间序列预测研究的模型代码和实验入口。核心实现 `TGCA_SOTA_Adapter_V2` 在基础预测器上加入残差适配器：趋势、局部卷积和频谱专家产生候选修正，由门控、方向控制和残差幅度约束调节最终预测。代码还包含可选的协方差状态、能量/熵与路由扩展；这些扩展是否启用取决于具体运行配置，不能默认所有历史结果都使用了它们。

### 已开展的工作

- 在 ETTh1、Weather、Traffic、Electricity 等数据集上开展长期预测实验，重点考察预测长度 720。
- 对 Traffic 探索输入长度、适配器幅度上限、时间池化和训练参数；现有记录中的一个 12-bin 配置测试 MSE 为 0.449498、MAE 为 0.302212。该数值来自历史单次实验，不代表本仓库当前源码已独立复现。
- 对 Traffic 和 Weather 的预测窗口计算协方差谱、有效秩、自由能等描述量，并探索专家权重变化。它们是机制诊断，不能单凭相关性证明物理相变或模块因果作用。
- 准备专家移除、均匀门控、路由、方向与幅度限制等消融。完整重训、多随机种子和多预测长度的验证仍需补充；请以各次实验日志和检查点为准。

### 目录与运行

`models/` 包含模型，`data_provider/`、`exp/`、`layers/` 和 `utils/` 包含数据处理、实验和训练代码，`run.py` 是训练入口。`ablations/` 保存隔离式消融入口；它通过环境变量选择变体，不改写共享模型源码。其中 `ablation_entry.py` 显式启用额外协方差组件并修正批次维度，`ablation_original_candidate_entry.py` 使用另一组候选配置：两者不应未经核验与历史检查点混用。

数据和模型权重未包含在本仓库中。安装 PyTorch 及 `requirements.txt` 中依赖后，将 ETTh1 数据放在 `dataset/ETT-small/ETTh1.csv`，例如：

```bash
python run.py --task_name long_term_forecast --is_training 1 \
  --model_id ETTh1_720 --model TGCA_SOTA_Adapter_V2 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7
```

实际复现实验时还须记录源码版本、模型初始化、数据划分、所有环境变量和完整训练参数；示例只是入口，不承诺复现上述历史指标。测试集按实验划分使用，不能仅依据展示过多个结果就认定数据泄漏。

## English

This repository collects the model code and experimental entry points for T-G-C-A, a long-horizon multivariate time-series forecasting project. `TGCA_SOTA_Adapter_V2` adds a residual adapter to a base forecaster. Trend, local convolutional and spectral experts propose corrections, which are controlled by gating, direction and a residual-magnitude constraint. Optional covariance-state, energy/entropy and routing extensions are configuration-dependent and were not necessarily active in every historical run.

### Work carried out

- Ran long-horizon forecasting experiments on ETTh1, Weather, Traffic and Electricity, with particular attention to horizon 720.
- Explored input length, adapter caps, temporal pooling and training settings on Traffic. One historical 12-bin run reports test MSE 0.449498 and MAE 0.302212. This is a single recorded run, not an independently verified reproduction by the code currently in this repository.
- Measured covariance spectra, effective rank and free-energy descriptors on Traffic and Weather windows, alongside expert-weight interventions. These are diagnostic observations; correlation alone establishes neither a physical phase transition nor a causal module effect.
- Prepared ablations for expert removal, uniform gating, routing, direction and magnitude constraints. Complete retraining, multi-seed and multi-horizon evaluation remain to be established from experiment logs and checkpoints.

### Layout and usage

`models/` contains model implementations; `data_provider/`, `exp/`, `layers/` and `utils/` contain data, experiment and training code; `run.py` is the main entry point. `ablations/` contains isolated ablation entry points selected via environment variables. `ablation_entry.py` enables additional covariance components and corrects a batch-dimension issue, whereas `ablation_original_candidate_entry.py` uses a different candidate configuration. Do not assume either matches a historical checkpoint without verification.

Datasets and weights are not included. After installing PyTorch and the dependencies in `requirements.txt`, place ETTh1 at `dataset/ETT-small/ETTh1.csv` and use the command above as a starting point. Record source version, initialization, data splits, environment variables and complete training settings for each reproduction; the example does not guarantee the historical metrics. The test set follows the experiment's partitioning protocol; displaying multiple results alone does not establish data leakage.

The included framework retains the upstream MIT license in `LICENSE`.
