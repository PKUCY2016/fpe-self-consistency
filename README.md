# FPE Self-Consistency Solver

工程化实现 Shen et al.（COLT 2022）Fokker-Planck 自洽思想的低维概率流求解器，并提供受控自适应演化、独立评价和可追溯实验记录。

> **当前状态：研究实现，完整科学验收未通过。**
>
> 15/15 个自适应开发基准通过，未参与调参的最终审查为 5/6；一次二维耦合审查超过冻结 KL/TV 阈值。真实 CIFAR-10 图像试验的两种容量均未生成可辨认自然物体。这里的结果适合继续研究和复核，不应表述为严格误差证书、通用高维求解器或成熟图片生成器。

## 能做什么

- 求解固定周期域 `[0, 2π)^d`、`d=1,2`、扩散系数为 1 的时间边缘分布。
- 使用 Fourier 空间基 × Bernstein 时间基表示速度场。
- 联合传播位置、log-density、score 和自洽残差，禁止自由 score 网络抵消残差。
- 从固定动作集合中一次选择一种修改：积分步数、batch、学习率、空间阶数或时间阶数。
- 记录候选、拒绝理由、检查点、随机状态、源码哈希、父版本和恢复状态。
- 用独立 NumPy/SciPy DOP853、解析解或有限体积参考解评价分布误差。

## 安装与快速运行

需要 Python 3.12 和 uv：

```bash
uv sync --frozen --python 3.12
uv run fpe doctor
uv run fpe train --problem heat1d --run runs/heat-fixed --seed 0
uv run fpe evolve --problem evolution --run runs/heat-adaptive --seed 0
uv run fpe evaluate --run runs/heat-adaptive
uv run fpe resume --run runs/heat-adaptive
```

Python 查询接口：

```python
from fpe_solver.api import ProbabilityFlow

flow = ProbabilityFlow.load("runs/heat-adaptive")
samples = flow.sample(1000, t=0.125, seed=11)
log_density = flow.log_density(samples, t=0.125)
```

运行目录必须是新目录。`--problem-json` 和 `--config` 可覆盖问题和求解配置；`doctor` 会报告版本、float64、设备和源码哈希。

## 数学对象与边界

增广状态满足

```text
x' = v
ℓ' = -div(v)
s' = -(Dv)^T s - grad(div(v))
q' = ||v + grad(V) + s||²
```

原论文的理论使用二阶 Sobolev 自洽目标；当前首版训练零阶残差，只做有限高阶诊断。因此实现中的训练损失、样本误差和网格细化都属于工程证据，不是严格数值认证。连续、光滑、严格正密度和精确流条件下的条件熵界记录在 `README` 与报告中。

## 验证结果

| 项目 | 结果 |
|---|---|
| 自适应开发基准 | 15/15 通过，5 类问题 × 3 个种子 |
| 最终独立审查 | 5/6 通过；失败案例最大 KL 0.00314443、TV 0.03076995 |
| 扩阶受控对照 | 3/3 种子改善，最终 KL 中位数降低 96.24% |
| 自动化测试 | 116 passed；1 个 Linux 专用测试在 macOS 跳过，已在 DSW 通过 |
| 源码外安装 | 新 Python 3.12 环境中的非 editable wheel 验证通过 |
| 图像分支 | rank 32 / 128 均为视觉生成失败；残差降低 33.5% 但无可辨认物体 |

图片实验在 3072 维欧氏 OU 分支中运行，与低维周期求解器验收分开。它使用已知高斯混合初值，因此不能证明从未知自然图像分布学习。

## 目录

- `src/fpe_solver/`：核心求解器、表示、训练、控制器、评价器和 CLI。
- `tests/`：数学核验、评价、控制器、恢复、图像和运行时测试。
- `reports/acceptance.md`：45 个预定低维任务的汇总。
- `reports/heldout-failure-review.md`：最终审查失败的独立复核。
- `reports/image-feasibility.md`：真实图片试验及限制。
- `reports/delivery.json`：源码外安装和 CLI/API 验证。
- `docs/FPE工程验证总结报告.pdf`：可直接转发的整体中文报告。
- `docs/assets/`：验收图和两组未挑选的图片样本。
- `deploy/README.md`：DSW 独立环境、资源限制和生命周期说明。

## 复现测试

```bash
uv run pytest -p no:cacheprovider
uv run ruff check src tests deploy
```

完整验收运行产生的 `runs/`、本地虚拟环境和临时文件不进入仓库；原始检查点和完整回执保留在项目归档中。最终 wheel 可用 `uv build` 重新生成。

## 参考

Shen, Z., Wang, Z., Kale, S., Ribeiro, A., Karbasi, A., and Hassani, H. *Self-Consistency of the Fokker-Planck Equation*. COLT 2022. [PMLR](https://proceedings.mlr.press/v178/shen22a.html)

CIFAR-10 official dataset page: [Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
