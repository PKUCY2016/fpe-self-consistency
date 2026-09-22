"""Render completed evidence; never modifies models, thresholds or selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/acceptance-v2"


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(CAMPAIGN / "summary.json")
    cases = ["uniform", "heat1d", "heat2d", "doublewell", "coupled2d"]
    lines = ["# FPE 工程验收记录", "", "本报告汇总全部预先声明的种子，不把运行成功或小训练残差等同于科学结论。",
             "", f"低维自适应测试通过：**{summary['adaptive_cases_passed']}**；保留审查批通过：**{summary['heldout_audit_passed']}**；共享预算上限的扩阶对照通过：**{summary['control']['passed']}**。",
             f"完整工程验收通过：**{summary['engineering_acceptance_passed']}**。连续模型的条件熵界、实测误差与未认证离散/统计误差分别记录。",
             "", "## 全部低维案例", "", "表中 KL、TV 和质量误差为 21 个预定时间点上的最大值；阈值分别为 0.001、0.02、0.0001。", "",
             "| 模式 | 案例 | 种子 | 状态 | 最大 KL | 最大 TV | 质量误差 | 评价通过 |", "|---|---|---:|---|---:|---:|---:|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#1565c0", "#e07b19", "#37966f"]
    for mode in ("fixed", "adaptive"):
        for i, case in enumerate(cases):
            for seed in range(3):
                name = f"{mode}-{case}-s{seed}"
                item = summary["results"].get(name, {})
                metrics = item.get("maxima")
                if metrics:
                    lines.append(f"| {mode} | {case} | {seed} | {item['status']} | {metrics['kl']:.6g} | {metrics['tv']:.6g} | {metrics['mass_error']:.3g} | {item['passed']} |")
                    for axis, metric, threshold in zip(axes, ("kl", "tv"), (.001, .02)):
                        axis.scatter(i+(seed-1)*.12+(-.025 if mode == "fixed" else .025), max(metrics[metric], 1e-8), color=colors[seed], marker="x" if mode == "fixed" else "o", s=45, alpha=.85)
                else:
                    lines.append(f"| {mode} | {case} | {seed} | {item.get('status', 'MISSING')} | — | — | — | False |")
    for axis, metric, threshold in zip(axes, ("KL", "TV"), (.001, .02)):
        axis.set_xticks(range(len(cases)), cases, rotation=20)
        axis.set_yscale("log")
        axis.axhline(threshold, color="#a63030", linestyle="--", label="Frozen threshold")
        axis.set_title(f"Maximum {metric} over 21 times")
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="best")
    fig.suptitle("Development benchmarks: blue=0, orange=1, green=2; x=fixed, circle=adaptive")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/acceptance.png", dpi=170)
    plt.close(fig)
    lines += ["", "![开发基准全部种子误差](acceptance.png)", "", "图中仅为五个开发基准；零误差为便于对数显示置于 1e-8，表格保留真实数值。保留审查批见下表。", "", "## 预声明扩阶对照", "",
              "两组共享 120 秒总预算上限，训练、编译、候选探索和验证均计入；各自的实际耗时同时报告。参考评价在模型冻结后执行。固定组保留 K=2，最多 12 个各 500 更新的块，复用编译和优化器。", "",
              "| 种子 | 自适应最终 KL | 固定 K=2 最终 KL | 自适应秒数 | 固定秒数 | 预算合规 |", "|---:|---:|---:|---:|---:|---|"]
    for item in summary["control"].get("seeds", []):
        lines.append(f"| {item['seed']} | {item['adaptive_final_kl']:.6g} | {item['fixed_final_kl']:.6g} | {item['adaptive_seconds']:.2f} | {item['fixed_seconds']:.2f} | {item.get('budget_compliant')} |")
    lines += ["", f"KL 中位数降低比例：{summary['control'].get('median_kl_reduction', 0):.2%}。验收要求至少 30%，并至少两个种子改善。", "",
              "## 独立审查与压力测试", "", "| 任务 | 状态 | 评价通过 | 最大 KL | 最大 TV |", "|---|---|---|---:|---:|"]
    for name, item in sorted(summary["results"].items()):
        if name.startswith(("audit-", "stress-")):
            m = item.get("maxima", {})
            lines.append(f"| {name} | {item.get('status')} | {item.get('passed')} | {format(m['kl'], '.6g') if 'kl' in m else '—'} | {format(m['tv'], '.6g') if 'tv' in m else '—'} |")
    lines += ["", "审查批：heat1d 幅度 0.27、相位 0.71；coupled2d 幅度 0.31、相位 1.17。若依据这些结果改算法，本批必须转为开发数据，重新生成审查批。压力例均未达到精度阈值：高势垒耗尽 12 轮上限，高频初值停在平台期，粗步长与小 batch 初始模型无法通过数值检查。它们分别明确退出，保留全部证据。", "",
              "## 实测成本", "", "目标为同一组 Fourier 观测量和势阱质量的绝对误差 0.01。这里不以 SDE 的观测量误差替代密度 KL 或 TV。", "",
              "| 案例，seed=0 | 求解器训练与验证秒数 | 密度评价秒数 | 独立采样 2048 秒数 | 参考解秒数 | SDE 含细化秒数 | 达到同观测量精度 |", "|---|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        path = CAMPAIGN / f"adaptive-{case}-s0/evaluation.json"
        if path.exists():
            r = read(path)
            cost, classical = r["costs_seconds"], r.get("classical_comparison", {})
            lines.append(f"| {case} | {cost['training_search_validation_total']:.3f} | {cost['density_queries_and_refinement']:.3f} | {cost['dop853_sample_2048']:.3f} | {cost['reference']:.3f} | {classical.get('sde_total_seconds', float('nan')):.3f} | {classical.get('same_observable_precision_reached')} |")
    lines += ["", "热方程参考为解析公式，势阱/耦合参考为有限体积加稀疏矩阵指数。表中参考成本包含细化；编译加首次执行的独立上界见各 evaluation.json，已包含于训练总时长，不重复相加。本轮低维 CPU 案例中，包含训练与验证的求解器总成本高于这些经典对照；快速重复采样不抵消首次训练成本。本轮不支持速度优势。", "",
              "## 证据与边界", "", f"冻结规格 SHA-256：`{summary['specification_sha256']}`。", "",
              "- 固定基线复用的源码与逐元素等价证据：`baseline-reuse.json`。早期版本保存在 `acceptance-v1`，没有混入新规则的自适应三种子结果。",
              "- 完整机器可读证据：`../runs/acceptance-v2/summary.json`，各实例 `state.json`、`evaluation.json`、逐候选检查点与收据。",
              "- 源码外 wheel 入口验证：`delivery.json`。最终数学、恢复和运行时测试：`full-tests-final.txt`；早期记录 `full-tests-v2.txt` 继续保留。",
              "- 控制器无法读取参考密度或离线 KL；评价阈值没有自动放宽。",
              "- 条件熵界要求连续、光滑、正密度及精确总体积分。样本残差、网格细化和三倍配对标准误是工程证据，不是严格误差证书。",
              "- 最终审查 `audit-1-s1` 超过冻结 KL/TV 阈值，见 [独立失效分析](heldout-failure-review.md)。15/15 开发基准通过不覆盖这次失败，完整工程验收未通过。",
              "- 3072 维真实图像实验两种容量均未生成可辨认复杂图片，见 [图像工程报告](image-feasibility.md) 与 [独立证据核验](image-evidence-review.md)。",
              f"- 预定任务共 {len(summary['results'])} 项；尚缺任务：{summary['missing']}；异常进程：{summary.get('process_failures', {})}。", ""]
    (ROOT / "reports/acceptance.md").write_text("\n".join(lines))
    print(ROOT / "reports/acceptance.md")


if __name__ == "__main__":
    main()
