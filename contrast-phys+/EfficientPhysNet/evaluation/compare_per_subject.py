# -*- coding: utf-8 -*-
"""
Per-subject 跨 scale 对比（取 subject 交集，控制变量）

用法:
  cd contrast-phys+
  python EfficientPhysNet/evaluation/compare_per_subject.py [t10_dir] [t30_dir] [t60_dir]
  不传参则用默认路径: .../curriculum/t{10,30,60}/1
"""
import os
import re
import sys

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)
_RESULTS = os.path.join(_CP, "results", "EfficientPhysNet", "label_ratio_0", "inference", "curriculum")


def _parse_per_subject_mae(summary_path):
    """从 summary.txt 解析 Per-subject 的 MAE，返回 {subject: mae}"""
    out = {}
    if not os.path.isfile(summary_path):
        return out
    with open(summary_path, "r", encoding="utf-8") as f:
        in_section = False
        for line in f:
            if line.strip().startswith("Per-subject:"):
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("-") or (line.strip() and not line.startswith(" ")):
                    break
                m = re.match(r"\s+(\S+):\s+.*MAE=([\d.]+)BPM", line)
                if m:
                    out[m.group(1)] = float(m.group(2))
    return out


def main():
    if len(sys.argv) >= 4:
        dirs = [sys.argv[1], sys.argv[2], sys.argv[3]]
    else:
        dirs = [
            os.path.join(_RESULTS, "t10", "1"),
            os.path.join(_RESULTS, "t30", "1"),
            os.path.join(_RESULTS, "t60", "1"),
        ]
    scales = ["t10", "t30", "t60"]
    data = {}
    for scale, d in zip(scales, dirs):
        summary = os.path.join(d, "eval", "summary.txt")
        data[scale] = _parse_per_subject_mae(summary)
        if not data[scale]:
            print(f"警告: 未找到或无法解析 {summary}")

    inter = set(data["t10"]) & set(data["t30"]) & set(data["t60"])
    inter = sorted(inter, key=lambda s: (int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 0))

    if not inter:
        print("无 subject 交集，无法对比")
        return

    lines = []
    lines.append("Per-subject 跨 scale 对比（subject 交集 n={}）".format(len(inter)))
    lines.append("=" * 56)
    lines.append("{:12} | {:>8} | {:>8} | {:>8} | {:>8}".format(
        "subject", "t10 MAE", "t30 MAE", "t60 MAE", "mean"))
    lines.append("-" * 56)
    maes = {s: [] for s in scales}
    for subj in inter:
        row = [data[s].get(subj, float("nan")) for s in scales]
        valid = [r for r in row if r == r]
        mean_val = sum(valid) / len(valid) if valid else float("nan")
        for s, v in zip(scales, row):
            maes[s].append(v)
        lines.append("{:12} | {:>7.2f} | {:>7.2f} | {:>7.2f} | {:>7.2f}".format(
            subj, row[0], row[1], row[2], mean_val))
    lines.append("-" * 56)
    import numpy as np
    means = [np.nanmean(maes[s]) for s in scales]
    lines.append("{:12} | {:>7.2f} | {:>7.2f} | {:>7.2f}".format("mean(n={})".format(len(inter)), means[0], means[1], means[2]))
    lines.append("")
    lines.append("说明: 仅对 t10/t30/t60 均有数据的 subject 做对比，控制 subject 变量")

    out_text = "\n".join(lines)
    print(out_text)

    out_dir = os.path.join(_EPN, "evaluation")
    out_path = os.path.join(out_dir, "per_subject_compare.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
