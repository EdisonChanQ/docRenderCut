"""验收度量：用数据集的 ground-truth 变换，精确计算配准残差。

为什么能算得准：合成样本的几何扰动是我们自己施加的，`H_ref_to_sample` 精确已知。
于是「引擎把样本映射回标准坐标系」与「真值映射」之差，就是坐标误差本身——
不需要人工标注，不需要抽检。

残差的物理含义：目标内容块在输出图上的位置，与标准坐标之差（像素）。
配准偏多少，块就偏多少。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _as3(m) -> np.ndarray:
    return np.asarray(m, dtype=np.float64)


def apply3(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,2) -> (N,2)，按 3x3 齐次映射。"""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    out = (M @ h.T).T
    return out[:, :2] / out[:, 2:3]


@dataclass
class SampleResidual:
    sample_id: str
    status: str
    dpi: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    per_point: list[float] = field(default_factory=list)
    reason: str = ""


def ground_truth_sample_to_canvas(H_sample_to_ref: np.ndarray,
                                  ref_size: tuple[int, int],
                                  canvas: tuple[int, int]) -> np.ndarray:
    """真值映射：样本坐标 -> 画布坐标。

    参考页(reference)与画布(canvas)之间的映射是纯像素尺度换算
    （参考页由 210mm 版面渲染而来，尺寸未必正好等于 A4 标称像素）。
    """
    rw, rh = ref_size
    cw, ch = canvas
    S = np.array([[cw / rw, 0.0, 0.0],
                  [0.0, ch / rh, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return S @ _as3(H_sample_to_ref)


def probe_points(canvas: tuple[int, int], *, grid: int = 5,
                 inset: float = 0.05) -> np.ndarray:
    """画布坐标下的探测点：整页 5x5 网格 + 四角 + 边中点。

    必须覆盖全幅：残余的透视/非刚性误差恰恰集中在外围，
    只测中心会给出虚假的乐观数字。
    """
    cw, ch = canvas
    xs = np.linspace(cw * inset, cw * (1 - inset), grid)
    ys = np.linspace(ch * inset, ch * (1 - inset), grid)
    grid_pts = [[x, y] for y in ys for x in xs]
    extra = [[0, 0], [cw, 0], [cw, ch], [0, ch],
             [cw / 2, 0], [0, ch / 2], [cw, ch / 2], [cw / 2, ch]]
    return np.asarray(grid_pts + extra, dtype=np.float64)


def evaluate_sample(sample: dict, record: dict, ref_size: tuple[int, int],
                    canvas: tuple[int, int]) -> SampleResidual:
    """残差 = 目标块在**输出画布**上的坐标误差，单位 px @ 画布尺度。

    度量方式必须直接在画布坐标系里做，不能「在样本坐标系量误差再乘一个系数」：
    样本坐标系到画布的换算系数是逐样本不同的（72dpi 样本是 4.17，300dpi 样本是 1.0），
    用常数近似会让低分辨率样本的误差被系统性低估 4 倍。
    """
    sid = sample["id"]
    if record.get("status") != "ok":
        return SampleResidual(sid, record.get("status", "missing"), sample["canvas"]["dpi"],
                              np.nan, np.nan, np.nan, np.nan, np.nan,
                              reason=record.get("reason", ""))

    G = ground_truth_sample_to_canvas(_as3(sample["H_sample_to_ref"]), ref_size, canvas)
    E = _as3(record["matrix_sample_to_canvas"])

    pts = probe_points(canvas)                       # 画布坐标下的探测点
    # 该画布位置「本应对应」的样本点（真值），引擎会把它放到哪里
    s_at_pts = apply3(np.linalg.inv(G), pts)
    placed = apply3(E, s_at_pts)
    d = np.linalg.norm(placed - pts, axis=1)         # 画布像素

    return SampleResidual(
        sid, "ok", sample["canvas"]["dpi"],
        float(d.mean()), float(np.percentile(d, 50)),
        float(np.percentile(d, 95)), float(np.percentile(d, 99)), float(d.max()),
        per_point=[round(float(v), 4) for v in d],
    )


def evaluate_dataset(dataset_dir: str | Path, out_dir: str | Path) -> dict:
    """把数据集的 manifest 与引擎输出的 sidecar 对照，产出残差报告。"""
    root = Path(dataset_dir)
    out_root = Path(out_dir)
    mf = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ref_size = (mf["reference"]["width"], mf["reference"]["height"])

    side_dir = out_root / "sidecar"
    results: list[SampleResidual] = []
    canvas = None

    for s in mf["samples"]:
        stem = f"{s['id']}_p0001"
        side = side_dir / f"{stem}.json"
        if not side.exists():
            results.append(SampleResidual(s["id"], "missing", s["canvas"]["dpi"],
                                         np.nan, np.nan, np.nan, np.nan, np.nan,
                                         reason="无 sidecar 输出"))
            continue
        rec = json.loads(side.read_text(encoding="utf-8"))
        canvas = (rec["canvas"]["width"], rec["canvas"]["height"])
        results.append(evaluate_sample(s, rec, ref_size, canvas))

    ok = [r for r in results if r.status == "ok"]
    summary = {
        "canvas": list(canvas) if canvas else None,
        "total": len(results),
        "ok": len(ok),
        "rejected": len(results) - len(ok),
    }
    if ok:
        arr = np.array([r.mean for r in ok])
        allp95 = np.array([r.p95 for r in ok])
        allmax = np.array([r.max for r in ok])
        summary.update({
            "residual_mean_avg": float(arr.mean()),
            "residual_mean_max": float(arr.max()),
            "p95_avg": float(allp95.mean()),
            "p95_max": float(allp95.max()),
            "max_over_pages": float(allmax.max()),
        })
        # 按 DPI 分组：看分辨率是否为残差主因
        by_dpi: dict[int, list[float]] = {}
        for r in ok:
            by_dpi.setdefault(r.dpi, []).append(r.mean)
        summary["by_dpi"] = {
            str(k): {"n": len(v), "mean": round(float(np.mean(v)), 4)}
            for k, v in sorted(by_dpi.items())
        }

    report = {"summary": summary, "samples": [r.__dict__ for r in results]}
    (out_root / "residual-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    s = report["summary"]
    lines = ["", "=" * 62,
             "配准残差（目标块坐标误差，单位 px @ 画布尺度）",
             "=" * 62,
             f"  画布            : {s.get('canvas')}",
             f"  总数/合格/拒收  : {s['total']} / {s['ok']} / {s['rejected']}"]
    if "residual_mean_avg" in s:
        lines += [
            f"  页均误差 均值   : {s['residual_mean_avg']:.3f} px",
            f"  页均误差 最大   : {s['residual_mean_max']:.3f} px",
            f"  P95 均值        : {s['p95_avg']:.3f} px",
            f"  P95 最大        : {s['p95_max']:.3f} px",
            f"  全页最大误差    : {s['max_over_pages']:.3f} px",
        ]
        if "by_dpi" in s:
            lines.append("  按输入 DPI 分组 :")
            for k, v in s["by_dpi"].items():
                lines.append(f"      {k:>3} dpi  n={v['n']:>2}  均值 {v['mean']:.3f} px")
    lines.append("=" * 62)
    return "\n".join(lines)
