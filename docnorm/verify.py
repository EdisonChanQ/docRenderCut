"""裁剪结果的验证物：把同一块从多张标准输出里按**同一组固定坐标**裁出来堆叠。

这是「坐标是否真的固定住了」的直接证据。它的逻辑很朴素：
如果整套流程真把每页映射进了同一个坐标系，那么用同一组坐标裁出来的同一块，
版式（表格边框、标签、栏目分隔线）应该逐像素重合，只有填写内容不同。
反之，只要有任何一页没对齐，堆叠图里那一行就会明显错位。

除了给人看，还给数：逐块与参考图（模板同坐标区域）做相位相关求相对位移，
位移量就是**这块内容在输出图上的坐标误差**，可直接和阈值比。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .crops import CropSpec, _to_gray, crop_one
from .loader import load_image


@dataclass
class Row:
    label: str          # 展示用（ASCII）
    name: str           # 完整来源名
    image: np.ndarray
    is_reference: bool = False


def _ascii(s: str, keep: int = 26) -> str:
    out = "".join(ch if 32 <= ord(ch) < 127 else "." for ch in s)
    return out[:keep]


def _bounded_shift(a: np.ndarray, b: np.ndarray, *,
                   max_shift: int | None = None) -> tuple[float, float, float]:
    """有界互相关求亚像素位移 → (dx, dy, 归一化峰)。

    为什么不用 cv2.phaseCorrelate：**它只保留相位，遇高周期性内容会混叠**。
    实测一个 `亿千百十万千百十元角分` 金额栏目（15+ 条等距竖线），
    两张明显错位的图被报成 dx=−355.66px——正好是栏目周期的整数倍。
    这类假读数比不报还危险：它看起来是个"具体的数"。

    这里改用幅度加权互相关，并把搜索**限制在 ±max_shift** 内：
    超出去的位置直接不可见，于是周期性内容无法把峰值吸到远处。
    代价是超过 max_shift 的真实位移会被截断显示——但那种量级的偏移
    本来就远大于容差，截断后依然是"大幅超限"，判定结论不受影响。

    峰用抛物线做亚像素细化（对 1px 分辨率的离散峰，精度到 0.1px 量级）。

    **返回值的含义**（实测校准，不要凭直觉改）：`_bounded_shift(X, Y)` 返回的是
    「X 的内容相对于 Y 的位移」。所以要比"这一块相对参考图偏了多少"，
    调用应当是 `_bounded_shift(该图, 参考图)`。
    自检记录：把 a 平移 (+7,+5) 得到 b 时，`_bounded_shift(a, b)` 返回 (−7,−5)。
    """
    fa = _to_gray(a).astype(np.float32)
    fb = _to_gray(b).astype(np.float32)
    if fa.shape != fb.shape or fa.size == 0:
        return (float("nan"), float("nan"), 0.0)

    h, w = fa.shape
    if max_shift is None:
        max_shift = int(max(16, min(64, 0.12 * min(h, w))))

    fa = fa - fa.mean()
    fb = fb - fb.mean()
    if fa.std() < 1e-6 or fb.std() < 1e-6:
        return (0.0, 0.0, 0.0)

    c = np.fft.irfft2(np.fft.fft2(fa) * np.conj(np.fft.fft2(fb)), s=fa.shape)
    c = np.fft.fftshift(c)
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - max_shift), min(h, cy + max_shift + 1)
    x0, x1 = max(0, cx - max_shift), min(w, cx + max_shift + 1)
    win = c[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)

    def _para(im1: float, i0: float, ip1: float, fallback: float) -> float:
        denom = (im1 - 2.0 * i0 + ip1)
        if abs(denom) < 1e-12:
            return fallback
        d = 0.5 * (im1 - ip1) / denom
        return fallback + (d if abs(d) <= 1.0 else 0.0)

    dy = float(iy - max_shift)
    dx = float(ix - max_shift)
    if 0 < iy < win.shape[0] - 1:
        dy = _para(float(win[iy - 1, ix]), float(win[iy, ix]), float(win[iy + 1, ix]), dy)
    if 0 < ix < win.shape[1] - 1:
        dx = _para(float(win[iy, ix - 1]), float(win[iy, ix]), float(win[iy, ix + 1]), dx)

    peak = float(win[iy, ix]) / max(float((fa * fa).sum() * (fb * fb).sum()) ** 0.5, 1e-9)
    return (dx, dy, peak)


def _peak_sharpness(a: np.ndarray, b: np.ndarray, *, max_shift: int | None = None) -> float:
    """互相关主峰与次峰的比值（>1.5 视为读数可信）。

    为什么需要它：**小尺寸 + 高周期内容**的块，互相关峰本身是宽的，
    于是读数会偏大且不可靠。实测一个 `亿千百十万千百十元角分` 金额栏目
    （400x60，15+ 条等距竖线）读数为 3.3px，而同一张图的整页位移只有 0.7px、
    相邻大块（676x46 收款人 / 716x74 大写金额）只有 0.2~0.3px——
    说明那 3.3px 是度量偏差，不是真实坐标误差。
    比值低就说明"这个峰有好几个竞争者"，读数不该被当成确定值。
    """
    fa = _to_gray(a).astype(np.float32)
    fb = _to_gray(b).astype(np.float32)
    if fa.shape != fb.shape or fa.size == 0:
        return float("nan")
    h, w = fa.shape
    if max_shift is None:
        max_shift = int(max(16, min(64, 0.12 * min(h, w))))
    fa, fb = fa - fa.mean(), fb - fb.mean()
    if fa.std() < 1e-6 or fb.std() < 1e-6:
        return float("nan")

    c = np.fft.fftshift(np.fft.irfft2(np.fft.fft2(fa) * np.conj(np.fft.fft2(fb)), s=fa.shape))
    cy, cx = h // 2, w // 2
    win = c[max(0, cy - max_shift):min(h, cy + max_shift + 1),
            max(0, cx - max_shift):min(w, cx + max_shift + 1)]
    if win.size == 0:
        return float("nan")
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    main = float(win[iy, ix])
    if main <= 0:
        return float("nan")
    y0, y1 = max(0, iy - 2), min(win.shape[0], iy + 3)
    x0, x1 = max(0, ix - 2), min(win.shape[1], ix + 3)
    masked = win.copy()
    masked[y0:y1, x0:x1] = -np.inf
    second = float(masked.max())
    if not np.isfinite(second) or second <= 0:
        return float("inf")
    return main / second


def _ink_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = _to_gray(a) < 128, _to_gray(b) < 128
    if ma.shape != mb.shape:
        return float("nan")
    union = int((ma | mb).sum())
    return float((ma & mb).sum()) / union if union else 1.0


def _row_name(p: Path) -> str:
    """给一行取个能区分的名字。

    同一个来源名在不同输出目录里会重名（实测两张支票的 stem 都是 `支票_p0001`），
    报告里出现两行同名会让人无法判断到底哪张有问题。所以把所属的运行目录带上。
    """
    parent = p.parent
    if parent.name in ("standardized", "low_confidence"):
        return f"{parent.parent.name}/{p.stem}"
    return f"{parent.name}/{p.stem}" if parent.name else p.stem


def build_rows(spec: CropSpec, paths: list[str | Path], *,
               template_png: str | Path | None = None,
               use_safe: bool = True) -> tuple[list[Row], list[dict]]:
    """把输入（标准输出图）与可选的模板基准读成行。"""
    rows: list[Row] = []
    diag: list[dict] = []

    if template_png is not None and Path(template_png).exists():
        tpl, _ = load_image(Path(template_png))
        rows.append(Row(label="[T] template", name="template",
                        image=tpl, is_reference=True))

    for i, p in enumerate(paths):
        p = Path(p)
        img, _ = load_image(p)
        # 标签带上所属运行目录：同一张图在不同输出目录里 stem 会重名，
        # 只看 stem 无法判断"到底哪一次运行出的问题"。
        rows.append(Row(label=f"[{i + 1}] {_ascii(_row_name(p), 24)}",
                        name=_row_name(p), image=img))
    return rows, diag


def verify(spec: CropSpec, paths: list[str | Path], *,
           template_png: str | Path | None = None,
           use_safe: bool = True, tol_px: float = 2.0,
           scale: int = 3, max_rows: int = 12) -> tuple[dict, dict[str, np.ndarray]]:
    """逐块裁剪 + 堆叠 + 量化。返回 (报告, {块 id: 堆叠图})。

    max_rows 只限制**堆叠图**放几行（30 页全放会得到一张高得没法看的图），
    统计仍然按全部输入算。
    """
    rows, _ = build_rows(spec, paths, template_png=template_png, use_safe=use_safe)
    if not rows:
        raise ValueError("没有可验证的输入")

    per_source: list[dict] = []
    blocks_by_row: list[dict[str, np.ndarray]] = []
    for r in rows:
        blocks, diag = crop_one(r.image, spec, use_safe=use_safe, strict=False)
        blocks_by_row.append(blocks)
        per_source.append({
            "label": r.label, "name": r.name,
            "canvas": [int(r.image.shape[1]), int(r.image.shape[0])],
            "blocks": {d["id"]: d for d in diag if d.get("id")},
            "problems": [d["message"] for d in diag if d.get("status") == "problem"],
        })

    ref_idx = 0 if rows[0].is_reference else 0
    ref_blocks = blocks_by_row[ref_idx]

    report_blocks: list[dict] = []
    sheets: dict[str, np.ndarray] = {}

    for it in spec.items:
        bid = it.id
        ref = ref_blocks.get(bid)
        if ref is None:
            report_blocks.append({"id": bid, "status": "missing",
                                  "reason": "参考图里没有裁出该块"})
            continue

        entries: list[dict] = []
        for i, (r, bl) in enumerate(zip(rows, blocks_by_row)):
            if i == ref_idx:
                continue
            b = bl.get(bid)
            if b is None:
                entries.append({"name": r.name, "status": "missing"})
                continue
            # 参数顺序：先被比较的图，后参考图 → 返回"被比较图相对参考图的位移"
            dx, dy, resp = _bounded_shift(b, ref)
            sharp = _peak_sharpness(b, ref)
            # 阈值 1.12 来自实测：读数可信的组峰锐度 ≥1.164，不可信的组 ≤1.082。
            # 歧义标记只是**标注**，不把读数排除出判定——实测排除会让
            # 真实问题页（读数 5.08px、锐度 1.004）被跳过，判定反而变成 PASS。
            ambiguous = not (isinstance(sharp, float) and np.isfinite(sharp)
                             and sharp >= 1.12)
            entries.append({
                "name": r.name, "label": r.label, "status": "ok",
                "shift_x": round(dx, 3), "shift_y": round(dy, 3),
                "shift_norm": round(float(np.hypot(dx, dy)), 3),
                "peak": round(resp, 4),
                "peak_sharpness": round(sharp, 3) if np.isfinite(sharp) else None,
                "shift_ambiguous": bool(ambiguous),
                "ink_iou": round(_ink_iou(ref, b), 4),
            })

        ok = [e for e in entries if e.get("status") == "ok"]
        worst = max((e["shift_norm"] for e in ok), default=float("nan"))
        pass_flag = bool(ok) and np.isfinite(worst) and worst <= tol_px
        verdict = "PASS" if pass_flag else "CHECK"
        report_blocks.append({
            "id": bid, "status": "ok",
            "roi": list(it.roi), "safe_margin": int(it.safe_margin),
            "block_size": [int(ref.shape[1]), int(ref.shape[0])],
            "reference": rows[ref_idx].name,
            "max_shift_px": round(worst, 3) if np.isfinite(worst) else None,
            "tolerance_px": float(tol_px),
            "n_ambiguous": sum(1 for e in ok if e.get("shift_ambiguous")),
            "verdict": verdict,
            "note": ("部分读数峰锐度不足，位移值可能偏大；以堆叠图为准"
                     if any(e.get("shift_ambiguous") for e in ok) else ""),
            "comparisons": entries,
        })
        rows_for_sheet = rows[:max(1, max_rows)]
        sheets[bid] = _stack(rows_for_sheet,
                             blocks_by_row[:max(1, max_rows)], bid, scale=scale)

    report = {
        "crops": spec.name,
        "canvas": [int(v) for v in spec.canvas],
        "tolerance_px": float(tol_px),
        "use_safe": bool(use_safe),
        "sources": per_source,
        "blocks": report_blocks,
        "summary": {
            "n_blocks": len(report_blocks),
            "n_pass": sum(1 for b in report_blocks if b.get("verdict") == "PASS"),
            "n_check": sum(1 for b in report_blocks if b.get("verdict") == "CHECK"),
            "n_ambiguous": sum(int(b.get("n_ambiguous") or 0)
                               for b in report_blocks),
            "n_missing": sum(1 for b in report_blocks if b.get("status") == "missing"),
        },
    }
    return report, sheets


def _stack(rows: list[Row], blocks_by_row: list[dict[str, np.ndarray]],
           bid: str, *, scale: int = 3, max_w: int = 2600) -> np.ndarray:
    """把同一块在每张图上裁出来的结果纵向堆叠，带标签与分隔线。"""
    cells: list[tuple[str, np.ndarray]] = []
    for r, blocks in zip(rows, blocks_by_row):
        b = blocks.get(bid)
        if b is None:
            cells.append((r.label, np.zeros((24, 200, 3), np.uint8)))
            continue
        g = _to_gray(b)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        k = scale
        if vis.shape[1] * k > max_w:
            k = max(1, max_w // max(1, vis.shape[1]))
        if k > 1:
            vis = cv2.resize(vis, (vis.shape[1] * k, vis.shape[0] * k),
                             interpolation=cv2.INTER_NEAREST)
        cells.append((r.label, vis))

    pad, head, gap = 10, 22, 8
    width = min(max_w, max(c.shape[1] for _, c in cells))
    height = pad * 2 + sum(c.shape[0] + head + gap for _, c in cells)
    sheet = np.full((height, width + pad * 2, 3), 246, np.uint8)

    y = pad
    for label, cell in cells:
        cv2.putText(sheet, label, (pad, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 30), 1, cv2.LINE_AA)
        y += head
        w = min(width, cell.shape[1])
        sheet[y:y + cell.shape[0], pad:pad + w] = cell[:, :w]
        cv2.rectangle(sheet, (pad - 1, y - 1), (pad + w, y + cell.shape[0]),
                      (185, 185, 185), 1)
        y += cell.shape[0] + gap
    return sheet


def write_verification(out_dir: str | Path, report: dict,
                       sheets: dict[str, np.ndarray]) -> dict[str, str]:
    from .render import write_png

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for bid, sheet in sheets.items():
        p = out / f"block-{bid}.png"
        write_png(p, sheet, 150)
        written[bid] = str(p)
    (out / "verify-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    written["_report"] = str(out / "verify-report.json")
    return written


def format_report(report: dict) -> str:
    lines: list[str] = []
    s = report["summary"]
    lines.append(f"裁剪坐标清单 : {report['crops']}")
    lines.append(f"画布         : {report['canvas'][0]}x{report['canvas'][1]}"
                 f"  外扩={'是' if report['use_safe'] else '否'}"
                 f"  容差={report['tolerance_px']}px")
    lines.append(f"块数 {s['n_blocks']}  通过 {s['n_pass']}  需复核 {s['n_check']}"
                 f"  读数歧义 {s.get('n_ambiguous', 0)}  缺失 {s['n_missing']}")
    lines.append("")
    lines.append(f"{'块':<16}{'尺寸':>12}{'最大位移':>10}  判定")
    for b in report["blocks"]:
        if b.get("status") == "missing":
            lines.append(f"{b['id']:<16}{'-':>12}{'-':>10}  缺失")
            continue
        sz = f"{b['block_size'][0]}x{b['block_size'][1]}"
        sh = b.get("max_shift_px")
        shs = f"{sh:.2f}px" if isinstance(sh, (int, float)) else "-"
        mark = b["verdict"]
        if b.get("n_ambiguous"):
            mark += f" ({b['n_ambiguous']} 项读数歧义)"
        lines.append(f"{b['id']:<16}{sz:>12}{shs:>10}  {mark}")
        for c in b.get("comparisons", []):
            if c.get("status") != "ok":
                continue
            flag = "  [读数歧义]" if c.get("shift_ambiguous") else ""
            lines.append(f"    vs {c['name'][:26]:<28}"
                         f"dx={c['shift_x']:+7.2f} dy={c['shift_y']:+7.2f}"
                         f"  IoU={c['ink_iuo'] if False else c['ink_iou']:.3f}"
                         f" 峰锐度={c.get('peak_sharpness')}{flag}")
    return "\n".join(lines)
