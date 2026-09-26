"""渲染层：把配准后的页面渲染成规格完全统一的输出，并写 DPI 元数据与伴随信息。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


def write_png(path: Path, img: np.ndarray, dpi: int, *,
              optimize: bool = False, compress_level: int = 6) -> None:
    """写 PNG 并写入真实 DPI 元数据（下游读图即知物理尺寸）。

    接受 BGR 三通道或灰度（二值输出用灰度保存，体积更小、语义更明确）。

    optimize 默认关闭：实测 2480x3508 的页面上 optimize=True 要 14.8s，
    占整页耗时的 85%；关掉只要 1.8s，文件大小差异可以忽略。
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(arr).save(
        path, format="PNG", dpi=(dpi, dpi),
        optimize=optimize, compress_level=compress_level,
    )


def render_to_canvas(page_bgr: np.ndarray, matrix: np.ndarray,
                     canvas: tuple[int, int], *, prefilter: bool = True) -> np.ndarray:
    """把页面按 matrix（样本坐标 -> 画布坐标）渲染到标准画布。

    降采样时先做抗混叠预滤波：这一步不改变几何（矩阵不变），只让像素质量更接近
    真实扫描仪的重采样效果。
    """
    from .align import as_affine, is_affine

    M = as_affine(matrix)
    src = page_bgr
    if prefilter:
        h, w = page_bgr.shape[:2]
        pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        proj = M @ pts
        dst = (proj[:2] / proj[2:3]).T
        span_x = np.hypot(*(dst[1] - dst[0]))
        span_y = np.hypot(*(dst[3] - dst[0]))
        scale = min(span_x / max(w, 1), span_y / max(h, 1))
        if scale < 0.98:
            sigma = min(2.5, 0.6 * (1.0 / scale - 1.0))
            if sigma > 0.05:
                src = cv2.GaussianBlur(src, (0, 0), sigma)

    if is_affine(M):
        return cv2.warpAffine(
            src, M[:2].astype(np.float32), canvas,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
    return cv2.warpPerspective(
        src, M.astype(np.float32), canvas,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


@dataclass
class RenderRecord:
    """一页的处理记录，写入 sidecar 供下游与验收脚本使用。"""

    source: str
    page_index: int
    label: str
    status: str                      # ok | rejected
    reason: str = ""
    output: str = ""
    canvas: dict = field(default_factory=dict)
    matrix_sample_to_canvas: list = field(default_factory=list)
    input_size: list = field(default_factory=list)
    paper: dict = field(default_factory=dict)
    skew_deg: float = 0.0
    cc: float = -1.0
    score: float = -1.0
    coarse_shift: list = field(default_factory=list)
    align_direction: dict = field(default_factory=dict)
    levels_used: int = 0
    locate: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    residual_skew_deg: float = 0.0
    residual_skew_lines: int = 0


def write_sidecar(path: Path, record: RenderRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")


def crop_blocks(image_bgr: np.ndarray, blocks: list[dict]) -> dict[str, np.ndarray]:
    """按模板登记的标准坐标裁剪目标块。

    用 roi_safe（μ±3σ 外扩）而不是 roi：坐标残差是**有界但非零**的，
    外扩余量正是用来吸收它的。这是「永远裁得全」的工程实现。
    """
    out: dict[str, np.ndarray] = {}
    h, w = image_bgr.shape[:2]
    for b in blocks:
        box = b.get("roi_safe") or b.get("roi")
        if not box:
            continue
        x, y, bw, bh = [int(round(v)) for v in box]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            out[b["id"]] = image_bgr[y0:y1, x0:x1]
    return out


# ---------------------------------------------------------------- 报告

def write_report(path: Path, records: list[RenderRecord], template_name: str) -> None:
    total = len(records)
    ok = [r for r in records if r.status == "ok"]
    warn = [r for r in records if r.status == "low_confidence"]
    bad = [r for r in records if r.status not in ("ok", "low_confidence")]

    def row(r: RenderRecord) -> str:
        cls = {"ok": "ok", "low_confidence": "warn"}.get(r.status, "bad")
        return (
            f"<tr class='{cls}'><td>{r.label}</td><td>{r.input_size[0]}x{r.input_size[1]}</td>"
            f"<td>{r.skew_deg:+.2f}</td><td>{r.cc:.4f}</td><td>{r.score:.4f}</td>"
            f"<td>{r.status}</td><td>{r.reason or '-'}</td></tr>"
        )

    rows = "\n".join(row(r) for r in records)
    scores = [r.score for r in ok] or [0.0]
    ccs = [r.cc for r in ok] or [0.0]

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>docnorm 渲染报告 - {template_name}</title>
<style>
 body {{ font-family: "Microsoft YaHei", sans-serif; margin: 32px; color: #1a1a1a; background:#fff; }}
 h1 {{ font-size: 20px; font-weight: 500; }}
 .cards {{ display: flex; gap: 12px; margin: 18px 0; flex-wrap: wrap; }}
 .card {{ border: 1px solid #dcdcdc; border-radius: 10px; padding: 12px 18px; min-width: 132px; }}
 .card .k {{ font-size: 12px; color: #666; }}
 .card .v {{ font-size: 22px; margin-top: 4px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 th, td {{ border-bottom: 1px solid #e6e6e6; padding: 7px 9px; text-align: left; }}
 th {{ background: #f5f5f5; font-weight: 500; }}
 tr.bad {{ background: #fdecec; }}
 tr.warn {{ background: #fff7e6; }}
</style></head><body>
<h1>docnorm 渲染报告 —— {template_name}</h1>
<div class="cards">
  <div class="card"><div class="k">总页数</div><div class="v">{total}</div></div>
  <div class="card"><div class="k">合格</div><div class="v">{len(ok)}</div></div>
  <div class="card"><div class="k">拒收</div><div class="v">{len(bad)}</div></div>
  <div class="card"><div class="k">低置信</div><div class="v">{len(warn)}</div></div>
  <div class="card"><div class="k">结构相关度 最小</div><div class="v">{min(scores):.3f}</div></div>
  <div class="card"><div class="k">结构相关度 均值</div><div class="v">{sum(scores)/len(scores):.3f}</div></div>
  <div class="card"><div class="k">ECC 均值</div><div class="v">{sum(ccs)/len(ccs):.4f}</div></div>
</div>
<table><thead><tr>
<th>来源</th><th>输入尺寸</th><th>去斜角</th><th>ECC</th><th>结构分</th><th>状态</th><th>原因</th>
</tr></thead><tbody>
{rows}
</tbody></table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
