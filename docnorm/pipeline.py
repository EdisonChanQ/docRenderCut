"""编排：模板 + 一批输入 -> 规格统一的输出。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .align import AlignConfig, align_to_template, as_affine, warp_forward
from .loader import PageUnit, iter_batch, iter_units
from .rectify import (
    PaperQuad,
    binarize_document,
    deskew,
    detect_paper_quad,
    normalize_illumination,
    paper_warp,
    to_gray,
)
from .render import (
    RenderRecord,
    render_to_canvas,
    write_png,
    write_report,
    write_sidecar,
)
from .template import Template


def _drop_stale(stem: str, dirs: list[Path]) -> None:
    """删掉同名页在其它目录里的旧产物，保证一页只存在于一处。"""
    for d in dirs:
        f = d / f"{stem}.png"
        try:
            if f.exists():
                f.unlink()
        except OSError:
            pass


class Engine:
    """标准化渲染引擎。

        eng = Engine.load("templates/bank-payment")
        recs = eng.process(["scans/a.pdf", "scans/b.png"], out_dir="out")

    可选前置层（按输入形态开）：
    - paper：纸张四边形检测 + 透视矫正。拍照件需要；扫描件整页完整时设 "off"
    - use_deskew：整体去斜
    - mono：输出纯黑白两色（OCR 用）
    """

    def __init__(self, template: Template, config: AlignConfig | None = None,
                 *, use_deskew: bool = True, render_dpi: int = 300,
                 paper: str = "off", mono: bool = True, despeckle: int = 0,
                 ink_dark_bias: int = 25, crops=None):
        self.template = template
        self.config = config or AlignConfig()
        self.use_deskew = use_deskew
        self.render_dpi = render_dpi
        self.paper = paper                  # "auto" | "off" | "x1,y1,...,x4,y4"
        self.mono = mono
        self.despeckle = despeckle
        self.ink_dark_bias = ink_dark_bias  # 输出二值化的深墨偏置
        self.crops = crops                  # CropSpec | None：给了就顺带输出块

    # ---- 构建 ----------------------------------------------------------------

    @classmethod
    def load(cls, template_dir: str | Path, **kwargs) -> "Engine":
        return cls(Template.load(template_dir), **kwargs)

    # ---- 前置层 --------------------------------------------------------------

    def prepare(self, page_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """纸张矫正 + 去斜，返回 (处理后的页, 原图->处理后 的 3x3, 诊断信息)。

        纸张层只需要「够用」：它负责把内容带进模板的捕获范围即可。
        坐标固定的精度由后面的内容配准提供（表格线/印刷边框），
        所以纸张角点差几个像素无害——精度不是从这一层来的。
        """
        diag: dict = {"paper": None}
        M_total = np.eye(3)
        work = page_bgr

        if self.paper != "off":
            quad = None
            if self.paper == "auto":
                quad = detect_paper_quad(page_bgr)
            else:
                quad = PaperQuad.from_corners(
                    [float(v) for v in self.paper.split(",")])

            if quad is not None and quad.ok:
                work, H = paper_warp(page_bgr, quad, self.template.canvas)
                M_total = as_affine(H)
                diag["paper"] = {
                    "ok": True, "confidence": quad.confidence,
                    "aspect": round(quad.aspect, 4),
                    "corners": [[round(float(v), 1) for v in p] for p in quad.corners],
                }
            else:
                diag["paper"] = {
                    "ok": False,
                    "confidence": getattr(quad, "confidence", None),
                    "reason": getattr(quad, "reason", "未检测到纸张"),
                }

        if self.use_deskew:
            # dark_bias 与配准层保持一致，否则深色桌面会被当成墨、去斜失效
            work_ds, M_ds, skew = deskew(work, dark_bias=self.config.ink_bias)
            M_total = as_affine(M_ds) @ M_total
            work = work_ds
            diag["skew_deg"] = round(float(skew.angle_deg), 3)
        else:
            diag["skew_deg"] = 0.0
        return work, M_total, diag

    # ---- 主流程 --------------------------------------------------------------

    def process_unit(self, unit: PageUnit) -> tuple[RenderRecord, np.ndarray | None]:
        rec = RenderRecord(
            source=str(unit.source), page_index=unit.page_index, label=unit.label,
            status="ok",
            canvas={"width": self.template.canvas[0], "height": self.template.canvas[1],
                    "dpi": self.template.dpi},
            input_size=[unit.size[0], unit.size[1]],
        )

        page, M_pre, diag = self.prepare(unit.image)
        rec.paper = diag.get("paper") or {}
        rec.skew_deg = float(diag.get("skew_deg", 0.0))

        res = align_to_template(
            self.template.struct, page, self.template.canvas, self.config,
            template_lines=self.template.lines(),
            template_skew_deg=self.template.skew_deg,
        )

        # 合成到「原始页坐标 -> 画布坐标」：从原图直接渲染，少一次重采样
        M_total = res.matrix @ M_pre

        rec.cc = round(res.cc, 6)
        rec.score = round(res.score, 6)
        rec.residual_skew_deg = round(float(res.residual_skew_deg), 4)
        rec.residual_skew_lines = int(res.residual_skew_lines)
        rec.coarse_shift = [round(v, 3) for v in res.coarse_shift]
        rec.align_direction = {
            k: (round(v, 6) if isinstance(v, (int, float)) else v)
            for k, v in res.direction.items()
        }
        rec.levels_used = res.levels_used
        rec.locate = res.locate
        rec.grid = res.grid
        rec.matrix_sample_to_canvas = [[round(float(v), 8) for v in row] for row in M_total]

        if not res.ok:
            rec.status = "rejected"
            rec.reason = res.reason
            return rec, None

        # 低置信预警：判据是**对齐后输出图上的残余倾斜**（分辨率无关的物理量）。
        # 内容歪了，指定坐标裁出来的块必然整体偏移，且偏移随离旋转中心距离线性放大。
        # 实测：A4 合格页最大 0.239°；坐标偏了 7~10px 的支票照片 0.99~1.11°。
        # 用结构相关度做这个判断是不行的——它随输入分辨率下降，
        # 而且偏了 10px 的照片得分（0.45）反而高于配准正确的 72dpi 页（0.27）。
        if abs(res.residual_skew_deg) > self.config.max_residual_skew_deg:
            rec.status = "low_confidence"
            rec.reason = (f"对齐后残余倾斜 {res.residual_skew_deg:+.3f}° 超过 "
                          f"{self.config.max_residual_skew_deg:.2f}°："
                          "坐标会有随位置放大的偏移，需人工复核")

        if self.mono:
            # 二值输出：先在源分辨率下做光照归一化，再 warp，最后阈值。
            # 归一化必须在源分辨率做（尺度才对）；阈值放在 warp 之后做，
            # 避免「先二值后插值」把细笔画糊成灰边、笔断。
            gray = normalize_illumination(to_gray(unit.image))
            warped_g = warp_forward(gray, M_total, self.template.canvas, border=255)
            out = binarize_document(warped_g, dark_bias=self.ink_dark_bias,
                                    despeckle=self.despeckle)
        else:
            out = render_to_canvas(unit.image, M_total, self.template.canvas)
        return rec, out

    def process_units(self, units: list[PageUnit], out_dir: str | Path,
                      *, rejected_dir: str | Path | None = None,
                      on_page=None) -> list[RenderRecord]:
        """on_page(已完成数, 总数, 当前记录) 在每页处理完后调用，用于上报进度。

        逐页回调而不是整体返回后才上报：批量任务动辄几十上百页，
        前端需要看到进度，而不是一个长时间的空白等待。
        """
        out_root = Path(out_dir)
        std_dir = out_root / "standardized"
        side_dir = out_root / "sidecar"
        rej_dir = Path(rejected_dir) if rejected_dir else out_root / "rejected"
        records: list[RenderRecord] = []
        crop_manifest: list[dict] = []

        for unit in units:
            rec, img = self.process_unit(unit)

            if rec.status in ("ok", "low_confidence") and img is not None:
                stem = f"{Path(unit.source).stem}_p{unit.page_index + 1:04d}"
                out_path = (std_dir if rec.status == "ok" else out_root / "low_confidence") \
                    / f"{stem}.png"
                # 同一页只应存在于一个目录。重跑时状态可能变化（ok <-> low_confidence），
                # 不清旧文件就会同一页同时躺在 standardized/ 和 low_confidence/ 里，
                # 下游会拿到过期的那一份——而且它看起来完全正常。实测因此让
                # verify 报出重复行，还差点把两个不同版本当成两张不同的输入图。
                _drop_stale(stem, [std_dir, out_root / "low_confidence", rej_dir])
                write_png(out_path, img, self.template.dpi)
                rec.output = str(out_path)
                write_sidecar(side_dir / f"{stem}.json", rec)

                if self.crops is not None:
                    from .crops import crop_one, write_blocks

                    blocks, diag = crop_one(img, self.crops, strict=False)
                    written = write_blocks(out_root / "blocks" / stem, blocks,
                                           dpi=self.template.dpi)
                    crop_manifest.append({
                        "source": stem, "status": rec.status,
                        "standardized": str(out_path), "blocks": written,
                        "diagnostics": diag,
                    })
            else:
                stem = f"{Path(unit.source).stem}_p{unit.page_index + 1:04d}"
                # 拒收页也原样落盘 + 记录，便于人工复核，绝不静默丢弃
                _drop_stale(stem, [std_dir, out_root / "low_confidence", rej_dir])
                write_png(rej_dir / f"{stem}.png", unit.image, self.render_dpi)
                rec.output = str(rej_dir / f"{stem}.png")
                write_sidecar(side_dir / f"{stem}.json", rec)

            records.append(rec)
            if on_page is not None:
                try:
                    on_page(len(records), len(units), rec)
                except Exception:  # noqa: BLE001
                    # 进度上报永远不能影响处理本身
                    pass

        if self.crops is not None:
            (out_root / "blocks").mkdir(parents=True, exist_ok=True)
            (out_root / "blocks" / "manifest.json").write_text(
                json.dumps({"crops": self.crops.to_dict(), "pages": crop_manifest},
                           ensure_ascii=False, indent=2), encoding="utf-8")

        write_report(out_root / "render-report.html", records, self.template.name)
        return records

    def process(self, paths, out_dir: str | Path, **kwargs) -> list[RenderRecord]:
        units = iter_batch(list(paths), render_dpi=self.render_dpi)
        return self.process_units(units, out_dir, **kwargs)

    def process_file(self, path: str | Path, out_dir: str | Path, **kwargs) -> list[RenderRecord]:
        units = iter_units(path, render_dpi=self.render_dpi)
        return self.process_units(units, out_dir, **kwargs)
