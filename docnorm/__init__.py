"""docnorm — 业务单据标准化渲染引擎。

输入：图片或 PDF 扫描件（PDF 逐页拆分，页 = 业务最小单元）
输出：规格尺寸/DPI 完全统一、内容块坐标固定在模板坐标系中的图片

    from docnorm.pipeline import Engine
    eng = Engine.load("templates/bank-payment")
    res = eng.process("scans/batch.pdf", out_dir="out")
"""

__version__ = "1.0.0"

CANVAS_PRESETS = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A5": (148.0, 210.0),
}


def canvas_px(preset: str, dpi: int) -> tuple[int, int]:
    """把纸张规格换算成目标 DPI 下的像素画布尺寸。"""
    if preset not in CANVAS_PRESETS:
        raise KeyError(f"未知纸张规格 {preset!r}，可选 {sorted(CANVAS_PRESETS)}")
    w_mm, h_mm = CANVAS_PRESETS[preset]
    return int(round(w_mm / 25.4 * dpi)), int(round(h_mm / 25.4 * dpi))
