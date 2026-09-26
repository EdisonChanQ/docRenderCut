"""统一加载器：图片 / PDF -> 逐页转换单元。

页（张）是业务处理的最小单元：一个 N 页 PDF 产出 N 个单元，互不影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXT = {".pdf"}


@dataclass
class PageUnit:
    """一个待处理的页面单元。"""

    source: Path
    page_index: int          # 0-based；图片文件恒为 0
    image: np.ndarray        # BGR
    page_count: int = 1      # 所在文件的总页数
    declared_dpi: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.page_count <= 1:
            return self.source.name
        return f"{self.source.name}#p{self.page_index + 1}"

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


def read_png_dpi(path: Path) -> float | None:
    """读图片里声明的 DPI（若有）。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if dpi and dpi[0]:
                return float(dpi[0])
    except Exception:
        pass
    return None


def load_image(path: Path) -> tuple[np.ndarray, float | None]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法解码图片: {path}")
    return img, read_png_dpi(path)


def load_pdf(path: Path, render_dpi: int) -> list[tuple[np.ndarray, float]]:
    """把 PDF 每页栅格化成 BGR。"""

    import pymupdf

    out: list[tuple[np.ndarray, float]] = []
    doc = pymupdf.open(str(path))
    try:
        for page in doc:
            pm = page.get_pixmap(dpi=int(render_dpi))
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
            if pm.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pm.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            out.append((np.ascontiguousarray(arr), float(render_dpi)))
    finally:
        doc.close()
    return out


def iter_units(path: str | Path, *, render_dpi: int = 300) -> list[PageUnit]:
    """把一个文件（图片或 PDF）展开成页面单元列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext in PDF_EXT:
        pages = load_pdf(path, render_dpi)
        n = len(pages)
        return [
            PageUnit(path, i, img, page_count=n, declared_dpi=dpi,
                     meta={"render_dpi": render_dpi})
            for i, (img, dpi) in enumerate(pages)
        ]

    if ext in IMAGE_EXT:
        img, dpi = load_image(path)
        return [PageUnit(path, 0, img, page_count=1, declared_dpi=dpi)]

    raise ValueError(f"不支持的文件类型: {path.suffix}")


def iter_batch(paths: list[str | Path], *, render_dpi: int = 300) -> list[PageUnit]:
    """展开一批文件（目录会自动展开为其中的图片/PDF）。"""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXT | PDF_EXT
            ))
        else:
            files.append(p)

    units: list[PageUnit] = []
    for f in files:
        units.extend(iter_units(f, render_dpi=render_dpi))
    return units
