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
    """把 PDF 每页栅格化成 BGR。

    注意：**整本 PDF 一次性解码**。批量处理请用 `iter_batch_stream`，
    它逐页 yield、内存与页数无关（大批次下这是 OOM 与否的分界）。
    """
    return list(iter_pdf_pages(path, render_dpi))


def iter_pdf_pages(path: Path, render_dpi: int):
    """**逐页**栅格化 PDF（生成器）。

    与 load_pdf 的区别只在内存：这里每 yield 一页就释放上一页的位图，
    峰值 ≈ 1 页（叠加上调用方持有的页）。50 页的批次从"一次 1.2 GB"
    降到"任意时刻几十 MB"。
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        for page in doc:
            pm = page.get_pixmap(dpi=int(render_dpi))
            n = pm.n
            buf = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, n)
            if n == 4:
                img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            elif n == 3:
                img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
            out = np.ascontiguousarray(img)
            # 显式断开对 pixmap 缓冲区的引用，让这一页的位图尽早被回收
            pm = None
            buf = None
            yield out, float(render_dpi)
    finally:
        doc.close()


def count_pages(path: str | Path) -> int:
    """只数页数、不栅格化（用于给任务一个准确的总页数，成本极低）。"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in PDF_EXT:
        import pymupdf

        with pymupdf.open(str(p)) as doc:
            return doc.page_count
    if ext in IMAGE_EXT:
        return 1
    raise ValueError(f"不支持的文件类型: {p.suffix}")


def expand_files(paths: list[str | Path]) -> list[Path]:
    """目录自动展开为其中的图片/PDF；顺序稳定（便于结果可复现）。"""
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
    return files


def iter_units(path: str | Path, *, render_dpi: int = 300) -> list[PageUnit]:
    """把一个文件（图片或 PDF）展开成页面单元列表。

    注意：**一次性**返回全部页（PDF 会整本解码）。批量处理请用
    `iter_batch_stream`。
    """
    return list(iter_units_stream(path, render_dpi=render_dpi))


def iter_units_stream(path: str | Path, *, render_dpi: int = 300):
    """逐页产出页面单元（生成器），一次只驻留一页。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext in PDF_EXT:
        n = count_pages(path)
        for i, (img, dpi) in enumerate(iter_pdf_pages(path, render_dpi)):
            yield PageUnit(path, i, img, page_count=n, declared_dpi=dpi,
                           meta={"render_dpi": render_dpi})

    elif ext in IMAGE_EXT:
        img, dpi = load_image(path)
        yield PageUnit(path, 0, img, page_count=1, declared_dpi=dpi)

    else:
        raise ValueError(f"不支持的文件类型: {path.suffix}")


def iter_batch(paths: list[str | Path], *, render_dpi: int = 300) -> list[PageUnit]:
    """展开一批文件（目录会自动展开为其中的图片/PDF）。

    注意：**一次性**返回全部页，内存与总页数成正比（300dpi A4 约 25 MB/页）。
    服务端的批量任务请用 `iter_batch_stream` + 有界在飞数，避免大批次 OOM。
    """
    return list(iter_batch_stream(paths, render_dpi=render_dpi))


def iter_batch_stream(paths: list[str | Path], *, render_dpi: int = 300):
    """逐页产出整批文件的页面单元（生成器）。内存与总页数无关。"""
    for f in expand_files(paths):
        yield from iter_units_stream(f, render_dpi=render_dpi)
