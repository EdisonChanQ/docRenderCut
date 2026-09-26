"""模板：定义标准坐标系、画布规格，以及用于配准的结构特征。

模板不是「一张参考图」那么弱——它是「一个坐标系 + 该坐标系下的画布规格 + 配准基准特征」。
后续所有页都被映射进这个坐标系，坐标才有意义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import canvas_px
from .grid import detect_grid_lines
from .rectify import binarize, normalize_illumination, structure_image, structure_mask, to_gray


@dataclass
class Template:
    name: str
    canvas: tuple[int, int]              # (宽, 高) 像素
    dpi: int
    paper: str = "A4"
    gray: np.ndarray = None              # 画布尺寸的灰度基准图
    struct: np.ndarray = None            # 画布尺寸的结构特征（float32 0~1）
    mask: np.ndarray = None              # 结构区域掩膜
    blocks: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    dark_bias: int = 0                   # 深墨偏置（拍照件/有防伪底纹的票据需要）
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def lines(self):
        """模板的表格线位置（画布坐标，亚像素）。结果缓存，只算一次。"""
        if "lines" not in self._cache:
            binimg = binarize(self.gray, dark_bias=self.dark_bias)
            self._cache["lines"] = detect_grid_lines(self.gray, binimg)
        return self._cache["lines"]

    @property
    def skew_deg(self) -> float:
        """模板基准图自身的倾斜角（度）。缓存，只算一次。

        门禁用的是「输入对齐后的残余倾斜 − 模板基准倾斜」：
        如果模板自己就是歪的，那输入对齐到它之后也会带着同样的角度，
        扣掉基准才能得到"输入相对模板偏了多少"。
        """
        if "skew" not in self._cache:
            from .align import _residual_skew

            self._cache["skew"] = _residual_skew(self.gray)[0]
        return float(self._cache["skew"])

    # ---- 构建 ----------------------------------------------------------------

    @classmethod
    def from_image(cls, name: str, image_bgr: np.ndarray, *,
                   paper: str = "A4", dpi: int = 300,
                   canvas: tuple[int, int] | None = None,
                   struct_blur: float = 3.0, dark_bias: int = 0) -> "Template":
        """用一张干净的基准页建立模板。

        画布尺寸可以不等于基准页尺寸——配准是坐标映射而非等比缩放，
        所以「A4 扫描件重渲染到自定义规格」是天然支持的。
        """
        target = canvas or canvas_px(paper, dpi)

        gray = normalize_illumination(to_gray(image_bgr))
        if (gray.shape[1], gray.shape[0]) != (target[0], target[1]):
            interp = cv2.INTER_AREA if gray.shape[1] > target[0] else cv2.INTER_CUBIC
            gray = cv2.resize(gray, target, interpolation=interp)

        binimg = binarize(gray, dark_bias=dark_bias)
        return cls(
            name=name, canvas=target, dpi=dpi, paper=paper,
            gray=gray, dark_bias=dark_bias,
            struct=structure_image(gray, binimg, blur=struct_blur),
            mask=structure_mask(gray, binimg),
            meta={"created_at": datetime.now().isoformat(timespec="seconds")},
        )

    # ---- 持久化 --------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        from .render import write_png
        write_png(directory / "template.png", cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR), self.dpi)

        spec = {
            "name": self.name,
            "paper": self.paper,
            "dpi": self.dpi,
            "canvas": {"width": self.canvas[0], "height": self.canvas[1]},
            "blocks": self.blocks,
            "dark_bias": self.dark_bias,
            "meta": self.meta,
        }
        (directory / "template.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, struct_blur: float = 3.0) -> "Template":
        directory = Path(directory)
        spec = json.loads((directory / "template.json").read_text(encoding="utf-8"))

        data = np.fromfile(str(directory / "template.png"), dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"模板图无法解码: {directory / 'template.png'}")

        dark_bias = int(spec.get("dark_bias", 0))
        binimg = binarize(gray, dark_bias=dark_bias)
        return cls(
            name=spec["name"], dark_bias=dark_bias,
            canvas=(spec["canvas"]["width"], spec["canvas"]["height"]),
            dpi=spec["dpi"],
            paper=spec.get("paper", "A4"),
            gray=gray,
            struct=structure_image(gray, binimg, blur=struct_blur),
            mask=structure_mask(gray, binimg),
            blocks=spec.get("blocks", []),
            meta=spec.get("meta", {}),
        )


def build_template_from_file(name: str, image_path: str | Path, out_dir: str | Path,
                             *, paper: str = "A4", dpi: int = 300,
                             canvas: tuple[int, int] | None = None,
                             dark_bias: int = 0) -> Template:
    from .loader import load_image

    img, _ = load_image(Path(image_path))
    tpl = Template.from_image(name, img, paper=paper, dpi=dpi, canvas=canvas,
                              dark_bias=dark_bias)
    tpl.save(out_dir)
    return tpl
