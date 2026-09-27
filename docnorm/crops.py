"""块裁剪坐标配置：与模板分离的、可人工维护的坐标清单。

为什么独立成文件，而不只用 template.json 里的 blocks：

1. **同一模板可以有多套清单**。同一张单据给 OCR 裁的字段、给人工抽检裁的区域、
   给归档裁的图章区，是不同用途的不同坐标集，不该互相挤在一个 blocks 数组里。
2. **调坐标不该重跑渲染**。渲染是秒级/页的重活，而坐标会反复微调。
   独立清单让「改坐标 -> 只重裁」成为一次几百毫秒的操作。
3. **生命周期不同**。模板绑定的是基准图（换图才要重建），
   坐标清单绑定的是版式约定（改字段就改坐标）。混在一起会互相牵制。

坐标系约定：**所有 roi 都在标准画布的像素坐标系里**，与输入图完全无关。
这正是整个引擎要提供给下游的东西——坐标只写一次，任何输入都能用。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CROPS_VERSION = 1


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class CropItem:
    id: str                     # 自动分配：field01 / field02 …（机器标识，不人工编辑）
    name: str = ""              # 字段说明：人工编辑，赋予 id 含义（如「付款账户」）
    roi: list[int] = field(default_factory=list)   # x, y, w, h —— 标准画布坐标
    safe_margin: int = 0        # roi_safe = roi 外扩这么多像素（吸收有界配准残差）

    def box(self, *, use_safe: bool = True) -> tuple[int, int, int, int]:
        x, y, w, h = (int(round(v)) for v in self.roi)
        m = int(self.safe_margin) if use_safe else 0
        return x - m, y - m, w + 2 * m, h + 2 * m

    @property
    def label(self) -> str:
        """给人看的标签：优先字段说明，没填就退回 id。"""
        return (self.name or "").strip() or self.id


@dataclass
class CropSpec:
    """一套裁剪坐标清单。"""

    name: str
    canvas: list[int]                     # [宽, 高]，必须与模板画布一致
    dpi: int
    items: list[CropItem] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- 索引 ----------------------------------------------------------------

    def find(self, item_id: str) -> CropItem | None:
        for it in self.items:
            if it.id == item_id:
                return it
        return None

    # ---- 字段 ID 分配 --------------------------------------------------------

    @staticmethod
    def _seq_of(item_id: str) -> int | None:
        m = re.fullmatch(r"field(\d+)", item_id or "")
        return int(m.group(1)) if m else None

    def next_id(self) -> str:
        """分配下一个字段 ID（fieldNN），并推进计数器。

        计数器**持久化在 meta.next_field_seq**，而不是"现有最大号 + 1"：
        后者在"删掉最大号再新建"时会复用已删除的 ID，
        而下游（ERP）可能已按旧 ID 存了对应关系——复用会让新旧语义撞车。
        所以这里是**单调递增、永不复用**。
        """
        n = int(self.meta.get("next_field_seq", 1) or 1)
        for it in self.items:
            s = self._seq_of(it.id)
            if s is not None:
                n = max(n, s + 1)
        self.meta["next_field_seq"] = n + 1
        return f"field{n:02d}"

    def upsert(self, item: CropItem) -> None:
        for i, it in enumerate(self.items):
            if it.id == item.id:
                self.items[i] = item
                return
        self.items.append(item)

    def remove(self, item_id: str) -> bool:
        before = len(self.items)
        self.items = [it for it in self.items if it.id != item_id]
        return len(self.items) != before

    # ---- 序列化 --------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": CROPS_VERSION,
            "name": self.name,
            "canvas": [int(v) for v in self.canvas],
            "dpi": int(self.dpi),
            "meta": self.meta,
            "items": [asdict(it) for it in self.items],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CropSpec":
        items = [
            CropItem(
                id=str(it["id"]),
                # 兼容旧数据：老版本用 note 存字段名，读到就并入 name
                name=str(it.get("name") or it.get("note") or ""),
                roi=[int(round(float(v))) for v in it["roi"]],
                safe_margin=int(it.get("safe_margin", 0) or 0),
            )
            for it in d.get("items", [])
        ]
        return cls(
            name=str(d.get("name", "crops")),
            canvas=[int(v) for v in d["canvas"]],
            dpi=int(d.get("dpi", 300)),
            items=items,
            meta=d.get("meta", {}) or {},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CropSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- 校验 ----------------------------------------------------------------

    def validate_for_canvas(self, size: tuple[int, int], *, strict: bool = True
                            ) -> list[str]:
        """核对清单与将要裁剪的图是否自洽。

        两条检查，性质不同：
        - **画布尺寸不符 = 硬错误**。坐标是在某个画布上量出来的，
          换一个画布它就没有意义了。这种情况必须拒绝，不能"尽力而为"地裁。
        - 单个 roi 越界 = 警告。可能是量坐标时手滑，也可能是刻意外扩到边缘。
          报告出来让人判断，但框会被裁到画布内（OpenCV 切片不会报错，会静默给出更小的图）。
        """
        problems: list[str] = []
        w, h = size
        if [int(w), int(h)] != [int(self.canvas[0]), int(self.canvas[1])]:
            msg = (f"画布不符：清单是在 {self.canvas[0]}x{self.canvas[1]} 画布上量的，"
                   f"当前图是 {w}x{h}。坐标不可跨画布复用。")
            if strict:
                raise ValueError(msg)
            problems.append(msg)

        for it in self.items:
            x, y, bw, bh = it.box(use_safe=True)
            if x < 0 or y < 0 or x + bw > w or y + bh > h:
                problems.append(
                    f"块 {it.label} 的裁剪框 [{x},{y},{bw},{bh}] 超出画布 {w}x{h}（会被截断）")
        return problems


# ---------------------------------------------------------------- 裁剪

def crop_one(image_bgr: np.ndarray, spec: CropSpec, *,
             use_safe: bool = True, strict: bool = True
             ) -> tuple[dict[str, np.ndarray], list[dict]]:
    """从一张标准输出图上按清单裁出所有块。

    返回 (块字典, 逐块诊断)。诊断里记录实际用的框与被截断情况——
    「安静地裁小了」是这类流程最典型的失败模式，必须让它可见。
    """
    h, w = image_bgr.shape[:2]
    problems = spec.validate_for_canvas((w, h), strict=strict)

    blocks: dict[str, np.ndarray] = {}
    diag: list[dict] = []
    for it in spec.items:
        bx, by, bw, bh = it.box(use_safe=use_safe)
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(w, bx + bw), min(h, by + bh)
        rec = {
            "id": it.id,
            "roi": [int(v) for v in it.roi],
            "roi_used": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
            "use_safe": bool(use_safe),
            "clipped": bool(x0 != bx or y0 != by or (x1 - x0) != bw or (y1 - y0) != bh),
        }
        if x1 <= x0 or y1 <= y0:
            rec["status"] = "empty_box"
            diag.append(rec)
            continue
        crop = image_bgr[y0:y1, x0:x1].copy()
        blocks[it.id] = crop

        gray = _to_gray(crop)
        rec["size"] = [int(crop.shape[1]), int(crop.shape[0])]
        rec["ink_ratio"] = round(float((gray < 128).mean()), 5)
        # 边缘触墨：内容贴到框边。表格行天然如此（左右边就是表格竖线），
        # 所以它只是**提示**，不是判据——真正要看的是 content 有没有被切断，
        # 那是人看图判断的（见 docnorm verify 的堆叠图）。
        ink = gray < 128
        rec["touches_border"] = bool(
            ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any()
        )
        rec["status"] = "ok"
        diag.append(rec)

    out = list(diag)
    for p in problems:
        out.append({"id": None, "status": "problem", "message": p})
    return blocks, out


def write_blocks(dst_dir: str | Path, blocks: dict[str, np.ndarray],
                 *, dpi: int) -> dict[str, str]:
    """把块写到 disk，返回 {块 id: 路径}。"""
    from .render import write_png

    dst = Path(dst_dir)
    written: dict[str, str] = {}
    for bid, img in blocks.items():
        p = dst / f"{bid}.png"
        write_png(p, img, dpi)
        written[bid] = str(p)
    return written
