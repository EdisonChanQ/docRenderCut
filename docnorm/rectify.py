"""矫正层：光照归一化、二值化、长直线检测、去斜、结构特征图。

设计要点：
- 输入页「整体完整」，所以不依赖纸张边缘检测——真实馈纸扫描件的图形边界往往就是纸边，
  没有背景余量可用来找纸张四边形。因此对齐基准取自**内容本身的结构线**，这也更稳。
- 长直线（表格线、边框线）是理想的对齐基准：位置与内容无关，
  手写金额写歪、印章盖在框上，都不影响线的位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------- 纸张检测

@dataclass
class PaperQuad:
    corners: np.ndarray          # (4,2) 顺序 tl,tr,br,bl，原图坐标
    area_ratio: float
    aspect: float
    ok: bool = True
    reason: str = ""
    confidence: str = "auto"     # auto | consensus | explicit | low

    def __len__(self) -> int:
        return 4

    def _aspect_from_sides(self) -> None:
        p = np.asarray(self.corners, dtype=np.float64)
        s = [float(np.hypot(*(p[(i + 1) % 4] - p[i]))) for i in range(4)]
        self.aspect = float((s[0] + s[2]) / max(1e-6, s[1] + s[3]))

    @classmethod
    def from_corners(cls, corners) -> "PaperQuad":
        """显式给定四角（文档扫描类 App 的"手动裁剪"路径）。

        自动检测在低质量照片上确实会失效：这里试过灰度大津、色度中性度、
        分块自适应、Canny+Hough 四种，全部不可靠——桌面木纹与票据防伪底纹
        都是长边缘，而左下角的阴影让纸的下边界在灰度上直接消失。

        所以模板登记时人工确认一次四角，是模板类的**必要步骤**，不是降级方案。
        显式角点走宽松校验：只挡真正的退化（非有限值、面积过小、边长比 > 2.5），
        不做"近似矩形"的假设——真实照片必然有透视，对边不等是正常的。
        """
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        q = cls(corners=_order_corners(pts), area_ratio=float("nan"), aspect=0.0,
                confidence="explicit")
        q._aspect_from_sides()
        q.ok, q.reason = True, "人工指定"
        return q


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """按 左上/右上/右下/左下 排序。"""
    pts = np.asarray(pts, dtype=np.float64)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float64)


def _fit_edge_lines(hull: np.ndarray, corners: np.ndarray
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """把凸包点分配到 4 条边，各自拟合直线。返回 [(点, 方向), ...]。"""
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ab = b - a
        n = np.hypot(*ab)
        if n < 1e-6:
            return []
        u = ab / n
        nrm = np.array([-u[1], u[0]])
        rel = hull - a
        t = rel @ u
        s = rel @ nrm
        # 落在该边范围内、且离该边足够近的点
        sel = (t > -0.05 * n) & (t < 1.05 * n) & (np.abs(s) < 0.08 * n)
        pts = hull[sel]
        if len(pts) < 8:
            return []
        # 稳健拟合：对主成分方向做最小二乘（迭代去离群）
        keep = np.ones(len(pts), dtype=bool)
        for _ in range(3):
            p = pts[keep]
            if len(p) < 6:
                break
            mean = p.mean(axis=0)
            uu, ss, _ = np.linalg.svd(p - mean, full_matrices=False)
            d = uu[0]
            nn = np.array([-d[1], d[0]])
            r = (p - mean) @ nn
            mad = np.median(np.abs(r - np.median(r)))
            thr = max(1.0, 3.5 * 1.4826 * mad)
            new_keep = np.abs(r - np.median(r)) < thr
            if new_keep.sum() < 6 or new_keep.all():
                break
            idx = np.where(keep)[0][new_keep]
            keep[:] = False
            keep[idx] = True
        p = pts[keep]
        mean = p.mean(axis=0)
        uu, _, _ = np.linalg.svd(p - mean, full_matrices=False)
        lines.append((mean, uu[0]))
    return lines


def _intersect_lines(l1, l2) -> np.ndarray | None:
    (p1, d1), (p2, d2) = l1, l2
    A = np.stack([d1, -d2], axis=1)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def binarize_document(gray: np.ndarray, *, dark_bias: int = 25,
                      despeckle: int = 0, use_clahe: bool = False) -> np.ndarray:
    """把页面转成纯黑白两色（纸=255，墨=0）。

    参数选择是有依据的，别随手改：

    - **默认不开 CLAHE**。CLAHE 会放大局部对比，把票据的防伪底纹（guilloche）
      从浅灰中调拉成强噪点——实测一张支票因此产生 19.7% 的墨占比，
      左半页几乎全被底纹噪点覆盖。开 CLAHE 只适合"纯文字、无底纹"的白纸。
    - **背景除法照旧保留**：它只压掉低频明暗不均（阴影、拍照渐晕），
      不放大局部纹理，是"灰度阈值能稳定工作"的前提。
    - **深墨偏置 dark_bias**：在归一化图上再取"比大津阈值更暗"的像素。
      实测 25 效果最好（墨占比 8.3%，文字清晰、底纹基本消失）；
      调到 40 会开始掉笔画（金额大写、MICR 磁码出现断笔），OCR 反而更差。
    - despeckle 默认关闭：它同样会误删标点和小字号笔画。

    放在渲染之后做：先归一化再二值化，阴影/底纹/色偏都不会变成噪点。
    """
    src = gray
    if use_clahe:
        src = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(src)

    k = max(31, (min(src.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(src, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4.0)
    norm = cv2.divide(src, np.maximum(bg, 1), scale=255)
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    level, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(1, min(254, level - dark_bias)))
    bw = cv2.threshold(norm, thr, 255, cv2.THRESH_BINARY)[1]

    if despeckle > 0:
        ink = cv2.bitwise_not(bw)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        if n > 1:
            small = np.where(stats[1:, cv2.CC_STAT_AREA] < despeckle)[0] + 1
            if small.size:
                ink[np.isin(lab, small)] = 0
                bw = cv2.bitwise_not(ink)
    return bw


def _poly_area(q: np.ndarray) -> float:
    """四边形面积（shoelace）。注意 np.cross 对 2 维向量已废弃，不能直接用。"""
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quad_sane(q: np.ndarray, shape: tuple[int, int]) -> bool:
    """候选合理性校验：把数值上"成功"但几何上荒谬的结果挡掉。

    没有这层校验，一个退化的候选（相邻边近乎平行 -> 交点飞到十几万像素外）
    会混进共识里，把中位数整体带偏——而且不报错。
    """
    if q is None or not np.isfinite(q).all():
        return False
    sides = [float(np.hypot(*(q[(i + 1) % 4] - q[i]))) for i in range(4)]
    if min(sides) < 1.0:
        return False
    if min(sides) / max(sides) < 0.15:          # 细长/退化
        return False
    h, w = shape
    if _poly_area(q) / (h * w) < 0.15:          # 面积太小
        return False
    return True


def _quad_from_method(bgr: np.ndarray, mode: str) -> np.ndarray | None:
    """用某种分割方式得到纸张四边形（输入为缩略图）。"""
    gray = to_gray(bgr)
    h, w = gray.shape[:2]

    if mode == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, max(31, (min(h, w) // 6) | 1), -8)
        if np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]]).mean() < 127:
            bw = cv2.bitwise_not(bw)
    else:
        src = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        if mode == "neutral":
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            src = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
        _, bw = cv2.threshold(src.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
        # 边界应以背景为主；若边界以纸为主，说明极性反了或纸铺满画面
        if border.mean() > 127:
            bw = cv2.bitwise_not(bw)

    k = max(3, (min(h, w) // 120) | 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((k * 3 + 1, k * 3 + 1), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return None
    comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(lab == comp, 255, 0).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    # 注意：cv2.convexHull 默认只返回**凸包顶点**（十几个点），
    # 不是边界上的完整点序列。拿它去分边拟合，每条边只有几个点，拟合无从谈起。
    # 所以要填充凸包再重新抽取稠密边界。
    hull_pts = cv2.convexHull(cnt)
    hull_mask = cv2.fillConvexPoly(np.zeros_like(gray), hull_pts, 255)
    hc, _ = cv2.findContours(hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not hc:
        return None
    hull = max(hc, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(hull) < 40:
        return None

    lines = _fit_edge_lines(hull, _order_corners(hull_pts.reshape(-1, 2).astype(np.float64)))
    if len(lines) != 4:
        return None
    quad = []
    for i in range(4):
        p = _intersect_lines(lines[i], lines[(i + 1) % 4])
        if p is None:
            return None
        quad.append(p)
    quad = _order_corners(np.asarray(quad))
    return quad if _quad_sane(quad, gray.shape[:2]) else None


def detect_paper_quad(bgr: np.ndarray, *, max_side: int = 1400,
                      min_area_ratio: float = 0.15,
                      agree_px: float = 5.0) -> PaperQuad | None:
    """检测纸张四边形：三种分割方式取共识，并过校验门禁。

    为什么用共识而不是挑一种：这个场景下没有单一方法可靠——
    全局灰度会被阴影骗（纸的下边界在灰度上消失），
    色度会被同色阴影骗，自适应阈值会被纸面防伪底纹骗。
    三法各自失败的方式不同，取中位数能把偶然失败平均掉，
    而**离散度**本身就是一个可用的置信度信号。

    四角用「凸包 -> 分边稳健拟合直线 -> 相邻边求交」，不用 approxPolyDP：
    后者对凸包上的细凸刺极敏感，一个背景高光三角就能让角点落到错误位置，
    而且不报错。
    """
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h0, w0)))
    if scale < 1.0:
        small = cv2.resize(bgr, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bgr

    cands = []
    for mode in ("gray", "neutral", "adaptive"):
        q = _quad_from_method(small, mode)
        if q is not None:
            cands.append(q)
    if not cands:
        return None

    spread = float("nan")
    if len(cands) >= 2:
        stack = np.stack(cands)
        med = np.median(stack, axis=0)
        spread = float(np.max(np.linalg.norm(stack - med[None], axis=2)))
        conf = "consensus" if spread <= agree_px else "low"
    else:
        med, conf = cands[0], "auto"

    quad = med / scale if scale < 1.0 else med
    h, w = h0, w0
    area = _poly_area(quad)
    res = PaperQuad(corners=quad, area_ratio=area / (h * w), aspect=0.0,
                    confidence=conf)
    if res.area_ratio < min_area_ratio:
        res.ok, res.reason = False, "四边形面积占比过小"
        return res
    _validate_paper_quad(res, (w, h))
    if conf == "low":
        res.ok = False
        res.reason = f"三种分割方式不一致（离散度 {spread:.1f}px），需人工确认角点"
    return res


def _validate_paper_quad(q: PaperQuad, size: tuple[int, int],
                         *, strict: bool = True) -> None:
    """闸门：不合理的四边形宁可不用，也不能悄悄扭歪。

    **不要假设"近似矩形"**：真实照片必然带透视，对边长度不等、角度偏离 90°
    都是正常的（本项目实测一张手机拍的支票照片，两侧高度 374 vs 430px，差 15%）。
    所以只判断"是不是被扭曲到荒谬"，而不是"像不像矩形"。

    strict=True（自动检测）用较紧的门限；strict=False（人工指定）只挡退化：
    人是权威，引擎不该因为真实透视而否掉人工给的角点。
    """
    p = np.asarray(q.corners, dtype=np.float64)
    w, h = size
    if not np.isfinite(p).all():
        q.ok, q.reason = False, "角点含非有限值"
        return

    sides = [float(np.hypot(*(p[(i + 1) % 4] - p[i]))) for i in range(4)]
    q.aspect = float((sides[0] + sides[2]) / max(1e-6, sides[1] + sides[3]))

    side_lim = 1.35 if strict else 2.5
    for i in range(2):
        a, b = sides[i], sides[i + 2]
        if min(a, b) < 1.0 or max(a, b) / min(a, b) > side_lim:
            q.ok, q.reason = False, f"对边长度比过大 {a:.0f} vs {b:.0f}px（疑似非纸张）"
            return

    lo, hi = (75.0, 105.0) if strict else (55.0, 125.0)
    for i in range(4):
        v1 = p[(i - 1) % 4] - p[i]
        v2 = p[(i + 1) % 4] - p[i]
        ang = float(np.degrees(np.arccos(np.clip(
            v1 @ v2 / (np.hypot(*v1) * np.hypot(*v2)), -1, 1))))
        if not (lo <= ang <= hi):
            q.ok, q.reason = False, f"角点 {i} 角度异常 {ang:.1f}deg"
            return

    if q.aspect < 1.05:
        q.ok, q.reason = False, f"宽高比异常 {q.aspect:.2f}"
        return

    if _poly_area(p) / (w * h) < 0.15:
        q.ok, q.reason = False, "四边形面积占比过小"
        return

    q.ok, q.reason = True, ""


def paper_warp(bgr: np.ndarray, quad: PaperQuad,
               target: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """把纸张四边形矫正成正面平行的 target 尺寸，返回 (图, 原图->输出 的 3x3)。"""
    cw, ch = target
    src = np.asarray(quad.corners, dtype=np.float32)
    dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(bgr, H, (cw, ch), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, H


# ---------------------------------------------------------------- 预处理

def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """抑制扫描/拍照的亮度不均：CLAHE + 背景除法。"""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    eq = clahe.apply(gray)

    # 背景估计：大核中值/模糊，用于去除低频渐变
    k = max(31, (min(gray.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4.0)
    bg = np.maximum(bg, 1)
    norm = cv2.divide(eq, bg, scale=255)
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray, *, dark_bias: int = 0) -> np.ndarray:
    """输出墨迹掩膜：墨=255，纸=0。

    dark_bias > 0 时把阈值往深色方向推，只保留**真正的深墨**。
    拍照件上必须这么调：票据的防伪底纹（guilloche）是浅灰中调，
    大津阈值会把它一起判成墨，于是底纹被当成结构线——
    实测一张支票模板因此检出 46 条"竖线"，全是纹理。
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    level, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(1, min(254, level - dark_bias)))
    return cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)[1]


# ---------------------------------------------------------------- 直线检测

def _kernel_len(span: int, ratio: float, floor: int | None = None) -> int:
    """线检测核长（必须随画面尺寸自适应）。

    下限的取值有实测依据，不要凭直觉调：
    - 固定下限 **25** 是合适的。它同时解决两件事：小画布文档（支票 1518x534）里
      防伪底纹的短笔画不被当成结构线；低分辨率页面（72dpi，宽 595px）里
      **文字笔画**不被当成结构线。
    - 试过按尺寸自适应 `clip(span/40, 8, 25)`（595px 页降到 15）：更差——
      72dpi 档残差从 10.4px 恶化到 14.9px，因为笔画的连续长度够过核长了。
    - 试过按比例只用 ratio（无下限）：同样把底纹和文字收成"结构线"，重复上述恶化。
    结论：**核长下限要够大，宁多滤勿乱收**。漏掉的真线会由直线精修的观测门禁挡住，
    而错收的假线会污染结构图，代价大得多。
    """
    if floor is None:
        floor = 25
    return max(floor, int(span * ratio) | 1)


def line_responses(binimg: np.ndarray, h_ratio: float = 0.025, v_ratio: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """分离水平线与垂直线的响应图。"""
    h, w = binimg.shape[:2]
    kh = _kernel_len(w, h_ratio)
    kv = _kernel_len(h, v_ratio)
    horiz = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
    vert = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    return horiz, vert


def detect_segments(resp: np.ndarray, *, min_len: int, max_gap: int) -> list[tuple[float, float, float, float]]:
    lines = cv2.HoughLinesP(
        resp, 1, np.pi / 720,
        threshold=max(30, min_len // 3),
        minLineLength=min_len, maxLineGap=max_gap,
    )
    if lines is None:
        return []
    # OpenCV 4 返回 (N,1,4)，OpenCV 5 返回 (N,4)，统一 reshape
    seg = np.asarray(lines).reshape(-1, 4)
    return [(float(a), float(b), float(c), float(d)) for a, b, c, d in seg]


@dataclass
class SkewEstimate:
    angle_deg: float
    n_h: int
    n_v: int
    dispersion_deg: float   # 角度离散度，用于判断估计是否可靠

    @property
    def reliable(self) -> bool:
        return (self.n_h + self.n_v) >= 3 and self.dispersion_deg < 1.5


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


def estimate_skew(binimg: np.ndarray, *, max_angle: float = 12.0) -> SkewEstimate:
    """用长直线估计页面内容的倾斜角（度，正值表示内容顺时针偏）。

    水平线与垂直线各自给出角度估计，合并为一个值；同时给出离散度用于可靠性判断。
    """
    h, w = binimg.shape[:2]
    horiz, vert = line_responses(binimg)

    angles: list[float] = []
    weights: list[float] = []

    for segs, base in (
        (detect_segments(horiz, min_len=int(w * 0.15), max_gap=int(w * 0.02)), 0.0),
        (detect_segments(vert, min_len=int(h * 0.10), max_gap=int(h * 0.02)), 90.0),
    ):
        for x1, y1, x2, y2 in segs:
            length = float(np.hypot(x2 - x1, y2 - y1))
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # 归一到基准方向附近的偏差
            dev = ang - base
            while dev > 90.0:
                dev -= 180.0
            while dev < -90.0:
                dev += 180.0
            if abs(dev) <= max_angle:
                angles.append(dev)
                weights.append(length)

    n_h = len(angles)
    n_v = 0
    if not angles:
        return SkewEstimate(0.0, 0, 0, 99.0)

    a = np.asarray(angles)
    wt = np.asarray(weights)
    deg = _weighted_median(a, wt)
    disp = float(np.sqrt(np.average((a - deg) ** 2, weights=wt)))
    return SkewEstimate(deg, n_h, n_v, disp)


# ---------------------------------------------------------------- 去斜

def rotate_image(img: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """绕图像中心旋转，输出同尺寸图与 3x3 变换矩阵（原图坐标 -> 输出坐标）。"""
    h, w = img.shape[:2]
    M2 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    out = cv2.warpAffine(
        img, M2, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if img.ndim == 3 else 255,
    )
    M = np.vstack([M2, [0.0, 0.0, 1.0]])
    return out, M


def deskew(bgr: np.ndarray, *, max_angle: float = 12.0, dark_bias: int = 0,
           verbose: bool = False) -> tuple[np.ndarray, np.ndarray, SkewEstimate]:
    """闭环去斜：先估计，再验证残余角是否真的变小；否则反向旋转。

    dark_bias 必须与配准层一致（由调用方传入）：拍照件里桌面是**深色**，
    dark_bias=0 时整片桌面被判成墨，线检测找的是桌边而不是单据的表格线，
    去斜估计直接失效（实测返回 0.000°，而实际有 1~2° 倾斜），
    残余旋转随后会在离旋转中心远的区域造成约 10px 的坐标偏差。

    单向试错而不是猜旋转方向的符号——符号约定在 OpenCV 里极易搞反，
    用「残余角是否下降」作为判据，逻辑上不可能错。
    """
    gray = normalize_illumination(to_gray(bgr))
    est = estimate_skew(binarize(gray, dark_bias=dark_bias), max_angle=max_angle)

    if abs(est.angle_deg) < 0.03:
        return bgr, np.eye(3), est

    best = None
    for cand in (-est.angle_deg, est.angle_deg):
        rot, M = rotate_image(bgr, cand)
        g2 = normalize_illumination(to_gray(rot))
        e2 = estimate_skew(binarize(g2, dark_bias=dark_bias), max_angle=max_angle)
        score = abs(e2.angle_deg)
        if best is None or score < best[0]:
            best = (score, rot, M, e2)

    assert best is not None
    score, rot, M, e2 = best
    if verbose:
        print(f"    deskew: {est.angle_deg:+.3f}deg -> {e2.angle_deg:+.3f}deg")
    return rot, M, e2


# ---------------------------------------------------------------- 结构特征

def structure_image(gray: np.ndarray, binimg: np.ndarray | None = None,
                    *, blur: float = 2.0, h_ratio: float = 0.025,
                    v_ratio: float = 0.02, dark_bias: int = 0) -> np.ndarray:
    """生成只保留长直线结构的 float32 特征图（0~1），供 ECC 配准使用。

    为什么不用原灰度图：手写金额、签名、印章是**可变内容**，会把相关性目标带偏。
    结构线位置与内容无关，用它做配准基准既稳又准。
    """
    if binimg is None:
        binimg = binarize(gray, dark_bias=dark_bias)
    horiz, vert = line_responses(binimg, h_ratio, v_ratio)
    struct = cv2.max(horiz, vert)
    if blur > 0.05:
        struct = cv2.GaussianBlur(struct, (0, 0), blur)
    return np.clip(struct.astype(np.float32) / 255.0, 0.0, 1.0)


def structure_mask(gray: np.ndarray, binimg: np.ndarray | None = None,
                   *, dilate_px: int = 9, dark_bias: int = 0) -> np.ndarray:
    """结构区域的膨胀掩膜（uint8 0/255），可用于遮挡可变内容。"""
    if binimg is None:
        binimg = binarize(gray, dark_bias=dark_bias)
    horiz, vert = line_responses(binimg)
    m = cv2.max(horiz, vert)
    k = max(3, dilate_px | 1)
    return cv2.dilate(m, np.ones((k, k), np.uint8))
