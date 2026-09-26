"""直线锚点精修：用表格线的位置做亚像素配准。

为什么需要它（而不是只靠 ECC）：
ECC 是逐像素梯度优化，低分辨率下表格线只有 1~2px 宽，目标函数条件数很差，
实测 72dpi 输入残差 4.7px、96dpi 4.99px，明显劣于 300dpi 的 1.7px。

而把一条长直线**沿其方向投影**，位置精度被整条线的长度平均——
一条 2000px 长的线，其垂直方向的位置可以估到 0.1px 量级，与像素分辨率关系不大。
这才是低分辨率下正确的工具。

为什么必须**分带**投影（踩过的坑）：
去斜之后仍会残留约 0.2° 的旋转。这看似极小，但作用在 3508px 高度上，
一条竖线的 x 位置会沿高度漂移约 12px。整条线一次投影会把这段漂移糊成一团，
线位置估计直接偏掉 4~8px，反而把已经很准的 300dpi 结果从 1.3px 推到 4.1px。

所以：把画布切成若干带，每带内单独投影取线位置，再拟合
    x' = a·X + c·y + e
    y' = b·Y + d·x + f
其中 c、d 正是残余旋转项。这样用 7×9 = 63 条（竖线）与 16×9 = 144 条（横线）
观测去解 3+3 个参数，既拿回了旋转自由度，又靠大量观测平均掉噪声。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .rectify import binarize, line_responses


# ---------------------------------------------------------------- 峰值提取

def _subpixel_peaks(profile: np.ndarray, *, min_dist: int, prom_ratio: float,
                    smooth: float) -> np.ndarray:
    """一维投影曲线找峰，抛物线插值到亚像素。"""
    if profile.size < 8:
        return np.empty(0, dtype=np.float64)

    p = gaussian_filter1d(profile.astype(np.float64), max(smooth, 0.1), mode="nearest")
    peak = float(p.max())
    if peak <= 1e-9:
        return np.empty(0, dtype=np.float64)

    idx, _ = find_peaks(p, distance=max(3, min_dist), prominence=prom_ratio * peak)
    if idx.size == 0:
        return np.empty(0, dtype=np.float64)

    out = []
    for i in idx:
        if 0 < i < p.size - 1:
            y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        out.append(float(i) + delta)
    return np.asarray(out, dtype=np.float64)


@dataclass
class GridLines:
    xs: np.ndarray      # 垂直线的 x 位置
    ys: np.ndarray      # 水平线的 y 位置

    def __len__(self) -> int:
        return len(self.xs) + len(self.ys)


def detect_grid_lines(gray: np.ndarray, binimg: np.ndarray | None = None, *,
                      profile_smooth: float = 1.2,
                      prom_ratio: float = 0.12,
                      dark_bias: int = 0) -> GridLines:
    """整体投影检测线位置（用于模板侧与粗定位）。"""
    if binimg is None:
        binimg = binarize(gray, dark_bias=dark_bias)
    horiz, vert = line_responses(binimg)
    h, w = gray.shape[:2]
    ys = _subpixel_peaks(horiz.sum(axis=1), min_dist=max(6, h // 120),
                         prom_ratio=prom_ratio, smooth=profile_smooth)
    xs = _subpixel_peaks(vert.sum(axis=0), min_dist=max(6, w // 120),
                         prom_ratio=prom_ratio, smooth=profile_smooth)
    return GridLines(xs=xs, ys=ys)


# ---------------------------------------------------------------- 分带检测

@dataclass
class BandedPositions:
    """某个方向上、逐带检测到的线位置。"""

    centers: np.ndarray            # 每个带的中心坐标（另一轴上的位置）
    positions: list[np.ndarray]    # 每带内的线位置列表

    def observations(self) -> int:
        return int(sum(len(p) for p in self.positions))


def detect_banded(gray: np.ndarray, *, axis: str, n_bands: int = 9,
                  min_dist: int = 18, prom_ratio: float = 0.12,
                  smooth: float = 1.2, dark_bias: int = 0) -> BandedPositions:
    """分带投影检测线位置。

    axis='x'：检测竖线的 x 位置，沿 y 方向分带（带中心 = y）
    axis='y'：检测横线的 y 位置，沿 x 方向分带（带中心 = x）
    """
    binimg = binarize(gray, dark_bias=dark_bias)
    horiz, vert = line_responses(binimg)
    resp = vert if axis == "x" else horiz        # 要投影的响应图
    h, w = resp.shape[:2]
    span = h if axis == "x" else w

    edges = np.linspace(0, span, n_bands + 1).astype(int)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float64)

    positions: list[np.ndarray] = []
    for i in range(n_bands):
        y0, y1 = edges[i], edges[i + 1]
        band = resp[y0:y1, :] if axis == "x" else resp[:, y0:y1]
        profile = band.sum(axis=0) if axis == "x" else band.sum(axis=1)
        positions.append(_subpixel_peaks(profile, min_dist=min_dist,
                                         prom_ratio=prom_ratio, smooth=smooth))
    return BandedPositions(centers=centers, positions=positions)


# ---------------------------------------------------------------- 匹配拟合

def _median_gap(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 100.0
    d = np.diff(np.sort(arr))
    d = d[d > 1.0]
    return float(np.median(d)) if d.size else 100.0


def fit_affine_axis(template_pos: np.ndarray, banded: BandedPositions, *,
                    tol: float, iters: int = 5, min_obs: int = 12
                    ) -> tuple[float, float, float, int, float]:
    """拟合 measured = p·T + q·band + r，抗漏检、抗错配。

    返回 (p, q, r, 观测数, 残差中位数)。
    """
    if template_pos.size < 2 or banded.observations() < min_obs:
        return 1.0, 0.0, 0.0, 0, float("inf")

    coef = np.array([1.0, 0.0, 0.0])
    obs: list[tuple[float, float, float]] = []

    for _ in range(iters):
        rows, rhs = [], []
        for c, pos in zip(banded.centers, banded.positions):
            if pos.size == 0:
                continue
            ps = np.sort(pos)
            for T in template_pos:
                pred = coef[0] * T + coef[1] * c + coef[2]
                j = int(np.searchsorted(ps, pred))
                cand = [k for k in (j - 1, j, j + 1) if 0 <= k < ps.size]
                if not cand:
                    continue
                k = min(cand, key=lambda t: abs(ps[t] - pred))
                if abs(ps[k] - pred) <= tol:
                    rows.append((T, c, 1.0))
                    rhs.append(float(ps[k]))
        if len(rows) < min_obs:
            break
        A = np.asarray(rows)
        y = np.asarray(rhs)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        new = np.asarray(sol, dtype=np.float64)
        converged = np.allclose(new, coef, atol=1e-10)
        coef = new
        obs = list(zip(A[:, 0], A[:, 1], y))
        if converged:
            break

    if len(obs) < min_obs:
        return 1.0, 0.0, 0.0, len(obs), float("inf")

    T = np.array([o[0] for o in obs])
    B = np.array([o[1] for o in obs])
    M = np.array([o[2] for o in obs])
    pred = coef[0] * T + coef[1] * B + coef[2]
    resid = float(np.median(np.abs(M - pred)))
    return float(coef[0]), float(coef[1]), float(coef[2]), len(obs), resid


@dataclass
class GridRefineResult:
    matrix: np.ndarray                   # 画布坐标系下的映射：理想输出 -> 当前渲染结果
    n_x: int = 0
    n_y: int = 0
    resid_x: float = float("inf")
    resid_y: float = float("inf")
    shear: tuple[float, float] = (0.0, 0.0)   # 残余旋转项 (c, d)

    @property
    def usable(self) -> bool:
        return self.n_x >= 12 and self.n_y >= 12

    @property
    def reliable(self) -> bool:
        """残差够小才敢用：大残差意味着错配，用它只会帮倒忙。

        阈值 4.0px 是「明显错配」与「低分辨率下的可接受拟合」的分界。
        不能设太小（1.5px）：72/96dpi 输入的线位置本身就有 1~3px 噪声，
        卡太严会把真正需要精修的页挡在门外——实测 s0004 因此从 0.64px 退回 5.7px。
        真正的安全网是闭环验证（修正后重测残差，变差就回退），不是这个阈值。
        """
        return (self.usable and np.isfinite(self.resid_x) and np.isfinite(self.resid_y)
                and self.resid_x <= 4.0 and self.resid_y <= 4.0)


def refine_by_grid(template_lines: GridLines, warped_gray: np.ndarray, *,
                   n_bands: int = 9, tol_cap: float = 10.0,
                   dark_bias: int = 0) -> GridRefineResult:
    """在画布坐标系下用表格线拟合修正映射（6 自由度仿射）。

    template_lines : 模板的线位置（画布坐标）
    warped_gray    : 用当前矩阵渲染到画布后的页面灰度图
    """
    bx = detect_banded(warped_gray, axis="x", n_bands=n_bands, dark_bias=dark_bias)
    by = detect_banded(warped_gray, axis="y", n_bands=n_bands, dark_bias=dark_bias)

    tol_x = float(np.clip(0.15 * _median_gap(template_lines.xs), 2.0, tol_cap))
    tol_y = float(np.clip(0.15 * _median_gap(template_lines.ys), 2.0, tol_cap))

    a, c, e, nx, rx = fit_affine_axis(np.asarray(template_lines.xs), bx, tol=tol_x)
    b, d, f, ny, ry = fit_affine_axis(np.asarray(template_lines.ys), by, tol=tol_y)

    C = np.array([[a, c, e],
                  [d, b, f],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return GridRefineResult(matrix=C, n_x=nx, n_y=ny,
                            resid_x=round(rx, 4), resid_y=round(ry, 4),
                            shear=(round(c, 8), round(d, 8)))


def soft_line_responses(gray: np.ndarray, h_ratio: float = 0.025,
                        v_ratio: float = 0.02,
                        noise_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """灰度形态学线响应（float32，保留强度剖面）。

    为什么不用二值响应：二值化把线的边缘量化了，质心精度被锁在 ±0.5px。
    灰度形态学保留线横截面的强度剖面，质心可以到 0.1px 量级——
    这是整个直线精修链路里最基础的精度来源。
    """
    inv = 255.0 - gray.astype(np.float32)
    h, w = gray.shape[:2]
    kh = max(15, int(w * h_ratio) | 1)
    kv = max(15, int(h * v_ratio) | 1)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    if noise_floor > 0:
        # 抑制浅色底纹：只有比 noise_floor 更深的响应才留下
        horiz = np.maximum(horiz - noise_floor, 0.0)
        vert = np.maximum(vert - noise_floor, 0.0)
    return horiz, vert


# ---------------------------------------------------------------- 单应精修

def _weighted_centroids(resp: np.ndarray, ts: np.ndarray, pred: np.ndarray,
                        half: int, *, along: str = "rows") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿采样位置取响应质心，得到亚像素的线位置。

    along="rows"：t 是**行**坐标，在列方向上定位竖线（resp 为竖直响应图）
    along="cols"：t 是**列**坐标，在行方向上定位横线（resp 为水平响应图）

    必须区分两个方向：早先只实现了 rows 一种，横线走的是转置访问，
    尺寸凑巧时不报错、只是静默地采样了错误的位置；遇到宽高比小的文档
    （如支票 1518x534）立刻越界。
    """
    n_other = resp.shape[1] if along == "rows" else resp.shape[0]
    lo = np.round(pred).astype(np.int64) - half
    offs = np.arange(2 * half + 1)
    idx = lo[:, None] + offs[None, :]
    valid = (idx >= 0) & (idx < n_other)
    idxc = np.clip(idx, 0, n_other - 1)

    if along == "rows":
        vals = np.where(valid, resp[ts[:, None], idxc], 0.0).astype(np.float64)
    else:
        vals = np.where(valid, resp[idxc, ts[:, None]], 0.0).astype(np.float64)

    wsum = vals.sum(axis=1)
    keep = wsum > 0
    if not np.any(keep):
        return ts[:0], pred[:0], wsum[:0]

    off_centered = (offs - half)[None, :].astype(np.float64)
    pos = pred[keep] + (vals[keep] * off_centered).sum(axis=1) / wsum[keep]
    return ts[keep], pos, wsum[keep]


def _fit_line(resp: np.ndarray, pred_positions: np.ndarray, *,
              half: int = 12, step: int = 4, min_pts: int = 8,
              along: str = "rows") -> list[tuple[float, float, int]]:
    """把每条线拟合成 x = m·y + p（逐条稳健拟合）。

    必须做两件事，否则直线方程会被带偏：
      1. 响应门限：一条线并不贯穿整页，在它不存在的区域里，
         质心窗口会抓到别的东西（文字、相邻线），产生完全错误的采样点。
      2. sigma 截断迭代：即使有线，局部干扰（印章、手写压线）也会留下离群点。
    """
    span = resp.shape[0] if along == "rows" else resp.shape[1]
    ts = np.arange(0, span, step, dtype=np.int64)
    out: list[tuple[float, float, int]] = []

    for p0 in pred_positions:
        pred = np.full(ts.shape, float(p0))
        t_ok, pos, w = _weighted_centroids(resp, ts, pred, half, along=along)
        if t_ok.size < min_pts:
            out.append((0.0, float(p0), 0))
            continue

        # 1. 响应门限：只保留线真实存在的采样位置
        thr = 0.25 * float(np.percentile(w, 90))
        keep = w >= max(thr, 1e-9)
        t1, p1, w1 = t_ok[keep], pos[keep], w[keep]
        if t1.size < min_pts:
            out.append((0.0, float(p0), 0))
            continue

        # 2. 加权最小二乘 + sigma 截断
        m, p = 0.0, float(p0)
        for _ in range(4):
            A = np.stack([t1.astype(np.float64), np.ones(t1.size)], axis=1)
            sol, *_ = np.linalg.lstsq(A * w1[:, None], p1 * w1, rcond=None)
            m_new, p_new = float(sol[0]), float(sol[1])
            r = p1 - (m_new * t1 + p_new)
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            sigma = max(1.5, 3.5 * 1.4826 * mad)
            keep2 = np.abs(r - med) < sigma
            if keep2.all() or keep2.sum() < min_pts:
                m, p = m_new, p_new
                break
            t1, p1, w1 = t1[keep2], p1[keep2], w1[keep2]
            m, p = m_new, p_new

        out.append((m, p, int(t1.size)))
    return out


@dataclass
class IntersectRefineResult:
    matrix: np.ndarray                 # 模板坐标 -> 当前渲染结果坐标（单应）
    n_points: int = 0
    n_inliers: int = 0
    resid: float = float("inf")

    @property
    def usable(self) -> bool:
        return self.n_points >= 24 and np.isfinite(self.resid)


def refine_by_intersections(template_lines: GridLines, warped_gray: np.ndarray, *,
                            half: int = 12, ransac_thresh: float = 2.5,
                            max_resid: float = 3.0,
                            noise_floor: float = 0.0) -> IntersectRefineResult:
    """用「表格线交点」的对应关系拟合单应矩阵（纳入透视分量）。

    与仿射版的区别：仿射只能表达两轴缩放+平移+旋转，吸收不了纸张/扫描带来的透视；
    实测透视分量是剩余误差的主因（全页最大误差 11.7px 集中在外围）。

    做法：把每条线拟合成解析直线方程，再解交点 —— 交点位置由整条线决定，
    比在像素图上找交点连通域稳得多。线位置用**灰度**响应求质心（亚像素）。
    """
    # 线位置用灰度响应（亚像素）；不二值化，避免把精度锁在 ±0.5px
    horiz, vert = soft_line_responses(warped_gray, noise_floor=noise_floor)

    vlines = _fit_line(vert, np.asarray(template_lines.xs), half=half, along="rows")
    hlines = _fit_line(horiz, np.asarray(template_lines.ys), half=half, along="cols")

    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    for i, X in enumerate(template_lines.xs):
        mv, pv, nv = vlines[i]
        if nv < 8:
            continue
        for j, Y in enumerate(template_lines.ys):
            mh, ph, nh = hlines[j]
            if nh < 8:
                continue
            # 竖线 x = mv·y + pv 与 横线 y = mh·x + ph 的交点
            denom = 1.0 - mv * mh
            if abs(denom) < 1e-9:
                continue
            x = (mv * ph + pv) / denom
            y = mh * x + ph
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            src_pts.append((float(X), float(Y)))
            dst_pts.append((float(x), float(y)))

    if len(src_pts) < 24:
        return IntersectRefineResult(matrix=np.eye(3), n_points=len(src_pts))

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh, maxIters=5000,
                                confidence=0.995)
    if H is None:
        return IntersectRefineResult(matrix=np.eye(3), n_points=len(src))

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    d = np.linalg.norm(proj - dst, axis=1)
    inliers = int(mask.sum()) if mask is not None else len(src)
    resid = float(np.median(d[mask.ravel() == 1])) if mask is not None else float(np.median(d))

    res = IntersectRefineResult(matrix=np.asarray(H, dtype=np.float64),
                                n_points=len(src), n_inliers=inliers, resid=resid)
    if (not np.isfinite(H).all()) or inliers < 20 or resid > max_resid:
        res.matrix = np.eye(3)
    return res
