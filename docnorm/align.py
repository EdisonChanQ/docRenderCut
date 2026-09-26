"""配准层：把每一页精确对齐到模板坐标系。

链条：粗略初值 -> 相位相关（亚像素平移）-> 多尺度 ECC 仿射 -> 方向/合理性校验

四个刻意的设计决定：

1. **配准在「结构特征图」上做，不在原灰度图上做。**
   手写金额、签名、印章是可变内容，会把相关性目标带偏；表格线的位置与内容无关。
   这是「锚点稳定」这一判断的工程落地形式。

2. **全程在降采样工作分辨率下计算，最后把矩阵抬回全分辨率。**
   全分辨率 8.7M 像素上做 ECC 是不可用的慢。工作尺度取 ~1100px 长边，
   精度损失换算到全分辨率远低于 0.1px，速度提升一个数量级。
   矩阵抬升是纯坐标换算：M_full = K⁻¹ · M_work · K。

3. **坐标方向不靠推导，靠闭环校验。**
   OpenCV 的 warp 方向约定、phaseCorrelate 的位移符号极易搞反且不会报错。
   凡涉及方向的步骤都同时构造两个候选、渲染后与模板比相关度、取优者。

4. **结果必须过合理性门禁。**
   ECC 是迭代优化，可能收敛到病态解。超限一律判为异常页并打标，绝不把畸形图片放进下游。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .rectify import normalize_illumination, structure_image, to_gray


# ---------------------------------------------------------------- 坐标工具

def as_affine(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M if M.shape == (3, 3) else np.vstack([M, [0.0, 0.0, 1.0]])


def is_affine(M3: np.ndarray, *, atol: float = 1e-10) -> bool:
    """判定是否为仿射（第三行未被透视项污染）。

    注意：把单应矩阵丢给 warpAffine 不会报错，只会静默丢掉透视分量，
    所以任何渲染入口都必须先判别。
    """
    M = as_affine(M3)
    return (abs(M[2, 0]) < atol and abs(M[2, 1]) < atol and abs(M[2, 2] - 1.0) < atol)


def warp_forward(src: np.ndarray, M3: np.ndarray, size: tuple[int, int],
                 *, border: tuple[int, int, int] | int = 255,
                 interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """按「源坐标 -> 目标坐标」的正向映射 M3 渲染（自动判别仿射/单应）。

    cv2.warpAffine 默认把矩阵当作 src->dst（内部自行求逆），与这里语义一致，
    已实测确认（translate(+50,0) 后内容确实右移 50px）。
    """
    M = as_affine(M3)
    if is_affine(M):
        return cv2.warpAffine(
            src, M[:2].astype(np.float32), size,
            flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border,
        )
    return cv2.warpPerspective(
        src, M.astype(np.float32), size,
        flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def initial_matrix(src_size: tuple[int, int], canvas: tuple[int, int]) -> np.ndarray:
    """初值：按两轴独立缩放把源页铺到画布。

    两轴独立（各向异性）是刻意的：真实扫描的缩放偏差在两个方向上并不相等，
    初值就该允许它不同，让后续仿射在此基础上微调。
    """
    sw, sh = src_size
    cw, ch = canvas
    sx, sy = cw / float(sw), ch / float(sh)
    M = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    M[0, 2] += (cw - sw * sx) / 2.0
    M[1, 2] += (ch - sh * sy) / 2.0
    return M


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_matrix(f: float) -> np.ndarray:
    return np.array([[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rescale_matrix(M: np.ndarray, k: float) -> np.ndarray:
    """把「全分辨率坐标 -> 全分辨率坐标」的映射换算到 k 倍工作分辨率下的等价映射。"""
    K = scale_matrix(k)
    return K @ as_affine(M) @ np.linalg.inv(K)


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    interp = cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_CUBIC
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


# ---------------------------------------------------------------- 粗定位

@dataclass
class CoarseLocate:
    matrix: np.ndarray      # 样本坐标 -> 画布坐标
    scale: float            # 单据在样本中的尺度（样本像素 / 画布像素）
    score: float            # 模板匹配归一化相关度
    angle_deg: float = 0.0  # 粗定位估计的旋转角
    n_scales: int = 0
    mean_dist: float = float("nan")   # 结构点的平均 Chamfer 距离（工作像素）


def _rotate_expand(img: np.ndarray, angle_deg: float):
    """绕中心旋转并扩边（保证内容不丢）。返回 (图, 3x3 变换, 新尺寸)。

    cv2.getRotationMatrix2D 的正角为逆时针（图像坐标，原点在左上）。
    这里不猜方向，把它当已知量写进公式，再用合成自检验证。
    """
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M2 = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M2[0, 0]), abs(M2[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    M2[0, 2] += nw / 2.0 - c[0]
    M2[1, 2] += nh / 2.0 - c[1]
    out = cv2.warpAffine(img, M2, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, np.vstack([M2, [0.0, 0.0, 1.0]]), (nw, nh)


def coarse_locate(template_struct: np.ndarray, page_struct: np.ndarray,
                  canvas: tuple[int, int], *, work_long: int = 800,
                  f_steps: int = 12, angle_range: float = 6.0,
                  angle_steps: int = 9, tol_px: float = 12.0
                  ) -> CoarseLocate | None:
    """在样本里**多尺度 + 多角度**搜索单据内容的位置，得到粗定位变换。

    为什么必须有这一层：拍照件的纸张边界检测不可靠（阴影、桌面纹理、防伪底纹
    都会骗过分割），而每页人工给四角又违背"输入尺寸/边距随便变"的诉求。
    换个思路——不找纸边，直接在样本里找**内容**。

    **角度也必须搜**：只搜尺度+平移时，实测一张带 1.9° 倾斜的支票照片，
    ECC 从差了 2° 的初值出发收敛不到正确解，输出仍带旋转，
    指定坐标裁出来的内容整体偏了 86px。

    **打分必须用 Chamfer 距离，不能用归一化相关**。结构图是稀疏二值的，
    归一化相关只要对上一条长线就能拿高分——实测合成自检里真值 f=0.70 被误判成 0.385。
    改用距离变换：把样本结构图变成"到最近结构像素的距离场"，
    让模板的结构点去找最近的样本结构，平均距离越小越好。
    离得远的点会被距离场如实惩罚，因此极值陡峭、尺度可辨。
    """
    a = work_long / float(max(canvas))
    tw, th = max(16, int(round(canvas[0] * a))), max(16, int(round(canvas[1] * a)))
    T0 = _resize_to(template_struct, (tw, th)).astype(np.float32)
    pw = max(16, int(round(page_struct.shape[1] * a)))
    ph = max(16, int(round(page_struct.shape[0] * a)))
    R = _resize_to(page_struct, (pw, ph)).astype(np.float32)

    t_thr = max(0.2, 0.35 * float(T0.max()))
    r_thr = max(0.2, 0.35 * float(R.max()))
    Tm = (T0 > t_thr).astype(np.uint8)
    Rm = (R > r_thr).astype(np.uint8)
    if Tm.sum() < 30 or Rm.sum() < 30:
        return None

    # 距离场：到最近"样本结构像素"的距离（单位：工作像素）
    dt = cv2.distanceTransform(1 - Rm, cv2.DIST_L2, 3).astype(np.float32)

    ratio = max(page_struct.shape[0], page_struct.shape[1]) / float(max(canvas))
    angles = np.linspace(-angle_range, angle_range, angle_steps)

    best = None      # (平均距离, -f, |ang|, 元组)
    n = 0
    for f in np.linspace(0.35 * ratio, 2.2 * ratio, f_steps):
        w, h = int(round(tw * f)), int(round(th * f))
        if w < 40 or h < 24:
            continue
        Tf = cv2.resize(Tm, (w, h), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        for ang in angles:
            if abs(ang) < 1e-9:
                Tt, Mt = Tf, np.eye(3)
            else:
                Tt, Mt, _sz = _rotate_expand(Tf.astype(np.float32) * 255.0, float(ang))
                Tt = (Tt > 127).astype(np.uint8)
            nw, nh = Tt.shape[1], Tt.shape[0]
            if nw > pw or nh > ph or nw < 24 or nh < 16:
                continue
            n += 1
            if Tt.sum() < 24:
                continue
            cm = cv2.matchTemplate(dt, Tt.astype(np.float32), cv2.TM_CCORR)
            if cm.size == 0:
                continue
            _mn, _mx, mnloc, _mxl = cv2.minMaxLoc(cm)
            mean_dist = float(cm.min()) / float(Tt.sum())

            # 同分优先取尺度更大、角度更小者（证据更充分、形变更小）
            key = (round(mean_dist, 6), -f, abs(float(ang)))
            if best is None or key < best[0]:
                best = (key, mean_dist, float(mnloc[0]), float(mnloc[1]), float(f), float(ang), Mt)

    if best is None or n == 0:
        return None

    _key, mean_dist, x0, y0, f, ang, Mt = best
    # 工作坐标下的正向链：canvas_w --(×f)--> T_f --(Mt)--> T_θ --(+x0,y0)--> page_w
    lin = Mt[:2, :2] * f
    off = (Mt[:2, 2] + np.array([x0, y0])) / a
    M_c2p = np.array([[lin[0, 0], lin[0, 1], off[0]],
                      [lin[1, 0], lin[1, 1], off[1]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    score = max(0.0, 1.0 - mean_dist / max(tol_px, 1e-6))
    return CoarseLocate(matrix=np.linalg.inv(M_c2p), scale=f, score=score,
                        angle_deg=ang, n_scales=n, mean_dist=mean_dist)


# ---------------------------------------------------------------- 打分

def struct_score(template_struct: np.ndarray, candidate: np.ndarray) -> float:
    a = template_struct.astype(np.float32)
    b = candidate.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return -1.0
    return float((a * b).sum() / denom)


def _score_matrix(template_w: np.ndarray, struct_w: np.ndarray,
                  M_w: np.ndarray, work_canvas: tuple[int, int]) -> float:
    warped = warp_forward(struct_w, M_w, work_canvas, border=0)
    return struct_score(template_w, warped)


def _best_of(template_w: np.ndarray, struct_w: np.ndarray,
             candidates: dict[str, np.ndarray], work_canvas: tuple[int, int],
             ) -> tuple[str, np.ndarray, float, dict[str, float]]:
    """在一组候选矩阵里选与模板最匹配的（用于消除方向约定歧义）。"""
    scores: dict[str, float] = {}
    best_name, best_M, best_s = "", None, -np.inf
    for name, M in candidates.items():
        if M is None:
            continue
        s = _score_matrix(template_w, struct_w, M, work_canvas)
        scores[name] = round(s, 6)
        if s > best_s:
            best_name, best_M, best_s = name, M, s
    assert best_M is not None, "候选矩阵为空"
    return best_name, best_M, best_s, scores


# ---------------------------------------------------------------- ECC

def _ecc(template: np.ndarray, img: np.ndarray, init: np.ndarray,
         motion: int, iters: int, eps: float) -> tuple[float, np.ndarray]:
    """单次 ECC。返回值语义与 findTransformECC 一致：模板坐标 -> 输入坐标。"""
    M3 = as_affine(init)
    wm = (M3 if motion == cv2.MOTION_HOMOGRAPHY else M3[:2]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, wm = cv2.findTransformECC(
            template.astype(np.float32), img.astype(np.float32),
            wm.copy(), motion, criteria, None, 5,
        )
    except cv2.error:
        return -1.0, as_affine(init)
    return float(cc), as_affine(wm)


def ecc_multiscale(template: np.ndarray, warped_struct: np.ndarray,
                   *, levels: tuple[int, ...] = (4, 2, 1),
                   motion: int = cv2.MOTION_AFFINE,
                   iters: int = 120, eps: float = 1e-8) -> tuple[float, np.ndarray, int]:
    """多尺度 ECC（在给定的工作分辨率下）。返回 (cc, W, levels_used)。

    W 为「模板坐标 -> 输入坐标」的 dst->src 映射（同 findTransformECC 语义）。
    """
    W = np.eye(3, dtype=np.float64)
    cc_best, used = -1.0, 0

    for f in levels:
        if f == 1:
            t, s = template, warped_struct
        else:
            t = cv2.resize(template, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA)
            s = cv2.resize(warped_struct, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA)
        h = min(t.shape[0], s.shape[0])
        w = min(t.shape[1], s.shape[1])
        if h < 32 or w < 32:
            continue
        t, s = t[:h, :w], s[:h, :w]

        S = scale_matrix(1.0 / f)
        S_inv = np.linalg.inv(S)
        init = S @ W @ S_inv

        cc, W_level = _ecc(t, s, init, motion, iters, eps)
        if cc < 0:
            continue
        W = S_inv @ W_level @ S
        cc_best, used = cc, f

    return cc_best, W, used


def _residual_skew(warped_gray: np.ndarray, *, long_side: int = 1200) -> tuple[float, float, int]:
    """在**对齐后的输出图**上测残余倾斜角（度）→ (角, 离散度, 线数)。

    这是目前唯一实测能把「对齐正确」和「对齐错了」分开的**分辨率无关**信号。
    实测（A4 30 页 vs 两张支票照片）：

        A4 合格页（27 页）      中位 0.000°   最大 0.239°
        支票·有四角（坐标正确）            -0.221°
        支票·去斜ON（偏 10px）             -1.107°
        支票·去斜OFF（偏 7px）             +0.990°

    为什么不用结构相关度做这个判断：那个分数**随输入分辨率单调下降**
    （同一批 A4 样本 300dpi 0.97 / 72dpi 0.29，残差却都是 0.5px 量级），
    而且偏了 10px 的照片得分（0.45）反而**高于**配准正确的 72dpi 页（0.27）——
    它无法区分这两类情况。倾斜角没有这个问题：内容歪了就是歪了。

    在降采样后的图上算：estimate_skew 靠长直线，降采样不影响直线方向，但快很多。
    """
    from .rectify import binarize, estimate_skew, to_gray

    g = to_gray(warped_gray)
    h, w = g.shape[:2]
    k = min(1.0, long_side / float(max(h, w)))
    if k < 1.0:
        g = cv2.resize(g, (max(64, int(round(w * k))), max(64, int(round(h * k)))),
                       interpolation=cv2.INTER_AREA)
    est = estimate_skew(binarize(g))
    return (float(est.angle_deg), float(est.dispersion_deg), int(est.n_h + est.n_v))


# ---------------------------------------------------------------- 结果

@dataclass
class AlignResult:
    matrix: np.ndarray                  # 3x3，样本坐标 -> 画布坐标（正向，全分辨率）
    cc: float = -1.0
    score: float = -1.0
    coarse_shift: tuple[float, float] = (0.0, 0.0)
    direction: dict[str, float] = field(default_factory=dict)
    levels_used: int = 0
    work_scale: float = 1.0
    locate: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    residual_skew_deg: float = 0.0      # 对齐后输出图上的残余倾斜（°），相对模板基准
    residual_skew_dispersion: float = 0.0
    residual_skew_lines: int = 0
    ok: bool = True
    reason: str = ""


@dataclass
class AlignConfig:
    # ECC 只需把误差压进直线精修的捕获范围（线间距的 ~1/4），
    # 所以工作分辨率可以低、迭代可以少——精度由直线精修负责。
    ecc_max_side: int = 900
    ecc_levels: tuple[int, ...] = (4, 2, 1)
    ecc_iters: int = 80
    ecc_eps: float = 1e-8
    struct_blur: float = 3.0
    ink_bias: int = 0                  # 深墨偏置：拍照件/有底纹的票据需要
    texture_floor: float = 0.0         # 灰度线响应的底噪扣除，抑制浅色底纹
    # 多尺度内容定位：**默认关闭**。
    # 目标场景是"单据只占画面一部分"的拍照件；但当前实现不可用——
    # 合成自检显示尺度判别不准（真值 f=0.70 被判成 0.495），
    # 一旦启用会把 A4 一类"内容铺满画面"的输入也带偏（实测 10 个样本全部拒收）。
    # 保留代码供后续改进：需要用 Chamfer 之外更稳的尺度判别（如对数极坐标相位相关，
    # 或先由样本边界估尺度再定角度）。启用前必须重跑 A4 数据集回归。
    coarse_locate: bool = False
    coarse_min_score: float = 0.15     # 粗定位相关度下限，低于则退回整图铺满初值
    coarse_angle_range: float = 6.0     # 粗定位的旋转搜索范围（度）
    phase_enabled: bool = True
    polish_translation: bool = True
    grid_refine: bool = True           # 直线锚点精修（低分辨率下尤其关键）
    grid_iters: int = 3
    grid_max_resid: float = 4.0        # 超过即判为错配，不采用
    grid_noise_floor: float = 0.4      # 已达此残差就停止迭代
    homography_refine: bool = True     # 交点单应精修（纳入透视分量）
    # 迭代次数偏多是刻意的：均值几乎不变，但极值显著下降
    # （实测 3 次 -> 最大 14.1px，5 次 -> 6.9px）。异常页的代价远高于多算两轮。
    homo_iters: int = 5
    homo_max_resid: float = 3.0
    homo_noise_floor: float = 0.25
    homo_max_shift_px: float = 25.0    # 单应只该做小修正，位移超限即判为错配
    min_axis_obs: int = 12             # 某一轴观测不足即判为低置信

    max_scale_delta: float = 0.15
    max_rotate_deg: float = 8.0
    max_translate_ratio: float = 0.15  # 相对画布尺寸的平移上限
    # 两道门禁，职责不同，不要混：
    #
    # (a) 结构相关度 min_score —— 只挡**明显错配**（把模板对到完全不相干的地方）。
    #     它的绝对值随输入分辨率下降，所以阈值要低（0.25），不能当质量判据。
    #     实测 A4 30 页：300dpi 0.97 / 72dpi 0.29，而残差都是 0.5px 量级。
    min_score: float = 0.25
    #
    # (b) 对齐后**残余倾斜** max_residual_skew_deg —— 真正的质量判据。
    #     它是分辨率无关的物理量，且实测能把两类情况分开：
    #     A4 合格页最大 0.239°；坐标偏了 7~10px 的支票照片 0.99~1.11°。
    #     超过就判"低置信"（输出仍生成，供人工抽检），不是拒收——
    #     它可能只是模板本身有轻微倾斜基准。
    max_residual_skew_deg: float = 0.35


def _max_corner_shift(a: np.ndarray, b: np.ndarray, canvas: tuple[int, int]) -> float:
    """两个映射在画布四角上的最大位移（px）。

    用来判断"这一步修正到底动了多大"——精细化环节的修正量应该是小的。
    """
    cw, ch = canvas
    pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64)
    pa = apply3(as_affine(a), pts)
    pb = apply3(as_affine(b), pts)
    return float(np.max(np.linalg.norm(pa - pb, axis=1)))


def apply3(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))])
    out = (as_affine(M) @ h.T).T
    return out[:, :2] / out[:, 2:3]


def _plausible(M: np.ndarray, base: np.ndarray, canvas: tuple[int, int],
               cfg: AlignConfig) -> tuple[bool, str]:
    if M is None or not np.isfinite(M).all():
        return False, "矩阵非有限值"

    M = as_affine(M)
    if abs(M[2, 2]) < 1e-12:
        return False, "矩阵退化"
    M = M / M[2, 2]

    rel = M @ np.linalg.inv(as_affine(base))
    if not np.isfinite(rel).all():
        return False, "相对变换非有限值"
    if abs(rel[2, 2]) < 1e-12:
        return False, "相对变换退化"
    rel = rel / rel[2, 2]

    sx = float(np.hypot(rel[0, 0], rel[1, 0]))
    sy = float(np.hypot(rel[0, 1], rel[1, 1]))
    if abs(sx - 1.0) > cfg.max_scale_delta or abs(sy - 1.0) > cfg.max_scale_delta:
        return False, f"缩放越界 sx={sx:.3f} sy={sy:.3f}"

    rot = float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))
    if abs(rot) > cfg.max_rotate_deg:
        return False, f"旋转越界 {rot:+.2f}deg"

    # 透视合理性：不看透视项的绝对大小，看**规范化分母**在画布四角是否远离 0。
    #
    # 踩过的坑：最初限制 |h20|*cw + |h21|*ch ≤ 0.03。但手机拍的单据本身就有明显透视梯形
    # （实测一张支票两侧高度差 15%，对应这个量约 0.15），于是**合法结果被门禁拒收**。
    # 透视项的大小不是异常信号，投影分母趋零才是——那才是真正的奇点。
    cw, ch = canvas
    pts = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    dens = [abs(rel[2, 0] * x + rel[2, 1] * y + rel[2, 2]) for x, y in pts]
    dmin, dmax = min(dens), max(dens)
    dmean = max(sum(dens) / len(dens), 1e-9)
    if dmin < 0.5 or (dmax - dmin) / dmean > 0.45:
        return False, f"透视畸变过大或接近奇点 (分母 {dmin:.3f}~{dmax:.3f})"

    tx, ty = float(rel[0, 2]), float(rel[1, 2])
    if abs(tx) > cw * cfg.max_translate_ratio or abs(ty) > ch * cfg.max_translate_ratio:
        return False, f"平移越界 dx={tx:+.1f} dy={ty:+.1f}"
    return True, ""


def align_to_template(template_struct: np.ndarray, page_bgr: np.ndarray,
                      canvas: tuple[int, int], cfg: AlignConfig | None = None,
                      *, page_gray: np.ndarray | None = None,
                      template_lines=None,
                      template_skew_deg: float | None = None) -> AlignResult:
    """把一页对齐到模板坐标系，返回 样本坐标 -> 画布坐标 的 3x3 矩阵。"""
    cfg = cfg or AlignConfig()
    cw, ch = canvas

    k = min(1.0, cfg.ecc_max_side / float(max(cw, ch)))
    work_canvas = (max(64, int(round(cw * k))), max(64, int(round(ch * k))))
    K = scale_matrix(k)
    locate_info: dict = {}

    # 工作坐标 = 全分辨率坐标 × k。因为域与值域**同时**被缩放 k，
    # 共轭变换是 K·M·K⁻¹（线性部分不变，平移项乘 k）——用 translate 的例子验算过。
    def to_work(M: np.ndarray) -> np.ndarray:
        return K @ as_affine(M) @ np.linalg.inv(K)

    def to_full(M: np.ndarray) -> np.ndarray:
        return np.linalg.inv(K) @ as_affine(M) @ K

    template_w = _resize_to(template_struct, work_canvas)

    if page_gray is None:
        page_gray = normalize_illumination(to_gray(page_bgr))
    struct_page = structure_image(page_gray, blur=cfg.struct_blur,
                                  dark_bias=cfg.ink_bias)
    pw = (max(32, int(round(page_gray.shape[1] * k))),
          max(32, int(round(page_gray.shape[0] * k))))
    struct_w = _resize_to(struct_page, pw)

    # 粗定位：整图铺满只是"没有更好信息时"的兜底初值。
    # 样本里有背景/边距时（拍照件），内容并不铺满画面，必须先把内容找出来定位。
    base_full = initial_matrix(page_gray.shape[1::-1], canvas)
    if cfg.coarse_locate:
        cl = coarse_locate(template_struct, struct_page, canvas,
                           angle_range=cfg.coarse_angle_range)
        if cl is not None:
            locate_info = {
                "score": round(cl.score, 4), "scale": round(cl.scale, 4),
                "angle_deg": round(cl.angle_deg, 3), "n_scales": cl.n_scales,
                "mean_dist": round(cl.mean_dist, 3),
                "used": cl.score >= cfg.coarse_min_score,
            }
            if locate_info["used"]:
                base_full = cl.matrix
    base_w = to_work(base_full)

    # --- 1. 相位相关：吃掉大位移 -------------------------------------------------
    coarse = (0.0, 0.0)
    work = base_w
    if cfg.phase_enabled:
        cur = warp_forward(struct_w, base_w, work_canvas, border=0)
        (dx, dy), _resp = cv2.phaseCorrelate(
            template_w.astype(np.float32), cur.astype(np.float32))
        name, work, _s, _sc = _best_of(
            template_w, struct_w,
            {"shift+": translation_matrix(dx, dy) @ base_w,
             "shift-": translation_matrix(-dx, -dy) @ base_w,
             "none": base_w},
            work_canvas,
        )
        coarse = (float(dx), float(dy)) if name == "shift+" else (float(-dx), float(-dy))

    # --- 2. 多尺度 ECC 仿射精配准 -----------------------------------------------
    cur = warp_forward(struct_w, work, work_canvas, border=0)
    cc, W, used = ecc_multiscale(
        template_w, cur, levels=cfg.ecc_levels,
        motion=cv2.MOTION_AFFINE, iters=cfg.ecc_iters, eps=cfg.ecc_eps,
    )

    # --- 3. 方向闭环校验（W⁻¹·P 与 W·P 取优）----------------------------------
    name, M_w, score, dir_scores = _best_of(
        template_w, struct_w,
        {"W_inv@P": np.linalg.inv(W) @ work,
         "W@P": W @ work,
         "no_ecc": work},
        work_canvas,
    )

    # --- 4. 工作分辨率下的平移抛光 -----------------------------------------------
    if cfg.polish_translation and name != "no_ecc":
        cur = warp_forward(struct_w, M_w, work_canvas, border=0)
        cc2, Wt, _ = ecc_multiscale(
            template_w, cur, levels=(1,),
            motion=cv2.MOTION_TRANSLATION, iters=80, eps=1e-9,
        )
        if cc2 > 0:
            n2, M2, s2, d2 = _best_of(
                template_w, struct_w,
                {"t_inv@P": np.linalg.inv(Wt) @ M_w, "t@P": Wt @ M_w, "keep": M_w},
                work_canvas,
            )
            M_w, score, name, dir_scores = M2, s2, n2, d2
            cc = max(cc, cc2)

    # --- 5. 直线锚点精修：分带投影 + 拟合仿射，把残余旋转一并解出来 -----------------
    #
    # 决策依据用**线拟合残差**而不是结构相关度：残差直接衡量「线对得准不准」，
    # 而相关度在修正量很小时无法分辨方向（实测会把 C·P 误判为优于 C⁻¹·P）。
    # 方向 C⁻¹·P 已实测确认（7 个样本全部优于反向），所以按理论取，并做闭环验证：
    # 修正后重测残差，没变好就回退。
    grid_info: dict = {}
    M_full_cur = to_full(M_w)
    if cfg.grid_refine and template_lines is not None and len(template_lines) >= 4:
        from .grid import refine_by_grid

        for it in range(max(1, cfg.grid_iters)):
            warped_full = warp_forward(page_gray, M_full_cur, canvas, border=255)
            gr = refine_by_grid(template_lines, warped_full, dark_bias=cfg.ink_bias)
            resid = max(gr.resid_x, gr.resid_y)
            info = {
                "iter": it + 1, "usable": bool(gr.usable),
                "n_x": gr.n_x, "n_y": gr.n_y,
                "resid_x": gr.resid_x, "resid_y": gr.resid_y,
                "shear": list(gr.shear),
                "scale": [round(float(gr.matrix[0, 0]), 6), round(float(gr.matrix[1, 1]), 6)],
                "accepted": False,
            }
            if not gr.usable:
                grid_info = {**info, "stop": "有效观测不足"}
                break
            if resid > cfg.grid_max_resid:
                grid_info = {**info, "stop": f"残差 {resid:.2f}px 过大，判为错配，不采用"}
                break
            # 通过门禁就应用。**不能**用「两次迭代残差谁更小」决定是否采用：
            # 残差已到噪声地板时第二次必然"未改善"，会把已经算对的修正误判为回退，
            # 而且此时 M_full_cur 已被修改、M_w 却没同步，导致修正被静默丢弃。
            M_full_cur = np.linalg.inv(gr.matrix) @ M_full_cur
            grid_info = {**info, "accepted": True}
            if resid <= cfg.grid_noise_floor:
                grid_info["stop"] = "已达噪声地板，停止迭代"
                break

        M_w = to_work(M_full_cur)
        cur = warp_forward(struct_w, M_w, work_canvas, border=0)
        score = struct_score(template_w, cur)
        dir_scores["grid"] = round(score, 6)
        name = "grid" if grid_info.get("accepted") else name

    # --- 6. 交点单应精修：纳入仿射表达不了的透视分量 -------------------------------
    #
    # 前置条件：仿射阶段必须已通过。单应是"最后一点误差"的精修，它预设前序环节已把
    # 误差压到几 px 量级；如果仿射阶段因观测不足/残差过大被跳过，剩余误差可能几十 px，
    # 此时单应会用"错误匹配但in-sample残差很低"的直线对拟合出一个全局错误的变换——
    # 实测一页 72dpi 样本因此从正常水平恶化到 19.5px，而它的残差指标看起来还很"好"。
    grid_ok = bool(grid_info.get("accepted")) or not cfg.grid_refine
    if cfg.homography_refine and template_lines is not None and grid_ok:
        from .grid import refine_by_intersections

        homo_info: dict = {}
        for it in range(max(1, cfg.homo_iters)):
            before = M_full_cur
            warped_full = warp_forward(page_gray, before, canvas, border=255)
            ir = refine_by_intersections(template_lines, warped_full,
                                         noise_floor=cfg.texture_floor)
            resid = float(ir.resid) if np.isfinite(ir.resid) else float("inf")
            info = {
                "iter": it + 1, "n_points": ir.n_points, "n_inliers": ir.n_inliers,
                "resid": resid if np.isfinite(resid) else None, "accepted": False,
            }
            if not ir.usable:
                homo_info = {**info, "stop": "交点数不足"}
                break
            if resid > cfg.homo_max_resid:
                homo_info = {**info, "stop": f"残差 {resid:.2f}px 过大，判为错配，不采用"}
                break
            M_full_cur = np.linalg.inv(ir.matrix) @ before
            homo_info = {**info, "accepted": True}
            if resid <= cfg.homo_noise_floor:
                homo_info["stop"] = "已达噪声地板，停止迭代"
                break

        grid_info["homo"] = homo_info
        if homo_info.get("accepted"):
            M_w = to_work(M_full_cur)
            cur = warp_forward(struct_w, M_w, work_canvas, border=0)
            score = struct_score(template_w, cur)
            dir_scores["homo"] = round(score, 6)
            name = "homo"

    M_full = to_full(M_w)

    # 残余倾斜：在**对齐后的输出**上测。内容歪了，指定坐标裁出来的块必然整体偏移，
    # 且偏移量随离旋转中心的距离线性放大（2° 在 A4 右边缘相当于约 100px）。
    warped_gray = warp_forward(page_gray, M_full, canvas, border=255)
    skew, skew_disp, skew_lines = _residual_skew(warped_gray)
    rel_skew = skew - float(template_skew_deg or 0.0)

    ok, reason = _plausible(M_full, base_full, canvas, cfg)
    if score < cfg.min_score:
        ok, reason = False, f"结构相关度过低 {score:.3f} < {cfg.min_score}"

    return AlignResult(
        matrix=M_full, cc=cc, score=score, coarse_shift=coarse,
        direction={**dir_scores, "winner": name}, levels_used=used,
        work_scale=k, locate=locate_info, grid=grid_info, ok=ok, reason=reason,
        residual_skew_deg=rel_skew, residual_skew_dispersion=skew_disp,
        residual_skew_lines=skew_lines,
    )