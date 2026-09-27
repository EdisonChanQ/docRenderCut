"""引擎编排：把 docnorm 的算法能力包成 Web 后端可直接调用的步骤。

- `build_template`：样张 + 参数 -> 模板标准模板（模板图 + 模板 JSON + 空坐标清单）
- `start_job`：分类 + 一批生产文件 -> 逐页识别模板 -> 按各自模板渲染 + 裁剪

全部走已验证的 docnorm 引擎，这里只做「参数校验、纸张层选择、警告收集、进度上报」。

数据模型两级：
    分类 Category（纯容器） -> 模板 Template（纸面参数 + 坐标，执行单元）
上传时选分类，系统逐页 classify_page 归到分类下某个模板；识别不出的页进 rejected。
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from docnorm.align import AlignConfig
from docnorm.classify import ClassifyConfig, classify_page
from docnorm.crops import CropItem, CropSpec
from docnorm.loader import (
    IMAGE_EXT,
    PDF_EXT,
    count_pages,
    expand_files,
    iter_batch_stream,
    load_image,
)
from docnorm.pipeline import Engine
from docnorm.render import write_page_pdf
from docnorm.rectify import (
    PaperQuad,
    _validate_paper_quad,
    deskew,
    detect_paper_quad,
    paper_warp,
)
from docnorm.template import Template

from . import store

# ---------------------------------------------------------------- 样张读取


def load_sample(path: Path, dpi: int) -> tuple[np.ndarray, tuple[float, float] | None, str]:
    """读样张的第一页。

    PDF 只栅格化第一页：模板用一张基准页建立就够，
    没必要为了一张图把整份 PDF 都渲染出来（大文件上这个差别很明显）。
    矢量 PDF 按模板 DPI 精确栅格化，不引入多余的重采样。
    返回 (BGR 图, 页面物理尺寸 mm 或 None, 类型)
    """
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            if doc.page_count < 1:
                raise ValueError("PDF 没有页面")
            page = doc[0]
            rect = page.rect
            pm = page.get_pixmap(dpi=int(dpi))
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
            import cv2

            if pm.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pm.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            mm = (rect.width / 72.0 * 25.4, rect.height / 72.0 * 25.4)
            return np.ascontiguousarray(arr), mm, "pdf"
        finally:
            doc.close()

    if ext in IMAGE_EXT:
        img, _ = load_image(path)
        return img, None, "image"

    raise ValueError(f"不支持的样张类型: {path.suffix}")


# ---------------------------------------------------------------- 样张探测

# 常见纸张（毫米），用于"图里没有 DPI 元数据"时按宽高比反推
_PAPERS = (
    ("A4", 210.0, 297.0),
    ("A5", 148.0, 210.0),
    ("A3", 297.0, 420.0),
    ("Letter", 215.9, 279.4),
    ("Legal", 215.9, 355.6),
)

# 物理尺寸超过这个值就认为 DPI 元数据不可信（A3 是 420mm，留些余量）
_MAX_MM = 600.0


def _read_image_dpi(path: Path) -> tuple[float, float] | None:
    """读图片内嵌 DPI（EXIF / PNG 元数据）。不可信/缺失返回 None。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if not dpi:
                return None
            dx = float(dpi[0]) if not isinstance(dpi, (int, float)) else float(dpi)
            dy = float(dpi[1]) if isinstance(dpi, (tuple, list)) and len(dpi) > 1 else dx
            if 10 < dx < 2400 and 10 < dy < 2400:
                return dx, dy
    except Exception:  # noqa: BLE001
        pass
    return None


def _guess_paper_mm(aspect: float, tol: float = 0.02) -> tuple[float, float, str] | None:
    """按宽高比猜纸张（横竖都试）。只用于给建议值，用户可改。"""
    for name, w, h in _PAPERS:
        for ww, hh in ((w, h), (h, w)):
            ref = ww / hh
            if abs(ref - aspect) / ref <= tol:
                return ww, hh, name
    return None


def probe_sample(path: Path, dpi_hint: int = 200) -> dict:
    """**只分析样张、不落盘**：返回检测到的尺寸/DPI 与建议的画布参数。

    为什么建模板要分两步：上传即落盘的话，用户每改一次尺寸就要重建一次模板
    （建模板要跑结构提取 + 表格线检测，秒级），而且每次尝试都会在数据目录里
    留下一个半成品模板。改成"先探测 -> 用户确认 -> 再落盘"后，
    数据目录里只会出现用户真正认可的模板。
    """
    warnings: list[str] = []
    img, mm, kind = load_sample(path, int(dpi_hint))
    px_w, px_h = int(img.shape[1]), int(img.shape[0])
    aspect = px_w / max(1, px_h)

    n_pages = None
    if kind == "pdf":
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            n_pages = doc.page_count
        finally:
            doc.close()
        if n_pages and n_pages > 1:
            warnings.append(f"样张是 {n_pages} 页，建模板只用第 1 页。")

    detected: int | None = None
    source = "unknown"
    img_dpi: list[float] | None = None
    paper: str | None = None

    if kind == "pdf" and mm:
        # 矢量 PDF 没有"原始 DPI"这个概念——物理尺寸已知，画布由 DPI 推导
        detected = int(dpi_hint)
        source = "pdf-page"
    else:
        d = _read_image_dpi(path)
        if d:
            cand_mm = (px_w / d[0] * 25.4, px_h / d[1] * 25.4)
            if cand_mm[0] <= _MAX_MM and cand_mm[1] <= _MAX_MM:
                img_dpi = [round(d[0], 2), round(d[1], 2)]
                mm = cand_mm
                detected = int(round((d[0] + d[1]) / 2))
                source = "image-meta"
            else:
                warnings.append(
                    f"样张内嵌 DPI 为 {d[0]:.0f}，据此推算物理尺寸 "
                    f"{cand_mm[0]:.0f}×{cand_mm[1]:.0f}mm，明显不合理——"
                    "该元数据不可信，已忽略。")
        if detected is None:
            g = _guess_paper_mm(aspect)
            if g:
                mm = (g[0], g[1])
                detected = int(round(px_w / (g[0] / 25.4)))
                paper = g[2]
                source = "paper-guess"
                warnings.append(
                    f"样张没有可信的 DPI 元数据，按宽高比看像 {g[2]}"
                    f"（{g[0]:.0f}×{g[1]:.0f}mm），据此推算约 {detected}dpi。请核对。")
            else:
                warnings.append(
                    "样张没有 DPI 元数据，也认不出标准纸张尺寸；"
                    "物理尺寸未知，DPI 与画布请手工确认。")

    suggest_dpi = int(detected or dpi_hint)
    if mm:
        sug_w = max(1, int(round(mm[0] / 25.4 * suggest_dpi)))
        sug_h = max(1, int(round(mm[1] / 25.4 * suggest_dpi)))
    else:
        sug_w, sug_h = px_w, px_h

    if abs(sug_w / max(1, sug_h) - aspect) / max(aspect, 1e-6) > 0.02:
        warnings.append("建议画布的宽高比与样张像素宽高比不一致，内容会被非等比拉伸。")

    return {
        "filename": path.name,
        "sample_kind": kind,
        "sample_size": [px_w, px_h],
        "sample_mm": [round(v, 2) for v in mm] if mm else None,
        "image_dpi": img_dpi,
        "detected_dpi": detected,
        "dpi_source": source,
        "paper": paper,
        "aspect": round(aspect, 4),
        "n_pages": n_pages,
        "suggest": {"dpi": suggest_dpi, "width": sug_w, "height": sug_h},
        "warnings": warnings,
    }


# ---------------------------------------------------------------- 建模板


def build_template(cid: str, tid: str, sample: Path, *, dpi: int,
                   canvas: tuple[int, int], paper_mode: str = "off",
                   corners: list[float] | None = None,
                   ink_bias: int = 0, deskew_sample: bool = True) -> dict:
    """用样张建立该模板的标准模板。返回 {warnings, info}。

    **deskew_sample 默认开启**，这是刻意的：模板是所有页的**基准坐标系**，
    基准自己歪着的话，每一张对齐过去的输出都会带着同样的倾斜，
    按坐标裁出来的块也是歪的（实测一张样例歪 0.79°，模板就歪 0.93°，
    在 1200px 宽上两端垂直漂移约 19px）。

    而且这个问题**质量门禁抓不到**：门禁用的是「输入残余倾斜 − 模板基准倾斜」，
    模板本来就歪时两者相减接近 0，一路显示正常。所以只能在建模板这一步归一化。
    """
    warnings: list[str] = []
    img, mm, kind = load_sample(sample, dpi)
    src = img

    quad = None
    if paper_mode == "auto":
        quad = detect_paper_quad(img)
        if quad is None or not quad.ok:
            warnings.append(
                "纸张自动检测失败（拍照件的阴影、桌面纹理、票据防伪底纹都会干扰分割），"
                "已按「整页」处理。若样张带背景，请改用「手工四角」。")
            quad = None
    elif paper_mode == "corners":
        if not corners or len(corners) != 8:
            raise ValueError("手工四角需要 8 个数：x1,y1,x2,y2,x3,y3,x4,y4")
        quad = PaperQuad.from_corners(corners)
        _validate_paper_quad(quad, (img.shape[1], img.shape[0]), strict=False)
        if not quad.ok:
            raise ValueError(f"四角无效：{quad.reason}")

    if quad is not None:
        try:
            src, _H = paper_warp(img, quad, canvas)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"纸张矫正失败：{exc}") from exc
        if abs((canvas[0] / canvas[1]) - quad.aspect) / max(quad.aspect, 1e-6) > 0.03:
            warnings.append(
                f"画布宽高比 {canvas[0] / canvas[1]:.3f} 与实测纸张宽高比 "
                f"{quad.aspect:.3f} 不符，纸张会被非等比拉伸。")

    # 纠偏：把样例摆正，让模板基准是水平的（否则整条链路都跟着歪）
    deskew_info: dict = {"applied": False, "residual_deg": None, "n_lines": None}
    if deskew_sample:
        src, _M_ds, sk = deskew(src, dark_bias=ink_bias)
        n_lines = int(sk.n_h + sk.n_v)
        deskew_info = {"applied": True, "residual_deg": round(float(sk.angle_deg), 3),
                       "n_lines": n_lines}
        if abs(sk.angle_deg) > 0.3:
            warnings.append(
                f"样例纠偏后残余倾斜仍有 {sk.angle_deg:+.3f}°（只检测到 {n_lines} 条表格线）——"
                "样例可能太糊或表格线太少，纠偏不可靠，建议换一张更清晰的样例。")

    # 宽高比核对：画布与内容宽高比不一致时内容会被拉伸
    ar_src = src.shape[1] / max(1, src.shape[0])
    ar_canvas = canvas[0] / max(1, canvas[1])
    if abs(ar_src - ar_canvas) / max(ar_canvas, 1e-6) > 0.02:
        warnings.append(
            f"内容宽高比 {ar_src:.3f} 与画布宽高比 {ar_canvas:.3f} 相差 "
            f"{abs(ar_src - ar_canvas) / ar_canvas:.1%}，内容会被非等比拉伸"
            "（表格线可能变粗/变细）。建议画布尺寸取 宽高比 ≈ "
            f"{ar_src:.3f}（例如按 {int(round(canvas[1] * ar_src))}x{canvas[1]} 或 "
            f"{canvas[0]}x{int(round(canvas[0] / ar_src))}）。")

    tpl = Template.from_image(store.load_template(cid, tid)["name"], src, paper="custom",
                              dpi=dpi, canvas=canvas, dark_bias=ink_bias)
    tpl.meta.update({
        "category_id": cid,
        "template_id": tid,
        "sample": sample.name,
        "sample_kind": kind,
        "paper_mode": paper_mode,
        "corners": corners,
        "ink_bias": ink_bias,
        "deskew": deskew_info,
        "sample_mm": [round(v, 2) for v in mm] if mm else None,
        "sample_size": [int(img.shape[1]), int(img.shape[0])],
    })
    tpl.save(store.tpl_dir(cid, tid) / "template")

    # 坐标清单：没有就建一个空的（含画布与 DPI，供前端标注器直接用）
    from datetime import datetime

    crops_path = store.tpl_dir(cid, tid) / "crops.json"
    if not crops_path.exists():
        CropSpec(name=store.load_template(cid, tid)["name"], canvas=list(canvas), dpi=dpi,
                 meta={"category_id": cid, "template_id": tid,
                       "created_at": datetime.now().isoformat(timespec="seconds")}
                 ).save(crops_path)

    lines = tpl.lines()
    png = store.tpl_dir(cid, tid) / "template" / "template.png"
    version = int(png.stat().st_mtime)
    return {
        "warnings": warnings,
        "info": {
            "canvas": list(canvas),
            "dpi": dpi,
            "sample_size": [int(img.shape[1]), int(img.shape[0])],
            "sample_kind": kind,
            "sample_mm": [round(v, 2) for v in mm] if mm else None,
            "paper_applied": quad is not None,
            "deskew": deskew_info,
            "n_lines_x": len(lines.xs),
            "n_lines_y": len(lines.ys),
            "template_url": store.to_url(png),
            # 界面预览用降采样版：全分辨率模板（拍照件）可能有几 MB
            "preview_url": f"/api/categories/{cid}/templates/{tid}/template.png?max=1800&v={version}",
            "bytes": png.stat().st_size,
        },
    }


# ---------------------------------------------------------------- 批量任务


def _paper_arg(tpl: dict) -> str:
    mode = (tpl.get("paper_mode") or "off").lower()
    if mode == "corners":
        cs = tpl.get("corners") or []
        if len(cs) == 8:
            return ",".join(str(float(v)) for v in cs)
        return "off"
    if mode == "auto":
        return "auto"
    return "off"


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


@dataclass
class JobConfig:
    """任务执行的资源预算：**要提速，但不能把线上机器的资源吃掉**。

    每个字段都可用环境变量覆盖（见 job_config）：

    - `workers`：页面级并行度（同时处理几页）。
      实测单页 ≈ 2.7 个核（20 核机器、OpenCV 未限线程时），
      所以 workers=2 大致落在 25%~30% CPU，留足余量给同机的其他服务。
    - `cv_threads`：OpenCV 内部线程上限。**必须显式设**：
      OpenCV 默认按 CPU 核数开线程（本机实测 20），
      worker 数 × 核数会变成线程风暴，反而拖慢并抢爆 CPU。
    - `max_inflight`：同时在飞的页数上限（含已提交未开跑的）。
      内存 ≈ `max_inflight × 单页解码大小`（300dpi A4 ≈ 25 MB）+ 基础开销。
      这是"内存不随批次页数增长"的关键——**不能**一次性把整批页 submit 给线程池。
    - `render_dpi`：上传件栅格化 DPI（与模板 DPI 一致最省，300 是精度优先）。
    - `max_jobs`：同时运行的任务数。1 = 后来的任务排队，
      否则连点两次上传会各开一套 worker，CPU 与内存同时翻倍。
    """
    workers: int = 2
    cv_threads: int = 4
    max_inflight: int = 3
    render_dpi: int = 300
    max_jobs: int = 1
    # 原始分页 PDF（standardized/pdf/）：original=忠实原样（默认）
    # gray=强制灰度（去掉无信息色偏，体积约省 40%）/ off=不产出
    page_pdf: str = "original"

    def as_dict(self) -> dict:
        return {"workers": self.workers, "cv_threads": self.cv_threads,
                "max_inflight": self.max_inflight, "render_dpi": self.render_dpi,
                "max_jobs": self.max_jobs, "page_pdf": self.page_pdf}


def job_config() -> JobConfig:
    """从环境变量解析资源预算；未设置时按「CPU 预算 50%」自动推算并行度。"""
    cores = os.cpu_count() or 4
    cpu_budget = min(1.0, max(0.05, _env_float("DOCRENDERCUT_CPU_BUDGET", 0.5)))
    cv_threads = max(1, min(64, _env_int("DOCRENDERCUT_CV_THREADS", 4)))
    # 单 worker 实际吃多少核：实测（cv_threads=4、20 核机）2 个 worker 合计 2.9 核
    # ≈ 1.45 核/worker。这里取 2.0 做保守估计，避免把机器吃满。
    # 注意不能用 cv_threads 当分母：那是"上限"不是"实测值"，
    # 会算出过度保守的并行度（2 核/worker 的公式在 20 核机上给 2，实测只用 14% CPU）。
    per_worker = max(0.5, _env_float("DOCRENDERCUT_CORES_PER_WORKER", 2.0))
    budget_cores = max(1.0, cores * cpu_budget)
    auto_workers = max(1, min(int(budget_cores // per_worker), 4))
    workers = max(1, min(16, _env_int("DOCRENDERCUT_WORKERS", auto_workers)))
    return JobConfig(
        workers=workers,
        cv_threads=cv_threads,
        max_inflight=max(workers, _env_int("DOCRENDERCUT_MAX_INFLIGHT", workers + 1)),
        render_dpi=max(50, min(1200, _env_int("DOCRENDERCUT_RENDER_DPI", 300))),
        max_jobs=max(1, min(8, _env_int("DOCRENDERCUT_MAX_JOBS", 1))),
        page_pdf=_page_pdf_mode(),
    )


def _page_pdf_mode() -> str:
    """原始分页 PDF 的产出方式：original（默认）/ gray / off。"""
    v = (os.environ.get("DOCRENDERCUT_PAGE_PDF") or "original").strip().lower()
    return v if v in ("original", "gray", "off") else "original"


# 任务级并发门：同一时刻最多跑 max_jobs 个任务，其余在"排队中"等待。
_gate_cond = threading.Condition()
_gate_running = 0
_gate_size: int | None = None


def _gate_acquire(max_jobs: int) -> None:
    global _gate_running, _gate_size
    with _gate_cond:
        if _gate_size is None:
            _gate_size = max(1, max_jobs)
        while _gate_running >= _gate_size:
            _gate_cond.wait()
        _gate_running += 1


def _gate_release() -> None:
    global _gate_running
    with _gate_cond:
        _gate_running = max(0, _gate_running - 1)
        _gate_cond.notify()


def start_job(cid: str, files: list[tuple[str, Path]], jid: str | None = None) -> dict:
    """登记任务并起后台线程。返回初始 job 记录。

    任务按**分类**发起：后台逐页识别每个页面属于分类下哪个模板，
    识别出的按各自模板渲染裁剪，识别不出的进 rejected（供审计）。
    """
    cat = store.load_category(cid)
    from datetime import datetime

    jid = jid or store.new_job_id()
    job = {
        "id": jid,
        "category_id": cid,
        "category_name": cat.get("name"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
        "message": "排队中",
        "total": 0,
        "done": 0,
        "files": [n for n, _ in files],
        "pages": [],
        "error": None,
        "verify": None,
    }
    store.save_job(jid, job)

    t = threading.Thread(target=_run_job, args=(jid, files), daemon=True)
    t.start()
    return job


def _run_job(jid: str, files: list[tuple[str, Path]]) -> None:
    cfg = job_config()
    job = store.load_job(jid)
    if cfg.max_jobs <= 1:
        job["message"] = "排队中（同一时刻只跑一个任务）"
        store.save_job(jid, job)
    # 任务级排队：否则连点两次上传会各开一套 worker，CPU 与内存同时翻倍
    _gate_acquire(cfg.max_jobs)
    try:
        _run_job_locked(jid, files, cfg)
    finally:
        _gate_release()


def _run_job_locked(jid: str, files: list[tuple[str, Path]],
                    cfg: JobConfig) -> None:
    """逐页识别并按各自模板渲染裁剪。

    并行模型（关键：**提速但资源有界**）：
    - 页面级并行 `cfg.workers` 条线程；
    - 提交是**有界**的：在飞页数不超过 `cfg.max_inflight`，所以解码驻留内存
      与批次总页数无关（这是大批次不 OOM 的关键，不能一次性 submit 全部页）；
    - `cv2.setNumThreads(cfg.cv_threads)` 给 OpenCV 内部线程封顶，
      否则每个 worker 都会按 CPU 核数开线程，叠起来是线程风暴。
    """
    job = store.load_job(jid)
    cid = job["category_id"]
    try:
        # 载入分类下全部模板
        tpl_meta = store.list_templates(cid)
        templates: dict[str, Template] = {}
        for tm in tpl_meta:
            tid = tm["id"]
            tdir = store.tpl_dir(cid, tid) / "template"
            if not (tdir / "template.png").exists():
                continue
            templates[tid] = Template.load(tdir)
        if not templates:
            raise ValueError("该分类下还没有任何可用模板，请先建模板")

        # 预热模板的惰性缓存（表格线 / 基准倾斜）：两者都不便宜，
        # 且 Template._cache 是惰性填充——并行下会重复计算。
        for t in templates.values():
            t.lines()
            t.skew_deg

        paths = [str(p) for _, p in files]
        total = sum(count_pages(p) for p in expand_files(paths))
        if not total:
            raise ValueError("上传的文件里没有可处理的页")

        out_root = store.job_dir(jid)
        job.update({"status": "running",
                    "message": f"处理中（{cfg.workers} 页并行）",
                    "total": total, "done": 0,
                    "runtime": {"workers": cfg.workers,
                                "cv_threads": cfg.cv_threads,
                                "max_inflight": cfg.max_inflight,
                                "render_dpi": cfg.render_dpi}})
        store.save_job(jid, job)

        class_cfg = ClassifyConfig()
        lock = threading.Lock()
        prog = {"done": 0, "last": 0.0}

        def bump() -> None:
            with lock:
                prog["done"] += 1
                now = time.monotonic()
                # 节流写盘：并行下每页都写 job.json 会变成 I/O 热点
                if now - prog["last"] >= 0.4 or prog["done"] >= total:
                    prog["last"] = now
                    job["done"] = prog["done"]
                    job["message"] = f"已处理 {prog['done']}/{total} 页"
                    store.save_job(jid, job)

        def work(idx: int, unit) -> tuple[int, dict]:
            stem = f"{Path(unit.source).stem}_p{unit.page_index + 1:04d}"

            # 1. 识别这一页属于哪个模板
            cr = classify_page(unit.image, templates, class_cfg)

            if not cr.matched or cr.template_id is None:
                # 识别不出 -> rejected，原样保留，供审计
                _drop_stale(stem, out_root)
                write_png(out_root / "rejected" / f"{stem}.png", unit.image, 300)
                # 原始分页 PDF 对**所有页**都产出：消费者要看的是整份原始文档，
                # 含识别不出的页（也便于对照"为什么这页被拒"）
                pdf_path = write_original_pdf(out_root, unit, stem, cfg.page_pdf)
                return idx, {
                    "stem": stem,
                    "source": Path(unit.source).name,
                    "page_index": unit.page_index + 1,
                    "status": "rejected",
                    "reason": f"识别不出：{cr.reason}",
                    "score": cr.score,
                    "scores": cr.scores,
                    "template_id": None,
                    "output_url": store.to_url(out_root / "rejected" / f"{stem}.png"),
                    "pdf_url": store.to_url(pdf_path) if pdf_path else "",
                    "blocks": [],
                }

            # 2. 命中 -> 用该模板渲染 + 裁剪
            tid = cr.template_id
            return idx, _render_with_template(
                out_root, unit, stem, tid, tpl_meta_by_id(tpl_meta, tid),
                templates[tid], cr, page_pdf=cfg.page_pdf)

        # OpenCV 内部线程封顶（全局生效，含 Engine 里的 warp/ECC）
        cv2.setNumThreads(cfg.cv_threads)
        try:
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                inflight: set = set()
                for idx, unit in enumerate(
                        iter_batch_stream(paths, render_dpi=cfg.render_dpi)):
                    # 有界提交：在飞到顶就等一个完成，内存因此与总页数无关
                    while len(inflight) >= cfg.max_inflight:
                        done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                        for f in done_set:
                            i, rec = f.result()
                            results[i] = rec
                            bump()
                    inflight.add(ex.submit(work, idx, unit))
                while inflight:
                    done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for f in done_set:
                        i, rec = f.result()
                        results[i] = rec
                        bump()
        finally:
            cv2.setNumThreads(0)   # 0 = 恢复 OpenCV 默认（按核数）

        pages = [results[i] for i in sorted(results)]
        job["pages"] = pages
        job["status"] = "done"
        job["done"] = len(pages)
        job["message"] = "完成"
        job["report_url"] = store.to_url(out_root / "render-report.html")
        store.save_job(jid, job)

    except Exception as exc:  # noqa: BLE001
        job = store.load_job(jid)
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc()[-2000:]
        job["message"] = "失败"
        store.save_job(jid, job)


def tpl_meta_by_id(meta_list: list[dict], tid: str) -> dict:
    for m in meta_list:
        if m["id"] == tid:
            return m
    return {}


def std_root(out_root: Path) -> Path:
    """标准输出根目录。"""
    return out_root / "standardized"


def std_image_dir(out_root: Path) -> Path:
    """配准后的标准图（**切片任务就吃这里的图**）。"""
    return std_root(out_root) / "image"


def std_pdf_dir(out_root: Path) -> Path:
    """原始分页：每页一个单页 PDF，给消费者查看原始文档用。"""
    return std_root(out_root) / "pdf"


def _page_pdf_dpi(unit) -> int:
    """原始页 PDF 的物理尺寸依据：优先栅格化 DPI，其次图片自带 DPI。"""
    return int(unit.meta.get("render_dpi") or unit.declared_dpi or 300)


def write_original_pdf(out_root: Path, unit, stem: str,
                       mode: str = "original") -> Path | None:
    """写这一页的「原始分页」单页 PDF（配准前的原始页）。mode=off 时不产出。"""
    if mode == "off":
        return None
    pdf_path = std_pdf_dir(out_root) / f"{stem}.pdf"
    write_page_pdf(pdf_path, unit.image, _page_pdf_dpi(unit),
                   force_gray=(mode == "gray"))
    return pdf_path


def _render_with_template(out_root: Path, unit, stem: str, tid: str,
                          tm: dict, tpl: Template, cr, *,
                          page_pdf: str = "original") -> dict:
    """用识别出的模板渲染一页并裁剪。返回前端要的逐页结构。"""
    cfg = AlignConfig()
    cfg.ink_bias = int(tm.get("ink_bias", 0) or 0)

    crops_path = store.tpl_dir(tm.get("category_id"), tid) / "crops.json"
    spec = CropSpec.load(crops_path) if crops_path.exists() else None

    eng = Engine(
        tpl, config=cfg, use_deskew=True,
        render_dpi=int(tm.get("dpi", 300)),
        paper=_paper_arg(tm),
        mono=bool(tm.get("mono", True)),
        ink_dark_bias=int(tm.get("ink_dark_bias", 25) or 25),
        crops=spec,
    )

    # 原始分页（配准前的原始页）先落盘：消费者要看的是"原样"，与是否识别成功无关
    pdf_path = write_original_pdf(out_root, unit, stem, page_pdf)

    rec, img = eng.process_unit(unit)
    blocks = []
    status = rec.status
    reason = rec.reason or ""

    if status in ("ok", "low_confidence") and img is not None:
        # 标准图（切片依据）：合格的进 standardized/image/，低置信的单独放，
        # 目录即状态，前端与验证都按它筛选
        out_dir = std_image_dir(out_root) if status == "ok" else (out_root / "low_confidence")
        _drop_stale(stem, out_root)
        out_path = out_dir / f"{stem}.png"
        write_png(out_path, img, tpl.dpi)
        rec.output = str(out_path)

        if spec is not None:
            from docnorm.crops import crop_one, write_blocks

            blocks_img, _diag = crop_one(img, spec, strict=False)
            written = write_blocks(out_root / "blocks" / stem, blocks_img,
                                   dpi=tpl.dpi)
            blocks = [{"id": bp.stem, "url": store.to_url(bp)}
                      for bp in sorted((out_root / "blocks" / stem).glob("*.png"))]
    else:
        # 配准不过关（错配/门禁）也落到 rejected，原样保留
        _drop_stale(stem, out_root)
        write_png(out_root / "rejected" / f"{stem}.png", unit.image, 300)
        status = "rejected"

    return {
        "stem": stem,
        "source": Path(unit.source).name,
        "page_index": unit.page_index + 1,
        "status": status,
        "reason": reason,
        "score": cr.score,
        "template_id": tid,
        "template_name": tm.get("name", ""),
        "output_url": store.to_url(rec.output) if rec.output else "",
        "pdf_url": store.to_url(pdf_path) if pdf_path else "",
        "blocks": blocks,
    }


def _drop_stale(stem: str, out_root: Path) -> None:
    """删掉同名页在其它目录里的旧产物，保证一页只存在于一处。"""
    for d in (out_root / "standardized",           # 兼容旧任务布局
              std_image_dir(out_root),
              out_root / "low_confidence",
              out_root / "rejected"):
        f = d / f"{stem}.png"
        try:
            if f.exists():
                f.unlink()
        except OSError:
            pass


def write_png(path: Path, img: np.ndarray, dpi: int) -> None:
    from docnorm.render import write_png as _wp

    path.parent.mkdir(parents=True, exist_ok=True)
    _wp(path, img, dpi)


# ---------------------------------------------------------------- 验证


def run_verify(jid: str, *, limit: int = 24, tol_px: float = 3.0) -> dict:
    """对一个任务的输出做块裁剪验证：堆叠对照图 + 位移量化。

    验证按页的 template_id 分组：不同模板坐标清单不同，不能混在一起堆叠。
    """
    from docnorm.verify import verify, write_verification

    job = store.load_job(jid)
    cid = job["category_id"]
    out_root = store.job_dir(jid)

    # 按 template_id 分组收集输出图
    groups: dict[str, list[Path]] = {}
    for page in job.get("pages", []):
        tid = page.get("template_id")
        if not tid:
            continue
        stem = page.get("stem")
        # 标准图现在在 standardized/image/；standardized/ 根路径保留以兼容旧任务
        for sub in ("standardized/image", "standardized", "low_confidence"):
            f = out_root / sub / f"{stem}.png"
            if f.exists():
                groups.setdefault(tid, []).append(f)
                break

    if not groups:
        raise FileNotFoundError("该任务还没有产出标准输出图")

    reports = []
    for tid, imgs in groups.items():
        crops_path = store.tpl_dir(cid, tid) / "crops.json"
        if not crops_path.exists():
            continue
        spec = CropSpec.load(crops_path)
        if not spec.items:
            continue
        imgs = imgs[:limit]
        tpl_png = store.tpl_dir(cid, tid) / "template" / "template.png"
        report, sheets = verify(spec, imgs, template_png=tpl_png,
                                use_safe=True, tol_px=tol_px, max_rows=12)
        vdir = out_root / "verify" / tid
        written = write_verification(vdir, report, sheets)
        report["sheets"] = {k: store.to_url(v) for k, v in written.items()
                            if k != "_report"}
        report["report_url"] = store.to_url(written["_report"])
        report["n_images"] = len(imgs)
        report["template_id"] = tid
        reports.append(report)

    job = store.load_job(jid)
    job["verify"] = {
        "groups": [{"template_id": r["template_id"], "summary": r.get("summary"),
                    "n_images": r["n_images"], "sheets": r["sheets"],
                    "report_url": r["report_url"]} for r in reports],
    }
    store.save_job(jid, job)
    return {"groups": reports}


# ---------------------------------------------------------------- 坐标清单 CRUD


def list_crops(cid: str, tid: str) -> dict:
    p = store.tpl_dir(cid, tid) / "crops.json"
    if not p.exists():
        tpl = store.load_template(cid, tid)
        return CropSpec(name=tpl["name"], canvas=tpl["canvas"], dpi=tpl["dpi"]).to_dict()
    spec = CropSpec.load(p)
    d = spec.to_dict()
    for it in d["items"]:
        it["preview_url"] = f"/api/categories/{cid}/templates/{tid}/crops/{it['id']}/preview"
    return d


def add_crop(cid: str, tid: str, item_id: str, name: str, roi: list[int],
             safe_margin: int = 0) -> dict:
    """新增/更新一个模板块。

    item_id 为空 = 新建 -> **服务端分配** fieldNN（单调递增、不复用）；
    item_id 非空 = 更新已有块（ID 不可变，只改 name/roi/margin）。
    返回值里带 saved_id，前端据此把"新建"的本地状态锚定到真实 ID。
    """
    tpl = store.load_template(cid, tid)
    p = store.tpl_dir(cid, tid) / "crops.json"
    spec = CropSpec.load(p) if p.exists() else CropSpec(
        name=tpl["name"], canvas=tpl["canvas"], dpi=tpl["dpi"])

    iid = (item_id or "").strip()
    if not iid:
        iid = spec.next_id()
    spec.upsert(CropItem(id=iid, name=(name or "").strip(),
                         roi=[int(v) for v in roi], safe_margin=int(safe_margin)))
    spec.save(p)

    out = spec.to_dict()
    out["saved_id"] = iid
    for it in out["items"]:
        it["preview_url"] = f"/api/categories/{cid}/templates/{tid}/crops/{it['id']}/preview"
    return out


def remove_crop(cid: str, tid: str, item_id: str) -> dict:
    p = store.tpl_dir(cid, tid) / "crops.json"
    if not p.exists():
        raise FileNotFoundError("该模板还没有坐标清单")
    spec = CropSpec.load(p)
    spec.remove(item_id)
    spec.save(p)
    return spec.to_dict()
