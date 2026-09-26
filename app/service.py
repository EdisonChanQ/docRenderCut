"""引擎编排：把 docnorm 的算法能力包成 Web 后端可直接调用的步骤。

- `build_template`：样张 + 参数 -> 模板标准模板（模板图 + 模板 JSON + 空坐标清单）
- `start_job`：分类 + 一批生产文件 -> 逐页识别模板 -> 按各自模板渲染 + 裁剪

全部走已验证的 docnorm 引擎，这里只做「参数校验、纸张层选择、警告收集、进度上报」。

数据模型两级：
    分类 Category（纯容器） -> 模板 Template（纸面参数 + 坐标，执行单元）
上传时选分类，系统逐页 classify_page 归到分类下某个模板；识别不出的页进 rejected。
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np

from docnorm.align import AlignConfig
from docnorm.classify import ClassifyConfig, classify_page
from docnorm.crops import CropItem, CropSpec
from docnorm.loader import IMAGE_EXT, PDF_EXT, iter_batch, load_image
from docnorm.pipeline import Engine
from docnorm.rectify import PaperQuad, _validate_paper_quad, detect_paper_quad, paper_warp
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


# ---------------------------------------------------------------- 建模板


def build_template(cid: str, tid: str, sample: Path, *, dpi: int,
                   canvas: tuple[int, int], paper_mode: str = "off",
                   corners: list[float] | None = None,
                   ink_bias: int = 0) -> dict:
    """用样张建立该模板的标准模板。返回 {warnings, info}。"""
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

        paths = [str(p) for _, p in files]
        units = iter_batch(paths, render_dpi=300)
        if not units:
            raise ValueError("上传的文件里没有可处理的页")

        out_root = store.job_dir(jid)
        job.update({"status": "running", "message": "处理中",
                    "total": len(units), "done": 0})
        store.save_job(jid, job)

        class_cfg = ClassifyConfig()

        def on_page(done: int, total: int, rec) -> None:
            job["done"] = done
            job["message"] = f"已处理 {done}/{total} 页"
            store.save_job(jid, job)

        pages: list[dict] = []
        n_done = 0
        for unit in units:
            stem = f"{Path(unit.source).stem}_p{unit.page_index + 1:04d}"

            # 1. 识别这一页属于哪个模板
            cr = classify_page(unit.image, templates, class_cfg)

            if not cr.matched or cr.template_id is None:
                # 识别不出 -> rejected，原样保留，供审计
                _drop_stale(stem, out_root)
                write_png(out_root / "rejected" / f"{stem}.png", unit.image, 300)
                pages.append({
                    "stem": stem,
                    "source": Path(unit.source).name,
                    "page_index": unit.page_index + 1,
                    "status": "rejected",
                    "reason": f"识别不出：{cr.reason}",
                    "score": cr.score,
                    "scores": cr.scores,
                    "template_id": None,
                    "output_url": store.to_url(out_root / "rejected" / f"{stem}.png"),
                    "blocks": [],
                })
                n_done += 1
                on_page(n_done, len(units), None)
                continue

            # 2. 命中 -> 用该模板渲染 + 裁剪
            tid = cr.template_id
            tm = tpl_meta_by_id(tpl_meta, tid)
            rec = _render_with_template(
                out_root, unit, stem, tid, tm, templates[tid], cr)
            pages.append(rec)
            n_done += 1
            on_page(n_done, len(units), None)

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


def _render_with_template(out_root: Path, unit, stem: str, tid: str,
                          tm: dict, tpl: Template, cr) -> dict:
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

    rec, img = eng.process_unit(unit)
    blocks = []
    status = rec.status
    reason = rec.reason or ""

    if status in ("ok", "low_confidence") and img is not None:
        sub = "standardized" if status == "ok" else "low_confidence"
        _drop_stale(stem, out_root)
        out_path = out_root / sub / f"{stem}.png"
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
        "blocks": blocks,
    }


def _drop_stale(stem: str, out_root: Path) -> None:
    """删掉同名页在其它目录里的旧产物，保证一页只存在于一处。"""
    for d in (out_root / "standardized", out_root / "low_confidence",
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
        for sub in ("standardized", "low_confidence"):
            f = out_root / sub / f"{stem}.png"
            if f.exists():
                groups.setdefault(tid, []).append(f)

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


def add_crop(cid: str, tid: str, item_id: str, roi: list[int],
             safe_margin: int = 0, note: str = "") -> dict:
    tpl = store.load_template(cid, tid)
    p = store.tpl_dir(cid, tid) / "crops.json"
    spec = CropSpec.load(p) if p.exists() else CropSpec(
        name=tpl["name"], canvas=tpl["canvas"], dpi=tpl["dpi"])
    spec.upsert(CropItem(item_id, [int(v) for v in roi],
                         safe_margin=int(safe_margin), note=note or ""))
    spec.save(p)
    out = spec.to_dict()
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
