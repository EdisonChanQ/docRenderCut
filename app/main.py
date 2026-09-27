"""docRenderCut 后端：分类 -> 模板 -> 坐标块 -> 批量识别渲染裁剪。

启动：
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8848

接口分组：
    /api/categories/**           分类（纯容器）与模板（纸面参数 + 坐标）
    /api/categories/{cid}/templates/{tid}/**  模板的模板图 / 坐标 / 裁剪预览
    /api/jobs/**                 批量任务（上传、识别、进度、逐页结果、验证）
    /files/**                    静态产物（模板图、标准输出图、裁剪块）
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import service, store

app = FastAPI(title="docRenderCut", version="2.0.0",
              description="业务单据标准模板渲染与模板块裁剪（分类 / 模板两级 + 自动识别）")


def _clean_name(name: str) -> str:
    """上传文件名只保留基础名，避免路径穿越。"""
    return Path(name.replace("\\", "/")).name or "upload.bin"


# ---------------------------------------------------------------- 初始化门禁


_DATA_PREFIXES = ("/api/categories", "/api/jobs", "/files")


@app.middleware("http")
async def _require_data_dir(request, call_next):
    """未配置数据目录时，所有数据相关接口一律拒绝。

    只在前端把界面藏起来是不够的——直接调 API 也能写数据，
    那就会在"没配置"的状态下产生一堆没有归属的文件。
    """
    if not store.is_configured():
        p = request.url.path
        if any(p.startswith(pref) for pref in _DATA_PREFIXES):
            return JSONResponse(
                {"detail": "尚未配置数据目录，请先完成初始化（配置并校验数据目录）"},
                status_code=409)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    cfg = service.job_config()
    return {
        "ok": True,
        "root": str(store.ROOT),
        "configured": store.is_configured(),
        "data": str(store.data_root()) if store.is_configured() else None,
        "data_source": store.DATA_SOURCE,
        # 任务并行与资源预算（运维要能看到"这个实例实际会用多少资源"）
        "job": cfg.as_dict(),
        "cpu_cores": os.cpu_count(),
    }


# ---------------------------------------------------------------- 数据目录配置


@app.get("/api/setup/state")
def setup_state() -> dict:
    """前端启动时先问这个：要不要走初始化流程。"""
    configured = store.is_configured()
    stats = None
    if configured:
        try:
            stats = store._count_existing(store.data_root())
        except OSError:
            stats = None
    return {
        "configured": configured,
        "data_dir": str(store.data_root()) if configured else None,
        "data_source": store.DATA_SOURCE,
        "has_override": store.has_override(),   # 来自 CLI/环境变量，前端改不动
        "suggested": str(store.suggested_data_dir()),
        "config_path": str(store.config_path()),
        "project_root": str(store.ROOT),
        "stats": stats,
    }


@app.post("/api/setup/probe")
def setup_probe(payload: dict = Body(...)) -> dict:
    """探测校验一个候选数据目录（无副作用，只写一个随即删除的探测文件）。"""
    return store.probe_data_dir(str(payload.get("path", "")))


@app.post("/api/setup/configure")
def setup_configure(payload: dict = Body(...)) -> dict:
    """校验通过后真正启用：写配置 + 运行时切换数据目录。"""
    if store.has_override():
        raise HTTPException(
            409, "数据目录由命令行参数或环境变量指定（--data / DOCRENDERCUT_DATA），"
                 "界面无法修改；请改用启动参数。")
    path = str(payload.get("path", "")).strip()
    rep = store.probe_data_dir(path)
    if not rep.get("ok"):
        raise HTTPException(400, "校验未通过：" + "；".join(rep.get("errors") or ["未知错误"]))
    target = store.set_data_root(rep["path"])
    store.save_config({"data_dir": str(target)})
    return {"ok": True, "data_dir": str(target), "warnings": rep.get("warnings", []),
            "existing": rep.get("existing")}


@app.get("/api/categories")
def categories() -> list[dict]:
    return store.list_categories()


@app.post("/api/categories")
def create_category(name: str = Form(...)) -> dict:
    """建分类：纯容器，只有名字。纸面参数在模板层。"""
    if not name.strip():
        raise HTTPException(400, "分类名称不能为空")
    cid = store.new_category_id()
    cat = {
        "id": cid,
        "name": name.strip(),
        "kind": "category",
        "created_at": store._now(),
    }
    store.save_category(cid, cat)
    return cat


@app.get("/api/categories/{cid}")
def get_category(cid: str) -> dict:
    try:
        cat = store.load_category(cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    cat["templates"] = store.list_templates(cid)
    cat["jobs"] = store.list_jobs(cid, limit=20)
    return cat


@app.delete("/api/categories/{cid}")
def delete_category(cid: str) -> dict:
    cleanup = store.delete_category(cid)
    return {"ok": True, "id": cid, "cleanup": cleanup}


# ---------------------------------------------------------------- 模板（纸面参数）


@app.post("/api/categories/{cid}/templates")
async def create_template(
    cid: str,
    name: str = Form(...),
    dpi: int = Form(200),
    width: int = Form(1600),
    height: int = Form(3400),
    paper_mode: str = Form("off"),
    corners: str = Form(""),
    ink_bias: int = Form(0),
    ink_dark_bias: int = Form(25),
    mono: bool = Form(True),
    deskew: bool = Form(True),
    sample: UploadFile = File(...),
) -> dict:
    """建模板：上传样张 + 参数 -> 后台按参数渲染出该模板标准模板。"""
    try:
        store.load_category(cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    if not name.strip():
        raise HTTPException(400, "模板名称不能为空")
    if dpi < 50 or dpi > 1200:
        raise HTTPException(400, "DPI 需要在 50~1200 之间")
    if not (100 <= width <= 20000 and 100 <= height <= 20000):
        raise HTTPException(400, "画布尺寸需要在 100~20000 像素之间")

    cs: list[float] | None = None
    if paper_mode == "corners":
        try:
            cs = [float(v) for v in corners.replace("，", ",").split(",") if v.strip()]
        except ValueError as exc:
            raise HTTPException(400, "四角格式应为 x1,y1,x2,y2,x3,y3,x4,y4") from exc
        if len(cs) != 8:
            raise HTTPException(400, "四角需要 8 个数：x1,y1,x2,y2,x3,y3,x4,y4")

    tid = store.new_template_id(cid)
    tdir = store.tpl_dir(cid, tid)
    sdir = tdir / "sample"
    sdir.mkdir(parents=True, exist_ok=True)

    fname = _clean_name(sample.filename or "sample")
    dst = sdir / fname
    with dst.open("wb") as fh:
        shutil.copyfileobj(sample.file, fh)

    tpl = {
        "id": tid,
        "category_id": cid,
        "name": name.strip(),
        "dpi": int(dpi),
        "canvas": {"width": int(width), "height": int(height)},
        "paper_mode": paper_mode,
        "corners": cs,
        "ink_bias": int(ink_bias),
        "ink_dark_bias": int(ink_dark_bias),
        "mono": bool(mono),
        "deskew": bool(deskew),
        "sample": {"filename": fname},
        "created_at": store._now(),
    }
    store.save_template(cid, tid, tpl)

    try:
        res = service.build_template(cid, tid, dst, dpi=int(dpi),
                                     canvas=(int(width), int(height)),
                                     paper_mode=paper_mode, corners=cs,
                                     ink_bias=int(ink_bias),
                                     deskew_sample=bool(deskew))
    except Exception as exc:  # noqa: BLE001
        store.delete_template(cid, tid)
        raise HTTPException(400, f"模板构建失败：{exc}") from exc

    tpl["sample"].update({k: res["info"][k] for k in
                          ("sample_size", "sample_kind", "sample_mm")})
    tpl["template"] = {"ready": True, **res["info"]}
    tpl["warnings"] = res["warnings"]
    store.save_template(cid, tid, tpl)

    out = store.load_template(cid, tid)
    out["template_ready"] = True
    out["n_blocks"] = 0
    return out


@app.post("/api/categories/{cid}/templates/probe")
async def probe_template(cid: str, sample: UploadFile = File(...),
                         dpi_hint: int = Form(200)) -> dict:
    """**只识别样张尺寸、不落盘**（建模板的第一步）。

    上传的文件写到系统临时目录、解析完即删，数据目录里不会留下任何中间态。
    返回检测到的 DPI / 物理尺寸 / 建议画布，供用户在弹窗里核对修改；
    用户点「确认构建」后再调 POST /templates 真正落盘。
    样张可以反复上传，每次都会重新识别。
    """
    try:
        store.load_category(cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    fname = _clean_name(sample.filename or "sample")
    with tempfile.TemporaryDirectory(prefix="drc-probe-") as td:
        dst = Path(td) / fname
        with dst.open("wb") as fh:
            shutil.copyfileobj(sample.file, fh)
        try:
            info = service.probe_sample(dst, dpi_hint=int(dpi_hint))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"样张无法解析：{exc}") from exc
    info["filename"] = fname
    return info


@app.get("/api/categories/{cid}/templates")
def list_templates(cid: str) -> list[dict]:
    return store.list_templates(cid)


@app.get("/api/categories/{cid}/templates/{tid}")
def get_template(cid: str, tid: str) -> dict:
    try:
        tpl = store.load_template(cid, tid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    tpl["template_ready"] = (store.tpl_dir(cid, tid) / "template" / "template.png").exists()
    tpl["n_blocks"] = store._count_blocks(store.tpl_dir(cid, tid) / "crops.json")
    return tpl


@app.delete("/api/categories/{cid}/templates/{tid}")
def delete_template(cid: str, tid: str) -> dict:
    cleanup = store.delete_template(cid, tid)
    return {"ok": True, "id": tid, "cleanup": cleanup}


_TPL_EDIT_LABEL = {
    "name": "名称", "dpi": "DPI", "paper_mode": "纸张处理",
    "corners": "四角", "ink_bias": "深墨偏置(建模板)",
    "ink_dark_bias": "深墨偏置(输出)", "mono": "黑白输出", "deskew": "自动纠偏",
}


@app.post("/api/categories/{cid}/templates/{tid}/edit")
@app.post("/api/categories/{cid}/templates/{tid}/rebuild")
async def edit_template(
    cid: str,
    tid: str,
    name: str | None = Form(None),
    dpi: int | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    paper_mode: str | None = Form(None),
    corners: str | None = Form(None),
    ink_bias: int | None = Form(None),
    ink_dark_bias: int | None = Form(None),
    mono: bool | None = Form(None),
    deskew: bool | None = Form(None),
    force: bool = Form(False),
    sample: UploadFile | None = File(None),
) -> dict:
    """**编辑模板**：改参数、换样张，然后按新参数重新渲染标准模板图。

    所有字段都是**可选**的——只传要改的，没传的沿用原值，所以同一个接口
    既能"只改个名字"（不重传样张，不重建）也能"换一张样张重新识别参数再建"。

    `/rebuild` 是同一实现的别名（老入口，行为向后兼容）。

    **坐标清单保留不动**——但它是在旧模板上量的，重建后内容可能整体转动或
    缩放，所以返回值里会明确提示需要逐个核对。
    """
    try:
        tpl = store.load_template(cid, tid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    before = {k: tpl.get(k) for k in _TPL_EDIT_LABEL}
    before_canvas = dict(tpl.get("canvas") or {})

    # ---- 1) 合并参数（只覆盖显式传上来的字段）----
    if name is not None:
        if not name.strip():
            raise HTTPException(400, "模板名称不能为空")
        tpl["name"] = name.strip()
    if dpi is not None:
        if dpi < 50 or dpi > 1200:
            raise HTTPException(400, "DPI 需要在 50~1200 之间")
        tpl["dpi"] = int(dpi)
    if width is not None:
        if not (100 <= width <= 20000):
            raise HTTPException(400, "画布宽需要在 100~20000 像素之间")
        tpl.setdefault("canvas", {})["width"] = int(width)
    if height is not None:
        if not (100 <= height <= 20000):
            raise HTTPException(400, "画布高需要在 100~20000 像素之间")
        tpl.setdefault("canvas", {})["height"] = int(height)
    if paper_mode is not None:
        if paper_mode not in ("off", "auto", "corners"):
            raise HTTPException(400, "纸张处理只能是 off / auto / corners")
        tpl["paper_mode"] = paper_mode
    if corners is not None:
        cs: list[float] | None = None
        if corners.strip():
            try:
                cs = [float(v) for v in corners.replace("，", ",").split(",") if v.strip()]
            except ValueError as exc:
                raise HTTPException(400, "四角格式应为 x1,y1,x2,y2,x3,y3,x4,y4") from exc
            if len(cs) != 8:
                raise HTTPException(400, "四角需要 8 个数：x1,y1,x2,y2,x3,y3,x4,y4")
        tpl["corners"] = cs
    if ink_bias is not None:
        tpl["ink_bias"] = int(ink_bias)
    if ink_dark_bias is not None:
        tpl["ink_dark_bias"] = int(ink_dark_bias)
    if mono is not None:
        tpl["mono"] = bool(mono)
    if deskew is not None:
        tpl["deskew"] = bool(deskew)

    # ---- 2) 换样张（可选）----
    changed: list[str] = []
    src = store.sample_file(cid, tid)
    sample_replaced = False
    if sample is not None and sample.filename:
        sdir = store.tpl_dir(cid, tid) / "sample"
        sdir.mkdir(parents=True, exist_ok=True)
        for old in sdir.iterdir():
            if old.is_file():
                old.unlink()
        dst = sdir / _clean_name(sample.filename)
        with dst.open("wb") as fh:
            shutil.copyfileobj(sample.file, fh)
        src = dst
        sample_replaced = True
    if src is None:
        raise HTTPException(400, "该模板还没有样张，请上传一张")

    tpl["sample"] = {"filename": src.name}

    # ---- 3) 差异清单（给前端说人话）----
    if sample_replaced:
        changed.append(f"样张 → {src.name}")
    for k, label in _TPL_EDIT_LABEL.items():
        if tpl.get(k) != before[k]:
            changed.append(f"{label}: {before[k]} → {tpl.get(k)}")
    canvas_now = tpl.get("canvas") or {}
    if canvas_now != before_canvas:
        changed.append(
            f"画布: {before_canvas.get('width')}×{before_canvas.get('height')}"
            f" → {canvas_now.get('width')}×{canvas_now.get('height')}")

    # 只改了名字这种"不影响渲染"的字段就不用重跑了，省几秒；
    # force=true 可强制重跑（老 /rebuild 的语义：无论如何都重建一次）
    needs_rebuild = force or sample_replaced or any(
        k in ("dpi", "paper_mode", "corners", "ink_bias", "deskew")
        for k in _TPL_EDIT_LABEL if tpl.get(k) != before[k]
    ) or canvas_now != before_canvas

    tpl["updated_at"] = store._now()
    store.save_template(cid, tid, tpl)

    if needs_rebuild:
        try:
            res = service.build_template(
                cid, tid, src,
                dpi=int(tpl.get("dpi", 200)),
                canvas=(int(canvas_now.get("width", 1600)),
                        int(canvas_now.get("height", 2600))),
                paper_mode=tpl.get("paper_mode") or "off",
                corners=tpl.get("corners"),
                ink_bias=int(tpl.get("ink_bias") or 0),
                deskew_sample=bool(tpl.get("deskew", True)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"模板重建失败：{exc}") from exc

        tpl["sample"].update({k: res["info"][k] for k in
                              ("sample_size", "sample_kind", "sample_mm")})
        tpl["template"] = {"ready": True, **res["info"]}
        tpl["warnings"] = res["warnings"]
        store.save_template(cid, tid, tpl)

    n_blocks = store._count_blocks(store.tpl_dir(cid, tid) / "crops.json")
    out = store.load_template(cid, tid)
    out["template_ready"] = (store.tpl_dir(cid, tid) / "template" / "template.png").exists()
    out["n_blocks"] = n_blocks
    out["changed"] = changed
    out["rebuilt"] = needs_rebuild
    if needs_rebuild and n_blocks:
        out.setdefault("warnings", []).append(
            f"保留了原有的 {n_blocks} 个坐标块——它们是在旧模板上量的。"
            "本次重建可能让内容整体转动或缩放，请逐个核对预览图，偏了就重新框选。")
    return out


@app.get("/api/categories/{cid}/templates/{tid}/template")
def template_info(cid: str, tid: str) -> dict:
    """模板信息 + 表格线位置（供坐标标注器做吸附参考）。"""
    from docnorm.template import Template

    tdir = store.tpl_dir(cid, tid) / "template"
    png = tdir / "template.png"
    if not png.exists():
        raise HTTPException(404, "该模板还没有模板图")
    tpl = Template.load(tdir)
    lines = tpl.lines()
    tm = store.load_template(cid, tid)
    version = int(png.stat().st_mtime)
    return {
        "canvas": {"width": tpl.canvas[0], "height": tpl.canvas[1]},
        "dpi": tpl.dpi,
        "template_url": store.to_url(png),
        "preview_url": f"/api/categories/{cid}/templates/{tid}/template.png?max=1800&v={version}",
        "preview_max": 1800,
        "bytes": png.stat().st_size,
        "lines_x": [round(float(v), 1) for v in lines.xs],
        "lines_y": [round(float(v), 1) for v in lines.ys],
        "warnings": tm.get("warnings", []),
        "paper_mode": tm.get("paper_mode"),
        "sample": tm.get("sample"),
    }


@app.get("/api/categories/{cid}/templates/{tid}/template.png")
def template_image(cid: str, tid: str,
                   max_px: int | None = Query(None, alias="max")) -> Response:
    """模板图。带 `max` 参数时降采样（界面用），不带时给全分辨率原图。"""
    p = store.tpl_dir(cid, tid) / "template" / "template.png"
    if not p.exists():
        raise HTTPException(404, "该模板还没有模板图")
    if max_px is None:
        return FileResponse(p, media_type="image/png")

    limit = max(200, min(6000, int(max_px)))
    gray = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(500, "模板图无法解码")
    h, w = gray.shape[:2]
    k = min(1.0, limit / float(max(h, w)))
    if k < 1.0:
        gray = cv2.resize(gray, (max(1, int(round(w * k))), max(1, int(round(h * k)))),
                          interpolation=cv2.INTER_AREA)

    cands: list[tuple[int, str, bytes]] = []
    ok_j, buf_j = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok_j:
        cands.append((buf_j.size, "image/jpeg", buf_j.tobytes()))
    ok_p, buf_p = cv2.imencode(".png", gray, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if ok_p:
        cands.append((buf_p.size, "image/png", buf_p.tobytes()))
    if not cands:
        raise HTTPException(500, "编码失败")
    _size, media, payload = min(cands, key=lambda c: c[0])
    return Response(content=payload, media_type=media,
                    headers={"Cache-Control": "public, max-age=3600"})


# ---------------------------------------------------------------- 坐标清单


@app.get("/api/categories/{cid}/templates/{tid}/crops")
def get_crops(cid: str, tid: str) -> dict:
    return service.list_crops(cid, tid)


@app.post("/api/categories/{cid}/templates/{tid}/crops")
def add_crop(cid: str, tid: str, payload: dict = Body(...)) -> dict:
    """新增/更新一个模板块。

    `id` 为空 = 新建（服务端自动分配 fieldNN）；非空 = 更新已有块。
    `name` 是人工编辑的字段说明，可选。
    """
    item_id = str(payload.get("id", "") or "").strip()
    name = str(payload.get("name", "") or "").strip()
    roi = payload.get("roi")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise HTTPException(400, "roi 需要 4 个数：x,y,w,h")
    try:
        roi_i = [int(round(float(v))) for v in roi]
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "roi 必须是数字") from exc
    if roi_i[2] <= 0 or roi_i[3] <= 0:
        raise HTTPException(400, "roi 的宽高必须大于 0")
    try:
        return service.add_crop(cid, tid, item_id, name, roi_i,
                                int(payload.get("safe_margin", 0) or 0))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/categories/{cid}/templates/{tid}/crops/{item_id}")
def delete_crop(cid: str, tid: str, item_id: str) -> dict:
    try:
        return service.remove_crop(cid, tid, item_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/categories/{cid}/templates/{tid}/crops/{item_id}/preview")
def crop_preview(cid: str, tid: str, item_id: str,
                 safe: int = 1, scale: float = 1.0) -> Response:
    """把该坐标块从**模板图**上裁出来，用于配置时立刻确认框得对不对。"""
    from docnorm.crops import CropSpec

    p = store.tpl_dir(cid, tid) / "crops.json"
    if not p.exists():
        raise HTTPException(404, "没有坐标清单")
    spec = CropSpec.load(p)
    it = spec.find(item_id)
    if it is None:
        raise HTTPException(404, f"没有这个块: {item_id}")

    tdir = store.tpl_dir(cid, tid) / "template"
    data = np.fromfile(str(tdir / "template.png"), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(404, "模板图无法读取")

    x, y, w, h = it.box(use_safe=bool(safe))
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(400, "该坐标块在画布之外")
    crop = gray[y0:y1, x0:x1]

    k = scale if scale > 0 else 1.0
    if k == 1.0 and crop.shape[1] < 400:
        k = min(6.0, 400.0 / max(1, crop.shape[1]))
    if k > 1.01:
        crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)),
                          interpolation=cv2.INTER_NEAREST)

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(500, "编码失败")
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- 批量任务


@app.post("/api/categories/{cid}/jobs")
async def create_job(cid: str, files: list[UploadFile] = File(...)) -> dict:
    """上传批量生产文件，按分类逐页识别模板、渲染、裁剪。

    PDF 自动分页；每页自动归到分类下某个模板；识别不出的页进 rejected。
    """
    try:
        cat = store.load_category(cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if store._count_templates(cid) == 0:
        raise HTTPException(400, "该分类下还没有任何模板，请先建模板")
    if not files:
        raise HTTPException(400, "请选择要上传的文件")

    jid = store.new_job_id()
    udir = store.job_dir(jid) / "uploads"
    udir.mkdir(parents=True, exist_ok=True)

    saved: list[tuple[str, Path]] = []
    for f in files:
        fname = _clean_name(f.filename or "file")
        dst = udir / fname
        n = 1
        while dst.exists():
            n += 1
            dst = udir / f"{Path(fname).stem}_{n}{Path(fname).suffix}"
        with dst.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved.append((dst.name, dst))

    if not saved:
        raise HTTPException(400, "没有收到有效文件")

    job = service.start_job(cid, saved, jid=jid)
    job["category_name"] = cat.get("name")
    return job


@app.get("/api/jobs")
def jobs(category_id: str | None = None, template_id: str | None = None,
         limit: int = 30) -> list[dict]:
    """任务列表。

    - 归属有**两级**：分类是任务级的，模板是**页级**的（一个任务可混多个模板），
      所以列表里同时给出 `category_id/name` 与 `templates: [{id,name,n}]` 汇总，
      不用点开详情就能反推出"这批是哪个分类、识别到了哪些模板"。
    - `?template_id=TPLxxx` 可按模板反查任务（哪些任务用过这个模板）。
    - 名称是建任务时的**快照**；稳定键是 id。
    """
    return store.list_jobs(category_id, limit=limit, template_id=template_id)


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    try:
        return store.load_job(jid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str) -> dict:
    """删除任务及其全部磁盘产物（标准图、裁剪块、拒收原图、验证报告）。

    返回 cleanup：deleted=已真正删除；trashed=环境拦截了批量删除，
    已整目录改名到 data/.trash（列表里已消失，磁盘清理可手动做）。
    """
    try:
        store.load_job(jid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    cleanup = store.delete_job(jid)
    return {"ok": True, "id": jid, "cleanup": cleanup}


@app.post("/api/jobs/{jid}/verify")
def verify_job(jid: str, limit: int = 24, tol: float = 3.0) -> dict:
    try:
        return service.run_verify(jid, limit=limit, tol_px=tol)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- 静态资源


@app.get("/files/{rel_path:path}")
def data_file(rel_path: str) -> FileResponse:
    """数据目录下的产物（模板图、标准输出图、裁剪块、验证报告）。

    刻意**不用启动时绑定的 StaticFiles**：数据目录可以在前端随时切换，
    静态挂载是启动时固定死的，切了目录之后图片会全部 404。
    这里每次请求都按当前数据目录解析，并挡住路径穿越。
    """
    if not store.is_configured():
        raise HTTPException(409, "尚未配置数据目录")
    root = store.data_root().resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)          # 路径穿越保护
    except ValueError:
        raise HTTPException(404, "not found") from None
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


if store.WEB.is_dir():
    app.mount("/", StaticFiles(directory=str(store.WEB), html=True), name="web")
