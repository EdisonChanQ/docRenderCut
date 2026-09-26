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

import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import service, store

app = FastAPI(title="docRenderCut", version="2.0.0",
              description="业务单据标准模板渲染与模板块裁剪（分类 / 模板两级 + 自动识别）")


def _clean_name(name: str) -> str:
    """上传文件名只保留基础名，避免路径穿越。"""
    return Path(name.replace("\\", "/")).name or "upload.bin"


# ---------------------------------------------------------------- 分类（纯容器）


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "root": str(store.ROOT)}


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
    store.delete_category(cid)
    return {"ok": True, "id": cid}


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

    tid = store.new_template_id()
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
        "sample": {"filename": fname},
        "created_at": store._now(),
    }
    store.save_template(cid, tid, tpl)

    try:
        res = service.build_template(cid, tid, dst, dpi=int(dpi),
                                     canvas=(int(width), int(height)),
                                     paper_mode=paper_mode, corners=cs,
                                     ink_bias=int(ink_bias))
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
    store.delete_template(cid, tid)
    return {"ok": True, "id": tid}


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
    item_id = str(payload.get("id", "")).strip()
    roi = payload.get("roi")
    if not item_id:
        raise HTTPException(400, "请填写字段名称（块 ID）")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise HTTPException(400, "roi 需要 4 个数：x,y,w,h")
    try:
        roi_i = [int(round(float(v))) for v in roi]
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "roi 必须是数字") from exc
    if roi_i[2] <= 0 or roi_i[3] <= 0:
        raise HTTPException(400, "roi 的宽高必须大于 0")
    try:
        return service.add_crop(cid, tid, item_id, roi_i,
                                int(payload.get("safe_margin", 0) or 0),
                                str(payload.get("note", "") or ""))
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
def jobs(category_id: str | None = None, limit: int = 30) -> list[dict]:
    return store.list_jobs(category_id, limit=limit)


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    try:
        return store.load_job(jid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/jobs/{jid}/verify")
def verify_job(jid: str, limit: int = 24, tol: float = 3.0) -> dict:
    try:
        return service.run_verify(jid, limit=limit, tol_px=tol)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- 静态资源

store._ensure()
app.mount("/files", StaticFiles(directory=str(store.DATA)), name="files")

if store.WEB.is_dir():
    app.mount("/", StaticFiles(directory=str(store.WEB), html=True), name="web")
