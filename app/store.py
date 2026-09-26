"""分类 / 模板 / 任务的文件存储布局。

数据模型（两级）：

    分类 Category —— 纯逻辑容器，只有 {id, name}，不带任何纸面参数。
        它的唯一作用是让上传时**不用人工拆分文件、不用指定模板**：
        系统按页自动识别每页属于分类下的哪个模板。

    模板 Template —— 完整的纸面配置 {id, name, dpi, canvas, paper_mode, ink_bias, ...}，
        是配准、渲染、裁剪的执行单元。相当于旧版里的"类别"。

设计原则：**磁盘就是数据库**。每个分类一个目录，分类下每个模板一个目录，
模板目录里放完整可复现的证据（样张、模板 JSON、模板图、坐标清单）；
每个任务一个目录，里面是引擎原样产出的 standardized / blocks / sidecar / report。

    data/categories/<分类ID>/
        category.json                分类元数据（纯容器）
        templates/<模板ID>/
            template.json            模板元数据（含 parent 分类 ID）
            sample/                  上传的样张原文件
            template/                template.json + template.png（引擎模板）
            crops.json               模板块坐标清单
    data/jobs/<任务ID>/
        job.json                     任务状态 + category_id + 逐页结果（每页含 template_id）
        uploads/                     上传的生产文件
        standardized/                标准输出图
        low_confidence/              低置信输出（内部一致，但需人工抽检）
        rejected/                    拒收原图（含"识别不出"的页，供审计）
        blocks/<页>/<块>.png         裁剪出的模板块
        sidecar/                     逐页配准诊断
        render-report.html
        verify/                      验证堆叠图与报告
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATEGORIES = DATA / "categories"
JOBS = DATA / "jobs"
WEB = ROOT / "web"


# ---------------------------------------------------------------- 基础


def _ensure() -> None:
    CATEGORIES.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(name: str) -> str:
    """把名称变成安全的目录名片段（保留中文，去掉路径分隔符等）。"""
    s = re.sub(r"[\\/:*?\"<>|\s]+", "-", name.strip())
    return (s[:24] or "CAT").strip("-")


def _next_id(prefix: str, base_dir: Path) -> str:
    """按「前缀 + 日期 + 当日序号」生成 ID，形如 CAT20260925-0001 / TPL20260925-0001。

    用"日期 + 当日序号"而不是随机串或 UUID：这类工具里人会念这个 ID
    （"把这个模板的模板图发我"），可读、可排序、能一眼看出建立时间。
    """
    _ensure()
    day = datetime.now().strftime("%Y%m%d")
    used = {p.name.split("-")[-1] for p in base_dir.glob(f"{prefix}{day}-*")}
    for i in range(1, 10000):
        seq = f"{i:04d}"
        if seq not in used:
            return f"{prefix}{day}-{seq}"
    return f"{prefix}{day}-{secrets.token_hex(3).upper()}"


def new_category_id() -> str:
    return _next_id("CAT", CATEGORIES)


def new_template_id() -> str:
    """模板 ID 全局唯一：模板分散在各分类下的 templates/ 目录里，
    不能只在单个目录里查重，必须扫描所有分类下的所有模板目录。"""
    _ensure()
    day = datetime.now().strftime("%Y%m%d")
    used: set[str] = set()
    for cdir in CATEGORIES.iterdir():
        if not cdir.is_dir():
            continue
        tdir = cdir / "templates"
        if not tdir.is_dir():
            continue
        for d in tdir.iterdir():
            if d.is_dir():
                used.add(d.name)
    for i in range(1, 10000):
        seq = f"{i:04d}"
        cand = f"TPL{day}-{seq}"
        if cand not in used:
            return cand
    return f"TPL{day}-{secrets.token_hex(3).upper()}"


def new_job_id() -> str:
    _ensure()
    return "JOB" + datetime.now().strftime("%Y%m%d%H%M%S") + "-" + secrets.token_hex(2).upper()


# ---------------------------------------------------------------- 分类


def cat_dir(cid: str) -> Path:
    return CATEGORIES / cid


def templates_dir(cid: str) -> Path:
    return CATEGORIES / cid / "templates"


def tpl_dir(cid: str, tid: str) -> Path:
    return CATEGORIES / cid / "templates" / tid


def load_category(cid: str) -> dict:
    p = cat_dir(cid) / "category.json"
    if not p.exists():
        raise FileNotFoundError(f"分类不存在: {cid}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_category(cid: str, d: dict) -> None:
    d["updated_at"] = _now()
    d_dir = cat_dir(cid)
    d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / "category.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def list_categories() -> list[dict]:
    _ensure()
    out: list[dict] = []
    for d in sorted(CATEGORIES.iterdir()):
        if not d.is_dir():
            continue
        f = d / "category.json"
        if not f.exists():
            continue
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        c["n_templates"] = _count_templates(c.get("id"))
        out.append(c)
    out.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return out


def delete_category(cid: str) -> None:
    import shutil

    d = cat_dir(cid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 模板


def _count_templates(cid: str) -> int:
    tdir = templates_dir(cid)
    if not tdir.is_dir():
        return 0
    n = 0
    for d in tdir.iterdir():
        if d.is_dir() and (d / "template.json").exists():
            n += 1
    return n


def load_template(cid: str, tid: str) -> dict:
    p = tpl_dir(cid, tid) / "template.json"
    if not p.exists():
        raise FileNotFoundError(f"模板不存在: {tid}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_template(cid: str, tid: str, d: dict) -> None:
    d["updated_at"] = _now()
    d_dir = tpl_dir(cid, tid)
    d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / "template.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def list_templates(cid: str) -> list[dict]:
    tdir = templates_dir(cid)
    if not tdir.is_dir():
        return []
    out: list[dict] = []
    for d in sorted(tdir.iterdir()):
        if not d.is_dir():
            continue
        f = d / "template.json"
        if not f.exists():
            continue
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        t["template_ready"] = (d / "template" / "template.png").exists()
        t["n_blocks"] = _count_blocks(d / "crops.json")
        out.append(t)
    out.sort(key=lambda t: t.get("created_at", ""))
    return out


def delete_template(cid: str, tid: str) -> None:
    import shutil

    d = tpl_dir(cid, tid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def sample_file(cid: str, tid: str) -> Path | None:
    d = tpl_dir(cid, tid) / "sample"
    if not d.is_dir():
        return None
    for f in sorted(d.iterdir()):
        if f.is_file():
            return f
    return None


def _count_blocks(crops_path: Path) -> int:
    if not crops_path.exists():
        return 0
    try:
        return len(json.loads(crops_path.read_text(encoding="utf-8")).get("items", []))
    except json.JSONDecodeError:
        return 0


# ---------------------------------------------------------------- 任务


def job_dir(jid: str) -> Path:
    return JOBS / jid


def load_job(jid: str) -> dict:
    p = job_dir(jid) / "job.json"
    if not p.exists():
        raise FileNotFoundError(f"任务不存在: {jid}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_job(jid: str, d: dict) -> None:
    d_dir = job_dir(jid)
    d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / "job.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _job_category(p: Path) -> str | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("category_id")
    except json.JSONDecodeError:
        return None


def list_jobs(category_id: str | None = None, limit: int = 30) -> list[dict]:
    _ensure()
    out: list[dict] = []
    for d in sorted(JOBS.iterdir(), reverse=True):
        f = d / "job.json"
        if not f.exists():
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if category_id and j.get("category_id") != category_id:
            continue
        out.append({
            "id": j.get("id"), "category_id": j.get("category_id"),
            "category_name": j.get("category_name"),
            "status": j.get("status"), "created_at": j.get("created_at"),
            "total": j.get("total", 0), "done": j.get("done", 0),
            "n_ok": sum(1 for p in j.get("pages", []) if p.get("status") == "ok"),
            "n_low": sum(1 for p in j.get("pages", [])
                         if p.get("status") == "low_confidence"),
            "n_rejected": sum(1 for p in j.get("pages", [])
                              if p.get("status") not in ("ok", "low_confidence")),
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 路径转 URL


def to_url(p: str | Path) -> str:
    """磁盘路径 -> 前端可访问的 URL（静态目录挂在 /files）。"""
    try:
        rel = Path(p).resolve().relative_to(DATA.resolve())
    except ValueError:
        return ""
    return "/files/" + rel.as_posix()
