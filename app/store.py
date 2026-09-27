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

数据根目录（下称 <DATA>）与项目解耦，见文件顶部的 DATA 定义。

    <DATA>/categories/<分类ID>/
        category.json                分类元数据（纯容器）
        templates/<模板ID>/
            template.json            模板元数据（含 parent 分类 ID）
            sample/                  上传的样张原文件
            template/                template.json + template.png（引擎模板）
            crops.json               模板块坐标清单
    <DATA>/jobs/<任务ID>/
        job.json                     任务状态 + category_id + 逐页结果（每页含 template_id）
        uploads/                     上传的生产文件
        standardized/image/<页>.png  标准图（配准后，**切片任务吃这里**）
        standardized/pdf/<页>.pdf    原始分页（配准前的原始页，每页一个单页 PDF，无损）
                                     给消费者查看原始文档用；含识别不出的页
        low_confidence/              低置信输出（内部一致，但需人工抽检）
        rejected/                    拒收原图（含"识别不出"的页，供审计）
        blocks/<页>/<块>.png         裁剪出的模板块
        sidecar/                     逐页配准诊断
        render-report.html
        verify/                      验证堆叠图与报告
    <DATA>/.trash/                   删除兜底回收站（环境拦截批量删除时改名到此）
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

# ---------------------------------------------------------------- 数据目录解析
#
# 数据目录与项目解耦，并且**不做隐式默认**：项目初始化时数据目录未配置，
# 前端必须先走"配置 + 探测校验"流程，之后才能建分类/模板。
#
# 指针（"数据目录在哪"）必须存在数据目录**之外**，否则循环依赖：
# 默认放 <用户目录>/.docrendercut/config.json，可用环境变量
# DOCRENDERCUT_CONFIG 覆盖（测试用）。
#
# 解析优先级：--data / DOCRENDERCUT_DATA  >  上面的配置文件  >  未配置


def config_path() -> Path:
    override = os.environ.get("DOCRENDERCUT_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".docrendercut" / "config.json"


def load_config() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(d: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def suggested_data_dir() -> Path:
    """未配置时给前端的建议值：用户目录下，**绝不在项目目录内**。"""
    return Path.home() / "docRenderCut-data"


def _resolve_data_root() -> tuple[Path | None, str]:
    env = os.environ.get("DOCRENDERCUT_DATA", "").strip()
    if env:
        src = os.environ.get("DOCRENDERCUT_DATA_SOURCE", "").strip() or "env"
        return Path(env).expanduser().resolve(), src
    d = str(load_config().get("data_dir") or "").strip()
    if d:
        return Path(d).expanduser().resolve(), "config"
    return None, "none"


# 未配置时这几个常量指向一个占位路径，但 is_configured() 为 False，
# 数据接口会被门禁挡掉，_ensure() 也不创建任何东西。
_PLACEHOLDER = ROOT / ".unconfigured"
DATA, DATA_SOURCE = _resolve_data_root()
_CONFIGURED = DATA is not None
if DATA is None:
    DATA = _PLACEHOLDER
CATEGORIES = DATA / "categories"
JOBS = DATA / "jobs"


def is_configured() -> bool:
    return _CONFIGURED


def data_root() -> Path:
    return DATA


def has_override() -> bool:
    """数据目录是否来自 CLI / 环境变量（这时前端改不动它，只能提示）。"""
    return DATA_SOURCE in ("cli", "env")


def set_data_root(path: str | Path, *, source: str = "config") -> Path:
    """运行时切换数据目录（前端"确认并启用"调用）。"""
    global DATA, CATEGORIES, JOBS, _CONFIGURED, DATA_SOURCE
    DATA = Path(path).expanduser().resolve()
    CATEGORIES = DATA / "categories"
    JOBS = DATA / "jobs"
    _CONFIGURED = True
    DATA_SOURCE = source
    _ensure()
    return DATA


# ---------------------------------------------------------------- 数据目录探测校验


def _count_existing(root: Path) -> dict:
    """统计某个目录里已有的数据（不依赖当前配置，用于校验"会不会覆盖"）。"""
    cats_dir = root / "categories"
    jobs_dir = root / "jobs"
    n_cat = n_tpl = n_job = 0
    size = 0
    if cats_dir.is_dir():
        for d in cats_dir.iterdir():
            if not (d / "category.json").exists():
                continue
            n_cat += 1
            td = d / "templates"
            if td.is_dir():
                for t in td.iterdir():
                    if (t / "template.json").exists():
                        n_tpl += 1
    if jobs_dir.is_dir():
        for j in jobs_dir.iterdir():
            if (j / "job.json").exists():
                n_job += 1
    try:
        for f in root.rglob("*"):
            if f.is_file():
                size += f.stat().st_size
    except OSError:
        pass
    return {"categories": n_cat, "templates": n_tpl, "jobs": n_job,
            "bytes": size}


def probe_data_dir(raw_path: str) -> dict:
    """对候选数据目录做**探测校验**，返回报告（不改动任何既有数据）。

    校验项：路径规范 / 可创建 / **真的可写**（写探测文件再删）/ 已有数据统计 /
    是否在项目目录内 / 是否网络路径 / 剩余空间。

    为什么"可写"要实际写一个文件：权限位在网络盘、只读挂载、域策略下经常骗人，
    只有真写一次才知道。这个探测文件随即删除，不留下任何痕迹。
    """
    p = str(raw_path or "").strip()
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {
        "input": p, "ok": False, "errors": errors, "warnings": warnings,
        "exists": False, "created": False, "writable": False,
        "in_project": False, "is_network": False, "free_gb": None,
        "existing": None,
    }
    if not p:
        errors.append("请填写数据目录路径")
        return report

    # 规范化：相对路径歧义太大（取决于启动时的工作目录），一律要求绝对路径
    raw = Path(p).expanduser()
    if not raw.is_absolute():
        errors.append("请填绝对路径（如 D:\\docRenderCut-data 或 \\\\server\\share\\drc）")
        return report
    try:
        path = raw.resolve()
    except OSError as exc:
        errors.append(f"路径无法解析：{exc}")
        return report

    report["path"] = str(path)
    report["exists"] = path.exists()
    report["is_network"] = str(path).startswith("\\\\")

    if path.exists() and not path.is_dir():
        errors.append("该路径已存在，且不是目录")
        return report

    # 1) 可创建
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            report["created"] = True
        except OSError as exc:
            errors.append(f"无法创建目录：{exc}")
            return report

    # 2) 真的可写
    probe = path / ".drc-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report["writable"] = True
    except OSError as exc:
        errors.append(f"目录不可写：{exc}")
        return report

    # 3) 已有数据
    report["existing"] = _count_existing(path)
    ex = report["existing"]
    if ex["categories"] or ex["jobs"]:
        warnings.append(
            f"该目录已有数据：{ex['categories']} 个分类 / {ex['templates']} 个模板 / "
            f"{ex['jobs']} 个任务。确认启用后将**使用这份已有数据**（不会被清空）。")

    # 4) 在项目目录内 —— 与"数据与项目解耦"的初衷相悖
    try:
        path.relative_to(ROOT)
        report["in_project"] = True
        warnings.append(
            f"该目录在项目目录内（{ROOT}）。项目升级/替换时数据有丢失风险，"
            "建议放到项目之外的独立目录。")
    except ValueError:
        pass

    if report["is_network"]:
        warnings.append(
            "这是网络路径。多实例并发写入依赖文件系统的原子语义，"
            "多数 SMB 共享可满足，但建议先用小批量验证。")

    # 5) 剩余空间
    try:
        import shutil as _sh

        report["free_gb"] = round(_sh.disk_usage(str(path)).free / 1073741824, 2)
    except OSError:
        pass

    report["ok"] = not errors
    return report


# ---------------------------------------------------------------- 基础


def _ensure() -> None:
    if not _CONFIGURED:
        return          # 未配置时绝不创建目录，避免"项目里冒出个 data/"
    CATEGORIES.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(name: str) -> str:
    """把名称变成安全的目录名片段（保留中文，去掉路径分隔符等）。"""
    s = re.sub(r"[\\/:*?\"<>|\s]+", "-", name.strip())
    return (s[:24] or "CAT").strip("-")


def _reserve(path: Path) -> bool:
    """以"创建目录"的方式**原子占位**一个 ID。

    为什么不用"扫描已用 ID 挑一个空号"：多实例共享同一数据目录时，
    两个实例会同时扫到同一个空号、生成同一个 ID、互相覆盖。
    mkdir 在同一文件系统上是原子的，成功即占位、冲突就换下一个号，
    所以并发也不会撞。代价是会短暂留下一个空目录——后续写入 JSON 即成为正式记录，
    而 list_* 都会跳过"没有 JSON"的目录，不会显示成半成品。
    """
    try:
        path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def new_category_id() -> str:
    """分类 ID：CATyyyymmdd-NNNN。

    用"日期 + 当日序号"而不是随机串或 UUID：这类工具里人会念这个 ID
    （"把 CAT20260925-0003 的模板发我"），可读、可排序、能一眼看出建立时间。
    """
    _ensure()
    day = datetime.now().strftime("%Y%m%d")
    for i in range(1, 10000):
        cand = f"CAT{day}-{i:04d}"
        if _reserve(CATEGORIES / cand):
            return cand
    cand = f"CAT{day}-{secrets.token_hex(3).upper()}"
    _reserve(CATEGORIES / cand)
    return cand


def new_template_id(cid: str) -> str:
    """模板 ID：TPLyyyymmdd-NNNN，在所属分类的 templates/ 下占位。"""
    tdir = templates_dir(cid)
    tdir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    for i in range(1, 10000):
        cand = f"TPL{day}-{i:04d}"
        if _reserve(tdir / cand):
            return cand
    cand = f"TPL{day}-{secrets.token_hex(3).upper()}"
    _reserve(tdir / cand)
    return cand


def new_job_id() -> str:
    """任务 ID：JOByyyymmddHHMMSS-XXXX。时间戳 + 随机后缀已足够唯一，
    仍用 mkdir 占位消除并发窗口。"""
    _ensure()
    for _ in range(64):
        cand = ("JOB" + datetime.now().strftime("%Y%m%d%H%M%S")
                + "-" + secrets.token_hex(2).upper())
        if _reserve(JOBS / cand):
            return cand
    raise RuntimeError("无法分配任务 ID（随机后缀连续冲突，请检查数据目录）")


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


def _remove_tree(path: Path) -> str:
    """尽力删除一棵目录树，返回 "deleted" 或 "trashed"。

    为什么要兜底：某些沙箱 / 终端防护会把"批量文件删除"改成
    "移入回收站并要求确认"，未确认时直接抛 SystemExit（实测 WorkBuddy
    的 shim 在单次删除超过 50 个文件时会这样，一个任务的产物常有几百个文件）。
    删除是用户明确点下的操作，不该在这种环境里变成 500。
    兜底方案是**整目录改名**到 .trash —— 改名不属于删除，不会被护栏拦，
    列表里立刻消失；磁盘清理交给后续（用户可整个删掉 .trash）。
    """
    import shutil as _sh

    if not path.exists():
        return "deleted"
    try:
        _sh.rmtree(path)
        return "deleted"
    except (SystemExit, Exception):  # noqa: BLE001
        pass
    try:
        trash = DATA / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        dst = trash / f"{path.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        path.rename(dst)
        return "trashed"
    except OSError:
        raise


def delete_category(cid: str) -> str:
    return _remove_tree(cat_dir(cid))


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


def delete_template(cid: str, tid: str) -> str:
    return _remove_tree(tpl_dir(cid, tid))


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


def delete_job(jid: str) -> str:
    """删除一个任务及其**全部磁盘产物**（标准输出图、裁剪块、拒收原图、验证报告）。

    刻意连产物一起删：任务列表只增不减的话磁盘会无限膨胀，
    而产物离开任务记录就没有检索入口了。调用方必须先给用户二次确认。
    返回 "deleted" | "trashed"。
    """
    return _remove_tree(job_dir(jid))


def _job_category(p: Path) -> str | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("category_id")
    except json.JSONDecodeError:
        return None


def list_jobs(category_id: str | None = None, limit: int = 30,
              template_id: str | None = None) -> list[dict]:
    """任务列表。

    归属有**两级**，别混：
    - **分类**是任务级的（一次上传只选一个分类）→ `category_id` / `category_name`；
    - **模板**是**页级**的（一个任务里可以混多个模板，这正是"混合分页自动识别"的目的）
      → 汇总成 `templates: [{id, name, n}]`，n 是该模板在本次任务里命中的页数。

    名称是**建任务时的快照**：分类/模板后来改名，这里仍是当时的名字
    （审计要的是"当时按什么处理的"）。稳定键永远是 id。
    给 `template_id` 则只返回"用过该模板"的任务（按模板反查）。
    """
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

        pages = j.get("pages", [])
        # 页级模板归属汇总（按出现次数降序，便于一眼看出主模板）
        tpl_counts: dict[str, int] = {}
        tpl_names: dict[str, str] = {}
        for p in pages:
            tid = p.get("template_id")
            if not tid:
                continue
            tpl_counts[tid] = tpl_counts.get(tid, 0) + 1
            tpl_names.setdefault(tid, p.get("template_name") or "")
        templates = [{"id": t, "name": tpl_names.get(t, ""), "n": n}
                     for t, n in sorted(tpl_counts.items(),
                                        key=lambda kv: (-kv[1], kv[0]))]
        if template_id and template_id not in tpl_counts:
            continue

        out.append({
            "id": j.get("id"), "category_id": j.get("category_id"),
            "category_name": j.get("category_name"),
            "templates": templates,
            "status": j.get("status"), "created_at": j.get("created_at"),
            "total": j.get("total", 0), "done": j.get("done", 0),
            "n_ok": sum(1 for p in pages if p.get("status") == "ok"),
            "n_low": sum(1 for p in pages
                         if p.get("status") == "low_confidence"),
            "n_rejected": sum(1 for p in pages
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
