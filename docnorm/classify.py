"""多模板分类：判断一页属于分类下的哪个模板。

背景：一个分类（如"支票"）下维护多个模板（不同银行、不同版本），
上传时用户**不拆分文件、不指定模板**，系统按页自动识别每页该归到哪个模板。

判据用配准层的**结构相关度**（`align_to_template` 返回的 `score`）：
它是在「结构特征图」（表格线 / 边框骨架）上算的皮尔逊相关，
天然忽略可变内容（金额、日期、签名、印章），只认版式骨架的空间分布。
同一银行内部版式一致 -> 对自家模板相关度高；不同银行内容块位置不同
（二维码在右下 vs 左下、付款账户在左上 vs 中间）-> 骨架空间错位，相关度低。

为什么用结构相关度而不是 OCR 读标题：版式差异是几何量，相关度直接度量它；
标题文字反而是最不稳定的（字体、缩放、扫描噪声、印章遮挡都影响 OCR）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .align import AlignConfig, align_to_template
from .rectify import normalize_illumination, to_gray


@dataclass
class ClassifyConfig:
    """识别门禁。两个条件同时满足才算「命中」：

    - min_score：最佳模板的结构相关度下限。太低说明这一页跟任何模板都不像
      （新银行、新版本、糊到骨架都认不出、或根本不是这个分类的单据）。
    - min_margin：最佳分与次高分的边距。两个模板骨架相近时分数会很接近，
      强行取最高者容易误判；要求拉开边距，拿不准就宁可判 unknown。
    """
    min_score: float = 0.35
    min_margin: float = 0.05


@dataclass
class ClassifyResult:
    template_id: str | None          # 命中的模板 ID；unknown 时为 None
    score: float = -1.0              # 最佳结构相关度
    second_score: float = -1.0       # 次高结构相关度（用于看区分度）
    scores: dict = field(default_factory=dict)   # 每个模板的分数，便于审计
    align: object = None             # 命中模板的 AlignResult（含对齐矩阵）
    matched: bool = False            # 是否命中（score/margin 都过门禁）
    reason: str = ""                 # 未命中原因


def classify_page(page_bgr, templates: dict, cfg: ClassifyConfig | None = None
                  ) -> ClassifyResult:
    """判断一页属于哪个模板。

    参数：
        page_bgr：输入页（BGR 图）。
        templates：{template_id: Template}，一个分类下的候选模板。
        cfg：门禁配置。

    返回 ClassifyResult。matched=False 时调用方把该页判为 rejected（供审计），
    不产出标准图、不裁块。
    """
    cfg = cfg or ClassifyConfig()
    if not templates:
        return ClassifyResult(reason="分类下没有模板")

    page_gray = normalize_illumination(to_gray(page_bgr))
    scores: dict = {}
    best_id, best_score = None, -1.0
    best_align = None

    for tid, tpl in templates.items():
        res = align_to_template(
            tpl.struct, page_bgr, tpl.canvas, AlignConfig(),
            page_gray=page_gray,
            template_lines=tpl.lines(),
            template_skew_deg=tpl.skew_deg,
        )
        s = float(res.score)
        scores[tid] = round(s, 6)
        if s > best_score:
            best_score, best_id, best_align = s, tid, res

    # 次高分：从其余模板里取最高
    second = -1.0
    for tid, s in scores.items():
        if tid != best_id and s > second:
            second = s

    matched = (best_id is not None
               and best_score >= cfg.min_score
               and (best_score - second) >= cfg.min_margin)
    reason = ""
    if best_id is None:
        reason = "无候选模板"
    elif not matched:
        if best_score < cfg.min_score:
            reason = (f"最佳相关度 {best_score:.3f} 低于门限 {cfg.min_score}，"
                      "该页不属于本分类任一模板")
        else:
            reason = (f"最佳 {best_score:.3f} 与次高 {second:.3f} 差距不足 "
                      f"{cfg.min_margin}，模板骨架过于接近，无法可靠区分")

    return ClassifyResult(
        template_id=best_id if matched else None,
        score=round(best_score, 6),
        second_score=round(second, 6),
        scores=scores,
        align=best_align,
        matched=matched,
        reason=reason,
    )
