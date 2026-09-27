/* docRenderCut 前端：分类 -> 模板 -> 坐标块 -> 批量识别渲染裁剪 */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const state = {
  cats: [],
  cat: null,          // 当前分类
  templates: [],      // 当前分类下的模板列表
  tpl: null,          // 当前选中的模板 {id, name, ...}
  crops: { items: [] },
  tplImg: null,       // 模板图 {img, canvas, linesX, linesY, url}
  zoom: 0.4,
  edit: null,         // 正在编辑的框 {id, name, roi, safe_margin, isNew, dirty}
  showAll: false,
  hoverId: null,
  filter: 'all',      // 任务记录筛选：all | ok | low_confidence | rejected
  jmJob: null,        // 任务弹窗里当前的任务
  jmFilter: 'all',
  probe: null,        // 建模板弹窗里最后一次样张探测结果（不落盘）
  sizeTouched: false, // 用户是否手工改过画布宽高（改过就不再随 DPI 自动联动）
  editTplId: null,    // 建模板弹窗当前是"编辑已有模板"还是"新建"（null = 新建）
  setupLocked: false, // 初始化未完成时锁住数据目录弹窗，不允许关掉
  poll: null,
};

const FILTER_LABEL = { all: '全部', ok: '合格', low_confidence: '低置信', rejected: '拒收' };

/* ---------------------------------------------------------------- API */

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status + ' ' + r.statusText;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}

const get = (p) => api(p);
const post = (p, body) => api(p, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});
const del = (p) => api(p, { method: 'DELETE' });

function postForm(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = { detail: xhr.responseText }; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error((data && data.detail) || ('HTTP ' + xhr.status)));
    };
    xhr.onerror = () => reject(new Error('网络错误'));
    xhr.send(form);
  });
}

/* ---------------------------------------------------------------- 通用 UI */

function openModal(sel) { $(sel).hidden = false; }
function closeModal(el) { (el.closest('.modal') || el).hidden = true; }

/** 初始化流程未完成时，配置弹窗**不允许被关掉**（否则用户会对着一个用不了的界面）。 */
function canDismiss(m) {
  return !(m.id === 'setupModal' && state.setupLocked);
}
$$('[data-close]').forEach((b) => b.addEventListener('click', () => {
  const m = b.closest('.modal');
  if (m && !canDismiss(m)) return;
  closeModal(b);
}));
$$('.modal').forEach((m) => m.addEventListener('mousedown', (e) => {
  if (e.target === m && canDismiss(m)) m.hidden = true;
}));
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  $$('.modal').forEach((m) => { if (canDismiss(m)) m.hidden = true; });
});

function lightbox(title, url) {
  $('#lbTitle').textContent = title;
  $('#lbImg').src = url;
  $('#lbOpen').href = url;
  openModal('#lightbox');
}

function setMsg(sel, kind, html) {
  const el = $(sel);
  if (!html) { el.hidden = true; return; }
  el.hidden = false;
  el.className = 'msg ' + kind;
  el.innerHTML = html;
}

/* ---------------------------------------------------------------- 顶栏 */

(async function health() {
  try {
    await get('/api/health');
    $('#health').textContent = '服务正常';
  } catch (e) {
    $('#health').textContent = '服务不可用';
    $('#health').classList.add('bad');
  }
})();

/* ---------------------------------------------------------------- 数据目录（初始化 + 配置）

 * 项目初始化时没有任何数据：必须先配置数据目录并**探测校验**通过，
 * 才能进入分类 / 模板的创建。未配置时后端也会拒绝所有数据接口（409），
 * 所以这里不是"把界面藏起来"而已。
 */

const SOURCE_LABEL = {
  cli: '命令行 --data',
  env: '环境变量 DOCRENDERCUT_DATA',
  config: '界面配置（已写入指针文件）',
  none: '未配置',
};

let setupReport = null;

function fmtBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return v + ' B';
  if (v < 1048576) return (v / 1024).toFixed(1) + ' KB';
  if (v < 1073741824) return (v / 1048576).toFixed(1) + ' MB';
  return (v / 1073741824).toFixed(2) + ' GB';
}

function renderSetup(st) {
  setupReport = null;
  $('#btnApplyDir').disabled = true;
  $('#setupReport').hidden = true;
  setMsg('#setupMsg', '', '');

  const locked = !!st.has_override;
  $('#setupLock').hidden = !locked;
  if (locked) {
    $('#setupLock').innerHTML =
      '数据目录已由启动参数指定（<code>--data</code> 或 <code>DOCRENDERCUT_DATA</code>），'
      + '界面无法修改。如需更换，请用启动参数重启服务。';
  }

  if (st.configured) {
    $('#setupTitle').textContent = '数据目录';
    $('#setupIntro').hidden = true;
    const s = st.stats || {};
    $('#setupCurrent').hidden = false;
    $('#setupCurrent').innerHTML = `
      <div>当前数据目录：<b>${esc(st.data_dir)}</b></div>
      <div class="dim">来源：${esc(SOURCE_LABEL[st.data_source] || st.data_source)}
        · 已有 ${s.categories || 0} 分类 / ${s.templates || 0} 模板 / ${s.jobs || 0} 任务
        · 占用 ${fmtBytes(s.bytes)}</div>
      <div class="dim">指针文件：<code>${esc(st.config_path)}</code></div>`;
    $('#setupPath').value = st.data_dir || '';
  } else {
    $('#setupTitle').textContent = '初始化 · 配置数据目录';
    $('#setupIntro').hidden = false;
    $('#setupCurrent').hidden = true;
    $('#setupPath').value = st.suggested || '';
  }

  $('#setupPath').disabled = locked;
  $('#btnProbeDir').disabled = locked;
}

function renderProbeReport(rep) {
  const box = $('#setupReport');
  const ex = rep.existing || {};
  const rows = [
    `路径 <b>${esc(rep.path || rep.input || '')}</b>`,
    `存在：${rep.exists ? '是' : '否'}${rep.created ? '（已自动创建）' : ''}`
      + ` · 可写：${rep.writable ? '✓ 是' : '✗ 否'}`,
    rep.free_gb != null ? `剩余空间：${rep.free_gb} GB` : null,
    (rep.existing && (ex.categories || ex.jobs))
      ? `已有数据：<b>${ex.categories} 分类 / ${ex.templates} 模板 / ${ex.jobs} 任务</b>`
        + `（${fmtBytes(ex.bytes)}）`
      : '目录为空（全新数据）',
    rep.is_network ? '网络路径' : null,
    rep.in_project ? '⚠ 位于项目目录内' : null,
  ].filter(Boolean);

  box.hidden = false;
  box.className = 'setup-report ' + (rep.ok ? 'ok' : 'bad');
  box.innerHTML = `
    <div class="probe-grid">${rows.map((r) => `<div>${r}</div>`).join('')}</div>
    ${(rep.errors || []).length
      ? `<ul class="probe-warn">${rep.errors.map((w) => `<li>✗ ${esc(w)}</li>`).join('')}</ul>` : ''}
    ${(rep.warnings || []).length
      ? `<ul class="probe-warn">${rep.warnings.map((w) => `<li>⚠ ${esc(w)}</li>`).join('')}</ul>` : ''}
    <div class="probe-tip">${rep.ok
      ? '校验通过。点「② 确认并启用」后数据目录立即生效。'
      : '校验未通过，请修正后重新校验。'}</div>`;
}

$('#setupPath').addEventListener('input', () => {
  if (setupReport) { setupReport = null; $('#setupReport').hidden = true; }
  $('#btnApplyDir').disabled = true;
});

$('#btnProbeDir').addEventListener('click', async () => {
  const p = $('#setupPath').value.trim();
  if (!p) return setMsg('#setupMsg', 'err', '请填写数据目录路径');
  $('#btnProbeDir').disabled = true;
  $('#btnApplyDir').disabled = true;
  setMsg('#setupMsg', 'info', '正在校验（写一个探测文件再删除，不影响已有数据）…');
  try {
    const rep = await post('/api/setup/probe', { path: p });
    setupReport = rep;
    renderProbeReport(rep);
    $('#btnApplyDir').disabled = !rep.ok;
    setMsg('#setupMsg', '', '');
  } catch (e) {
    setupReport = null;
    $('#setupReport').hidden = true;
    setMsg('#setupMsg', 'err', '校验失败：' + esc(e.message));
  }
  $('#btnProbeDir').disabled = false;
});

$('#btnApplyDir').addEventListener('click', async () => {
  $('#btnApplyDir').disabled = true;
  setMsg('#setupMsg', 'info', '正在启用…');
  try {
    const r = await post('/api/setup/configure', { path: $('#setupPath').value.trim() });
    state.setupLocked = false;
    setMsg('#setupMsg', 'ok', `已启用：${esc(r.data_dir)} · 正在重新加载…`);
    setTimeout(() => location.reload(), 800);
  } catch (e) {
    setMsg('#setupMsg', 'err', '启用失败：' + esc(e.message));
    $('#btnApplyDir').disabled = false;
  }
});

$('#btnDataDir').addEventListener('click', async () => {
  const st = await get('/api/setup/state');
  state.setupLocked = false;
  $('#setupCancel').hidden = false;
  $('#setupX').hidden = false;
  renderSetup(st);
  openModal('#setupModal');
});

/* ---------------------------------------------------------------- 分类列表 */

async function loadCats() {
  state.cats = await get('/api/categories');
  renderCats();
}

function renderCats() {
  const q = ($('#catSearch').value || '').trim().toLowerCase();
  const list = state.cats.filter((c) =>
    !q || (c.name || '').toLowerCase().includes(q) || (c.id || '').toLowerCase().includes(q));
  $('#catCount').textContent = `${state.cats.length} 个分类`;
  $('#catList').innerHTML = list.map((c) => `
    <li data-id="${esc(c.id)}" class="${state.cat && state.cat.id === c.id ? 'active' : ''}">
      <div class="n">${esc(c.name)}</div>
      <div class="m">${esc(c.id)} · ${c.n_templates || 0} 个模板</div>
    </li>`).join('') || '<li class="dim" style="cursor:default">暂无分类</li>';
  $$('#catList li[data-id]').forEach((li) => li.addEventListener('click', () => selectCat(li.dataset.id)));
}

$('#catSearch').addEventListener('input', renderCats);

/* ---------------------------------------------------------------- 创建分类 */

$('#btnNewCat').addEventListener('click', () => {
  $('#fName').value = '';
  $('#createMsg').hidden = true;
  $('#btnDoCreate').disabled = false;
  openModal('#modalCreate');
});

$('#btnDoCreate').addEventListener('click', async () => {
  const name = $('#fName').value.trim();
  if (!name) return setMsg('#createMsg', 'err', '请填写分类名称');
  const btn = $('#btnDoCreate');
  btn.disabled = true;
  setMsg('#createMsg', 'info', '创建中…');
  try {
    const fd = new FormData();
    fd.append('name', name);
    const cat = await postForm('/api/categories', fd);
    closeModal($('#btnDoCreate'));
    await loadCats();
    await selectCat(cat.id);
  } catch (e) {
    setMsg('#createMsg', 'err', '创建失败：' + esc(e.message));
    btn.disabled = false;
  }
});

/* ---------------------------------------------------------------- 创建模板 */

/* 打开建模板弹窗：t = null 表示新建；否则表示"编辑该模板"（可换样张、改参数） */
function openTplModal(t) {
  state.editTplId = t ? t.id : null;
  state.probe = null;
  // 编辑态把已存参数视为"用户已确认过"，别让 DPI 联动把画布改掉
  state.sizeTouched = !!t;

  $('#tplModalTitle').textContent = t ? `编辑模板 · ${t.id}` : '创建模板';
  $('#tfIdHint').textContent = t
    ? '模板 ID 不可改；改完点「保存并重建模板」才落盘。'
    : '模板 ID 由后台自动生成（形如 TPL20260925-0001）。';

  $('#tfName').value = t ? (t.name || '') : '';
  $('#tfSample').value = '';
  $('#tfProbe').hidden = true;
  $('#tfProbe').innerHTML = '';
  $('#tfDpi').value = t ? (t.dpi || 200) : 200;
  $('#tfW').value = (t && t.canvas && t.canvas.width) || 1600;
  $('#tfH').value = (t && t.canvas && t.canvas.height) || 3400;
  $('#tfPaper').value = (t && t.paper_mode) || 'off';
  $('#tfCorners').value = t && t.corners ? t.corners.join(',') : '';
  $('#tfCornersRow').hidden = $('#tfPaper').value !== 'corners';
  $('#tfInkBias').value = t ? (t.ink_bias ?? 0) : 0;
  $('#tfInkDark').value = t ? (t.ink_dark_bias ?? 25) : 25;
  $('#tfDeskew').checked = t ? t.deskew !== false : true;

  // 编辑态：已存样张的物理尺寸算已知量，这样改 DPI 仍能联动画布
  if (t && t.sample && t.sample.sample_mm) {
    state.probe = { sample_mm: t.sample.sample_mm };
  }
  renderCurSample(t);

  tplSizeHint();
  $('#tplMsg').hidden = true;
  $('#btnDoCreateTpl').hidden = false;
  $('#btnDoCreateTpl').disabled = false;
  $('#btnDoCreateTpl').textContent = t ? '保存并重建模板' : '确认构建模板';
  $('#btnPreviewTpl').hidden = true;
  $('#btnFinishTpl').hidden = true;
  openModal('#modalTpl');
}

/* 编辑态下把"当前样张"摆出来：不换文件就沿用，换文件会重新识别 */
function renderCurSample(t) {
  const box = $('#tfCurSample');
  if (!t || !t.sample || !t.sample.filename) {
    box.hidden = true;
    box.innerHTML = '';
    return;
  }
  const s = t.sample;
  const size = (s.sample_size || []).join('×') || '—';
  const mm = (s.sample_mm || []).map((v) => Number(v).toFixed(1)).join('×');
  box.hidden = false;
  box.innerHTML =
    `<b>当前样张</b>：${esc(s.filename)} · ${size} px` + (mm ? ` · ${mm} mm` : '') +
    '<div class="dim sm">不选新文件就沿用这张；选了会覆盖它，并<b>重新识别参数</b>。</div>';
}

$('#btnNewTpl').addEventListener('click', () => {
  if (!state.cat) return;
  openTplModal(null);
});

$('#btnEditTpl').addEventListener('click', () => {
  if (!state.cat || !state.tpl) return;
  openTplModal(state.tpl);
});

$('#tfPaper').addEventListener('change', (e) => {
  $('#tfCornersRow').hidden = e.target.value !== 'corners';
});

/* ---- 样张探测：选文件即识别尺寸，**不落盘**；可反复换文件重测 ---- */

async function probeSample() {
  const f = $('#tfSample').files[0];
  if (!f) {
    // 编辑态没选新文件：保留"已存样张的物理尺寸"，DPI 联动仍可用
    state.probe = null;
    const t = state.editTplId
      ? (state.templates || []).find((x) => x.id === state.editTplId) : null;
    if (t && t.sample && t.sample.sample_mm) {
      state.probe = { sample_mm: t.sample.sample_mm };
    }
    $('#tfProbe').hidden = true;
    return;
  }
  const fd = new FormData();
  fd.append('sample', f);
  fd.append('dpi_hint', $('#tfDpi').value || '200');
  setMsg('#tplMsg', 'info', '正在识别样张尺寸…（只解析，不落盘）');
  try {
    const p = await postForm(
      `/api/categories/${state.cat.id}/templates/probe`, fd);
    state.probe = p;
    // 自动填充：DPI 用识别值，画布 = 物理尺寸 × DPI
    $('#tfDpi').value = p.suggest.dpi;
    $('#tfW').value = p.suggest.width;
    $('#tfH').value = p.suggest.height;
    state.sizeTouched = false;
    renderProbe(p);
    tplSizeHint();
    setMsg('#tplMsg', '', '');
  } catch (e) {
    state.probe = null;
    $('#tfProbe').hidden = true;
    setMsg('#tplMsg', 'err', '样张识别失败：' + esc(e.message));
  }
}

$('#tfSample').addEventListener('change', probeSample);

const DPI_SOURCE_LABEL = {
  'pdf-page': '矢量 PDF（物理尺寸取自页面，无原始 DPI）',
  'image-meta': '图片内嵌 DPI 元数据',
  'paper-guess': '按宽高比推断的纸张尺寸',
  'unknown': '无（未能识别）',
};

function renderProbe(p) {
  const box = $('#tfProbe');
  const mm = p.sample_mm;
  const rows = [
    `类型 <b>${p.sample_kind === 'pdf' ? 'PDF' : '图片'}</b>`
      + (p.n_pages ? ` · ${p.n_pages} 页（只用第 1 页）` : ''),
    `样张像素 <b>${p.sample_size[0]}×${p.sample_size[1]}</b>`
      + ` · 宽高比 ${p.aspect}`,
    mm ? `物理尺寸 <b>${mm[0]}×${mm[1]} mm</b>`
       + (p.paper ? `（像 ${p.paper}）` : '') : '物理尺寸 <b>未知</b>',
    `识别 DPI <b>${p.detected_dpi ?? '—'}</b> —— 来源：${DPI_SOURCE_LABEL[p.dpi_source] || p.dpi_source}`,
    p.image_dpi ? `内嵌 DPI：${p.image_dpi[0]}×${p.image_dpi[1]}` : null,
  ].filter(Boolean);

  const warns = (p.warnings || []);
  const editing = !!state.editTplId;
  box.hidden = false;
  box.innerHTML = `
    <div class="probe-title">已识别新样张（尚未保存）</div>
    <div class="probe-grid">${rows.map((r) => `<div>${r}</div>`).join('')}</div>
    ${warns.length ? `<ul class="probe-warn">${warns.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}
    <div class="probe-tip">参数已按识别结果填好，可直接改；改完点「${editing ? '保存并重建模板' : '确认构建模板'}」才会真正落盘。
      想换样张重新识别，重新选文件即可。</div>`;
}

function tplSizeHint() {
  const w = +$('#tfW').value || 0, h = +$('#tfH').value || 0;
  if (!w || !h) { $('#tfSizeHint').textContent = ''; return; }
  const dpi = +$('#tfDpi').value || 200;
  $('#tfSizeHint').textContent =
    `画布 ${w}×${h} 像素，宽高比 ${(w / h).toFixed(3)}，` +
    `物理尺寸约 ${(w / dpi * 25.4).toFixed(1)}×${(h / dpi * 25.4).toFixed(1)} mm。上传样张后会核对宽高比。`;
}
// DPI 与画布联动：已知物理尺寸时，改 DPI 会按比例重算画布；
// 一旦用户手工改过 W/H，就不再自动覆盖（别跟用户的输入打架）。
$('#tfDpi').addEventListener('input', () => {
  const mm = state.probe && state.probe.sample_mm;
  if (mm && !state.sizeTouched) {
    const dpi = +$('#tfDpi').value || 200;
    $('#tfW').value = Math.max(1, Math.round(mm[0] / 25.4 * dpi));
    $('#tfH').value = Math.max(1, Math.round(mm[1] / 25.4 * dpi));
  }
  tplSizeHint();
});
['#tfW', '#tfH'].forEach((s) => $(s).addEventListener('input', () => {
  state.sizeTouched = true;
  tplSizeHint();
}));
tplSizeHint();

$('#btnDoCreateTpl').addEventListener('click', async () => {
  const name = $('#tfName').value.trim();
  const f = $('#tfSample').files[0];
  const editing = !!state.editTplId;
  if (!name) return setMsg('#tplMsg', 'err', '请填写模板名称');
  if (!f && !editing) return setMsg('#tplMsg', 'err', '请选择一张样张');

  const fd = new FormData();
  fd.append('name', name);
  fd.append('dpi', $('#tfDpi').value || '200');
  fd.append('width', $('#tfW').value || '1600');
  fd.append('height', $('#tfH').value || '3400');
  fd.append('paper_mode', $('#tfPaper').value);
  fd.append('corners', $('#tfCorners').value || '');
  fd.append('ink_bias', $('#tfInkBias').value || '0');
  fd.append('ink_dark_bias', $('#tfInkDark').value || '25');
  fd.append('deskew', $('#tfDeskew').checked ? 'true' : 'false');
  if (f) fd.append('sample', f);   // 编辑态不选文件 = 沿用原样张

  const btn = $('#btnDoCreateTpl');
  btn.disabled = true;
  setMsg('#tplMsg', 'info', (editing ? '正在按新参数重建模板…' : '正在按确认的参数构建模板…')
    + '（结构提取 + 表格线检测，稍慢）');
  try {
    const url = editing
      ? `/api/categories/${state.cat.id}/templates/${encodeURIComponent(state.editTplId)}/edit`
      : `/api/categories/${state.cat.id}/templates`;
    const tpl = await postForm(url, fd,
      (p) => setMsg('#tplMsg', 'info', `上传中 ${(p * 100).toFixed(0)}%`));

    const warns = tpl.warnings || [];
    const dk = (tpl.template && tpl.template.deskew) || {};
    const changed = tpl.changed || [];
    const nLines = tpl.template
      ? (tpl.template.n_lines_x + tpl.template.n_lines_y) : null;

    let third;
    if (editing && !tpl.rebuilt) {
      third = '未改动影响渲染的参数，模板图保持不变。';
    } else if (tpl.template && dk.applied) {
      third = `纠偏：已执行，残余倾斜 <b>${dk.residual_deg}°</b>（${dk.n_lines} 条线）`;
    } else if (tpl.template) {
      third = '<b>纠偏：未执行</b>（模板基准可能带倾斜）';
    } else {
      third = '';
    }

    setMsg('#tplMsg', warns.length ? 'err' : 'ok',
      `<b>${editing ? '模板已更新' : '模板已构建'}</b> —— ${esc(tpl.id)} · ${esc(tpl.name)}<br>` +
      `画布 ${tpl.canvas.width}×${tpl.canvas.height} @${tpl.dpi}dpi · ` +
      `样张 ${tpl.sample && tpl.sample.sample_size ? tpl.sample.sample_size.join('×') : '—'}` +
      (nLines !== null ? ` · 检测到表格线 ${nLines} 条` : '') + '<br>' +
      (editing && changed.length
        ? `本次改动：${changed.map((c) => esc(c)).join('；')}<br>` : '') +
      third +
      (warns.length ? `<ul>${warns.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : ''));
    $('#btnDoCreateTpl').hidden = true;
    $('#btnPreviewTpl').hidden = false;
    $('#btnFinishTpl').hidden = false;
    $('#btnPreviewTpl').onclick = () => lightbox('标准模板 · ' + tpl.name,
      (tpl.template.preview_url || tpl.template.template_url));
    state.newTplId = tpl.id;
  } catch (e) {
    setMsg('#tplMsg', 'err', (editing ? '保存失败：' : '构建失败：') + esc(e.message));
    btn.disabled = false;
  }
});

$('#btnFinishTpl').addEventListener('click', async () => {
  closeModal($('#btnFinishTpl'));
  const tid = state.newTplId;
  state.newTplId = null;
  await reloadCategory();
  if (tid) await selectTpl(tid);
});

/* ---------------------------------------------------------------- 选中分类 */

function resetBatchPane() {
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  $('#progressWrap').hidden = true;
  $('#progressBar').style.width = '0%';
  $('#progressText').textContent = '';
  clearBatchFiles();
  $('#btnRunBatch').disabled = false;
}

async function selectCat(cid) {
  resetBatchPane();
  state.tplImg = null;
  state.crops = { items: [] };
  state.tpl = null;
  state.edit = null;
  state.hoverId = null;
  state.jmJob = null;
  canvasNotice('模板加载中…');
  $('#cropList').innerHTML = '<div class="dim sm">加载中…</div>';
  $('#jobsList').innerHTML = '<div class="dim">加载中…</div>';
  $('#batchJobList').innerHTML = '<div class="dim">加载中…</div>';

  state.cat = await get(`/api/categories/${cid}`);
  state.templates = state.cat.templates || [];
  $('#empty').hidden = true;
  $('#catView').hidden = false;
  $('#catBar').hidden = false;   // 分类标题与操作按钮挂在顶栏，随选中态显示

  $('#catName').textContent = state.cat.name;
  $('#catMeta').innerHTML = `<code>${esc(state.cat.id)}</code> · ${state.templates.length} 个模板`;
  $('#batchScope').innerHTML =
    `上传目标分类：<b>${esc(state.cat.name)}</b> <code>${esc(state.cat.id)}</code>` +
    ` · ${state.templates.length} 个模板 · 系统自动识别每页归属`;

  $('#catWarn').hidden = true;
  $('#catWarn').innerHTML = '';

  renderCats();
  renderTplChips();
  syncUrl();

  if (state.templates.length) {
    await selectTpl(state.templates[0].id);
  } else {
    $('#tplStrip').hidden = true;
    state.tpl = null;
    $('#btnDelTpl').disabled = true;
    $('#btnEditTpl').disabled = true;
    canvasNotice('该分类下还没有模板，点「+ 新建模板」创建');
    $('#cropList').innerHTML = '<div class="dim sm">先创建模板</div>';
  }

  await refreshJobLists();
}

function renderTplChips() {
  const strip = $('#tplStrip');
  const chips = $('#tplChips');
  if (!state.templates.length) { strip.hidden = true; chips.innerHTML = ''; return; }
  strip.hidden = false;
  chips.innerHTML = state.templates.map((t) => `
    <button class="tpl-chip ${state.tpl && state.tpl.id === t.id ? 'on' : ''}"
            data-id="${esc(t.id)}">
      ${esc(t.name)}
      ${t.template_ready ? '' : ' <span class="dim">(未构建)</span>'}
    </button>`).join('');
  $$('#tplChips .tpl-chip').forEach((b) => b.addEventListener('click', () => selectTpl(b.dataset.id)));
}

async function selectTpl(tid) {
  if (!state.cat) return;
  state.tpl = state.templates.find((t) => t.id === tid) || null;
  if (!state.tpl) return;
  state.edit = null;
  state.hoverId = null;
  $('#btnDelTpl').disabled = false;
  $('#btnDelTpl').title = `删除模板「${state.tpl.name}」`;
  $('#btnEditTpl').disabled = false;
  $('#btnEditTpl').title =
    `编辑模板「${state.tpl.name}」：改参数，或换一张样张重新识别`;
  renderTplChips();
  canvasNotice('模板加载中…');
  loadTemplate();
  loadCrops();
  syncUrl();
}

/** 重新拉取分类（含模板列表），保持当前模板选中态。 */
async function reloadCategory() {
  if (!state.cat) return;
  const catId = state.cat.id;
  const curTpl = state.tpl ? state.tpl.id : null;
  state.cat = await get(`/api/categories/${catId}`);
  state.templates = state.cat.templates || [];
  $('#catName').textContent = state.cat.name;
  $('#catMeta').innerHTML = `<code>${esc(state.cat.id)}</code> · ${state.templates.length} 个模板`;
  $('#batchScope').innerHTML =
    `上传目标分类：<b>${esc(state.cat.name)}</b> <code>${esc(state.cat.id)}</code>` +
    ` · ${state.templates.length} 个模板 · 系统自动识别每页归属`;
  renderTplChips();
  if (curTpl && state.templates.some((t) => t.id === curTpl)) {
    await selectTpl(curTpl);
  } else if (state.templates.length) {
    await selectTpl(state.templates[0].id);
  } else {
    state.tpl = null;
    $('#tplStrip').hidden = true;
    $('#btnDelTpl').disabled = true;
    $('#btnEditTpl').disabled = true;
    canvasNotice('该分类下还没有模板');
  }
}

$('#btnDelCat').addEventListener('click', async () => {
  if (!state.cat) return;
  if (!confirm(`删除分类「${state.cat.name}」？\n分类下的所有模板、坐标清单与样张都会被删除，任务记录保留。`)) return;
  await del(`/api/categories/${state.cat.id}`);
  state.cat = null;
  state.tpl = null;
  resetBatchPane();
  state.tplImg = null;
  state.crops = { items: [] };
  $('#catView').hidden = true;
  $('#catBar').hidden = true;
  $('#empty').hidden = false;
  await loadCats();
});

$('#btnDelTpl').addEventListener('click', async () => {
  if (!state.cat || !state.tpl) return;
  const t = state.tpl;
  const nBlocks = (state.crops.items || []).length;
  if (!confirm(`删除模板「${t.name}」？\n它的模板图、样张与 ${nBlocks} 个坐标块都会被一并删除，不可恢复。\n（分类与任务记录保留。）`)) return;
  try {
    await del(`/api/categories/${state.cat.id}/templates/${encodeURIComponent(t.id)}`);
    state.tpl = null;
    state.tplImg = null;
    state.crops = { items: [] };
    state.edit = null;
    await reloadCategory();
  } catch (e) { alert('删除模板失败：' + e.message); }
});

/* ---------------------------------------------------------------- 模板与标注器 */

function canvasNotice(text, color = '#6b7280') {
  const cv = $('#tplCanvas');
  cv.width = 460; cv.height = 90;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = color;
  ctx.font = '14px -apple-system, "Microsoft YaHei", sans-serif';
  ctx.fillText(text, 14, 50);
}

async function loadTemplate() {
  if (!state.cat || !state.tpl) return;
  const catId = state.cat.id, tplId = state.tpl.id;
  try {
    const info = await get(`/api/categories/${catId}/templates/${tplId}/template`);
    if (!state.tpl || state.tpl.id !== tplId) return;
    canvasNotice(`模板加载中…（${(info.bytes / 1048576).toFixed(1)} MB）`);
    const img = new Image();
    img.src = info.preview_url || (info.template_url + '?t=' + Date.now());
    await img.decode();
    if (!state.tpl || state.tpl.id !== tplId) return;
    state.tplImg = {
      img, canvas: info.canvas,
      linesX: info.lines_x || [], linesY: info.lines_y || [],
      url: info.template_url,
    };
    const link = $('#linkFullTpl');
    link.href = info.template_url;
    link.hidden = false;
    $('#tplSizeHint').textContent =
      ` 画布 ${info.canvas.width}×${info.canvas.height} · 显示用降采样图`
      + `（原图 ${(info.bytes / 1048576).toFixed(1)} MB）`;
    fitZoom();
    draw();
  } catch (e) {
    if (!state.tpl || state.tpl.id !== tplId) return;
    state.tplImg = null;
    canvasNotice('模板未构建：' + e.message, '#b42318');
  }
}

function setZoom(z) {
  state.zoom = Math.max(0.1, Math.min(8, Math.round(z * 100) / 100));
  $('#zoom').value = state.zoom;
  $('#zoomVal').textContent = state.zoom.toFixed(2) + 'x';
}

function fitZoom() {
  if (!state.tplImg) return;
  const wrap = $('#canvasWrap');
  const avail = Math.max(240, wrap.clientWidth - 24 - PAD * 2);
  const availH = Math.max(240, wrap.clientHeight - 24 - PAD * 2);
  const z = Math.min(avail / state.tplImg.canvas.width,
    availH / state.tplImg.canvas.height, 1);
  setZoom(Math.max(0.1, z));
}

/* ================= 标注器：可编辑的坐标框 ================= */

function handlePoints(roi) {
  const [x, y, w, h] = roi;
  const x1 = x + w, y1 = y + h, cx = x + w / 2, cy = y + h / 2;
  return {
    nw: [x, y], n: [cx, y], ne: [x1, y],
    w: [x, cy], e: [x1, cy],
    sw: [x, y1], s: [cx, y1], se: [x1, y1],
  };
}

const HANDLE_CURSOR = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize',
};

function inside(p, roi) {
  const [x, y, w, h] = roi;
  return p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
}

function hitHandle(p, roi) {
  const tol = 7 / Math.max(state.zoom, 0.05);
  let best = null, bd = tol;
  for (const [name, [hx, hy]] of Object.entries(handlePoints(roi))) {
    const d = Math.hypot(p.x - hx, p.y - hy);
    if (d <= bd) { bd = d; best = name; }
  }
  return best;
}

function clampRoi(roi) {
  if (!state.tplImg) return roi;
  const cw = state.tplImg.canvas.width, ch = state.tplImg.canvas.height;
  let [x, y, w, h] = roi;
  w = Math.max(4, Math.min(Math.round(w), cw));
  h = Math.max(4, Math.min(Math.round(h), ch));
  x = Math.max(0, Math.min(Math.round(x), cw - w));
  y = Math.max(0, Math.min(Math.round(y), ch - h));
  return [x, y, w, h];
}

function clampPt(v, axis) {
  if (!state.tplImg) return v;
  const lim = axis === 'x' ? state.tplImg.canvas.width : state.tplImg.canvas.height;
  return Math.max(0, Math.min(v, lim));
}

function applyResize(roi, handle, px, py) {
  let [x, y, w, h] = roi;
  let x1 = x + w, y1 = y + h;
  if (handle.includes('w')) x = px;
  if (handle.includes('e')) x1 = px;
  if (handle.includes('n')) y = py;
  if (handle.includes('s')) y1 = py;
  return [Math.min(x, x1), Math.min(y, y1), Math.abs(x1 - x), Math.abs(y1 - y)];
}

function snapVal(v, axis) {
  if (!$('#ckSnap').checked || !state.tplImg) return v;
  const arr = axis === 'x' ? state.tplImg.linesX : state.tplImg.linesY;
  if (!arr || !arr.length) return v;
  const tol = Math.max(4, 8 / Math.max(state.zoom, 0.15));
  let best = v, bd = tol;
  for (const a of arr) { const d = Math.abs(a - v); if (d < bd) { bd = d; best = a; } }
  return best;
}

const PAD = 14;

function px(v) { return v * state.zoom + PAD; }

function draw() {
  if (!state.tplImg) return;
  const cv = $('#tplCanvas'), ctx = cv.getContext('2d');
  const s = state.zoom, t = state.tplImg;
  const W = Math.max(1, Math.round(t.canvas.width * s));
  const H = Math.max(1, Math.round(t.canvas.height * s));
  cv.width = W + PAD * 2; cv.height = H + PAD * 2;
  ctx.imageSmoothingEnabled = s < 1;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#eceff3'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(PAD, PAD, W, H);
  ctx.drawImage(t.img, PAD, PAD, W, H);

  if ($('#ckLines').checked) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(30,111,235,.35)';
    ctx.beginPath();
    t.linesX.forEach((x) => { ctx.moveTo(px(x), PAD); ctx.lineTo(px(x), PAD + H); });
    t.linesY.forEach((y) => { ctx.moveTo(PAD, px(y)); ctx.lineTo(PAD + W, px(y)); });
    ctx.stroke();
  }

  const edit = state.edit;
  const editId = edit && edit.id;
  const items = state.crops.items || [];

  if (state.showAll) {
    items.forEach((it) => {
      if (editId && it.id === editId) return;
      drawSavedBox(ctx, it, s);
    });
  }

  if (edit && edit.dirty && editId) {
    const it = items.find((x) => x.id === editId);
    if (it) drawSavedBox(ctx, it, s, 0.3);
  }

  if (state.hoverId && state.hoverId !== editId) {
    const it = items.find((x) => x.id === state.hoverId);
    if (it) {
      const [x, y, w, h] = it.roi;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 2; ctx.strokeStyle = '#e0651a';
      ctx.strokeRect(px(x), px(y), w * s, h * s);
      ctx.setLineDash([]);
    }
  }

  if (edit) drawEditBox(ctx, edit, s);
}

function drawSavedBox(ctx, it, s, alpha = 1) {
  const [x, y, w, h] = it.roi;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#e0651a';
  ctx.strokeRect(px(x), px(y), w * s, h * s);
  if (alpha >= 1) label(ctx, it.name || it.id, px(x), px(y), '#e0651a');
  ctx.globalAlpha = 1;
}

function drawEditBox(ctx, edit, s) {
  const [x, y, w, h] = edit.roi;
  const color = edit.dirty ? '#127f3f' : '#1e6feb';

  const m = Number(edit.safe_margin) || 0;
  if (m > 0) {
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(18,127,63,.75)';
    ctx.strokeRect(px(x - m), px(y - m), (w + 2 * m) * s, (h + 2 * m) * s);
    ctx.setLineDash([]);
  }

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = color;
  ctx.strokeRect(px(x), px(y), w * s, h * s);

  // 标签优先显示"字段说明"（人看的），没填就退回 fieldNN / 新建
  const bare = edit.name || edit.id || '新建（未命名）';
  const capt = edit.isNew && !edit.name ? '新建（未填写说明）' : bare;
  label(ctx, edit.dirty ? `${capt} · 未保存` : capt, px(x), px(y), color);

  const hsz = 5;
  ctx.fillStyle = '#fff';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  for (const [, [hx, hy]] of Object.entries(handlePoints(edit.roi))) {
    ctx.beginPath();
    ctx.rect(px(hx) - hsz, px(hy) - hsz, hsz * 2, hsz * 2);
    ctx.fill(); ctx.stroke();
  }
}

function label(ctx, text, x, y, color) {
  ctx.font = '12px ui-monospace, Consolas, monospace';
  const tw = ctx.measureText(text).width;
  const ty = Math.max(PAD + 12, y - 6);
  ctx.fillStyle = color;
  ctx.fillRect(x, ty - 12, tw + 8, 15);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, x + 4, ty);
}

/* ---- 表单 <-> 框 双向同步 ---- */

function syncFormFromEdit() {
  const e = state.edit;
  if (!e) {
    $('#inId').value = '';
    $('#inName').value = '';
    $('#inX').value = ''; $('#inY').value = ''; $('#inW').value = ''; $('#inH').value = '';
    $('#inMargin').value = 8;
    updateEditStatus();
    return;
  }
  $('#inId').value = e.id || '';           // 只读：自动分配的 fieldNN
  $('#inName').value = e.name || '';        // 可编辑：字段说明
  $('#inX').value = e.roi[0]; $('#inY').value = e.roi[1];
  $('#inW').value = e.roi[2]; $('#inH').value = e.roi[3];
  $('#inMargin').value = e.safe_margin || 0;
  updateEditStatus();
}

function syncEditFromForm() {
  const e = state.edit;
  if (!e) return;
  const num = (sel) => { const v = Number($(sel).value); return Number.isFinite(v) ? v : 0; };
  e.name = $('#inName').value.trim();       // id 不从表单读取（不可编辑）
  e.safe_margin = Math.max(0, Math.round(num('#inMargin')));
  const want = [num('#inX'), num('#inY'), num('#inW'), num('#inH')];
  if (want[2] > 0 && want[3] > 0) {
    const got = clampRoi(want);
    e.clamped = got.some((v, i) => v !== want[i]);
    e.roi = got;
  }
  e.dirty = true;
  updateEditStatus();
  draw();
}

function writeRoiToForm() {
  const e = state.edit;
  if (!e) return;
  $('#inX').value = e.roi[0]; $('#inY').value = e.roi[1];
  $('#inW').value = e.roi[2]; $('#inH').value = e.roi[3];
  e.clamped = false;
  updateEditStatus();
}

function markDirty() {
  if (state.edit) { state.edit.dirty = true; updateEditStatus(); }
}

function updateEditStatus() {
  const e = state.edit;
  const badge = $('#btnAddCrop');
  if (!e) {
    $('#editStatus').innerHTML = '<span class="dim">未选中。在模板图上拖出一个新框，或点右下列表里的某一项。</span>';
    badge.classList.remove('dirty');
    badge.textContent = '保存该块';
    return;
  }
  const [x, y, w, h] = e.roi;
  const idTxt = e.id ? esc(e.id) : '<b>待分配</b>';
  const nameTxt = e.name ? esc(e.name) : '<span class="dim">未填写说明</span>';
  $('#editStatus').innerHTML =
    `${idTxt} · ${nameTxt} · 坐标 x=${x} y=${y} w=${w} h=${h}`
    + (e.dirty ? ' · <b style="color:#127f3f">有未保存改动</b>' : ' · <span class="dim">已保存</span>')
    + (e.clamped
      ? '<br><b style="color:#9a6400">输入的坐标超出画布，已按边界钳位</b>（离开输入框后会写回实际值）'
      : '');
  badge.classList.toggle('dirty', !!e.dirty);
  badge.textContent = e.dirty ? '保存该块 *' : '保存该块';
}

function selectCrop(id) {
  const it = (state.crops.items || []).find((x) => x.id === id);
  if (!it) return;
  state.edit = {
    id: it.id, name: it.name || '', roi: [...it.roi],
    safe_margin: it.safe_margin || 0, isNew: false, dirty: false,
  };
  syncFormFromEdit();
  renderCrops();
  draw();
}

/* ---- 工具条 ---- */

$('#zoom').addEventListener('input', () => {
  setZoom(+$('#zoom').value);
  draw();
});
$('#btnFit').addEventListener('click', () => { fitZoom(); draw(); });
$('#btnZoom1').addEventListener('click', () => { setZoom(1.0); draw(); });
$('#ckLines').addEventListener('change', draw);
$('#ckSnap').addEventListener('change', draw);
$('#ckShowAll').addEventListener('change', () => {
  state.showAll = $('#ckShowAll').checked;
  draw();
});
window.addEventListener('resize', () => { if (state.tplImg) fitZoom(); draw(); });

/* ---- 鼠标：新建 / 移动 / 改宽高 / 平移 ---- */

function toImg(e) {
  const cv = $('#tplCanvas'), r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - PAD) / state.zoom,
           y: (e.clientY - r.top - PAD) / state.zoom };
}

let drag = null;
let pan = null;
let spaceDown = false;

// 平移用"滚动容器"实现：不碰任何坐标换算，放大后能平移到任意局部
document.addEventListener('keydown', (e) => { if (e.code === 'Space') spaceDown = true; });
document.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceDown = false; });

$('#tplCanvas').addEventListener('mousedown', (e) => {
  if (!state.tplImg) return;

  // 平移优先：中键拖动，或 空格+左键拖动
  if (e.button === 1 || (e.button === 0 && spaceDown)) {
    const wrap = $('#canvasWrap');
    pan = { x: e.clientX, y: e.clientY, sl: wrap.scrollLeft, st: wrap.scrollTop };
    $('#tplCanvas').style.cursor = 'grabbing';
    e.preventDefault();
    return;
  }
  if (e.button !== 0) return;

  const p = toImg(e);

  if (state.edit) {
    const h = hitHandle(p, state.edit.roi);
    if (h) {
      drag = { mode: 'resize', handle: h, startRoi: [...state.edit.roi] };
      e.preventDefault();
      return;
    }
    if (inside(p, state.edit.roi)) {
      drag = { mode: 'move', start: p, startRoi: [...state.edit.roi] };
      e.preventDefault();
      return;
    }
  }

  if (state.showAll) {
    const items = state.crops.items || [];
    for (let i = items.length - 1; i >= 0; i--) {
      if (inside(p, items[i].roi)) { selectCrop(items[i].id); return; }
    }
  }

  const x = snapVal(p.x, 'x'), y = snapVal(p.y, 'y');
  state.edit = {
    id: '', name: '', roi: clampRoi([x, y, 0, 0]),
    safe_margin: Number($('#inMargin').value) || 0, isNew: true, dirty: true,
  };
  syncFormFromEdit();
  drag = { mode: 'new', start: { x, y } };
  renderCrops();
  e.preventDefault();
});

$('#tplCanvas').addEventListener('mousemove', (e) => {
  if (!state.tplImg) return;
  // 实时坐标读数：放大后靠它精确对位
  const p = toImg(e);
  const t = state.tplImg.canvas;
  const cx = Math.round(p.x), cy = Math.round(p.y);
  $('#cursorPos').textContent =
    (p.x >= 0 && p.y >= 0 && p.x <= t.width && p.y <= t.height)
      ? `x ${cx} · y ${cy}` : '—';

  if (drag || pan) return;
  const cv = $('#tplCanvas');
  if (state.edit) {
    const h = hitHandle(p, state.edit.roi);
    if (h) { cv.style.cursor = HANDLE_CURSOR[h]; return; }
    if (inside(p, state.edit.roi)) { cv.style.cursor = 'move'; return; }
  }
  cv.style.cursor = spaceDown ? 'grab' : 'crosshair';
});

window.addEventListener('mousemove', (e) => {
  if (pan) {
    const wrap = $('#canvasWrap');
    wrap.scrollLeft = pan.sl - (e.clientX - pan.x);
    wrap.scrollTop = pan.st - (e.clientY - pan.y);
    return;
  }
  if (!drag || !state.tplImg) return;
  const p = toImg(e);

  if (drag.mode === 'resize') {
    const nx = /[we]/.test(drag.handle) ? snapVal(p.x, 'x') : p.x;
    const ny = /[ns]/.test(drag.handle) ? snapVal(p.y, 'y') : p.y;
    state.edit.roi = clampRoi(applyResize(drag.startRoi, drag.handle,
      clampPt(nx, 'x'), clampPt(ny, 'y')));
  } else if (drag.mode === 'move') {
    const dx = p.x - drag.start.x, dy = p.y - drag.start.y;
    let [x, y, w, h] = drag.startRoi;
    x = snapVal(x + dx, 'x');
    y = snapVal(y + dy, 'y');
    state.edit.roi = clampRoi([x, y, w, h]);
  } else {
    const x1 = clampPt(snapVal(p.x, 'x'), 'x');
    const y1 = clampPt(snapVal(p.y, 'y'), 'y');
    state.edit.roi = clampRoi([Math.min(drag.start.x, x1), Math.min(drag.start.y, y1),
      Math.abs(x1 - drag.start.x), Math.abs(y1 - drag.start.y)]);
  }
  markDirty();
  syncFormFromEdit();
  draw();
});

window.addEventListener('mouseup', () => {
  if (pan) { pan = null; $('#tplCanvas').style.cursor = 'crosshair'; return; }
  if (!drag) return;
  const wasNew = drag.mode === 'new';
  drag = null;
  const e = state.edit;
  if (!e) { draw(); return; }

  const tooSmall = e.roi[2] <= 4 || e.roi[3] <= 4;
  if (wasNew && e.isNew && tooSmall) {
    state.edit = null;
    syncFormFromEdit();
    renderCrops();
  } else if (wasNew && e.isNew) {
    $('#inName').focus();          // 框画完了，直接让用户填字段说明
  }
  draw();
});

/* ---- 表单 ---- */

['#inX', '#inY', '#inW', '#inH', '#inMargin', '#inName'].forEach((sel) => {
  $(sel).addEventListener('input', () => {
    if (!state.edit) {
      // 没选中任何框就直接改数字：当成新建，方便"按已知数值录入"
      const roi = [Number($('#inX').value) || 0, Number($('#inY').value) || 0,
        Number($('#inW').value) || 0, Number($('#inH').value) || 0];
      state.edit = {
        id: '', name: $('#inName').value.trim(), roi,
        safe_margin: Number($('#inMargin').value) || 0, isNew: true, dirty: true,
      };
      updateEditStatus(); draw(); renderCrops();
      return;
    }
    syncEditFromForm();
  });
  $(sel).addEventListener('change', () => {
    if (!$('#inX').value && !$('#inW').value) return;
    writeRoiToForm();
  });
});

document.addEventListener('keydown', (e) => {
  if (!state.edit || !state.tplImg) return;
  const tag = (document.activeElement || {}).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  const step = e.shiftKey ? 10 : 1;
  const dx = e.key === 'ArrowLeft' ? -step : (e.key === 'ArrowRight' ? step : 0);
  const dy = e.key === 'ArrowUp' ? -step : (e.key === 'ArrowDown' ? step : 0);
  if (!dx && !dy) return;
  e.preventDefault();
  const [x, y, w, h] = state.edit.roi;
  state.edit.roi = clampRoi([x + dx, y + dy, w, h]);
  markDirty(); syncFormFromEdit(); draw();
});

async function saveCrop() {
  if (!state.tpl) { alert('请先选中一个模板'); return null; }
  if (!state.edit) { alert('请先在模板图上拖出一个框，或点右下列表里的一项'); return null; }
  const e = state.edit;
  if (!(e.name || '').trim()) {
    alert('请填写「字段说明」——它是这个字段对人的含义（如 付款账户）。');
    $('#inName').focus();
    return null;
  }
  if (!(e.roi[2] > 0) || !(e.roi[3] > 0)) {
    alert('宽高必须大于 0。拖动框边上的小方块可以调整宽高。');
    return null;
  }
  try {
    const res = await post(`/api/categories/${state.cat.id}/templates/${state.tpl.id}/crops`, {
      id: e.id || '', name: e.name.trim(), roi: e.roi, safe_margin: e.safe_margin,
    });
    state.crops = res;
    const sid = res.saved_id || e.id;
    state.edit = {
      id: sid, name: e.name.trim(), roi: [...e.roi],
      safe_margin: e.safe_margin || 0, isNew: false, dirty: false,
    };
    renderCrops();
    syncFormFromEdit();
    draw();
    return sid;
  } catch (err) {
    alert('保存失败：' + err.message);
    return null;
  }
}

$('#btnAddCrop').addEventListener('click', saveCrop);

$('#btnClearSel').addEventListener('click', () => {
  state.edit = null;
  state.hoverId = null;
  syncFormFromEdit();
  renderCrops();
  draw();
});

/* ---- 坐标清单 ---- */

async function loadCrops() {
  if (!state.cat || !state.tpl) return;
  const catId = state.cat.id, tplId = state.tpl.id;
  const data = await get(`/api/categories/${catId}/templates/${tplId}/crops`);
  if (!state.tpl || state.tpl.id !== tplId) return;
  state.crops = data;
  if (state.edit && state.edit.id) {
    const it = (data.items || []).find((x) => x.id === state.edit.id);
    if (it) {
      state.edit = {
        id: it.id, name: it.name || '', roi: [...it.roi],
        safe_margin: it.safe_margin || 0, isNew: false, dirty: false,
      };
    } else {
      state.edit = null;
    }
  }
  syncFormFromEdit();
  renderCrops();
  draw();
}

function renderCrops() {
  const items = state.crops.items || [];
  const selId = state.edit && state.edit.id ? state.edit.id : null;
  $('#cropCount').textContent = items.length;
  $('#cropList').innerHTML = items.map((it) => `
    <div class="crop-item ${it.id === selId ? 'sel' : ''}" data-id="${esc(it.id)}">
      <img src="${esc(it.preview_url)}?t=${Date.now()}" alt="">
      <div class="info">
        <div class="id">${esc(it.name || '（未命名）')}</div>
        <div class="roi"><b>${esc(it.id)}</b> · x=${it.roi[0]} y=${it.roi[1]} w=${it.roi[2]} h=${it.roi[3]}
          · 外扩 ${it.safe_margin || 0}</div>
      </div>
      <button class="del" title="删除">×</button>
    </div>`).join('')
    || '<div class="dim sm">还没有配置模板块。在左边模板图上拖出一个框，填「字段说明」后保存。</div>';

  $$('#cropList .crop-item').forEach((el) => {
    el.addEventListener('click', (ev) => {
      if (ev.target.classList.contains('del')) return;
      selectCrop(el.dataset.id);
    });
    el.addEventListener('mouseenter', () => { state.hoverId = el.dataset.id; draw(); });
    el.addEventListener('mouseleave', () => { state.hoverId = null; draw(); });
    el.querySelector('.del').addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const id = el.dataset.id;
      if (!confirm(`删除模板块「${id}」？`)) return;
      try {
        await del(`/api/categories/${state.cat.id}/templates/${state.tpl.id}/crops/${encodeURIComponent(id)}`);
        if (state.edit && state.edit.id === id) state.edit = null;
        await loadCrops();
      } catch (err) { alert('删除失败：' + err.message); }
    });
  });
  updateEditStatus();
}

$('#btnReloadCrops').addEventListener('click', loadCrops);

/* ---------------------------------------------------------------- 标签页 */

function switchTab(name) {
  const t = $(`.tab[data-tab="${name}"]`);
  if (!t) return;
  $$('.tab').forEach((x) => x.classList.remove('active'));
  $$('.tabpane').forEach((x) => x.classList.remove('active'));
  t.classList.add('active');
  $(`.tabpane[data-pane="${name}"]`).classList.add('active');
  syncUrl();
  if (name === 'tpl') { fitZoom(); draw(); }
  if (name === 'jobs') loadJobs();
  if (name === 'batch') loadBatchJobs();
}

$$('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

/* ---------------------------------------------------------------- 任务：列表 / 过滤 / 弹窗 */

function statusBadge(st) {
  if (st === 'ok') return '<span class="badge ok">合格</span>';
  if (st === 'low_confidence') return '<span class="badge low">低置信</span>';
  if (st === 'rejected') return '<span class="badge bad">拒收</span>';
  return `<span class="badge bad">${esc(st)}</span>`;
}

function jobRunBadge(j) {
  if (j.status === 'failed') return '<span class="badge bad">失败</span>';
  if (j.status !== 'done') return '<span class="badge low">进行中</span>';
  return '<span class="badge ok">完成</span>';
}

/** 归属摘要：分类（任务级）+ 识别到的模板（页级汇总）。
 *  名称是建任务时的快照，展示时把 id 一起带上——改名/删模板后仍能追溯。 */
function jobScopeHTML(j) {
  const cat = j.category_name
    ? `<span class="tpl-tag" title="${esc(j.category_id || '')}">${esc(j.category_name)}</span>`
    : '';
  const tpls = (j.templates || []).map((t) =>
    `<span class="tpl-tag" title="${esc(t.id)}">${esc(t.name || t.id)}` +
    `${t.n > 1 ? ` ×${t.n}` : ''}</span>`).join('');
  const none = (j.templates || []).length ? ''
    : '<span class="dim sm">未识别到模板</span>';
  return cat + tpls + none;
}

function jobRowHTML(j) {
  return `<div class="job" data-jid="${esc(j.id)}">
    <span class="id">${esc(j.id)}</span>
    ${jobRunBadge(j)}
    <span class="dim">${esc(j.created_at || '')}</span>
    <span class="grow"></span>
    <span>${j.total || 0} 页 · 合格 ${j.n_ok || 0} · 低置信 ${j.n_low || 0} · 拒收 ${j.n_rejected || 0}</span>
    <button class="ghost sm" data-open="${esc(j.id)}">查看</button>
    <button class="danger ghost sm" data-del="${esc(j.id)}">删除</button>
    <div class="job-scope" title="分类（任务级）+ 识别到的模板（页级）">${jobScopeHTML(j)}</div>
  </div>`;
}

/** 过滤是「原子」的：任务只要含 ≥1 张该状态的页就算命中。 */
function jobMatchesFilter(j, f) {
  if (f === 'all') return true;
  if (f === 'ok') return (j.n_ok || 0) > 0;
  if (f === 'low_confidence') return (j.n_low || 0) > 0;
  if (f === 'rejected') return (j.n_rejected || 0) > 0;
  return true;
}

function wireJobRows(sel, filter) {
  $$(`${sel} [data-open]`).forEach((b) => b.addEventListener('click', () => openJob(b.dataset.open, filter)));
  $$(`${sel} [data-del]`).forEach((b) => b.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    const jid = b.dataset.del;
    if (!confirm(`删除任务「${jid}」？\n该任务的标准输出图、裁剪块、拒收原图与验证报告都会一并删除，不可恢复。`)) return;
    try {
      await del(`/api/jobs/${encodeURIComponent(jid)}`);
      if (state.jmJob && state.jmJob.id === jid) { state.jmJob = null; closeModal($('#jobModal')); }
      await refreshJobLists();
    } catch (e) { alert('删除失败：' + e.message); }
  }));
}

async function fetchJobs() {
  if (!state.cat) return [];
  const catId = state.cat.id;
  const jobs = await get(`/api/jobs?category_id=${encodeURIComponent(catId)}`);
  if (!state.cat || state.cat.id !== catId) return [];
  return jobs;
}

/** 批量处理页的任务列表：**不过滤**。 */
async function loadBatchJobs() {
  const jobs = await fetchJobs();
  if (!state.cat) return [];
  $('#batchJobList').innerHTML = jobs.map(jobRowHTML).join('')
    || '<div class="dim">该分类还没有任务记录</div>';
  wireJobRows('#batchJobList', 'all');
  return jobs;
}

/** 任务记录页：按当前筛选过滤。 */
async function loadJobs() {
  if (!state.cat) return [];
  $('#jobsScope').innerHTML =
    `只显示分类 <b>${esc(state.cat.name)}</b> <code>${esc(state.cat.id)}</code> 的任务记录`;
  const all = await fetchJobs();
  if (!state.cat) return [];
  const jobs = all.filter((j) => jobMatchesFilter(j, state.filter));
  $('#filterHint').textContent =
    state.filter === 'all' ? `共 ${all.length} 个任务`
      : `筛出 ${jobs.length} / ${all.length} 个任务（含${FILTER_LABEL[state.filter]}页的任务）`;
  $('#jobsList').innerHTML = jobs.map(jobRowHTML).join('')
    || `<div class="dim">没有含「${FILTER_LABEL[state.filter]}」页的任务</div>`;
  wireJobRows('#jobsList', state.filter);
  return all;
}

async function refreshJobLists() {
  await Promise.all([loadBatchJobs(), loadJobs()]);
}

/* ---- 任务详情弹窗：逐页卡片 + 裁切块 ---- */

async function openJob(jid, filter = 'all') {
  let job;
  try { job = await get(`/api/jobs/${encodeURIComponent(jid)}`); }
  catch (e) { alert('打开任务失败：' + e.message); return; }
  if (state.cat && job.category_id && job.category_id !== state.cat.id) {
    alert('该任务属于分类 ' + (job.category_id || '未知') + '，与当前分类不符。');
    return;
  }
  state.jmJob = job;
  state.jmFilter = filter;
  state.filter = filter;          // 让筛选按钮状态与弹窗一致
  $$('.fbtn').forEach((x) => x.classList.toggle('on', x.dataset.f === filter));
  renderJobModal();
  openModal('#jobModal');
}

function pageCardHTML(p) {
  return `<div class="page">
    <div class="ph">
      <span class="nm">${esc(p.source)} · p${p.page_index}</span>
      <span class="grow"></span>
      ${p.template_name ? `<span class="tpl-tag">${esc(p.template_name)}</span>` : ''}
      ${statusBadge(p.status)}
    </div>
    ${p.output_url
      ? `<img class="full" src="${esc(p.output_url)}?t=${Date.now()}" data-title="${esc(p.stem)}" alt="">`
      : '<div class="dim sm">无输出（已拒收）</div>'}
    ${p.reason ? `<div class="reason">${esc(p.reason)}</div>` : ''}
    <div class="dim sm" style="margin-top:6px">
      相关度 ${p.score ?? '—'}
      ${p.pdf_url
        ? ` · <a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">原始分页 PDF ↗</a>`
        : ''}
    </div>
    <div class="block-chips">
      ${(p.blocks || []).map((b) => `
        <span class="block-chip" data-url="${esc(b.url)}" data-title="${esc(b.id)}">
          <img src="${esc(b.url)}?t=${Date.now()}" alt="">${esc(b.id)}
        </span>`).join('') || '<span class="dim sm">没有模板块</span>'}
    </div>
  </div>`;
}

function renderJobModal() {
  const job = state.jmJob;
  if (!job) return;
  const pages = job.pages || [];
  const f = state.jmFilter || 'all';
  const shown = pages.filter((p) => f === 'all' || p.status === f);
  const okN = pages.filter((p) => p.status === 'ok').length;
  const lowN = pages.filter((p) => p.status === 'low_confidence').length;
  const rejN = pages.filter((p) => p.status === 'rejected').length;

  $('#jmTitle').innerHTML = `任务 ${esc(job.id)} · 共 ${pages.length} 页`;

  const cards = shown.map(pageCardHTML).join('')
    || `<div class="dim">没有符合「${FILTER_LABEL[f]}」的页</div>`;

  $('#jmBody').innerHTML = `
    <p class="dim">分类 <b>${esc(job.category_name || '')}</b>
      <code>${esc(job.category_id || '')}</code> · 合格 ${okN} · 低置信 ${lowN}
      · 拒收 ${rejN} · ${esc(job.created_at || '')}
      ${f !== 'all' ? ` · <b>已筛选：${FILTER_LABEL[f]}（${shown.length} 页）</b>` : ''}</p>
    <div class="pages">${cards}</div>
    <div id="jmVerifyBox"></div>`;

  $$('#jmBody img.full, #jmBody .block-chip').forEach((el) => {
    el.addEventListener('click', () =>
      lightbox(el.dataset.title || '预览', el.dataset.url
        || el.getAttribute('src').split('?')[0]));
  });
  $('#jmVerify').disabled = job.status !== 'done';
}

$('#jmVerify').addEventListener('click', async () => {
  const job = state.jmJob;
  if (!job) return;
  $('#jmVerify').disabled = true;
  $('#jmVerifyBox').innerHTML = '<div class="card dim">正在验证：逐块裁剪 + 堆叠 + 位移量化…</div>';
  try {
    const rep = await post(`/api/jobs/${job.id}/verify`);
    const groups = rep.groups || [];
    $('#jmVerifyBox').innerHTML = groups.map((g) => {
      const s = g.summary || {};
      return `<div class="card">
        <h3>块裁剪验证 · ${esc(g.template_id)}（${g.n_images} 张输出）</h3>
        <p class="dim">块数 ${s.n_blocks} · 通过 ${s.n_pass} · 需复核 ${s.n_check}
          · 读数歧义 ${s.n_ambiguous || 0}
          ${g.report_url ? ` · <a href="${esc(g.report_url)}" target="_blank">完整报告</a>` : ''}</p>
        <div class="block-chips">
          ${Object.entries(g.sheets || {}).map(([k, u]) =>
            `<span class="block-chip" data-url="${esc(u)}" data-title="${esc(k)}">
               <img src="${esc(u)}?t=${Date.now()}" alt="">${esc(k)} 堆叠图</span>`).join('')}
        </div>
      </div>`;
    }).join('') || '<div class="card dim">没有可验证的输出（全部被拒收）。</div>';
    $$('#jmVerifyBox .block-chip').forEach((el) => el.addEventListener('click', () =>
      lightbox(el.dataset.title, el.dataset.url)));
  } catch (e) {
    $('#jmVerifyBox').innerHTML = `<div class="card"><b>验证失败：</b>${esc(e.message)}</div>`;
  }
  $('#jmVerify').disabled = false;
});

$('#jmDelete').addEventListener('click', async () => {
  const job = state.jmJob;
  if (!job) return;
  if (!confirm(`删除任务「${job.id}」？\n该任务的标准输出图、裁剪块、拒收原图与验证报告都会一并删除，不可恢复。`)) return;
  try {
    await del(`/api/jobs/${encodeURIComponent(job.id)}`);
    state.jmJob = null;
    closeModal($('#jobModal'));
    await refreshJobLists();
  } catch (e) { alert('删除失败：' + e.message); }
});

$$('.fbtn').forEach((b) => b.addEventListener('click', () => {
  state.filter = b.dataset.f;
  $$('.fbtn').forEach((x) => x.classList.toggle('on', x === b));
  loadJobs();
}));

/* ---------------------------------------------------------------- 批量处理 */

$('#btnRunBatch').addEventListener('click', async () => {
  if (!state.cat) { alert('请先在左侧选中一个分类'); return; }
  const files = Array.from($('#batchFiles').files || []);
  if (!files.length) { alert('请选择要上传的文件'); return; }
  if (!(state.templates.length)) { alert('该分类下还没有模板，请先建模板'); return; }
  const catId = state.cat.id;

  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));

  $('#progressWrap').hidden = false;
  $('#progressBar').style.width = '0%';
  $('#progressText').textContent = '上传中…';
  $('#btnRunBatch').disabled = true;

  try {
    const job = await postForm(`/api/categories/${catId}/jobs`, fd,
      (p) => {
        if (!state.cat || state.cat.id !== catId) return;
        $('#progressBar').style.width = (p * 60).toFixed(0) + '%';
        $('#progressText').textContent = `上传中 ${(p * 100).toFixed(0)}%`;
      });
    pollJob(job.id, catId, files);
  } catch (e) {
    $('#progressText').textContent = '失败：' + e.message;
    $('#btnRunBatch').disabled = false;
    // 上传就没成功，保留已选文件方便重试
  }
});

/** 清空批量上传的文件选择（清完浏览器会显示"未选择任何文件"）。 */
function clearBatchFiles() {
  const el = $('#batchFiles');
  if (el) el.value = '';
}

/** 判断当前文件框里的是不是"我们刚提交的那一批"（换过就别动用户的新选择）。 */
function filesAreSame(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].name !== b[i].name || a[i].size !== b[i].size
      || a[i].lastModified !== b[i].lastModified) return false;
  }
  return true;
}

function pollJob(jid, catId, submitted) {
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try { job = await get(`/api/jobs/${jid}`); }
    catch (e) { clearInterval(state.poll); state.poll = null; return; }

    const onSameCat = state.cat && state.cat.id === catId;
    if (onSameCat) {
      const pct = job.total ? Math.round(job.done / job.total * 100) : 0;
      $('#progressBar').style.width = (60 + pct * 0.4).toFixed(0) + '%';
      $('#progressText').textContent = job.message || job.status;
    }
    if (job.status === 'done' || job.status === 'failed') {
      clearInterval(state.poll); state.poll = null;
      if (!onSameCat) return;
      $('#progressBar').style.width = '100%';
      $('#btnRunBatch').disabled = false;

      // 任务跑完就清空已选文件：否则再点一次「上传并处理」会把同一批重传一遍。
      // 只在"还是刚提交的那一批"时清——用户中途换了选择就别动他的。
      const cur = $('#batchFiles').files;
      if (!cur || !cur.length || filesAreSame(cur, submitted || [])) {
        clearBatchFiles();
        if (job.status === 'done') {
          $('#progressText').textContent = '已完成，已清空文件选择（避免重复上传）。';
        }
      } else {
        $('#progressText').textContent = '已完成（你换过文件，选择已保留）。';
      }

      await refreshJobLists();
      // 处理完直接把这次的结果用弹窗打开（不过滤），一眼看到产出
      openJob(jid, 'all');
    }
  }, 700);
}

/* ---------------------------------------------------------------- 启动 */

function syncUrl() {
  if (!state.cat) return;
  const tab = ($('.tab.active') || {}).dataset;
  const q = new URLSearchParams({ cat: state.cat.id });
  if (state.tpl) q.set('tpl', state.tpl.id);
  if (tab && tab.tab && tab.tab !== 'tpl') q.set('tab', tab.tab);
  history.replaceState(null, '', location.pathname + '?' + q.toString());
}

(async function boot() {
  // 先看数据目录是否已配置：未配置则只允许走初始化流程，其它一概不加载
  let st = null;
  try { st = await get('/api/setup/state'); } catch (e) { /* 接口异常时按已配置处理，别把用户卡死 */ }
  if (st && !st.configured) {
    state.setupLocked = true;
    $('#setupCancel').hidden = true;
    $('#setupX').hidden = true;
    renderSetup(st);
    openModal('#setupModal');
    return;
  }
  if (st) $('#btnDataDir').hidden = false;

  await loadCats();
  const q = new URLSearchParams(location.search);
  const want = q.get('cat');
  const target = (want && state.cats.some((c) => c.id === want))
    ? want : (state.cats[0] && state.cats[0].id);
  if (!target) return;
  await selectCat(target);
  const tplWant = q.get('tpl');
  if (tplWant && state.templates.some((t) => t.id === tplWant)) {
    await selectTpl(tplWant);
  }
  const tab = q.get('tab');
  if (tab) switchTab(tab);
  syncUrl();
})();
