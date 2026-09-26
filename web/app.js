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
  edit: null,         // 正在编辑的框 {id, roi, safe_margin, note, isNew, dirty}
  showAll: false,
  hoverId: null,
  job: null,
  poll: null,
};

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
$$('[data-close]').forEach((b) => b.addEventListener('click', () => closeModal(b)));
$$('.modal').forEach((m) => m.addEventListener('mousedown', (e) => {
  if (e.target === m) m.hidden = true;
}));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $$('.modal').forEach((m) => { m.hidden = true; });
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

$('#btnNewTpl').addEventListener('click', () => {
  if (!state.cat) return;
  $('#tfName').value = '';
  $('#tplMsg').hidden = true;
  $('#btnDoCreateTpl').hidden = false;
  $('#btnPreviewTpl').hidden = true;
  $('#btnFinishTpl').hidden = true;
  $('#btnDoCreateTpl').disabled = false;
  openModal('#modalTpl');
});

$('#tfPaper').addEventListener('change', (e) => {
  $('#tfCornersRow').hidden = e.target.value !== 'corners';
});

function tplSizeHint() {
  const w = +$('#tfW').value || 0, h = +$('#tfH').value || 0;
  if (!w || !h) { $('#tfSizeHint').textContent = ''; return; }
  const dpi = +$('#tfDpi').value || 200;
  $('#tfSizeHint').textContent =
    `画布 ${w}×${h} 像素，宽高比 ${(w / h).toFixed(3)}，` +
    `物理尺寸约 ${(w / dpi * 25.4).toFixed(1)}×${(h / dpi * 25.4).toFixed(1)} mm。上传样张后会核对宽高比。`;
}
['#tfW', '#tfH', '#tfDpi'].forEach((s) => $(s).addEventListener('input', tplSizeHint));
tplSizeHint();

$('#btnDoCreateTpl').addEventListener('click', async () => {
  const name = $('#tfName').value.trim();
  const f = $('#tfSample').files[0];
  if (!name) return setMsg('#tplMsg', 'err', '请填写模板名称');
  if (!f) return setMsg('#tplMsg', 'err', '请选择一张样张');

  const fd = new FormData();
  fd.append('name', name);
  fd.append('dpi', $('#tfDpi').value || '200');
  fd.append('width', $('#tfW').value || '1600');
  fd.append('height', $('#tfH').value || '3400');
  fd.append('paper_mode', $('#tfPaper').value);
  fd.append('corners', $('#tfCorners').value || '');
  fd.append('ink_bias', $('#tfInkBias').value || '0');
  fd.append('ink_dark_bias', $('#tfInkDark').value || '25');
  fd.append('sample', f);

  const btn = $('#btnDoCreateTpl');
  btn.disabled = true;
  setMsg('#tplMsg', 'info', '正在上传并构建标准模板…（首次会稍慢，需做结构提取与线检测）');
  try {
    const tpl = await postForm(`/api/categories/${state.cat.id}/templates`, fd,
      (p) => setMsg('#tplMsg', 'info', `上传中 ${(p * 100).toFixed(0)}%`));
    const warns = tpl.warnings || [];
    setMsg('#tplMsg', warns.length ? 'err' : 'ok',
      `<b>模板已构建</b> —— ${esc(tpl.id)} · ${esc(tpl.name)}<br>` +
      `画布 ${tpl.canvas.width}×${tpl.canvas.height} @${tpl.dpi}dpi · ` +
      `样张 ${tpl.sample && tpl.sample.sample_size ? tpl.sample.sample_size.join('×') : '—'} · ` +
      `检测到表格线 ${tpl.template.n_lines_x + tpl.template.n_lines_y} 条` +
      (warns.length ? `<ul>${warns.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : ''));
    $('#btnDoCreateTpl').hidden = true;
    $('#btnPreviewTpl').hidden = false;
    $('#btnFinishTpl').hidden = false;
    $('#btnPreviewTpl').onclick = () => lightbox('标准模板 · ' + tpl.name,
      (tpl.template.preview_url || tpl.template.template_url));
    state.newTplId = tpl.id;
  } catch (e) {
    setMsg('#tplMsg', 'err', '构建失败：' + esc(e.message));
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
  state.job = null;
  $('#jobResult').innerHTML = '';
  $('#progressWrap').hidden = true;
  $('#progressBar').style.width = '0%';
  $('#progressText').textContent = '';
  $('#batchFiles').value = '';
  $('#btnRunBatch').disabled = false;
  $('#btnVerify').disabled = true;
}

async function selectCat(cid) {
  resetBatchPane();
  state.tplImg = null;
  state.crops = { items: [] };
  state.tpl = null;
  canvasNotice('模板加载中…');
  $('#cropList').innerHTML = '<div class="dim sm">加载中…</div>';
  $('#jobsList').innerHTML = '<div class="dim">加载中…</div>';

  state.cat = await get(`/api/categories/${cid}`);
  state.templates = state.cat.templates || [];
  state.edit = null;
  state.hoverId = null;
  $('#empty').hidden = true;
  $('#catView').hidden = false;

  $('#catName').textContent = state.cat.name;
  $('#catMeta').innerHTML =
    `<code>${esc(state.cat.id)}</code> · ${state.templates.length} 个模板`;

  $('#batchScope').innerHTML =
    `上传目标分类：<b>${esc(state.cat.name)}</b> <code>${esc(state.cat.id)}</code>` +
    ` · ${state.templates.length} 个模板 · 系统自动识别每页归属`;

  $('#catWarn').hidden = true;
  $('#catWarn').innerHTML = '';

  renderCats();
  renderTplChips();
  syncUrl();

  // 自动选中第一个模板（若有）
  if (state.templates.length) {
    await selectTpl(state.templates[0].id);
  } else {
    $('#tplStrip').hidden = true;
    state.tpl = null;
    canvasNotice('该分类下还没有模板，点「+ 新建模板」创建');
    $('#cropList').innerHTML = '<div class="dim sm">先创建模板</div>';
  }

  loadJobs();
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
  $('#catMeta').innerHTML =
    `<code>${esc(state.cat.id)}</code> · ${state.templates.length} 个模板`;
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
  $('#empty').hidden = false;
  await loadCats();
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

function fitZoom() {
  if (!state.tplImg) return;
  const wrap = $('#canvasWrap');
  const avail = Math.max(240, wrap.clientWidth - 24 - PAD * 2);
  const availH = Math.max(240, wrap.clientHeight - 24 - PAD * 2);
  const z = Math.min(avail / state.tplImg.canvas.width,
    availH / state.tplImg.canvas.height, 1);
  state.zoom = Math.max(0.1, Math.round(z * 100) / 100);
  $('#zoom').value = state.zoom;
  $('#zoomVal').textContent = state.zoom.toFixed(2) + 'x';
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
  if (alpha >= 1) label(ctx, it.id, px(x), px(y), '#e0651a');
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

  const name = edit.id || '新建（未命名）';
  label(ctx, edit.dirty ? `${name} · 未保存` : name, px(x), px(y), color);

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
  if (!e) return;
  $('#inId').value = e.id || '';
  $('#inX').value = e.roi[0]; $('#inY').value = e.roi[1];
  $('#inW').value = e.roi[2]; $('#inH').value = e.roi[3];
  $('#inMargin').value = e.safe_margin || 0;
  $('#inNote').value = e.note || '';
  updateEditStatus();
}

function syncEditFromForm() {
  const e = state.edit;
  if (!e) return;
  const num = (sel) => { const v = Number($(sel).value); return Number.isFinite(v) ? v : 0; };
  e.id = $('#inId').value.trim();
  e.safe_margin = Math.max(0, Math.round(num('#inMargin')));
  e.note = $('#inNote').value.trim();
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
  const tag = e.id ? esc(e.id) : '<b>未命名</b>';
  $('#editStatus').innerHTML =
    `已选中 ${tag} · 坐标 x=${x} y=${y} w=${w} h=${h}`
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
    id: it.id, roi: [...it.roi], safe_margin: it.safe_margin || 0,
    note: it.note || '', isNew: false, dirty: false,
  };
  syncFormFromEdit();
  renderCrops();
  draw();
}

/* ---- 工具条 ---- */

$('#zoom').addEventListener('input', () => {
  state.zoom = +$('#zoom').value;
  $('#zoomVal').textContent = state.zoom.toFixed(2) + 'x';
  draw();
});
$('#btnFit').addEventListener('click', () => { fitZoom(); draw(); });
$('#ckLines').addEventListener('change', draw);
$('#ckSnap').addEventListener('change', draw);
$('#ckShowAll').addEventListener('change', () => {
  state.showAll = $('#ckShowAll').checked;
  draw();
});
window.addEventListener('resize', () => { if (state.tplImg) fitZoom(); draw(); });

/* ---- 鼠标 ---- */

function toImg(e) {
  const cv = $('#tplCanvas'), r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - PAD) / state.zoom,
           y: (e.clientY - r.top - PAD) / state.zoom };
}

let drag = null;

$('#tplCanvas').addEventListener('mousedown', (e) => {
  if (!state.tplImg) return;
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
    id: '', roi: clampRoi([x, y, 0, 0]), safe_margin: Number($('#inMargin').value) || 0,
    note: '', isNew: true, dirty: true,
  };
  syncFormFromEdit();
  drag = { mode: 'new', start: { x, y } };
  renderCrops();
  e.preventDefault();
});

$('#tplCanvas').addEventListener('mousemove', (e) => {
  if (drag || !state.tplImg) return;
  const p = toImg(e);
  const cv = $('#tplCanvas');
  if (state.edit) {
    const h = hitHandle(p, state.edit.roi);
    if (h) { cv.style.cursor = HANDLE_CURSOR[h]; return; }
    if (inside(p, state.edit.roi)) { cv.style.cursor = 'move'; return; }
  }
  cv.style.cursor = 'crosshair';
});

window.addEventListener('mousemove', (e) => {
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
    $('#inId').focus();
  }
  draw();
});

/* ---- 表单 ---- */

['#inX', '#inY', '#inW', '#inH', '#inMargin', '#inId', '#inNote'].forEach((sel) => {
  $(sel).addEventListener('input', () => {
    if (!state.edit) {
      const roi = [Number($('#inX').value) || 0, Number($('#inY').value) || 0,
        Number($('#inW').value) || 0, Number($('#inH').value) || 0];
      state.edit = {
        id: $('#inId').value.trim(), roi, safe_margin: Number($('#inMargin').value) || 0,
        note: $('#inNote').value.trim(), isNew: true, dirty: true,
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
  const id = (e.id || '').trim();
  if (!id) { alert('请填写字段名称（它会作为裁剪产物的文件名）'); $('#inId').focus(); return null; }
  if (!(e.roi[2] > 0) || !(e.roi[3] > 0)) {
    alert('宽高必须大于 0。拖动框边上的小方块可以调整宽高。');
    return null;
  }
  try {
    state.crops = await post(`/api/categories/${state.cat.id}/templates/${state.tpl.id}/crops`, {
      id, roi: e.roi, safe_margin: e.safe_margin, note: e.note,
    });
    state.edit = {
      id, roi: [...e.roi], safe_margin: e.safe_margin || 0,
      note: e.note || '', isNew: false, dirty: false,
    };
    renderCrops();
    syncFormFromEdit();
    draw();
    return id;
  } catch (err) {
    alert('保存失败：' + err.message);
    return null;
  }
}

$('#btnAddCrop').addEventListener('click', saveCrop);

$('#btnClearSel').addEventListener('click', () => {
  state.edit = null;
  state.hoverId = null;
  ['#inId', '#inX', '#inY', '#inW', '#inH', '#inNote'].forEach((s) => { $(s).value = ''; });
  $('#inMargin').value = 8;
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
        id: it.id, roi: [...it.roi], safe_margin: it.safe_margin || 0,
        note: it.note || '', isNew: false, dirty: false,
      };
    } else {
      state.edit = null;
    }
    syncFormFromEdit();
  }
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
        <div class="id">${esc(it.id)}</div>
        <div class="roi">x=${it.roi[0]} y=${it.roi[1]} w=${it.roi[2]} h=${it.roi[3]}
          · 外扩 ${it.safe_margin || 0}</div>
        ${it.note ? `<div class="roi">${esc(it.note)}</div>` : ''}
      </div>
      <button class="del" title="删除">×</button>
    </div>`).join('')
    || '<div class="dim sm">还没有配置模板块。在左边模板图上拖出一个框，填字段名称后保存。</div>';

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
}

$$('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

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
  $('#jobResult').innerHTML = '';

  try {
    const job = await postForm(`/api/categories/${catId}/jobs`, fd,
      (p) => {
        if (!state.cat || state.cat.id !== catId) return;
        $('#progressBar').style.width = (p * 60).toFixed(0) + '%';
        $('#progressText').textContent = `上传中 ${(p * 100).toFixed(0)}%`;
      });
    pollJob(job.id, catId);
  } catch (e) {
    $('#progressText').textContent = '失败：' + e.message;
    $('#btnRunBatch').disabled = false;
  }
});

function pollJob(jid, catId) {
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try { job = await get(`/api/jobs/${jid}`); }
    catch (e) { clearInterval(state.poll); state.poll = null; return; }

    const onSameCat = state.cat && state.cat.id === catId;
    if (onSameCat) {
      state.job = job;
      const pct = job.total ? Math.round(job.done / job.total * 100) : 0;
      $('#progressBar').style.width = (60 + pct * 0.4).toFixed(0) + '%';
      $('#progressText').textContent = job.message || job.status;
    }
    if (job.status === 'done' || job.status === 'failed') {
      clearInterval(state.poll); state.poll = null;
      if (!onSameCat) return;
      $('#progressBar').style.width = '100%';
      $('#btnRunBatch').disabled = false;
      $('#btnVerify').disabled = (job.status !== 'done');
      renderJob(job);
      loadJobs();
    }
  }, 700);
}

function badge(st) {
  if (st === 'ok') return '<span class="badge ok">合格</span>';
  if (st === 'low_confidence') return '<span class="badge low">低置信</span>';
  if (st === 'rejected') return '<span class="badge bad">拒收</span>';
  return `<span class="badge bad">${esc(st)}</span>`;
}

function renderJob(job, historical = false) {
  if (state.cat && job.category_id && job.category_id !== state.cat.id) {
    $('#jobResult').innerHTML = '';
    return;
  }
  if (job.status === 'failed') {
    $('#jobResult').innerHTML =
      `<div class="card"><h3>任务失败</h3><pre class="mono sm">${esc(job.error || '')}</pre></div>`;
    return;
  }
  const pages = job.pages || [];
  const n = pages.length;
  const okN = pages.filter((p) => p.status === 'ok').length;
  const lowN = pages.filter((p) => p.status === 'low_confidence').length;
  const rejN = pages.filter((p) => p.status === 'rejected').length;

  const head = `<div class="card">
    <h3>任务 ${esc(job.id)} —— ${n} 页${historical ? '（该分类最近一次）' : ''}</h3>
    <p class="dim">分类 <b>${esc(job.category_name || '')}</b>
      <code>${esc(job.category_id || '')}</code> · 合格 ${okN} · 低置信 ${lowN}
      · 拒收 ${rejN} · ${esc(job.created_at || '')}</p>
  </div>`;

  const cards = pages.map((p) => `
    <div class="page">
      <div class="ph">
        <span class="nm">${esc(p.source)} · p${p.page_index}</span>
        <span class="grow"></span>
        ${p.template_name ? `<span class="tpl-tag">${esc(p.template_name)}</span>` : ''}
        ${badge(p.status)}
      </div>
      ${p.output_url
        ? `<img class="full" src="${esc(p.output_url)}?t=${Date.now()}" data-title="${esc(p.stem)}" alt="">`
        : '<div class="dim sm">无输出（已拒收）</div>'}
      ${p.reason ? `<div class="reason">${esc(p.reason)}</div>` : ''}
      <div class="dim sm" style="margin-top:6px">
        相关度 ${p.score ?? '—'}${p.scores ? ` · <span title="${esc(JSON.stringify(p.scores))}">各模板分见悬停</span>` : ''}
      </div>
      <div class="block-chips">
        ${(p.blocks || []).map((b) => `
          <span class="block-chip" data-url="${esc(b.url)}" data-title="${esc(b.id)}">
            <img src="${esc(b.url)}?t=${Date.now()}" alt="">${esc(b.id)}
          </span>`).join('') || '<span class="dim sm">没有模板块</span>'}
      </div>
    </div>`).join('');

  // 拒收清单单列，方便审计
  const rejPages = pages.filter((p) => p.status === 'rejected');
  const rejBox = rejPages.length ? `<div class="card">
    <h3>拒收清单（${rejPages.length} 页 · 供审计）</h3>
    <p class="dim sm">这些页识别不出属于分类下哪个模板（或配准不过关），原样保留、不产出标准图、不裁块。</p>
    <ul class="rej-list">
      ${rejPages.map((p) => `<li>
        <b>${esc(p.source)} · p${p.page_index}</b>
        <span class="dim">${esc(p.reason || '')}</span>
        ${p.output_url ? `<a href="${esc(p.output_url)}" target="_blank">查看原图</a>` : ''}
      </li>`).join('')}
    </ul>
  </div>` : '';

  $('#jobResult').innerHTML = head + `<div class="pages">${cards}</div>` + rejBox +
    `<div id="verifyBox"></div>`;

  $$('#jobResult img.full, #jobResult .block-chip').forEach((el) => {
    el.addEventListener('click', () =>
      lightbox(el.dataset.title || '预览', el.dataset.url
        || el.getAttribute('src').split('?')[0]));
  });
}

$('#btnVerify').addEventListener('click', async () => {
  if (!state.job) { alert('请先跑一次批量处理，或在「任务记录」里打开一个任务'); return; }
  if (state.cat && state.job.category_id && state.job.category_id !== state.cat.id) {
    alert('当前显示的任务不属于选中的分类，请重新从「任务记录」打开。');
    resetBatchPane();
    return;
  }
  $('#btnVerify').disabled = true;
  $('#verifyBox').innerHTML = '<div class="card dim">正在验证：逐块裁剪 + 堆叠 + 位移量化…</div>';
  try {
    const rep = await post(`/api/jobs/${state.job.id}/verify`);
    const groups = rep.groups || [];
    $('#verifyBox').innerHTML = groups.map((g) => {
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
    $$('#verifyBox .block-chip').forEach((el) => el.addEventListener('click', () =>
      lightbox(el.dataset.title, el.dataset.url)));
  } catch (e) {
    $('#verifyBox').innerHTML = `<div class="card"><b>验证失败：</b>${esc(e.message)}</div>`;
  }
  $('#btnVerify').disabled = false;
});

/* ---------------------------------------------------------------- 任务记录 */

async function loadJobs() {
  if (!state.cat) return [];
  const catId = state.cat.id;
  $('#jobsScope').innerHTML =
    `只显示分类 <b>${esc(state.cat.name)}</b> <code>${esc(catId)}</code> 的任务记录`;
  const jobs = await get(`/api/jobs?category_id=${encodeURIComponent(catId)}`);
  if (!state.cat || state.cat.id !== catId) return [];
  $('#jobsList').innerHTML = jobs.map((j) => `
    <div class="job">
      <span class="id">${esc(j.id)}</span>
      ${badge(j.status === 'done' ? 'ok' : (j.status === 'failed' ? 'rejected' : 'low_confidence'))}
      <span class="dim">${esc(j.created_at || '')}</span>
      <span class="grow"></span>
      <span>${j.total || 0} 页 · 合格 ${j.n_ok || 0} · 低置信 ${j.n_low || 0} · 拒收 ${j.n_rejected || 0}</span>
      <button class="ghost sm" data-open="${esc(j.id)}">查看</button>
    </div>`).join('') || '<div class="dim">该分类还没有任务记录</div>';

  $$('#jobsList [data-open]').forEach((b) => b.addEventListener('click', async () => {
    const job = await get(`/api/jobs/${b.dataset.open}`);
    if (!state.cat || job.category_id !== state.cat.id) {
      alert('该任务属于分类 ' + (job.category_id || '未知') + '，与当前分类不符。');
      await loadJobs();
      return;
    }
    state.job = job;
    $$('.tab').forEach((x) => x.classList.remove('active'));
    $$('.tabpane').forEach((x) => x.classList.remove('active'));
    $('.tab[data-tab="batch"]').classList.add('active');
    $('.tabpane[data-pane="batch"]').classList.add('active');
    $('#btnVerify').disabled = (job.status !== 'done');
    $('#progressWrap').hidden = true;
    renderJob(job);
  }));
  return jobs;
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
