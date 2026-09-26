/* docRenderCut 标注器交互测试
 * 驱动真实 Chrome，验证：默认不显示框 / 点清单显示框 / 拖动改坐标 /
 * 拖手柄改宽高 / 数字输入联动 / 保存落到后端。
 *
 * 运行： NODE_PATH=<playwright-core 所在 node_modules> node tools/test_annotator.js
 */
const path = require('path');
const { chromium } = require('playwright-core');

const BASE = process.env.BASE || 'http://127.0.0.1:8848';
const CAT = process.env.CAT || 'CAT20260925-0001';
const CHROME = process.env.CHROME
  || 'C:\\Users\\Administrator\\.agent-browser\\browsers\\chrome-154.0.8037.57\\chrome.exe';

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? '  —— ' + detail : ''}`);
}

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1680, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

  await page.goto(`${BASE}/?cat=${CAT}`, { waitUntil: 'load' });
  // 等模板真正就绪。
  // 注意不能用「canvas.width > 100」判——加载提示也会把 canvas 设成 460x90，
  // 那个条件会提前通过。要判 state.tpl 是否已赋值。
  await page.waitForFunction(
    () => typeof state !== 'undefined' && state.tpl !== null,
    null, { timeout: 30000 });

  /** 图像坐标 -> 屏幕坐标。直接问页面要，别在测试里重算一遍：
   *  画布内有 PAD 边距，测试里漏算它会让落点整体偏几十像素，
   *  从而"点到了框外面"却看起来像功能坏了。 */
  const imgToClient = (ix, iy) => page.evaluate(([a, b]) => {
    const r = document.querySelector('#tplCanvas').getBoundingClientRect();
    return { x: r.left + a * state.zoom + PAD, y: r.top + b * state.zoom + PAD };
  }, [ix, iy]);

  const S = () => page.evaluate(() => {
    const e = state.edit;
    return {
      hasEdit: !!e,
      id: e && e.id,
      roi: e && [...e.roi],
      dirty: e && e.dirty,
      showAll: state.showAll,
      canvas: state.tpl ? [state.tpl.canvas.width, state.tpl.canvas.height] : null,
      zoom: state.zoom,
      formX: document.querySelector('#inX').value,
      formW: document.querySelector('#inW').value,
      status: document.querySelector('#editStatus').innerText,
    };
  });

  console.log('\n[1] 默认状态：应不显示任何框');
  let s = await S();
  check('初始未选中任何块', s.hasEdit === false);
  check('全部显示默认关闭', s.showAll === false);

  console.log('\n[2] 点击清单项 → 显示对应框');
  const first = (await page.$$('#cropList .crop-item'))[0];
  const firstName = await first.getAttribute('data-id');
  await first.click();
  s = await S();
  check('点清单后选中了该项', s.hasEdit && s.id === firstName, `选中 ${s.id}`);
  check('表单已同步坐标', s.formX === String(s.roi[0]), `X=${s.formX} roi=${s.roi}`);
  check('未改动时不算 dirty', s.dirty === false);

  console.log('\n[3] 拖动框内部 → 改坐标（宽高不变）');
  const roiBefore = await page.evaluate(() => [...state.edit.roi]);
  const box = await imgToClient(roiBefore[0] + roiBefore[2] / 2, roiBefore[1] + roiBefore[3] / 2);
  box.before = roiBefore;
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 60, box.y + 30, { steps: 8 });
  await page.mouse.up();
  s = await S();
  check('拖动后坐标变化', s.roi[0] !== box.before[0] || s.roi[1] !== box.before[1],
    `${box.before} -> ${s.roi}`);
  check('拖动后宽高不变', s.roi[2] === box.before[2] && s.roi[3] === box.before[3]);
  check('拖动后标记为未保存', s.dirty === true);
  check('数字输入框跟着更新', s.formX === String(s.roi[0]), `X=${s.formX}`);

  console.log('\n[4] 拖右下角手柄 → 改宽高');
  const roiForHandle = await page.evaluate(() => [...state.edit.roi]);
  const hnd = await imgToClient(roiForHandle[0] + roiForHandle[2], roiForHandle[1] + roiForHandle[3]);
  hnd.before = roiForHandle;
  await page.mouse.move(hnd.x, hnd.y);
  await page.mouse.down();
  // 往左上收：这一个框的右边已经贴住画布右边界，往右拖会被边界正确截住（另有专门用例）
  await page.mouse.move(hnd.x - 80, hnd.y + 50, { steps: 8 });
  await page.mouse.up();
  s = await S();
  check('拖手柄后宽变小、高变大', s.roi[2] < hnd.before[2] && s.roi[3] > hnd.before[3],
    `${hnd.before} -> ${s.roi}`);
  check('拖手柄不移动左上角', s.roi[0] === hnd.before[0] && s.roi[1] === hnd.before[1],
    `x=${s.roi[0]} y=${s.roi[1]}`);

  console.log('\n[4b] 把手柄拖出画布边界 → 应停在边界，而不是整个框被推回去');
  const roi4b = await page.evaluate(() => [...state.edit.roi]);
  const h4b = await imgToClient(roi4b[0] + roi4b[2], roi4b[1] + roi4b[3]);
  await page.mouse.move(h4b.x, h4b.y);
  await page.mouse.down();
  await page.mouse.move(h4b.x + 400, h4b.y + 20, { steps: 10 });
  await page.mouse.up();
  const r4b = await page.evaluate(() => [...state.edit.roi]);
  check('左上角没被推走', r4b[0] === roi4b[0] && r4b[1] === roi4b[1],
    `${roi4b} -> ${r4b}`);
  check('宽高被边界截住', r4b[0] + r4b[2] <= (await page.evaluate(() => state.tpl.canvas.width)),
    `x+w=${r4b[0] + r4b[2]}`);

  console.log('\n[5] 改数字输入 → 框跟着移动');
  let before5 = (await S()).roi;
  // 往回移而不是往前：框已经被拖到右边界了，再往右会被钳位（那是另一条用例）
  const targetX = Math.max(0, before5[0] - 120);
  await page.fill('#inX', String(targetX));
  await page.waitForTimeout(150);
  s = await S();
  check('改 X 后框的位置变了', s.roi[0] === targetX, `${before5[0]} -> ${s.roi[0]}`);

  console.log('\n[5b] 输入超出画布 → 钳位并在离开输入框后写回实际值');
  const cw = (await page.evaluate(() => state.tpl.canvas.width));
  await page.fill('#inX', String(cw + 500));
  await page.waitForTimeout(150);
  s = await S();
  check('超出画布的输入被钳住', s.roi[0] + s.roi[2] <= cw, `x=${s.roi[0]} w=${s.roi[2]} cw=${cw}`);
  check('状态区提示了钳位', /钳位/.test(s.status), s.status.split('\n').pop());
  await page.dispatchEvent('#inX', 'change');
  await page.waitForTimeout(120);
  const after5b = await S();
  check('失焦后输入框写回实际值', after5b.formX === String(after5b.roi[0]),
    `formX=${after5b.formX} roi[0]=${after5b.roi[0]}`);
  before5 = after5b.roi;

  console.log('\n[6] 方向键微调 ±1px');
  await page.evaluate(() => document.activeElement.blur());
  const y0 = (await S()).roi[1];
  await page.keyboard.press('ArrowDown');
  s = await S();
  check('方向键下移 1px', s.roi[1] === y0 + 1, `${y0} -> ${s.roi[1]}`);

  console.log('\n[7] 全部显示开关');
  await page.click('#ckShowAll');
  check('全部显示已开启', (await S()).showAll === true);
  await page.click('#ckShowAll');
  check('全部显示已关闭', (await S()).showAll === false);

  console.log('\n[8] 保存落盘 → 重新加载后仍是新坐标');
  const saved = (await S()).roi;
  await page.click('#btnAddCrop');
  await page.waitForTimeout(900);
  s = await S();
  check('保存后 dirty 清除', s.dirty === false);
  check('保存后坐标保持', JSON.stringify(s.roi) === JSON.stringify(saved),
    `${saved} vs ${s.roi}`);
  const api = await page.evaluate(async (id) => {
    const r = await fetch(`/api/categories/${state.cat.id}/crops`);
    const d = await r.json();
    return d.items.find((x) => x.id === id);
  }, firstName);
  check('后端已持久化新坐标', JSON.stringify(api.roi) === JSON.stringify(saved),
    `后端 ${api.roi}`);

  console.log('\n[9] 点空白处不会留下垃圾框');
  // 找一块确定没有任何框的空白：画布右下角留出足够距离
  const cvs = await imgToClient(40, (await page.evaluate(() => state.tpl.canvas.height)) - 40);
  await page.mouse.click(cvs.x, cvs.y);
  await page.waitForTimeout(120);
  const after = await page.evaluate(() => (state.edit ? state.edit.roi : null));
  check('点击空白清除选择或留下有效框', after === null, `edit=${JSON.stringify(after)}`);

  console.log('\n[10] 控制台无报错');
  check('无 JS 运行时错误', errors.length === 0, errors.slice(0, 3).join(' | '));

  await page.screenshot({ path: path.join(__dirname, '_annotator_test.png') });
  await browser.close();

  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${results.length - failed.length}/${results.length} 通过`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error('测试异常:', e); process.exit(2); });
