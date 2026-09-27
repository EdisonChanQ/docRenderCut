/* docRenderCut 批量上传「跑完即清空文件选择」测试
 *
 * 验证：上传一批文件跑完后，文件框被清空（避免同一批被重复上传）；
 *       且提示文案明确告知已清空。
 *
 * 运行： NODE_PATH=<playwright-core 所在 node_modules> \
 *        SAMPLE=<一个可用作上传的 pdf/jpg> CAT=<分类ID> node tools/test_batch_clear.js
 */
const { chromium } = require('playwright-core');
const { resolveChrome } = require('./find_chrome');

const BASE = process.env.BASE || 'http://127.0.0.1:8848';
const CAT = process.env.CAT || 'CAT20260927-0001';
const SAMPLE = process.env.SAMPLE;

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? '  —— ' + detail : ''}`);
}

(async () => {
  if (!SAMPLE) {
    console.error('缺少 SAMPLE 环境变量（要上传的样例文件路径）');
    process.exit(2);
  }
  let chromeExe;
  try { chromeExe = resolveChrome(); }
  catch (e) { console.error(e.message); process.exit(2); }
  const browser = await chromium.launch({ executablePath: chromeExe, headless: true });
  const page = await browser.newPage({ viewport: { width: 1680, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(`${BASE}/?cat=${CAT}&tab=batch`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => typeof state !== 'undefined' && state.cat !== null, null, { timeout: 30000 });
  await page.waitForFunction(
    () => typeof state !== 'undefined' && (state.templates || []).length > 0,
    null, { timeout: 30000 });

  // ---- 1) 选文件：清空前应能读到我们选的这批 ----
  await page.setInputFiles('#batchFiles', [SAMPLE]);
  const picked = await page.$eval('#batchFiles', (el) => el.files.length);
  check('选文件后文件框有内容', picked === 1, `files.length = ${picked}`);

  // ---- 2) 跑一轮 ----
  await page.click('#btnRunBatch');
  const started = await page.waitForFunction(
    () => typeof state !== 'undefined' && state.poll !== null, null, { timeout: 20000 })
    .then(() => true).catch(() => false);
  check('任务已提交并进入轮询', started);

  // 等任务跑完：poll 归零 且 结果弹窗已打开（openJob 在完成分支里被调用）
  const done = await page.waitForFunction(
    () => typeof state !== 'undefined' && state.poll === null
      && !document.querySelector('#jobModal').hidden,
    null, { timeout: 240000 }).then(() => true).catch(() => false);
  check('任务执行完成且结果弹窗已打开', done);

  // ---- 3) 核心断言：文件框被清空 ----
  const after = await page.$eval('#batchFiles', (el) => el.files.length);
  check('任务完成后文件框已清空', after === 0, `files.length = ${after}`);

  const tip = await page.$eval('#progressText', (el) => el.textContent || '');
  check('提示文案说明已清空', tip.includes('已清空文件选择'), JSON.stringify(tip));

  // ---- 4) 再点一次：应被前端拦住（没有文件） ----
  // 结果弹窗盖住了批量面板，弹窗遮罩会拦 pointer 事件，先关掉它
  await page.keyboard.press('Escape');
  await page.waitForFunction(
    () => document.querySelector('#jobModal').hidden, null, { timeout: 10000 });

  let alerted = '';
  page.once('dialog', async (d) => { alerted = d.message(); await d.dismiss(); });
  await page.click('#btnRunBatch');
  await page.waitForTimeout(800);
  check('空文件时点上传会被拦下', alerted.includes('请选择要上传的文件'), JSON.stringify(alerted));

  check('页面无 JS 报错', errors.length === 0, errors.slice(0, 2).join(' | '));

  await browser.close();
  const bad = results.filter((r) => !r.ok);
  console.log(`\n${results.length - bad.length}/${results.length} 项通过`);
  process.exit(bad.length ? 1 : 0);
})().catch((e) => { console.error('测试异常：', e); process.exit(1); });
