/* 解析可用的 Chrome / Chromium 可执行文件。
 *
 * 为什么需要它：playwright-core 不自带浏览器，必须给出 executablePath。
 * **不要硬编码某个人的机器路径**——原来写死了 `C:\Users\<某人>\...`，
 * 别人克隆下来跑测试必然失败。这里按优先级自动找：
 *
 *   1. 环境变量 CHROME（最高优先级，CI 里指定即可）
 *   2. 常见安装位置（Windows / macOS / Linux 各一套）
 *   3. agent-browser 的浏览器缓存目录（版本号会变，扫目录取最新的）
 *
 * 都找不到就抛出带操作指引的错误，而不是抛一个看不懂的 ENOENT。
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

function safeDirs(dir) {
  try {
    return fs.readdirSync(dir).sort().reverse();
  } catch (e) {
    return [];
  }
}

function candidates() {
  const home = os.homedir();
  const list = [];
  if (process.platform === 'win32') {
    list.push(
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
      path.join(home, 'AppData', 'Local', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    );
    const cache = path.join(home, '.agent-browser', 'browsers');
    for (const d of safeDirs(cache)) {
      list.push(path.join(cache, d, 'chrome.exe'));
      list.push(path.join(cache, d, 'chrome-win64', 'chrome.exe'));
    }
  } else if (process.platform === 'darwin') {
    list.push(
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
    );
  } else {
    list.push(
      '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
      '/usr/bin/chromium', '/usr/bin/chromium-browser', '/snap/bin/chromium',
    );
  }
  return list;
}

function resolveChrome() {
  const env = (process.env.CHROME || '').trim();
  if (env) {
    if (!fs.existsSync(env)) {
      throw new Error(`CHROME 指向的文件不存在：${env}`);
    }
    return env;
  }
  for (const c of candidates()) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error(
    '找不到 Chrome，请用 CHROME 环境变量指定可执行文件：\n'
    + '  Windows : set CHROME=C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\n'
    + '  macOS   : export CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"\n'
    + '  Linux   : export CHROME=/usr/bin/google-chrome',
  );
}

module.exports = { resolveChrome };
