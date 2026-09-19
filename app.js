import { Wllama } from './vendor/wllama.js';
import { remoteSource, localSource, acquireRemote, safeSourceLabel, redactURLs } from './sources.js';
const asset = path => new URL(path, import.meta.url).href;
const SERVER_MODELS_AVAILABLE = false; // Build sets false: weights are never published.
import { RunMetrics, msText, rateText } from './metrics.js';
const $ = id => document.getElementById(id);
let model = null, ready = false, busy = false, hasVision = false;
let imageData = null, imageURL = null, originalImage = null, imageStats = null;
let run = null, loadStats = null, loadedConfig = null, loadNumber = 0, runNumber = 0;
let output = '', gpuInfo = null;
const history = [], evidence = [], logLines = [];
const loadFields = ['backend', 'context', 'vision-gpu', 'batch', 'ubatch', 'flash', 'source-kind', 'model-url', 'mmproj-url', 'model-files', 'mmproj-file'];
const inputFields = ['image', 'example', 'clear', 'image-size', 'prompt', 'tokens', 'cache-prompt'];
const time = () => performance.now();
// Minimal WASM module that requires the Memory64 feature (memory section, flags 0x04).
const HAS_MEM64 = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 1]));
if (!HAS_MEM64 && 'Suspending' in WebAssembly) {
  // Safari 26+: has JSPI, lacks Memory64, and its Memory constructor silently ignores
  // address:'i64', so wllama 3.6.1's support probe misreports and picks the Memory64
  // build ("CompileError: Memory64 is not enabled"). Hiding Suspending makes
  // needCompat() select the Asyncify compat build instead.
  delete WebAssembly.Suspending;
}
function update() {
  $('load').disabled = busy || ready;
  $('unload').disabled = busy || !model;
  $('run').disabled = busy || !ready;
  $('probe').disabled = busy;
  for (const id of loadFields) $(id).disabled = busy || ready;
  for (const id of inputFields) $(id).disabled = busy;
  $('export').disabled = !history.length;
}
function number(id, min, max) {
  const n = Number($(id).value);
  if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${id} 必须在 ${min} 到 ${max} 之间`);
  return n;
}
function log(...args) {
  const text = redactURLs(args.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' '));
  logLines.push(text);
  if (logLines.length > 500) logLines.splice(0, logLines.length - 500);
  if (run && run.finished === null) {
    const decoded = text.match(/image decoded \(batch \d+\/\d+\) in ([\d.]+) ms/);
    if (decoded) run.event('图像 token prefill 完成', time(), Number(decoded[1]));
    if (/decoding image batch/.test(text)) run.event('图像 token prefill 开始（日志到达）', time());
  }
  if (/using .*backend|offloaded .*layers|ggml_webgpu:|WebGPU.*adapter|device.*lost/i.test(text)) {
    evidence.push(text);
    if (evidence.length > 35) evidence.shift();
    $('backend-evidence').textContent = evidence.join('\n');
  }
}
function fail(error) {
  $('status').textContent = `失败：${redactURLs(String(error.message || error))}`;
  log(error.stack || String(error));
  console.error(redactURLs(String(error.message || error)));
}
async function withBusy(fn) {
  busy = true; update();
  try { await fn(); } catch (error) { fail(error); }
  finally { busy = false; update(); render(); }
}
function render() {
  if ($('output').textContent !== output) $('output').textContent = output;
  if ($('log-panel').open) $('logs').textContent = logLines.join('\n');
  if (!run) return;
  const s = run.snapshot(time());
  $('metric-ttft').textContent = msText(s.ttftMs);
  $('metric-speed').textContent = rateText(s.engine.predicted_per_second);
  $('metric-total').textContent = msText(s.totalMs);
  $('metric-after').textContent = msText(s.afterFirstTextMs);
  $('metric-prefill').textContent = msText(s.engine.prompt_ms);
  const encoded = s.events.filter(e => e.name === '图像 token prefill 完成' && Number.isFinite(e.reportedMs));
  $('metric-vision').textContent = encoded.length ? msText(encoded.reduce((n, e) => n + e.reportedMs, 0)) : '—';
  $('metric-tokens').textContent = s.engine.predicted_n ?? s.usage?.completion_tokens ?? '—';
  $('metric-cache').textContent = `${s.engine.prompt_n ?? '—'} / ${s.engine.cache_n ?? '—'}`;
  $('timeline').textContent = s.events.map(e => `${msText(e.atMs).padStart(10)}  ${e.name}${e.reportedMs === null ? '' : `（底层报告 ${msText(e.reportedMs)}）`}`).join('\n');
  if (!s.completed) {
    const progress = s.progress ? ` · prompt ${s.progress.processed}/${s.progress.total}（缓存 ${s.progress.cache}）` : '';
    $('phase').textContent = s.ttftMs === null ? `等待首字 ${msText(s.totalMs)}${progress}` : `正在生成 · ${msText(s.totalMs)}`;
  }
}
setInterval(render, 150); // Batch UI and log work; never rewrite the DOM on every native log/token.
$('log-panel').addEventListener('toggle', render);
$('capabilities').textContent = `跨源隔离：${crossOriginIsolated} · WebGPU API：${!!navigator.gpu} · JSPI：${typeof WebAssembly.Suspending === 'function'} · Memory64：${HAS_MEM64}`;
async function probeGPU() {
  if (!navigator.gpu) throw new Error('浏览器没有 WebGPU API；请使用安全上下文或选择 CPU');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter 返回 null：没有可用 WebGPU adapter，不会静默当作 GPU 测试');
  const info = adapter.info;
  gpuInfo = {
    vendor: info?.vendor, architecture: info?.architecture, device: info?.device,
    description: info?.description, isFallbackAdapter: adapter.isFallbackAdapter ?? info?.isFallbackAdapter ?? null,
    shaderF16: adapter.features.has('shader-f16'), timestampQuery: adapter.features.has('timestamp-query'),
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
  $('gpu-info').textContent = `${JSON.stringify(gpuInfo, null, 2)}\n此处为独立 adapter 探测，不代表 wllama 实际使用相同设备；以下方 native 日志为准。`;
  return gpuInfo;
}
$('probe').onclick = () => withBusy(probeGPU);
function sourceChanged() {
  for (const kind of ['url', 'files', 'server']) $(`source-${kind}`).hidden = $('source-kind').value !== kind;
}
$('source-kind').onchange = sourceChanged;
if (!SERVER_MODELS_AVAILABLE || !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) $('server-option').remove();
sourceChanged();
// Both wllama 3.6.1 WASM builds import shared memory, so SharedArrayBuffer (cross-origin
// isolation) is required even for single-thread CPU runs. GitHub Pages cannot set COOP/COEP
// headers, so a minimal service worker injects them and the page reloads itself (bounded).
async function ensureCrossOriginIsolated() {
  if (crossOriginIsolated) { sessionStorage.removeItem('coi-reloads'); return; }
  if (!('serviceWorker' in navigator)) {
    $('coi-note').textContent = '此页面未跨源隔离，且浏览器不支持 Service Worker：无法启用 SharedArrayBuffer，wllama 无法运行。请使用开发服务器或支持 Service Worker 的浏览器。';
    return;
  }
  const attempts = Number(sessionStorage.getItem('coi-reloads') || 0);
  if (attempts >= 3) {
    $('coi-note').textContent = '跨源隔离启用失败（已自动重试 3 次）：请清除本站点数据后刷新，或改用开发服务器。';
    return;
  }
  $('coi-note').textContent = '正在启用跨源隔离（注入 COOP/COEP），页面将自动刷新…';
  try {
    await navigator.serviceWorker.register(asset('./coi-sw.js'));
    sessionStorage.setItem('coi-reloads', String(attempts + 1));
    location.reload();
  } catch (error) {
    $('coi-note').textContent = `跨源隔离启用失败：${error.message}`;
  }
}
await ensureCrossOriginIsolated();
if (crossOriginIsolated) $('coi-note').textContent = '已跨源隔离（SharedArrayBuffer 可用）。CPU 固定单线程；WebGPU 可选。';
$('load').onclick = () => withBusy(async () => {
  const start = time();
  evidence.length = 0;
  $('backend-evidence').textContent = '等待底层 backend 日志';
  $('status').textContent = '检查配置…';
  try {
    const gpu = $('backend').value === 'gpu';
    const options = {
      n_ctx: number('context', 1024, 16384), n_threads: 1, n_parallel: 1,
      n_gpu_layers: gpu ? 99999 : 0, mmproj_offload: gpu && $('vision-gpu').checked,
      n_batch: number('batch', 32, 4096), n_ubatch: number('ubatch', 32, 2048),
      jinja: true, reasoning: false, log_level: 2,
    };
    if (options.n_ubatch > options.n_batch) throw new Error('n_ubatch 不应大于 n_batch');
    if (gpu) {
      const info = await probeGPU();
      if (!info.shaderF16) throw new Error('当前 GPU adapter 缺少 shader-f16，wllama 无法启用 WebGPU；请改用 CPU 或检查浏览器 GPU 配置');
      options.offload_kqv = true;
    }
    if ($('flash').value !== 'auto') options.flash_attn = $('flash').value === 'on';
    loadStats = { totalMs: null, acquireMs: null, initializeMs: null };
    model = new Wllama({ default: asset('./vendor/wllama.wasm') }, {
      logger: { log, debug: log, warn: log, error: log },
    });
    model.setCompat({ wasm: asset('./vendor/compat.wasm'), worker: asset('./vendor/compat.js') });
    const acquireStart = time();
    let cachedModel, sourceSummary, requiresVision;
    const kind = $('source-kind').value;
    if (kind === 'files') {
      const files = [...$('model-files').files], projector = $('mmproj-file').files[0];
      cachedModel = await localSource(files, projector);
      sourceSummary = { kind, files: cachedModel.map(f => ({ name: f.name, bytes: f.size })) };
      requiresVision = !!projector;
      $('status').textContent = '读取本地 GGUF（不上传）…';
    } else {
      let source;
      if (kind === 'url') source = remoteSource({ url: $('model-url').value, mmprojUrl: $('mmproj-url').value }, location.href);
      else if (kind === 'server' && SERVER_MODELS_AVAILABLE && ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) source = { url: asset('./MiniCPM-V-4_6-Q8_0.gguf'), mmprojUrl: asset('./mmproj-MiniCPM-V-4_6-F16.gguf') };
      else throw new Error('不支持的模型来源');
      sourceSummary = { kind, ...safeSourceLabel(source) };
      requiresVision = !!source.mmprojUrl;
      $('status').textContent = '获取模型（下载或浏览器缓存）…';
      cachedModel = await acquireRemote(model.modelManager, source, { progressCallback: ({ loaded, total }) => {
        if (total > 0) $('progress').value = loaded / total; else $('progress').removeAttribute('value');
        $('status').textContent = `获取模型：${(loaded / 1e6).toFixed(0)} / ${total > 0 ? (total / 1e6).toFixed(0) : '未知'} MB`;
      } });
    }
    loadStats.acquireMs = time() - acquireStart;
    $('load-metrics').textContent = `模型获取 ${msText(loadStats.acquireMs)} · WASM / 模型 / GPU 初始化中…`;
    $('status').textContent = '初始化 WASM、模型与计算设备…';
    const initStart = time();
    await model.loadModel(cachedModel, options);
    loadStats.initializeMs = time() - initStart;
    loadStats.totalMs = time() - start;
    hasVision = model.supportInputModality('image');
    if (requiresVision && !hasVision) throw new Error('已指定 mmproj，但模型未报告图像支持，请核对模型配对和格式');
    loadedConfig = { source: sourceSummary, hasVision, backend: gpu ? 'WebGPU requested' : 'CPU', ...options, actualThreads: model.getNumThreads(), multithread: model.isMultithread(), loadNumber: ++loadNumber };
    runNumber = 0;
    $('load-metrics').textContent = `模型获取（含缓存） ${msText(loadStats.acquireMs)} · 初始化 ${msText(loadStats.initializeMs)} · 加载总计 ${msText(loadStats.totalMs)} · 实际 CPU 线程 ${loadedConfig.actualThreads}`;
    ready = true; $('progress').value = 1;
    $('status').textContent = `已加载 · ${gpu ? 'WebGPU 请求模式，实际卸载见日志' : 'CPU'}`;
  } catch (error) {
    if (model) { try { await model.exit(); } catch (e) { log(String(e)); } }
    model = null; ready = false; throw error;
  }
});
$('reset').onclick = () => location.reload();
$('unload').onclick = () => withBusy(async () => {
  await model.exit(); model = null; ready = false;
  $('status').textContent = '模型已卸载，下载缓存保留';
});
async function setImage(blob) {
  if (!blob) {
    originalImage = null; imageData = null; imageStats = null;
    if (imageURL) URL.revokeObjectURL(imageURL);
    imageURL = null; $('preview').removeAttribute('src'); $('preview').style.display = 'none';
    $('image-metrics').textContent = '无图片 · 纯文本模式'; return;
  }
  const start = time();
  const bitmap = await createImageBitmap(blob);
  const decodeEnd = time();
  const maxSide = Number($('image-size').value);
  const scale = maxSide ? Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height)) : 1;
  const original = { width: bitmap.width, height: bitmap.height, bytes: blob.size };
  const width = Math.max(1, Math.round(bitmap.width * scale)), height = Math.max(1, Math.round(bitmap.height * scale));
  let prepared = blob, transformMs = 0;
  try {
    // Avoid a PNG encode + decode roundtrip if the original already fits.
    // WebP is normalized to PNG because support in the WASM decoder can differ.
    if (scale < 1 || !['image/png', 'image/jpeg'].includes(blob.type)) {
      const transformStart = time();
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0, width, height);
      prepared = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!prepared) throw new Error('图片转换失败');
      transformMs = time() - transformStart;
    }
  } finally { bitmap.close(); }
  const readStart = time();
  const data = await prepared.arrayBuffer();
  const end = time();
  if (imageURL) URL.revokeObjectURL(imageURL);
  originalImage = blob; imageData = data; imageURL = URL.createObjectURL(prepared);
  imageStats = { original, width, height, bytes: data.byteLength, decodeMs: decodeEnd - start, transformMs, readMs: end - readStart, totalMs: end - start, passthrough: prepared === blob };
  $('preview').src = imageURL; $('preview').style.display = 'block';
  $('image-metrics').textContent = `${original.width}×${original.height} → ${width}×${height} · 浏览器解码 ${msText(imageStats.decodeMs)} · 缩放/重编码 ${msText(transformMs)} · 读取 ${msText(imageStats.readMs)} · 预处理合计 ${msText(imageStats.totalMs)}${imageStats.passthrough ? ' · 原编码直传' : ''}（不计入 TTFT）`;
}
$('image').onchange = () => withBusy(() => setImage($('image').files[0] || null));
$('image-size').onchange = () => withBusy(() => setImage(originalImage));
$('clear').onclick = () => withBusy(async () => { await setImage(null); $('image').value = ''; });
$('example').onclick = () => withBusy(async () => {
  const res = await fetch(asset('./example.png')); if (!res.ok) throw new Error('示例图片读取失败');
  await setImage(await res.blob()); $('image').value = '';
});
function addHistory(snapshot) {
  history.push(snapshot);
  const tr = document.createElement('tr');
  const values = [history.length, `${snapshot.config.backend} / CPU ${snapshot.config.n_threads} / batch ${snapshot.config.n_batch}:${snapshot.config.n_ubatch}`, snapshot.config.image ? `${snapshot.config.image.width}×${snapshot.config.image.height}` : '文本', msText(snapshot.ttftMs), msText(snapshot.totalMs), rateText(snapshot.engine.predicted_per_second), snapshot.engine.cache_n ?? '—', snapshot.error ? '失败' : `完成 · 此次加载第 ${snapshot.config.runNumber} 次`];
  for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
  $('history').prepend(tr);
}
$('run').onclick = () => withBusy(async () => {
  const prompt = $('prompt').value.trim();
  if (!prompt) throw new Error('请输入提示词');
  if (imageData && !hasVision) throw new Error('当前模型没有视觉输入能力，请清除图片或加载配套 mmproj');
  const maxTokens = number('tokens', 1, 2048);
  const cachePrompt = $('cache-prompt').checked;
  output = ''; $('timings').textContent = '';
  run = new RunMetrics(time(), { ...loadedConfig, image: imageStats ? structuredClone(imageStats) : null, cachePrompt, maxTokens, runNumber: ++runNumber });
  run.event('请求开始', time());
  $('status').textContent = '正在推理…';
  try {
    const content = imageData ? [{ type: 'image', data: imageData.slice(0) }, { type: 'text', text: prompt }] : prompt;
    const stream = await model.createChatCompletion({
      messages: [{ role: 'user', content }], max_tokens: maxTokens, temperature: 0.2, seed: 42,
      cache_prompt: cachePrompt, stream: true, timings_per_token: true, return_progress: true,
    });
    for await (const chunk of stream) {
      const now = time(), first = run.firstText === null;
      const text = run.chunk(chunk, now);
      if (first && text) run.event('首个非空文本 delta（TTFT）', now);
      output += text;
    }
    run.finished = time(); run.event('流结束', run.finished);
    $('status').textContent = '生成完成'; $('phase').textContent = '完成';
    $('timings').textContent = `总耗时 ${msText(run.finished - run.started)} · TTFT ${msText(run.firstText === null ? null : run.firstText - run.started)}`;
  } catch (error) {
    run.error = redactURLs(String(error.message || error)); run.finished = time();
    run.event('请求失败', run.finished); $('phase').textContent = '失败'; throw error;
  } finally { addHistory(run.snapshot(time())); render(); }
});
const report = () => ({ schemaVersion: 1, userAgent: navigator.userAgent, gpuProbe: gpuInfo, load: loadStats, backendEvidence: [...evidence], runs: history, current: run?.snapshot(time()) ?? null });
window.getPerformanceReport = report; // Read-only diagnostic snapshot; no prompt/image contents.
window.getDiagnosticLogs = () => logLines.join('\n');
$('export').onclick = () => {
  const blob = new Blob([JSON.stringify(report(), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'minicpm-performance.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
update();
