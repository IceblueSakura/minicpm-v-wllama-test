function firstShard(url) {
  const match = new URL(url).pathname.match(/-(\d{5})-of-(\d{5})\.gguf$/i);
  if (match && (match[1] !== '00001' || Number(match[2]) < 1)) throw new Error('远程分片模型请填写第一片 -00001-of-NNNNN.gguf');
}
export function remoteSource({ url, mmprojUrl }, baseURL) {
  const base = new URL(baseURL);
  function check(value) {
    let u;
    try { u = new URL(value); } catch { throw new Error('请输入完整的 HTTP(S) 模型直链'); }
    if (!['https:', 'http:'].includes(u.protocol) || u.username || u.password || u.hash) throw new Error('只接受无用户名、密码和片段的 HTTP(S) 直链');
    if (base.protocol === 'https:' && u.protocol !== 'https:') throw new Error('HTTPS 页面不能下载 HTTP 模型：请使用 HTTPS 直链或本地文件');
    if (!u.pathname.toLowerCase().endsWith('.gguf')) throw new Error('请填写以 .gguf 结尾的文件直链，而不是模型网页');
    return u.href;
  }
  const source = { url: check(url) };
  firstShard(source.url);
  if (mmprojUrl) source.mmprojUrl = check(mmprojUrl);
  if (source.url === source.mmprojUrl) throw new Error('主模型和 mmproj 不能是同一个 URL');
  return source;
}
export async function localSource(modelFiles, projector) {
  const files = [...modelFiles].sort((a, b) => a.name.localeCompare(b.name, 'en'));
  if (!files.length) throw new Error('请选择主模型 GGUF；分片模型须一次选齐所有分片');
  for (const file of [...files, ...(projector ? [projector] : [])]) {
    if (!file.name.toLowerCase().endsWith('.gguf') || file.size < 4) throw new Error(`不是有效 GGUF 文件：${file.name}`);
    const magic = new Uint8Array(await file.slice(0, 4).arrayBuffer());
    if (String.fromCharCode(...magic) !== 'GGUF') throw new Error(`GGUF 文件头不正确：${file.name}`);
  }
  const splits = files.map(f => f.name.match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/i));
  if (files.length > 1 || splits[0]) {
    if (splits.some(s => !s) || splits.some(s => s[1] !== splits[0][1] || s[3] !== splits[0][3]) || Number(splits[0][3]) !== files.length || splits.some((s, i) => Number(s[2]) !== i + 1)) throw new Error('模型分片不完整或来自不同模型；请选择同一组全部分片');
  }
  if (projector && files.some(f => f.name === projector.name)) throw new Error('mmproj 请单独选择，不要重复放入主模型列表');
  return projector ? [...files, projector] : files;
}
export async function acquireRemote(manager, source, options = {}) {
  // wllama 3.6.1 getModelOrDownload only compares the main URL. Match the
  // projector too, otherwise changing/clearing it silently reuses the old one.
  const cached = (await manager.getModels()).find(m => m.url === source.url && (m.mmprojUrl || '') === (source.mmprojUrl || ''));
  if (cached) {
    options.progressCallback?.({ loaded: cached.size, total: cached.size });
    return cached;
  }
  return manager.downloadModel(source, options);
}
export function safeSourceLabel(source) {
  return Object.fromEntries(Object.entries(source).map(([key, value]) => { const u = new URL(value); return [key, `${u.origin}${u.pathname}`]; }));
}
export function redactURLs(text) {
  return text.replace(/https?:\/\/[^\s"'<>]+/g, value => {
    try { const u = new URL(value); return `${u.origin}${u.pathname}${u.search ? '?[redacted]' : ''}`; } catch { return '[URL]'; }
  });
}
