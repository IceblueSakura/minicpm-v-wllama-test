// All durations are CPU monotonic wall-clock or explicitly labeled engine values.
// Stream chunks are NOT tokens. Never infer token counts from text length.
export class RunMetrics {
  constructor(start, config) {
    this.started = start;
    this.config = config;
    this.firstChunk = null;
    this.firstText = null;
    this.lastText = null;
    this.finished = null;
    this.chunks = 0;
    this.textChunks = 0;
    this.engine = {};
    this.usage = null;
    this.events = [];
    this.error = null;
  }
  chunk(chunk, now) {
    this.chunks++;
    this.firstChunk ??= now;
    const text = chunk.choices?.[0]?.delta?.content;
    if (text) {
      this.firstText ??= now;
      this.lastText = now;
      this.textChunks++;
    }
    if (chunk.timings) Object.assign(this.engine, chunk.timings);
    if (chunk.usage) this.usage = chunk.usage;
    if (chunk.prompt_progress) this.progress = { ...chunk.prompt_progress };
    return text || '';
  }
  event(name, now, reportedMs = null) {
    this.events.push({ name, atMs: now - this.started, reportedMs });
  }
  snapshot(now) {
    return {
      config: this.config,
      totalMs: (this.finished ?? now) - this.started,
      ttftMs: this.firstText === null ? null : this.firstText - this.started,
      firstChunkMs: this.firstChunk === null ? null : this.firstChunk - this.started,
      afterFirstTextMs: this.firstText === null ? null : (this.finished ?? now) - this.firstText,
      chunks: this.chunks, textChunks: this.textChunks,
      engine: { ...this.engine }, usage: this.usage, progress: this.progress ? { ...this.progress } : null,
      events: this.events.map(e => ({ ...e })), error: this.error,
      completed: this.finished !== null,
    };
  }
}
export const msText = value => Number.isFinite(value) ? `${(value / 1000).toFixed(2)} s` : '—';
export const rateText = value => Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—';
