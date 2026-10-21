"use strict";
// Local microphone capture -> 16 kHz mono PCM WAV. No network calls.
// Uses ScriptProcessorNode for compatibility with the supplied offline browsers.

function pcmWavFromChunks(chunks, sampleRate) {
  const total = chunks.reduce((s, x) => s + x.length, 0);
  if (!total) throw new Error("Микрофон не записал звуковые данные.");
  const raw = new Float32Array(total);
  let cursor = 0;
  for (const chunk of chunks) { raw.set(chunk, cursor); cursor += chunk.length; }
  const rate = 16000;
  const outputLength = Math.floor(raw.length * rate / sampleRate);
  const values = new Int16Array(outputLength);
  const ratio = sampleRate / rate;
  for (let i = 0; i < outputLength; i++) {
    const start = i * ratio;
    let value = 0;
    if (ratio >= 1) {
      const a = Math.floor(start);
      const b = Math.min(raw.length, Math.max(a + 1, Math.floor((i + 1) * ratio)));
      for (let j = a; j < b; j++) value += raw[j];
      value /= b - a;
    } else {
      const a = Math.min(raw.length - 1, Math.floor(start));
      const b = Math.min(raw.length - 1, a + 1);
      const alpha = start - Math.floor(start);
      value = raw[a] * (1 - alpha) + raw[b] * alpha;
    }
    value = Math.max(-1, Math.min(1, value));
    values[i] = Math.round(value < 0 ? value * 32768 : value * 32767);
  }
  const buffer = new ArrayBuffer(44 + values.byteLength);
  const view = new DataView(buffer);
  const text = (offset, s) => { for (let j = 0; j < s.length; j++) view.setUint8(offset+j, s.charCodeAt(j)); };
  text(0, "RIFF"); view.setUint32(4, 36 + values.byteLength, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, values.byteLength, true);
  for (let i = 0; i < values.length; i++) view.setInt16(44 + 2*i, values[i], true);
  return new Blob([buffer], { type: "audio/wav" });
}

class VoiceRecorder {
  constructor() { this.active = null; this.starting = false; }
  get isRecording() { return !!this.active; }

  async start({ maxSeconds = 5, onLimit = () => {} } = {}) {
    if (this.active || this.starting) throw new Error("Запись уже идёт.");
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Открой страницу через http://localhost:8000 и разреши микрофон.");
    this.starting = true;
    let stream, context;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
      });
      context = new (window.AudioContext || window.webkitAudioContext)();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const muted = context.createGain();
      muted.gain.value = 0;
      const session = { stream, context, source, processor, muted, chunks: [], count: 0, limitCalled: false };
      processor.onaudioprocess = (e) => {
        if (this.active !== session) return;
        const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
        session.chunks.push(chunk);
        session.count += chunk.length;
        if (!session.limitCalled && session.count >= maxSeconds * context.sampleRate) {
          session.limitCalled = true;
          queueMicrotask(onLimit);
        }
      };
      source.connect(processor);
      processor.connect(muted);
      muted.connect(context.destination);
      this.active = session;
    } catch (err) {
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (context) await context.close().catch(() => {});
      throw err;
    } finally { this.starting = false; }
  }

  async stop() {
    const s = this.active;
    if (!s) throw new Error("Запись ещё не началась.");
    this.active = null;
    s.processor.onaudioprocess = null;
    s.source.disconnect(); s.processor.disconnect(); s.muted.disconnect();
    s.stream.getTracks().forEach(track => track.stop());
    await s.context.close().catch(() => {});
    return pcmWavFromChunks(s.chunks, s.context.sampleRate);
  }
}
window.VoiceRecorder = VoiceRecorder;
