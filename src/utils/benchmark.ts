// Micro-benchmark helpers: warm-up, auto-calibrated batch size, interleaved A/B samples, median.

export interface MeasureResult {
  medianMs: number; // time per single call of fn
  minMs: number;
  maxMs: number;
  iterations: number; // calls of fn per timed sample
  samples: number;
  belowResolution: boolean;
}

export interface CompareResult {
  active: MeasureResult;
  baseline: MeasureResult;
  speedup: number | null; // baseline / active, null when either side is not measurable
}

export interface BenchmarkOptions {
  warmup?: number;
  samples?: number;
  minSampleMs?: number;
  maxIterations?: number;
}

// Number of timed samples per benchmark (the median of these is reported)
export const BENCH_SAMPLES = 7;

// Accumulates every return value so the JIT cannot discard the benchmarked work as dead code.
let benchSink = 0;
export const readBenchSink = (): number => benchSink;

let cachedResolutionMs: number | null = null;

// Smallest observable step of performance.now() (≈0.005–0.1 ms in Chrome, up to 1 ms in Safari/Firefox).
export function getTimerResolutionMs(): number {
  if (cachedResolutionMs !== null) return cachedResolutionMs;
  let best = Infinity;
  for (let k = 0; k < 10; k++) {
    const a = performance.now();
    let b = performance.now();
    while (b === a) b = performance.now();
    best = Math.min(best, b - a);
  }
  cachedResolutionMs = best;
  return best;
}

function timeBatch(fn: () => number, iterations: number): number {
  let acc = 0;
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) acc += fn();
  const elapsed = performance.now() - t0;
  benchSink += acc;
  return elapsed;
}

// Doubles the batch size until one timed batch lasts at least minSampleMs.
function calibrate(fn: () => number, minSampleMs: number, maxIterations: number) {
  let iterations = 1;
  let batchMs = timeBatch(fn, iterations);
  while (batchMs < minSampleMs && iterations < maxIterations) {
    iterations *= 2;
    batchMs = timeBatch(fn, iterations);
  }
  return { iterations, batchMs };
}

function summarize(perCallMs: number[], iterations: number, belowResolution: boolean): MeasureResult {
  const sorted = [...perCallMs].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const medianMs = sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return {
    medianMs,
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    iterations,
    samples: sorted.length,
    belowResolution,
  };
}

export function measure(
  fn: () => number,
  options: BenchmarkOptions = {}
): MeasureResult {
  const resolution = getTimerResolutionMs();
  const warmup = options.warmup ?? 5;
  const samples = options.samples ?? BENCH_SAMPLES;
  const minSampleMs = options.minSampleMs ?? Math.max(5, 20 * resolution);
  const maxIterations = options.maxIterations ?? 1 << 20;

  for (let i = 0; i < warmup; i++) {
    benchSink += fn();
  }

  const c = calibrate(fn, minSampleMs, maxIterations);
  const sampleMs: number[] = [];
  for (let s = 0; s < samples; s++) {
    sampleMs.push(timeBatch(fn, c.iterations) / c.iterations);
  }
  return summarize(sampleMs, c.iterations, c.batchMs < 2 * resolution);
}

export function compareBenchmarks(
  active: () => number,
  baseline: () => number,
  options: BenchmarkOptions = {}
): CompareResult {
  const resolution = getTimerResolutionMs();
  const warmup = options.warmup ?? 5;
  const samples = options.samples ?? BENCH_SAMPLES;
  // ≥ 20 timer ticks per sample keeps quantization error ≤ 5 %
  const minSampleMs = options.minSampleMs ?? Math.max(5, 20 * resolution);
  const maxIterations = options.maxIterations ?? 1 << 20;

  for (let i = 0; i < warmup; i++) {
    benchSink += active();
    benchSink += baseline();
  }

  const ca = calibrate(active, minSampleMs, maxIterations);
  const cb = calibrate(baseline, minSampleMs, maxIterations);

  // Interleave A/B samples so GC pauses and CPU frequency drift hit both sides equally
  const activeMs: number[] = [];
  const baselineMs: number[] = [];
  for (let s = 0; s < samples; s++) {
    activeMs.push(timeBatch(active, ca.iterations) / ca.iterations);
    baselineMs.push(timeBatch(baseline, cb.iterations) / cb.iterations);
  }

  const a = summarize(activeMs, ca.iterations, ca.batchMs < 2 * resolution);
  const b = summarize(baselineMs, cb.iterations, cb.batchMs < 2 * resolution);
  const speedup =
    a.belowResolution || b.belowResolution || a.medianMs <= 0 ? null : b.medianMs / a.medianMs;
  return { active: a, baseline: b, speedup };
}
