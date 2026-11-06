import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Cpu,
  ChevronUp,
  ChevronDown,
  RotateCcw,
  Zap,
  Gauge,
  BarChart3,
  Play,
  BookmarkCheck,
  CheckCircle2,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { compareBenchmarks } from '../utils/benchmark';
import { mulberry32, hashSeed } from '../utils/random';
import {
  perfTracker,
  SimulationPerfSnapshot,
  TSComplexSNN,
  ComplexLIFNeuron,
  generateSpectralSignal,
  complex,
  Complex,
  computeSurrogateDerivative,
} from '../simulationEngine';

export type ActiveModelTabId =
  | 'theory'
  | 'comparison'
  | 'neuron'
  | 'network'
  | 'vision'
  | 'hardware'
  | 'choi-lab'
  | 'pytorch';

interface PerformanceMonitorProps {
  activeTab: ActiveModelTabId;
}

interface ChromePerformanceMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

interface ModelSpec {
  id: ActiveModelTabId;
  code: string;
  nameEn: string;
  nameVi: string;
  opDescEn: string;
  opDescVi: string;
  baselineDescEn: string;
  baselineDescVi: string;
  topology: string;
  complexParams: number;
  baseTensorKB: number;
  timeSteps: number;
}

export interface ModuleBenchmarkResult {
  id: ActiveModelTabId;
  code: string;
  nameEn: string;
  nameVi: string;
  avgExecMs: number; // MEDIAN time per call (ms) of the active kernel
  baselineMs: number; // baseline shown (custom or measured)
  measuredBaselineMs: number; // median time per call of the measured baseline
  minExecMs: number;
  maxExecMs: number;
  opsCount: number;
  speedupRatio: number | null; // null = below timer resolution
  deltaPercent: number | null; // negative = faster than baseline
  activeBelowResolution: boolean;
  baselineBelowResolution: boolean;
  timestamp: number;
}

const ALL_TAB_IDS: ActiveModelTabId[] = [
  'theory',
  'comparison',
  'neuron',
  'network',
  'vision',
  'hardware',
  'choi-lab',
  'pytorch',
];

const MODEL_SPECS: Record<ActiveModelTabId, ModelSpec> = {
  theory: {
    id: 'theory',
    code: 'M01',
    nameEn: 'Phasor Algebra Kernel',
    nameVi: 'Nhân Đại Số Phasor',
    opDescEn: '2,500× Karatsuba 3M Complex Mul + Von Mises Surrogate',
    opDescVi: '2.500× Nhân phức Karatsuba 3M + Surrogate Von Mises',
    baselineDescEn: 'Same loop, Standard 4M Complex Mul',
    baselineDescVi: 'Cùng vòng lặp, nhân phức chuẩn 4M',
    topology: '1×1 Complex Phasor',
    complexParams: 4,
    baseTensorKB: 2.4,
    timeSteps: 50,
  },
  comparison: {
    id: 'comparison',
    code: 'M02',
    nameEn: 'Dual LIF vs cLIF Soma',
    nameVi: 'Đối Chiếu Soma LIF vs cLIF',
    opDescEn: '400-step cLIF recurrence in Cartesian form',
    opDescVi: '400 bước cLIF dạng Descartes',
    baselineDescEn: 'Same recurrence in polar form (cos/sin/atan2 per step)',
    baselineDescVi: 'Cùng hệ thức dạng cực (cos/sin/atan2 mỗi bước)',
    topology: '2×1 Parallel Soma',
    complexParams: 8,
    baseTensorKB: 4.8,
    timeSteps: 20,
  },
  neuron: {
    id: 'neuron',
    code: 'M03',
    nameEn: 'cLIF + Bode Sweep',
    nameVi: 'Neuron cLIF & Quét Bode',
    opDescEn: '20-Freq Resonance Sweep (T=100), recursive phasor input',
    opDescVi: 'Quét cộng hưởng 20 tần số (T=100), đầu vào phasor đệ quy',
    baselineDescEn: 'Same sweep, input via cos/sin per step',
    baselineDescVi: 'Cùng phép quét, đầu vào tính cos/sin mỗi bước',
    topology: '1×1 cLIF (T=100)',
    complexParams: 6,
    baseTensorKB: 8.6,
    timeSteps: 100,
  },
  network: {
    id: 'network',
    code: 'M04',
    nameEn: '16×2 Spectral Forward',
    nameVi: 'Lan Truyền Thuận Phổ 16×2',
    opDescEn: '16×2 Karatsuba CV-SNN Forward (T=30)',
    opDescVi: 'Lan truyền thuận CV-SNN 16×2 Karatsuba (T=30)',
    baselineDescEn: 'Same weights, Standard 4-MAC Forward (T=30)',
    baselineDescVi: 'Cùng trọng số, nhân chuẩn 4-MAC (T=30)',
    topology: '16×2 Complex BPTT',
    complexParams: 32,
    baseTensorKB: 18.4,
    timeSteps: 30,
  },
  vision: {
    id: 'vision',
    code: 'M05',
    nameEn: '8×8 Spatial Vision SNN',
    nameVi: 'Thị Giác Pha Không Gian 8×8',
    opDescEn: '64×4 Crossbar (T=15), precomputed phase LUT',
    opDescVi: 'Crossbar 64×4 (T=15), bảng tra pha tính trước',
    baselineDescEn: 'Same crossbar, atan2/cos/sin per pixel per step',
    baselineDescVi: 'Cùng crossbar, tính atan2/cos/sin từng pixel mỗi bước',
    topology: '64×4 Crossbar Array',
    complexParams: 256,
    baseTensorKB: 42.0,
    timeSteps: 15,
  },
  hardware: {
    id: 'hardware',
    code: 'M06',
    nameEn: '4-Quadrant PCM Tile',
    nameVi: 'Ô Nhớ PCM 4-Góc Phần Tư',
    opDescEn: '1,200-cell differential ΔG tile, 1 complex MAC/cell (digital emulation)',
    opDescVi: 'Tile vi sai ΔG 1.200 ô, 1 MAC phức/ô (mô phỏng số)',
    baselineDescEn: '4 single-ended arrays read separately + digital subtraction (emulation)',
    baselineDescVi: 'Đọc riêng 4 mảng đơn cực + trừ số (mô phỏng)',
    topology: '4M Differential Tile',
    complexParams: 128,
    baseTensorKB: 24.5,
    timeSteps: 16,
  },
  'choi-lab': {
    id: 'choi-lab',
    code: 'M07',
    nameEn: 'KAIST Write-Verify',
    nameVi: 'Hiệu Chuẩn Ghi-Đọc KAIST',
    opDescEn: '800× Closed-Loop 1-Step Linear Conductance Calibration',
    opDescVi: '800× Hiệu chuẩn dẫn nạp vòng kín 1 bước KAIST',
    baselineDescEn: 'Iterative Blind Multi-Pulse Programming Loop',
    baselineDescVi: 'Vòng lập trình đa xung mù lặp nhiều chu kỳ',
    topology: '4-Cell Write-Verify',
    complexParams: 16,
    baseTensorKB: 6.2,
    timeSteps: 10,
  },
  pytorch: {
    id: 'pytorch',
    code: 'M08',
    nameEn: 'Wirtinger BPTT Engine',
    nameVi: 'Nhân Huấn Luyện Wirtinger BPTT',
    opDescEn: 'Wirtinger BPTT trainStep, fast-sigmoid surrogate (16×2, T=30)',
    opDescVi: 'Một bước BPTT Wirtinger, surrogate fast-sigmoid (16×2, T=30)',
    baselineDescEn: 'Same trainStep, Von Mises surrogate',
    baselineDescVi: 'Cùng trainStep, surrogate Von Mises',
    topology: '16×2 Complex (T=30)',
    complexParams: 32,
    baseTensorKB: 5.9,
    timeSteps: 30,
  },
};

interface BenchmarkPair {
  active: () => number;
  baseline: () => number;
  opsPerCall: number;
}

/**
 * Builds an (active, baseline) pair that does IDENTICAL work on IDENTICAL seeded inputs,
 * differing only in the variable named in MODEL_SPECS[tabId].opDesc / baselineDesc.
 * Each function returns a number that depends on all its work (consumed by the bench sink).
 */
function buildBenchmarkPair(tabId: ActiveModelTabId): BenchmarkPair {
  const rng = mulberry32(hashSeed(2026, ALL_TAB_IDS.indexOf(tabId)));

  if (tabId === 'theory') {
    const n = 2500;
    const run = (mul: (a: Complex, b: Complex) => Complex) => () => {
      let accRe = 0.4;
      let accIm = -0.3;
      for (let i = 0; i < n; i++) {
        const [pRe, pIm] = mul([accRe, accIm], [0.98, 0.15]);
        const surr = computeSurrogateDerivative(
          Math.sqrt(pRe * pRe + pIm * pIm),
          1.0,
          'von_mises',
          2.5
        );
        accRe = pRe * 0.99 + surr * 0.01;
        accIm = pIm * 0.99 - surr * 0.01;
      }
      return accRe + accIm;
    };
    return { active: run(complex.mulKaratsuba), baseline: run(complex.mul), opsPerCall: n };
  }

  if (tabId === 'comparison') {
    const n = 400;
    const leakMag = 0.88;
    const leakPhase = 0.25;
    const threshold = 1.0;
    const inRe = new Float64Array(n);
    const inIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      inRe[i] = (rng() - 0.5) * 0.7;
      inIm[i] = (rng() - 0.5) * 0.7;
    }
    const neuron = new ComplexLIFNeuron(leakMag, leakPhase, threshold, 'soft');
    return {
      // Cartesian state u = beta*u + x, soft reset along the spike phase
      active: () => {
        neuron.reset();
        let spikes = 0;
        for (let i = 0; i < n; i++) {
          if (neuron.step(inRe[i], inIm[i], false).spiked) spikes++;
        }
        return spikes;
      },
      // Identical recurrence kept in polar form, re-projected with cos/sin/atan2 every step
      baseline: () => {
        let mag = 0;
        let ang = 0;
        let spikes = 0;
        for (let i = 0; i < n; i++) {
          const re = leakMag * mag * Math.cos(ang + leakPhase) + inRe[i];
          const im = leakMag * mag * Math.sin(ang + leakPhase) + inIm[i];
          mag = Math.sqrt(re * re + im * im);
          ang = Math.atan2(im, re);
          if (mag >= threshold) {
            spikes++;
            mag -= threshold;
          }
        }
        return spikes;
      },
      opsPerCall: n,
    };
  }

  if (tabId === 'neuron') {
    const freqs = 20;
    const steps = 100;
    const amp = 0.35;
    const neuron = new ComplexLIFNeuron(0.92, 0.35, 1.1, 'soft');
    return {
      // Input phasor advanced by a precomputed unit rotation (1 complex mul per step)
      active: () => {
        let peak = 0;
        for (let f = 1; f <= freqs; f++) {
          neuron.reset();
          const omega = (2 * Math.PI * f) / steps;
          const cosW = Math.cos(omega);
          const sinW = Math.sin(omega);
          let xRe = amp;
          let xIm = 0;
          for (let t = 0; t < steps; t++) {
            peak = Math.max(peak, neuron.step(xRe, xIm, false).preResetMag);
            const nextRe = xRe * cosW - xIm * sinW;
            xIm = xRe * sinW + xIm * cosW;
            xRe = nextRe;
          }
        }
        return peak;
      },
      // Same sweep, input synthesized with cos/sin at every step
      baseline: () => {
        let peak = 0;
        for (let f = 1; f <= freqs; f++) {
          neuron.reset();
          for (let t = 0; t < steps; t++) {
            const phase = (2 * Math.PI * f * t) / steps;
            const r = neuron.step(amp * Math.cos(phase), amp * Math.sin(phase), false);
            peak = Math.max(peak, r.preResetMag);
          }
        }
        return peak;
      },
      opsPerCall: freqs * steps,
    };
  }

  if (tabId === 'network') {
    const sample = generateSpectralSignal(1, 30, 16, 1.2, 4.5, 1.2, 0.15, rng);
    const netKaratsuba = new TSComplexSNN(16, 2, 0.85, 0.12, 1.0, 'soft', true, 'von_mises');
    const netStandard = new TSComplexSNN(16, 2, 0.85, 0.12, 1.0, 'soft', false, 'von_mises');
    // Same weights on both sides: only the 3M vs 4M synapse kernel differs
    netStandard.weights = netKaratsuba.weights.map(w => w.slice());
    const spikesOf = (net: TSComplexSNN) => () => {
      const res = net.forward(sample.inputSpikeSequence, false);
      return res.spikeCounts[0] + res.spikeCounts[1];
    };
    return { active: spikesOf(netKaratsuba), baseline: spikesOf(netStandard), opsPerCall: 30 * 16 * 2 };
  }

  if (tabId === 'vision') {
    const steps = 15;
    const outputs = 4;
    const pixels = 64;
    const cosLut = new Float64Array(pixels);
    const sinLut = new Float64Array(pixels);
    for (let idx = 0; idx < pixels; idx++) {
      const ang = Math.atan2(Math.floor(idx / 8) - 3.5, (idx % 8) - 3.5);
      cosLut[idx] = Math.cos(ang);
      sinLut[idx] = Math.sin(ang);
    }
    const wRe = new Float64Array(outputs);
    const wIm = new Float64Array(outputs);
    for (let o = 0; o < outputs; o++) {
      wRe[o] = Math.cos(o * 0.4);
      wIm[o] = Math.sin(o * 0.4);
    }
    return {
      // Precomputed spatial phase LUT, rotated by one temporal phasor per step
      active: () => {
        let acc = 0;
        for (let t = 0; t < steps; t++) {
          const cT = Math.cos(t * 0.3);
          const sT = Math.sin(t * 0.3);
          for (let o = 0; o < outputs; o++) {
            let sumRe = 0;
            let sumIm = 0;
            for (let idx = 0; idx < pixels; idx++) {
              const xRe = cosLut[idx] * cT - sinLut[idx] * sT;
              const xIm = sinLut[idx] * cT + cosLut[idx] * sT;
              sumRe += xRe * wRe[o] - xIm * wIm[o];
              sumIm += xRe * wIm[o] + xIm * wRe[o];
            }
            acc += Math.sqrt(sumRe * sumRe + sumIm * sumIm);
          }
        }
        return acc;
      },
      // Same crossbar and weights, spatial phase recomputed with atan2/cos/sin per pixel
      baseline: () => {
        let acc = 0;
        for (let t = 0; t < steps; t++) {
          for (let o = 0; o < outputs; o++) {
            let sumRe = 0;
            let sumIm = 0;
            for (let idx = 0; idx < pixels; idx++) {
              const ang = Math.atan2(Math.floor(idx / 8) - 3.5, (idx % 8) - 3.5) + t * 0.3;
              const xRe = Math.cos(ang);
              const xIm = Math.sin(ang);
              sumRe += xRe * wRe[o] - xIm * wIm[o];
              sumIm += xRe * wIm[o] + xIm * wRe[o];
            }
            acc += Math.sqrt(sumRe * sumRe + sumIm * sumIm);
          }
        }
        return acc;
      },
      opsPerCall: steps * outputs * pixels,
    };
  }

  if (tabId === 'hardware') {
    const cells = 1200;
    const vRe = 0.25;
    const vIm = -0.18;
    const gRePos = new Float64Array(cells);
    const gReNeg = new Float64Array(cells);
    const gImPos = new Float64Array(cells);
    const gImNeg = new Float64Array(cells);
    for (let i = 0; i < cells; i++) {
      gRePos[i] = 20 + rng() * 20;
      gReNeg[i] = 10 + rng() * 10;
      gImPos[i] = 20 + rng() * 20;
      gImNeg[i] = 10 + rng() * 10;
    }
    return {
      // Differential tile: ΔG formed per cell, then one complex MAC
      active: () => {
        let iRe = 0;
        let iIm = 0;
        for (let i = 0; i < cells; i++) {
          const dRe = gRePos[i] - gReNeg[i];
          const dIm = gImPos[i] - gImNeg[i];
          iRe += dRe * vRe - dIm * vIm;
          iIm += dRe * vIm + dIm * vRe;
        }
        return iRe + iIm;
      },
      // Single-ended: 4 arrays read separately (8 partial currents), subtracted digitally
      baseline: () => {
        let iRe = 0;
        let iIm = 0;
        for (let i = 0; i < cells; i++) {
          const a1 = gRePos[i] * vRe;
          const a2 = gReNeg[i] * vRe;
          const a3 = gImPos[i] * vIm;
          const a4 = gImNeg[i] * vIm;
          const b1 = gRePos[i] * vIm;
          const b2 = gReNeg[i] * vIm;
          const b3 = gImPos[i] * vRe;
          const b4 = gImNeg[i] * vRe;
          iRe += a1 - a2 - (a3 - a4);
          iIm += b1 - b2 + (b3 - b4);
        }
        return iRe + iIm;
      },
      opsPerCall: cells,
    };
  }

  if (tabId === 'choi-lab') {
    const cells = 800;
    const pulses = 4;
    const targets = new Float64Array(cells);
    const initial = new Float64Array(cells);
    const pulseNoise = new Float64Array(cells * pulses);
    for (let i = 0; i < cells; i++) {
      targets[i] = 24 + rng() * 16;
      initial[i] = targets[i] + (rng() - 0.5) * 4.8;
      for (let p = 0; p < pulses; p++) pulseNoise[i * pulses + p] = (rng() - 0.5) * 0.3;
    }
    return {
      // 1-step closed-loop linear calibration (same programming noise model as baseline)
      active: () => {
        let errSum = 0;
        for (let i = 0; i < cells; i++) {
          const g = initial[i] - 0.88 * (initial[i] - targets[i]) + pulseNoise[i * pulses];
          errSum += Math.abs(g - targets[i]);
        }
        return errSum;
      },
      // 4-pulse iterative blind programming loop
      baseline: () => {
        let errSum = 0;
        for (let i = 0; i < cells; i++) {
          let g = initial[i];
          for (let p = 0; p < pulses; p++) {
            g -= 0.35 * (g - targets[i]) + pulseNoise[i * pulses + p];
          }
          errSum += Math.abs(g - targets[i]);
        }
        return errSum;
      },
      opsPerCall: cells,
    };
  }

  // pytorch: one Wirtinger BPTT trainStep; only the surrogate differs
  const sample = generateSpectralSignal(0, 30, 16, 1.0, 4.2, 1.3, 0.12, rng);
  const netFast = new TSComplexSNN(16, 2, 0.85, 0.12, 1.0, 'soft', true, 'fast_sigmoid');
  const netVonMises = new TSComplexSNN(16, 2, 0.85, 0.12, 1.0, 'soft', true, 'von_mises');
  const initialWeights = netFast.weights.map(w => w.slice());
  // Every call restarts from the same weights so each call does identical work
  const trainOnce = (net: TSComplexSNN) => () => {
    net.weights = initialWeights.map(w => w.slice());
    return net.trainStep(sample, 0.02, false);
  };
  return { active: trainOnce(netFast), baseline: trainOnce(netVonMises), opsPerCall: 30 * 16 * 2 };
}

// Recomputes speedup/delta against a custom baseline (or the measured one when undefined)
function withBaseline(r: ModuleBenchmarkResult, customBaselineMs?: number): ModuleBenchmarkResult {
  const baselineMs = customBaselineMs ?? r.measuredBaselineMs;
  const measurable =
    !r.activeBelowResolution && (customBaselineMs !== undefined || !r.baselineBelowResolution);
  return {
    ...r,
    baselineMs,
    speedupRatio: measurable ? baselineMs / r.avgExecMs : null,
    deltaPercent: measurable ? ((r.avgExecMs - baselineMs) / baselineMs) * 100 : null,
  };
}

/**
 * Measures the active kernel of a module against its baseline with the calibrated,
 * interleaved median benchmark. No floors, no synthetic telemetry.
 */
function executeModuleBenchmark(
  tabId: ActiveModelTabId,
  customBaselineMap: Partial<Record<ActiveModelTabId, number>>
): ModuleBenchmarkResult {
  const spec = MODEL_SPECS[tabId];
  const pair = buildBenchmarkPair(tabId);
  const { active, baseline } = compareBenchmarks(pair.active, pair.baseline);
  const measured: ModuleBenchmarkResult = {
    id: tabId,
    code: spec.code,
    nameEn: spec.nameEn,
    nameVi: spec.nameVi,
    avgExecMs: active.medianMs,
    baselineMs: baseline.medianMs,
    measuredBaselineMs: baseline.medianMs,
    minExecMs: active.minMs,
    maxExecMs: active.maxMs,
    opsCount: pair.opsPerCall,
    speedupRatio: null,
    deltaPercent: null,
    activeBelowResolution: active.belowResolution,
    baselineBelowResolution: baseline.belowResolution,
    timestamp: Date.now(),
  };
  return withBaseline(measured, customBaselineMap[tabId]);
}

const formatMs = (ms: number): string =>
  ms >= 1 ? `${ms.toFixed(2)} ms` : ms >= 0.001 ? `${(ms * 1000).toFixed(1)} µs` : `${(ms * 1e6).toFixed(0)} ns`;

const formatSpeedup = (r: ModuleBenchmarkResult, isEn: boolean): string =>
  r.speedupRatio === null || r.deltaPercent === null
    ? isEn
      ? 'below timer resolution'
      : 'dưới độ phân giải timer'
    : `${r.speedupRatio.toFixed(2)}× (${r.deltaPercent > 0 ? '+' : ''}${r.deltaPercent.toFixed(1)}%)`;

export default function PerformanceMonitor({ activeTab }: PerformanceMonitorProps) {
  const { language } = useLanguage();
  const isEn = language === 'en';

  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [monitorMode, setMonitorMode] = useState<'telemetry' | 'benchmark'>('telemetry');

  const [fps, setFps] = useState<number>(60);
  const [frameTimeMs, setFrameTimeMs] = useState<number>(16.6);
  const [fpsHistory, setFpsHistory] = useState<number[]>(Array(24).fill(60));
  const [heapUsedMB, setHeapUsedMB] = useState<number | null>(null);
  const [heapTotalMB, setHeapTotalMB] = useState<number | null>(null);
  const [hasNativeMemoryApi, setHasNativeMemoryApi] = useState<boolean>(false);
  const [simSnap, setSimSnap] = useState<SimulationPerfSnapshot>(() =>
    perfTracker.getSnapshot()
  );

  // Benchmarking Suite State
  const [benchmarkResults, setBenchmarkResults] = useState<
    Partial<Record<ActiveModelTabId, ModuleBenchmarkResult>>
  >({});
  const [customBaselines, setCustomBaselines] = useState<
    Partial<Record<ActiveModelTabId, number>>
  >({});
  const customBaselinesRef = useRef(customBaselines);
  const benchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isRunningSuite, setIsRunningSuite] = useState<boolean>(false);
  const [runningModuleId, setRunningModuleId] = useState<ActiveModelTabId | null>(null);
  const [savedBaselineToast, setSavedBaselineToast] = useState<boolean>(false);

  const frameCountRef = useRef<number>(0);
  const lastRafTimeRef = useRef<number>(performance.now());
  const rafIdRef = useRef<number | null>(null);

  const activeSpec = MODEL_SPECS[activeTab] || MODEL_SPECS.theory;
  const activeBenchResult = benchmarkResults[activeTab];

  useEffect(() => {
    customBaselinesRef.current = customBaselines;
  }, [customBaselines]);

  const cancelBenchmarkQueue = () => {
    if (benchTimerRef.current !== null) {
      clearTimeout(benchTimerRef.current);
      benchTimerRef.current = null;
    }
  };

  // Runs one module per macrotask so the browser can paint between modules
  const runBenchmarkQueue = (ids: ActiveModelTabId[], onDone?: () => void) => {
    cancelBenchmarkQueue();
    let idx = 0;
    const stepNext = () => {
      if (idx >= ids.length) {
        benchTimerRef.current = null;
        setRunningModuleId(null);
        setSimSnap(perfTracker.getSnapshot());
        onDone?.();
        return;
      }
      const tabId = ids[idx];
      setRunningModuleId(tabId);
      benchTimerRef.current = setTimeout(() => {
        const res = executeModuleBenchmark(tabId, customBaselinesRef.current);
        setBenchmarkResults(prev => ({ ...prev, [tabId]: res }));
        idx += 1;
        stepNext();
      }, 16);
    };
    stepNext();
  };

  // Initial measurement after first paint; cancelled on unmount (and on StrictMode re-mount)
  useEffect(() => {
    runBenchmarkQueue(ALL_TAB_IDS);
    return () => {
      cancelBenchmarkQueue();
      if (toastTimerRef.current !== null) clearTimeout(toastTimerRef.current);
    };
  }, []);

  // Real-time RAF loop for FPS, Frame Time, Memory Usage, and SimulationEngine polling
  useEffect(() => {
    const updateLoop = (now: number) => {
      frameCountRef.current += 1;
      const elapsed = now - lastRafTimeRef.current;

      if (elapsed >= 500) {
        const currentFps = Math.min(
          120,
          Math.max(1, Math.round((frameCountRef.current * 1000) / elapsed))
        );
        const avgFrameMs = elapsed / Math.max(1, frameCountRef.current);

        setFps(currentFps);
        setFrameTimeMs(avgFrameMs);
        setFpsHistory(prev => [...prev.slice(-23), currentFps]);

        const snap = perfTracker.getSnapshot();
        setSimSnap(snap);

        const perfWithMem = performance as Performance & {
          memory?: ChromePerformanceMemory;
        };
        if (perfWithMem.memory && perfWithMem.memory.usedJSHeapSize > 0) {
          setHasNativeMemoryApi(true);
          setHeapUsedMB(perfWithMem.memory.usedJSHeapSize / (1024 * 1024));
          setHeapTotalMB(perfWithMem.memory.totalJSHeapSize / (1024 * 1024));
        } else {
          // performance.memory is Chrome-only; never synthesize a value
          setHasNativeMemoryApi(false);
          setHeapUsedMB(null);
          setHeapTotalMB(null);
        }

        frameCountRef.current = 0;
        lastRafTimeRef.current = now;
      }

      rafIdRef.current = requestAnimationFrame(updateLoop);
    };

    rafIdRef.current = requestAnimationFrame(updateLoop);
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, []);

  const handleBenchmarkActiveModule = () => {
    if (runningModuleId !== null || isRunningSuite) return;
    runBenchmarkQueue([activeTab]);
  };

  const handleRunAllModulesSuite = () => {
    if (runningModuleId !== null || isRunningSuite) return;
    setIsRunningSuite(true);
    setMonitorMode('benchmark');
    runBenchmarkQueue(ALL_TAB_IDS, () => setIsRunningSuite(false));
  };

  const handleLockCurrentAsBaseline = () => {
    const nextBaselines: Partial<Record<ActiveModelTabId, number>> = {};
    ALL_TAB_IDS.forEach(id => {
      const r = benchmarkResults[id];
      if (r) nextBaselines[id] = r.avgExecMs;
    });
    setCustomBaselines(nextBaselines);
    customBaselinesRef.current = nextBaselines;
    setBenchmarkResults(prev => {
      const updated: Partial<Record<ActiveModelTabId, ModuleBenchmarkResult>> = {};
      ALL_TAB_IDS.forEach(id => {
        const r = prev[id];
        if (r) updated[id] = withBaseline(r, nextBaselines[id]);
      });
      return updated;
    });
    setSavedBaselineToast(true);
    if (toastTimerRef.current !== null) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setSavedBaselineToast(false), 1800);
  };

  const handleResetTelemetry = () => {
    perfTracker.resetPeak();
    setCustomBaselines({});
    customBaselinesRef.current = {};
    // Restore measured baselines; re-measuring is an explicit user action ("Run suite")
    setBenchmarkResults(prev => {
      const restored: Partial<Record<ActiveModelTabId, ModuleBenchmarkResult>> = {};
      ALL_TAB_IDS.forEach(id => {
        const r = prev[id];
        if (r) restored[id] = withBaseline(r);
      });
      return restored;
    });
    setSimSnap(perfTracker.getSnapshot());
  };

  const activeTensorKB = Math.max(
    activeSpec.baseTensorKB,
    simSnap.activeTensorBytes / 1024
  );

  const fpsColor =
    fps >= 50
      ? 'text-emerald-400'
      : fps >= 30
        ? 'text-amber-400'
        : 'text-rose-400';

  const computeColor =
    simSnap.lastComputeMs < 2.0
      ? 'text-elegant-cyan'
      : simSnap.lastComputeMs < 8.0
        ? 'text-amber-400'
        : 'text-rose-400';

  const formatCompactNumber = (num: number): string => {
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
    return `${num}`;
  };

  // Aggregate suite statistics (geometric mean of per-module speedups)
  const allResultsList = ALL_TAB_IDS.map(id => benchmarkResults[id]).filter(
    (r): r is ModuleBenchmarkResult => r !== undefined
  );
  const maxChartMs = Math.max(
    1e-6,
    ...allResultsList.map(r => Math.max(r.avgExecMs, r.baselineMs))
  );
  const suiteTotalActiveMs = allResultsList.reduce((acc, r) => acc + r.avgExecMs, 0);
  const suiteTotalBaselineMs = allResultsList.reduce((acc, r) => acc + r.baselineMs, 0);
  const measurableSpeedups = allResultsList
    .map(r => r.speedupRatio)
    .filter((s): s is number => s !== null && s > 0);
  const suiteOverallSpeedup =
    measurableSpeedups.length > 0
      ? Math.exp(measurableSpeedups.reduce((acc, s) => acc + Math.log(s), 0) / measurableSpeedups.length)
      : null;
  const suiteOverallDeltaPercent = suiteOverallSpeedup !== null ? (1 / suiteOverallSpeedup - 1) * 100 : null;

  return (
    <aside
      aria-label={isEn ? 'SNN Performance Monitor' : 'Bộ Giám Sát Hiệu Năng SNN'}
      className="fixed bottom-3 right-3 z-50 font-mono text-xs select-none max-w-[calc(100vw-1.5rem)]"
    >
      {/* EXPANDED SCIENTIFIC TELEMETRY & BENCHMARK SUITE PANEL */}
      {isExpanded && (
        <div className="mb-2 w-[370px] sm:w-[440px] bg-elegant-card/95 backdrop-blur-md border border-slate-700/80 rounded-xl shadow-2xl p-3.5 space-y-3 text-elegant-text">
          {/* Header Row with Mode Switcher */}
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5 gap-2">
            <div className="flex items-center gap-1 bg-elegant-bg p-0.5 rounded-lg border border-slate-800/80">
              <button
                onClick={() => setMonitorMode('telemetry')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-colors cursor-pointer ${
                  monitorMode === 'telemetry'
                    ? 'bg-elegant-card text-elegant-cyan border border-slate-700/70'
                    : 'text-elegant-muted hover:text-white'
                }`}
              >
                <Gauge className="w-3 h-3" />
                <span>{isEn ? 'Telemetry' : 'Thời Gian Thực'}</span>
              </button>
              <button
                onClick={() => setMonitorMode('benchmark')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-colors cursor-pointer ${
                  monitorMode === 'benchmark'
                    ? 'bg-elegant-card text-elegant-cyan border border-slate-700/70'
                    : 'text-elegant-muted hover:text-white'
                }`}
              >
                <BarChart3 className="w-3 h-3" />
                <span>{isEn ? 'Benchmark Suite' : 'Đo Kiểm Chuẩn'}</span>
              </button>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={handleRunAllModulesSuite}
                disabled={isRunningSuite}
                className="px-2 py-1 text-[10px] font-semibold rounded bg-elegant-cyan/15 hover:bg-elegant-cyan/25 text-elegant-cyan border border-elegant-cyan/30 flex items-center gap-1 transition-colors cursor-pointer"
                title={
                  isEn
                    ? 'Run predefined benchmark operations across all 8 SNN modules'
                    : 'Chạy chuỗi tác vụ đo kiểm chuẩn trên toàn bộ 8 phân hệ SNN'
                }
              >
                <Play className="w-2.5 h-2.5 fill-current" />
                <span>
                  {isRunningSuite
                    ? isEn
                      ? 'Running...'
                      : 'Đang chạy...'
                    : isEn
                      ? 'Run 8-Mod Suite'
                      : 'Chạy 8 Module'}
                </span>
              </button>
              <button
                onClick={handleResetTelemetry}
                className="p-1 rounded hover:bg-elegant-bg text-elegant-muted hover:text-white transition-colors cursor-pointer"
                title={isEn ? 'Reset counters & baselines' : 'Đặt lại bộ đếm & chuẩn cơ sở'}
              >
                <RotateCcw className="w-3 h-3" />
              </button>
              <button
                onClick={() => setIsExpanded(false)}
                className="p-1 rounded hover:bg-elegant-bg text-elegant-muted hover:text-white transition-colors cursor-pointer"
                title={isEn ? 'Minimize overlay' : 'Thu gọn'}
              >
                <ChevronDown className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {monitorMode === 'telemetry' ? (
            <>
              {/* Active Model Context Box */}
              <div className="p-2.5 rounded-lg bg-elegant-bg/80 border border-slate-800/80 space-y-1">
                <div className="flex items-center justify-between text-[10px] text-elegant-muted">
                  <span>{isEn ? 'ACTIVE SNN MODEL' : 'MÔ HÌNH SNN ĐANG HOẠT ĐỘNG'}</span>
                  <span className="text-elegant-cyan font-semibold">
                    {activeSpec.code} · {activeSpec.topology}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-white font-semibold truncate">
                    {isEn ? activeSpec.nameEn : activeSpec.nameVi}
                  </span>
                  <span className="text-elegant-muted text-[10px] shrink-0">
                    {activeSpec.complexParams} ℂ-params · T={activeSpec.timeSteps}
                  </span>
                </div>
              </div>

              {/* 3-Column Core Metrics Grid */}
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70 space-y-0.5">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Frame Rate' : 'Tốc Độ Khung'}
                  </div>
                  <div className={`text-sm font-bold ${fpsColor}`}>
                    {fps} <span className="text-[10px] font-normal">FPS</span>
                  </div>
                  <div className="text-[9px] text-elegant-muted">
                    {frameTimeMs.toFixed(1)} ms/f
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70 space-y-0.5">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Step Compute' : 'Thời Gian Tính'}
                  </div>
                  <div className={`text-sm font-bold ${computeColor}`}>
                    {simSnap.lastComputeMs.toFixed(2)}{' '}
                    <span className="text-[10px] font-normal">ms</span>
                  </div>
                  <div className="text-[9px] text-elegant-muted">
                    {isEn ? 'Avg' : 'TB'} {simSnap.avgComputeMs.toFixed(2)} ms
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70 space-y-0.5">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Memory / Buffers' : 'Bộ Nhớ Đệm'}
                  </div>
                  <div className="text-sm font-bold text-indigo-400">
                    {activeTensorKB.toFixed(1)}{' '}
                    <span className="text-[10px] font-normal">KB</span>
                  </div>
                  <div className="text-[9px] text-elegant-muted">
                    Heap {heapUsedMB !== null ? `${heapUsedMB.toFixed(1)} MB` : 'N/A'}
                  </div>
                </div>
              </div>

              {/* Dual Mini Sparklines: FPS Stability & Compute Latency */}
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70 space-y-1">
                  <div className="flex justify-between text-[9px] text-elegant-muted">
                    <span>{isEn ? 'FPS Trace (12s)' : 'Chuỗi FPS (12s)'}</span>
                    <span className="text-emerald-400">60 Hz Target</span>
                  </div>
                  <svg width="100%" height="26" viewBox="0 0 120 26" className="overflow-visible">
                    <line
                      x1="0"
                      y1="4"
                      x2="120"
                      y2="4"
                      stroke="#334155"
                      strokeWidth="0.5"
                      strokeDasharray="2 2"
                    />
                    <polyline
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="1.5"
                      points={fpsHistory
                        .map((val, idx) => {
                          const x = (idx / Math.max(1, fpsHistory.length - 1)) * 120;
                          const y = 24 - Math.min(1, val / 65) * 20;
                          return `${x.toFixed(1)},${y.toFixed(1)}`;
                        })
                        .join(' ')}
                    />
                  </svg>
                </div>

                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70 space-y-1">
                  <div className="flex justify-between text-[9px] text-elegant-muted">
                    <span>{isEn ? 'Kernel Latency' : 'Độ Trễ Nhân'}</span>
                    <span className="text-elegant-cyan">
                      {isEn ? 'Peak' : 'Đỉnh'} {simSnap.peakComputeMs.toFixed(2)}ms
                    </span>
                  </div>
                  <svg width="100%" height="26" viewBox="0 0 120 26" className="overflow-visible">
                    {simSnap.recentComputeHistory.map((val, idx) => {
                      const barW = 120 / 20 - 1.5;
                      const x = idx * (120 / 20);
                      const maxScale = Math.max(1.0, simSnap.peakComputeMs);
                      const h = Math.max(2, Math.min(22, (val / maxScale) * 22));
                      return (
                        <rect
                          key={idx}
                          x={x}
                          y={24 - h}
                          width={barW}
                          height={h}
                          rx="1"
                          fill={
                            idx === simSnap.recentComputeHistory.length - 1
                              ? '#06b6d4'
                              : '#0284c7'
                          }
                          opacity={idx === simSnap.recentComputeHistory.length - 1 ? 1 : 0.55}
                        />
                      );
                    })}
                  </svg>
                </div>
              </div>

              {/* Quick Active Module Benchmark Comparison Preview */}
              {activeBenchResult && (
                <div className="p-2.5 rounded-lg bg-elegant-bg/70 border border-slate-800/80 flex items-center justify-between gap-2">
                  <div className="space-y-0.5 min-w-0">
                    <div className="text-[9px] text-elegant-muted uppercase">
                      {isEn
                        ? `${activeSpec.code} vs Baseline Benchmark`
                        : `Đo Kiểm ${activeSpec.code} So Với Chuẩn`}
                    </div>
                    <div className="text-[11px] text-white font-semibold truncate">
                      {formatMs(activeBenchResult.avgExecMs)}{' '}
                      <span className="text-elegant-muted font-normal">
                        vs {formatMs(activeBenchResult.baselineMs)}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className={`text-[11px] font-bold ${
                        (activeBenchResult.deltaPercent ?? 0) <= 0
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {formatSpeedup(activeBenchResult, isEn)}
                    </span>
                    <button
                      onClick={() => setMonitorMode('benchmark')}
                      className="px-2 py-1 text-[10px] rounded bg-elegant-card border border-slate-700/80 hover:border-elegant-cyan text-elegant-cyan transition-colors cursor-pointer"
                    >
                      {isEn ? 'Chart' : 'Biểu Đồ'}
                    </button>
                  </div>
                </div>
              )}

              {/* Detailed Kernel & Memory Footer */}
              <div className="pt-1 border-t border-slate-800/70 space-y-1 text-[10px] text-elegant-muted">
                <div className="flex items-center justify-between">
                  <span>{isEn ? 'Last SNN Kernel:' : 'Tác vụ SNN gần nhất:'}</span>
                  <span className="text-white font-medium truncate max-w-[200px]">
                    {simSnap.lastOpLabel}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>{isEn ? 'MAC Ops / Spikes:' : 'Phép nhân MAC / Xung:'}</span>
                  <span className="text-elegant-text">
                    {formatCompactNumber(simSnap.totalMacs)} MACs ·{' '}
                    {formatCompactNumber(simSnap.totalSpikes)} spikes
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>
                    {hasNativeMemoryApi
                      ? isEn ? 'V8 JS Heap (Used / Total):' : 'Bộ nhớ V8 JS Heap:'
                      : isEn ? 'JS Heap (not exposed by this browser):' : 'JS Heap (trình duyệt không cung cấp):'}
                  </span>
                  <span className="text-indigo-400">
                    {heapUsedMB !== null && heapTotalMB !== null
                      ? `${heapUsedMB.toFixed(1)} / ${heapTotalMB.toFixed(0)} MB`
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </>
          ) : (
            /* BENCHMARK SUITE & COMPARATIVE CHART VIEW */
            <div className="space-y-3">
              {/* Active Module Predefined Operation Inspector + Single-Module Trigger */}
              <div className="p-2.5 rounded-lg bg-elegant-bg/85 border border-slate-800/80 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-bold text-elegant-cyan uppercase">
                    {activeSpec.code} · {isEn ? activeSpec.nameEn : activeSpec.nameVi}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleBenchmarkActiveModule}
                      disabled={runningModuleId !== null || isRunningSuite}
                      className="px-2 py-0.5 text-[9.5px] font-semibold rounded bg-elegant-card hover:border-elegant-cyan text-white border border-slate-700/80 transition-colors cursor-pointer"
                    >
                      {runningModuleId === activeTab
                        ? isEn
                          ? 'Testing...'
                          : 'Đang đo...'
                        : isEn
                          ? `Test ${activeSpec.code} (6 Trials)`
                          : `Đo ${activeSpec.code} (6 Lần)`}
                    </button>
                    <button
                      onClick={handleLockCurrentAsBaseline}
                      className="px-2 py-0.5 text-[9.5px] font-semibold rounded bg-elegant-card hover:border-emerald-400/60 text-elegant-muted hover:text-emerald-400 border border-slate-700/80 flex items-center gap-1 transition-colors cursor-pointer"
                      title={
                        isEn
                          ? 'Snapshot current execution times as reference baseline'
                          : 'Lưu thời gian thực thi hiện tại làm mức cơ sở (Baseline)'
                      }
                    >
                      {savedBaselineToast ? (
                        <>
                          <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                          <span className="text-emerald-400">
                            {isEn ? 'Saved' : 'Đã lưu'}
                          </span>
                        </>
                      ) : (
                        <>
                          <BookmarkCheck className="w-2.5 h-2.5" />
                          <span>{isEn ? 'Set Baseline' : 'Lưu Chuẩn'}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
                <div className="text-[10px] text-elegant-text/90 truncate">
                  <span className="text-elegant-cyan font-semibold">
                    {isEn ? 'Kernel: ' : 'Tác vụ: '}
                  </span>
                  {isEn ? activeSpec.opDescEn : activeSpec.opDescVi}
                </div>
                <div className="text-[10px] text-elegant-muted truncate">
                  <span className="text-slate-400 font-semibold">
                    {isEn ? 'Baseline: ' : 'Chuẩn so sánh: '}
                  </span>
                  {isEn ? activeSpec.baselineDescEn : activeSpec.baselineDescVi}
                </div>
              </div>

              {/* Suite Summary KPI Strip */}
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Suite Total (CV-SNN)' : 'Tổng 8 Module (Tối Ưu)'}
                  </div>
                  <div className="text-xs font-bold text-elegant-cyan mt-0.5">
                    {formatMs(suiteTotalActiveMs)}
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Reference Baseline' : 'Tổng Chuẩn Cơ Sở'}
                  </div>
                  <div className="text-xs font-bold text-slate-300 mt-0.5">
                    {formatMs(suiteTotalBaselineMs)}
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-elegant-bg/60 border border-slate-800/70">
                  <div className="text-[9px] text-elegant-muted uppercase">
                    {isEn ? 'Geo-mean Speedup / Δ' : 'Tăng Tốc TB Nhân'}
                  </div>
                  <div
                    className={`text-xs font-bold mt-0.5 ${
                      (suiteOverallDeltaPercent ?? 0) <= 0 ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  >
                    {suiteOverallSpeedup === null || suiteOverallDeltaPercent === null
                      ? isEn ? 'below timer resolution' : 'dưới độ phân giải timer'
                      : `${suiteOverallSpeedup.toFixed(2)}× (${suiteOverallDeltaPercent > 0 ? '+' : ''}${suiteOverallDeltaPercent.toFixed(1)}%)`}
                  </div>
                </div>
              </div>

              {/* COMPARATIVE CHART ACROSS ALL 8 SNN MODULES */}
              <div className="p-2.5 rounded-lg bg-elegant-bg/75 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-[9.5px] text-elegant-muted border-b border-slate-800/70 pb-1.5">
                  <span className="font-bold uppercase tracking-wider text-white">
                    {isEn
                      ? 'Median Time per Call vs Baseline (Lower is Faster)'
                      : 'Thời gian trung vị mỗi lần gọi so với chuẩn (thấp hơn là nhanh hơn)'}
                  </span>
                  <div className="flex items-center gap-2.5">
                    <span className="flex items-center gap-1 text-elegant-cyan">
                      <span className="w-2 h-2 rounded-xs bg-elegant-cyan inline-block" />
                      {isEn ? 'Active' : 'Tối ưu'}
                    </span>
                    <span className="flex items-center gap-1 text-slate-400">
                      <span className="w-2 h-2 rounded-xs bg-slate-500 inline-block" />
                      {isEn ? 'Baseline' : 'Chuẩn'}
                    </span>
                  </div>
                </div>

                <div className="space-y-1.5 max-h-[210px] overflow-y-auto pr-1">
                  {ALL_TAB_IDS.map(tabId => {
                    const res = benchmarkResults[tabId];
                    if (!res) {
                      return (
                        <div key={tabId} className="p-1.5 rounded-md border bg-elegant-bg/40 border-slate-800/60 text-[10px] text-elegant-muted">
                          {MODEL_SPECS[tabId].code} · {isEn ? 'measuring…' : 'đang đo…'}
                        </div>
                      );
                    }
                    const isCurrentTab = tabId === activeTab;
                    const isCurrentlyRunning = runningModuleId === tabId;
                    const activeWidthPct = Math.max(
                      4,
                      Math.min(100, (res.avgExecMs / maxChartMs) * 100)
                    );
                    const baselineWidthPct = Math.max(
                      4,
                      Math.min(100, (res.baselineMs / maxChartMs) * 100)
                    );
                    const isFaster = (res.deltaPercent ?? 0) <= 0;

                    return (
                      <div
                        key={tabId}
                        className={`p-1.5 rounded-md border transition-colors ${
                          isCurrentTab
                            ? 'bg-elegant-card/90 border-elegant-cyan/40'
                            : 'bg-elegant-bg/40 border-slate-800/60'
                        }`}
                      >
                        <div className="flex items-center justify-between text-[10px] mb-1">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span
                              className={`font-bold ${
                                isCurrentTab ? 'text-elegant-cyan' : 'text-white'
                              }`}
                            >
                              {res.code}
                            </span>
                            <span className="text-elegant-muted truncate max-w-[135px]">
                              {isEn ? res.nameEn : res.nameVi}
                            </span>
                            {isCurrentlyRunning && (
                              <span className="text-[9px] text-amber-400">
                                {isEn ? 'running...' : 'đang đo...'}
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-white font-semibold">
                              {formatMs(res.avgExecMs)}
                            </span>
                            <span className="text-elegant-muted text-[9.5px]">
                              / {formatMs(res.baselineMs)}
                            </span>
                            <span
                              className={`text-[9.5px] font-bold w-[48px] text-right ${
                                isFaster ? 'text-emerald-400' : 'text-amber-400'
                              }`}
                            >
                              {res.deltaPercent === null ? '—' : `${res.deltaPercent > 0 ? '+' : ''}${res.deltaPercent.toFixed(1)}%`}
                            </span>
                          </div>
                        </div>

                        {/* Dual Comparative Horizontal Bars (Active Kernel vs Baseline) */}
                        <div className="space-y-1">
                          <div className="w-full h-1.5 bg-slate-900/90 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-elegant-cyan rounded-full transition-all duration-300"
                              style={{ width: `${activeWidthPct}%` }}
                            />
                          </div>
                          <div className="w-full h-1 bg-slate-900/90 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-slate-500/80 rounded-full transition-all duration-300"
                              style={{ width: `${baselineWidthPct}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* SUBTLE PERSISTENT OVERLAY BAR (Always visible in bottom-right) */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="ml-auto flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-elegant-card/85 hover:bg-elegant-card backdrop-blur-md border border-slate-800/90 hover:border-elegant-cyan/40 shadow-lg transition-all cursor-pointer group"
        title={
          isEn
            ? 'Click to toggle SNN Performance Monitor & Benchmark Suite'
            : 'Nhấn để mở/đóng Giám Sát Hiệu Năng & Bộ Đo Kiểm Chuẩn SNN'
        }
      >
        <span className="flex items-center gap-1.5 text-[11px] font-semibold text-white">
          <Activity className="w-3.5 h-3.5 text-elegant-cyan shrink-0" />
          <span>{activeSpec.code}</span>
        </span>

        <span className="text-slate-600" aria-hidden="true">
          |
        </span>

        <span className={`text-[11px] font-semibold ${fpsColor}`}>{fps} FPS</span>

        <span className="text-slate-600" aria-hidden="true">
          ·
        </span>

        <span className="flex items-center gap-1 text-[11px] text-elegant-cyan">
          <Zap className="w-3 h-3 shrink-0" />
          <span>{simSnap.lastComputeMs.toFixed(2)} ms</span>
        </span>

        {activeBenchResult && (
          <>
            <span className="text-slate-600 hidden md:inline" aria-hidden="true">
              ·
            </span>
            <span
              className={`hidden md:inline text-[10.5px] font-semibold ${
                (activeBenchResult.deltaPercent ?? 0) <= 0
                  ? 'text-emerald-400'
                  : 'text-amber-400'
              }`}
            >
              {activeBenchResult.speedupRatio === null ? (isEn ? 'timer res.' : 'dưới ngưỡng timer') : `${activeBenchResult.speedupRatio.toFixed(2)}× vs Base`}
            </span>
          </>
        )}

        <span className="text-slate-600 hidden sm:inline" aria-hidden="true">
          ·
        </span>

        <span className="hidden sm:flex items-center gap-1 text-[11px] text-indigo-400">
          <Cpu className="w-3 h-3 shrink-0" />
          <span>
            {activeTensorKB.toFixed(1)} KB / {heapUsedMB !== null ? `${heapUsedMB.toFixed(1)} MB` : 'heap N/A'}
          </span>
        </span>

        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-elegant-muted group-hover:text-white ml-0.5" />
        ) : (
          <ChevronUp className="w-3.5 h-3.5 text-elegant-muted group-hover:text-white ml-0.5" />
        )}
      </button>
    </aside>
  );
}
