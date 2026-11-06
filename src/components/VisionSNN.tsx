import { useState, useEffect, useMemo } from 'react';
import {
  RotateCcw,
  Sparkles,
  Cpu,
  Check,
  Info,
  Brain,
  Terminal,
  Zap,
  Shuffle,
  ShieldCheck,
  Activity,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { complex, Complex, ComplexLIFNeuron, perfTracker } from '../simulationEngine';
import { mulberry32, hashSeed, seededGaussian } from '../utils/random';

interface VisionSNNProps {
  onNavigateToHardware?: () => void;
}

const GRID_SIZE = 8;

const TEMPLATES: Record<string, number[]> = {
  T: [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  L: [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  X: [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  O: [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
};

const CLASS_NAMES = ['T', 'L', 'X', 'O'];

type PhaseEncodingMode = 'radial' | 'linear' | 'uniform';

export default function VisionSNN({ onNavigateToHardware }: VisionSNNProps = {}) {
  const { language } = useLanguage();
  const isEn = language === 'en';

  const [grid, setGrid] = useState<number[]>(TEMPLATES['T']);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawValue, setDrawValue] = useState<number>(1);

  // Phase Encoding config
  const [encodingMode, setEncodingMode] = useState<PhaseEncodingMode>('radial');
  const [globalPhaseScale, setGlobalPhaseScale] = useState<number>(1.0);
  const [inputPhaseJitter, setInputPhaseJitter] = useState<number>(0.0);

  // SNN configuration
  const [simSteps] = useState<number>(15);
  const [leakMag] = useState<number>(0.85);
  const [leakPhase] = useState<number>(0.12);
  const [threshold] = useState<number>(1.2);

  // Hardware compiler config (Memristor emulation)
  const [hardwareBits, setHardwareBits] = useState<number>(4);
  const [writeNoise, setWriteNoise] = useState<number>(0.15);
  const [enableSelfCorrection, setEnableSelfCorrection] = useState<boolean>(true);
  const [noiseSeed, setNoiseSeed] = useState<number>(1);

  // Simulation results
  const [simResult, setSimResult] = useState<{
    spikeRates: number[];
    potentials: Complex[][];
    spikes: boolean[][];
    phaseCoherence: number[];
    meanCurrentMag: number[];
  } | null>(null);

  // 1. Ideal matched-filter Complex Synaptic Weights
  const { idealWeights, maxWeightMag } = useMemo(() => {
    const weights: Complex[][] = [];
    let maxMag = 0;

    for (let c = 0; c < 4; c++) {
      const classTemplate = TEMPLATES[CLASS_NAMES[c]];
      const row: Complex[] = [];

      let activePixels = 0;
      let inactivePixels = 0;
      for (let i = 0; i < 64; i++) {
        if (classTemplate[i] === 1) activePixels++;
        else inactivePixels++;
      }

      for (let idx = 0; idx < 64; idx++) {
        const x = idx % GRID_SIZE;
        const y = Math.floor(idx / GRID_SIZE);

        let inPhase = 0;
        if (encodingMode === 'radial') {
          inPhase = Math.atan2(y - 3.5, x - 3.5);
        } else if (encodingMode === 'linear') {
          inPhase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
        } else {
          inPhase = 0.5;
        }

        // Conjugate phase alignment so W_ci * X_i sums constructively at 0 radians
        const targetWPhase = -inPhase * globalPhaseScale;

        let w: Complex;
        if (classTemplate[idx] === 1) {
          w = complex.fromPolar(2.2 / Math.max(1, activePixels), targetWPhase);
        } else {
          // Out-of-phase (+pi) inhibitory weight cancels false-positive pixels
          w = complex.fromPolar(1.6 / Math.max(1, inactivePixels), targetWPhase + Math.PI);
        }
        maxMag = Math.max(maxMag, Math.abs(w[0]), Math.abs(w[1]));
        row.push(w);
      }
      weights.push(row);
    }
    return { idealWeights: weights, maxWeightMag: maxMag || 0.1 };
  }, [encodingMode, globalPhaseScale]);

  // 2. Compiled Hardware Weights: properly scaled to full-scale conductance G_max!
  const compiledWeights = useMemo(() => {
    const weights: Complex[][] = [];
    const gMax = maxWeightMag;

    for (let c = 0; c < 4; c++) {
      const row: Complex[] = [];
      for (let idx = 0; idx < 64; idx++) {
        const origW = idealWeights[c][idx];
        let wr = origW[0];
        let wi = origW[1];

        // 1. Quantization normalized by full-scale conductance [-gMax, +gMax]
        if (hardwareBits > 0) {
          const levels = Math.pow(2, hardwareBits) - 1;
          const normRe = Math.max(-1, Math.min(1, wr / gMax));
          const normIm = Math.max(-1, Math.min(1, wi / gMax));
          wr = (Math.round(((normRe + 1) / 2) * levels) / levels * 2 - 1) * gMax;
          wi = (Math.round(((normIm + 1) / 2) * levels) / levels * 2 - 1) * gMax;
        }

        // 2. Conductance programming noise relative to full-scale conductance gMax
        if (writeNoise > 0) {
          const effectiveNoise = (enableSelfCorrection ? writeNoise * 0.15 : writeNoise) * gMax;
          const nRe = seededGaussian(hashSeed(noiseSeed, 1, c, idx));
          const nIm = seededGaussian(hashSeed(noiseSeed, 2, c, idx));
          wr += nRe * effectiveNoise;
          wi += nIm * effectiveNoise;
        }

        row.push(complex.create(wr, wi));
      }
      weights.push(row);
    }
    return weights;
  }, [idealWeights, maxWeightMag, hardwareBits, writeNoise, enableSelfCorrection, noiseSeed]);

  // Helper to run inference on any 64-pixel pattern
  const evaluatePattern = (inputGrid: number[], phaseJitter: number) => {
    const neurons = Array.from({ length: 4 }).map(
      () => new ComplexLIFNeuron(leakMag, leakPhase, threshold, 'soft')
    );

    const potentials: Complex[][] = [];
    const spikes: boolean[][] = [];
    const spikeCounts = [0, 0, 0, 0];
    const currentMagSums = [0, 0, 0, 0];
    const phaseSumRe = [0, 0, 0, 0];
    const phaseSumIm = [0, 0, 0, 0];

    for (let t = 0; t < simSteps; t++) {
      const timeFactor = t / simSteps;
      const temporalPhase = 2 * Math.PI * 1.5 * timeFactor;
      const amplitude = 0.85 + 0.15 * Math.sin(2 * Math.PI * 2 * timeFactor);

      const stepPotentials: Complex[] = [];
      const stepSpikes: boolean[] = [];

      for (let o = 0; o < 4; o++) {
        let sumRe = 0;
        let sumIm = 0;
        let pixelPhasorRe = 0;
        let pixelPhasorIm = 0;
        let activeCount = 0;

        for (let idx = 0; idx < 64; idx++) {
          if (inputGrid[idx] === 0) continue;
          activeCount++;

          const x = idx % GRID_SIZE;
          const y = Math.floor(idx / GRID_SIZE);
          let spatialPhase = 0;
          if (encodingMode === 'radial') {
            spatialPhase = Math.atan2(y - 3.5, x - 3.5);
          } else if (encodingMode === 'linear') {
            spatialPhase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
          } else {
            spatialPhase = 0.5;
          }

          const jitter =
            phaseJitter > 0 ? seededGaussian(hashSeed(noiseSeed, 3, t, idx)) * phaseJitter : 0;
          const finalPhase = spatialPhase * globalPhaseScale + temporalPhase + jitter;

          const xRe = amplitude * Math.cos(finalPhase);
          const xIm = amplitude * Math.sin(finalPhase);
          const w = compiledWeights[o][idx];

          const prodRe = w[0] * xRe - w[1] * xIm;
          const prodIm = w[0] * xIm + w[1] * xRe;

          sumRe += prodRe;
          sumIm += prodIm;

          const pMag = Math.sqrt(prodRe * prodRe + prodIm * prodIm) || 1e-9;
          pixelPhasorRe += prodRe / pMag;
          pixelPhasorIm += prodIm / pMag;
        }

        const cMag = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
        currentMagSums[o] += cMag;

        if (activeCount > 0) {
          phaseSumRe[o] += pixelPhasorRe / activeCount;
          phaseSumIm[o] += pixelPhasorIm / activeCount;
        }

        const res = neurons[o].step(sumRe, sumIm, false);
        stepPotentials.push([neurons[o].preResetRe, neurons[o].preResetIm]);
        stepSpikes.push(res.spiked);
        if (res.spiked) spikeCounts[o]++;
      }

      potentials.push(stepPotentials);
      spikes.push(stepSpikes);
    }

    const phaseCoherence = [0, 1, 2, 3].map(o =>
      Math.min(
        1.0,
        Math.sqrt(phaseSumRe[o] * phaseSumRe[o] + phaseSumIm[o] * phaseSumIm[o]) / simSteps
      )
    );

    return {
      spikeRates: spikeCounts.map(count => count / simSteps),
      potentials,
      spikes,
      phaseCoherence,
      meanCurrentMag: currentMagSums.map(s => s / simSteps),
    };
  };

  useEffect(() => {
    const t0 = performance.now();
    const res = evaluatePattern(grid, inputPhaseJitter);
    const totalSpikes = res.spikes.reduce(
      (acc, step) => acc + step.filter(Boolean).length,
      0
    );
    perfTracker.recordSample(
      Math.max(0.04, performance.now() - t0),
      simSteps * 64 * 4 * 4,
      totalSpikes,
      43008,
      `VisionSNN.forward (${hardwareBits > 0 ? `${hardwareBits}-bit` : 'FP32'})`
    );
    setSimResult(res);
  }, [
    grid,
    encodingMode,
    globalPhaseScale,
    inputPhaseJitter,
    compiledWeights,
    simSteps,
    hardwareBits,
  ]);

  // Empirical Hardware Crossbar Accuracy across 20 perturbed 4-class patterns
  const hardwareMetrics = useMemo(() => {
    let correct = 0;
    const totalTrials = 20; // 5 noisy trials per template class (T, L, X, O)
    let avgRmsError = 0;

    for (let c = 0; c < 4; c++) {
      for (let idx = 0; idx < 64; idx++) {
        const diffRe = compiledWeights[c][idx][0] - idealWeights[c][idx][0];
        const diffIm = compiledWeights[c][idx][1] - idealWeights[c][idx][1];
        avgRmsError += (diffRe * diffRe + diffIm * diffIm) / (maxWeightMag * maxWeightMag);
      }
    }
    // RMS of complex weight error, normalized to full-scale conductance
    avgRmsError = Math.sqrt(avgRmsError / (4 * 64));

    for (let c = 0; c < 4; c++) {
      const baseGrid = TEMPLATES[CLASS_NAMES[c]];
      for (let trial = 0; trial < 5; trial++) {
        // Flip each pixel independently with p = 2/64 (expected 2 flips per noisy trial)
        const rng = mulberry32(hashSeed(noiseSeed, 4, c, trial));
        const testGrid = baseGrid.map(val => {
          if (trial === 0) return val;
          return rng() < 2 / 64 ? (val === 1 ? 0 : 1) : val;
        });
        const res = evaluatePattern(testGrid, inputPhaseJitter);
        // Score combines spike rate and mean current magnitude
        let bestIdx = 0;
        let bestScore = -1;
        for (let o = 0; o < 4; o++) {
          const score = res.spikeRates[o] * 2.0 + res.meanCurrentMag[o];
          if (score > bestScore) {
            bestScore = score;
            bestIdx = o;
          }
        }
        if (bestIdx === c) correct++;
      }
    }

    return {
      empiricalAccuracy: correct / totalTrials,
      rmsWeightErrorPercent: avgRmsError * 100,
    };
  }, [compiledWeights, idealWeights, maxWeightMag, inputPhaseJitter, noiseSeed]);

  // Corrupt current canvas with 10% random pixel flips to test robustness
  const handleAddSpatialNoise = () => {
    const next = grid.map(v => (Math.random() < 0.1 ? (v === 1 ? 0 : 1) : v));
    setGrid(next);
  };

  // Drawing event handlers
  const handlePixelMouseDown = (idx: number) => {
    setIsDrawing(true);
    const newValue = grid[idx] === 1 ? 0 : 1;
    setDrawValue(newValue);
    const newGrid = [...grid];
    newGrid[idx] = newValue;
    setGrid(newGrid);
  };

  const handlePixelMouseEnter = (idx: number) => {
    if (!isDrawing) return;
    setGrid(prev => {
      if (prev[idx] === drawValue) return prev;
      const next = [...prev];
      next[idx] = drawValue;
      return next;
    });
  };

  // End drawing on mouseup anywhere in the window, not only inside the component
  useEffect(() => {
    if (!isDrawing) return;
    const stop = () => setIsDrawing(false);
    window.addEventListener('mouseup', stop);
    window.addEventListener('blur', stop);
    return () => {
      window.removeEventListener('mouseup', stop);
      window.removeEventListener('blur', stop);
    };
  }, [isDrawing]);

  const getPixelColorStyle = (idx: number) => {
    if (grid[idx] === 0) return 'bg-[#15202B]/60 border border-elegant-muted/5';

    const x = idx % GRID_SIZE;
    const y = Math.floor(idx / GRID_SIZE);
    let phase = 0;
    if (encodingMode === 'radial') {
      phase = Math.atan2(y - 3.5, x - 3.5);
    } else if (encodingMode === 'linear') {
      phase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
    }

    const hue = Math.round(((phase + Math.PI) / (2 * Math.PI)) * 360);
    return {
      background: `hsla(${hue}, 85%, 45%, 0.9)`,
      boxShadow: `0 0 8px hsla(${hue}, 85%, 55%, 0.4)`,
      border: `1px solid hsla(${hue}, 90%, 65%, 0.6)`,
    };
  };

  const predictionIdx = useMemo(() => {
    if (!simResult) return -1;
    let bestScore = -1;
    let bestIdx = -1;
    for (let idx = 0; idx < 4; idx++) {
      const score = simResult.spikeRates[idx] * 2.0 + simResult.meanCurrentMag[idx];
      if (score > bestScore && score > 0.05) {
        bestScore = score;
        bestIdx = idx;
      }
    }
    return bestIdx;
  }, [simResult]);

  return (
    <div className="space-y-6">

      {/* Title block */}
      <div className="bg-gradient-to-r from-elegant-card to-elegant-card/40 border border-elegant-muted/15 p-5 rounded-xl shadow-xl space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
              <Cpu className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h2 className="text-xl font-bold font-display text-white">
                {isEn
                  ? 'Computer Vision Experiment & CV-SNN Hardware Compiler'
                  : 'Thử Nghiệm Thị Giác Máy Tính & Biên Dịch Phần Cứng CV-SNN'}
              </h2>
              <p className="text-xs text-elegant-text/80 leading-relaxed mt-0.5">
                {isEn
                  ? 'Spatial phase encoding on 8×8 images, conjugate matched-filter crossbar weights, and empirical robustness testing.'
                  : 'Mã hóa sóng pha không gian 8×8, bộ lọc phối hợp liên hợp phức trên mảng Crossbar, và kiểm định chống nhiễu thực nghiệm.'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] bg-emerald-400/10 text-emerald-400 border border-emerald-400/20 px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1">
              <Check className="w-3.5 h-3.5" />
              <span>
                {isEn
                  ? `G_max Normalized (${maxWeightMag.toFixed(3)} S)`
                  : `Đã Chuẩn Hóa Dải Dẫn G_max (${maxWeightMag.toFixed(3)} S)`}
              </span>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* COLUMN 1: SANDBOX CANVAS DRAWING & STRESS TEST */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl flex flex-col">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 rounded-full bg-elegant-cyan" />
              <span>
                {isEn
                  ? 'Drawing Canvas & Phase Perturbation'
                  : 'Bản Vẽ & Kiểm Thử Nhiễu Pha'}
              </span>
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={handleAddSpatialNoise}
                className="text-xs text-amber-400 hover:text-amber-300 transition flex items-center gap-1 font-mono bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20 cursor-pointer"
                title={isEn ? 'Flip 10% random pixels' : 'Đảo ngẫu nhiên 10% pixel'}
              >
                <Shuffle className="w-3 h-3" />
                <span>{isEn ? '+10% Noise' : '+10% Nhiễu Ảnh'}</span>
              </button>
              <button
                onClick={() => setGrid(new Array(64).fill(0))}
                className="text-xs text-rose-400 hover:text-rose-300 transition flex items-center gap-1 font-mono cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>{isEn ? 'Clear' : 'Xóa'}</span>
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <span className="text-[11px] font-mono text-elegant-muted block font-bold uppercase tracking-wider">
              {isEn ? 'Standard Character Templates:' : 'Mẫu Ký Tự Chuẩn (Templates):'}
            </span>
            <div className="grid grid-cols-4 gap-2">
              {CLASS_NAMES.map(name => (
                <button
                  key={`preset-${name}`}
                  onClick={() => setGrid(TEMPLATES[name])}
                  className="py-1.5 px-3 bg-elegant-bg hover:bg-elegant-card hover:border-elegant-cyan/40 border border-elegant-muted/10 rounded-md text-xs font-bold text-white transition flex items-center justify-center gap-1 cursor-pointer"
                >
                  <Sparkles className="w-3 h-3 text-elegant-cyan" />
                  <span>{isEn ? `Char ${name}` : `Chữ ${name}`}</span>
                </button>
              ))}
            </div>
          </div>

          {/* 8x8 Canvas Box */}
          <div className="flex justify-center py-3 bg-elegant-bg/40 rounded-xl border border-elegant-muted/10">
            <div className="grid grid-cols-8 gap-1.5 p-3 bg-elegant-bg/90 rounded-xl shadow-inner select-none">
              {grid.map((val, idx) => {
                const style = getPixelColorStyle(idx);
                const isObject = typeof style === 'object';

                return (
                  <div
                    key={`pixel-${idx}`}
                    onMouseDown={() => handlePixelMouseDown(idx)}
                    onMouseEnter={() => handlePixelMouseEnter(idx)}
                    style={isObject ? style : undefined}
                    className={`w-8 h-8 rounded-md cursor-pointer transition-all duration-100 flex items-center justify-center ${
                      !isObject ? style : ''
                    }`}
                  >
                    {val === 1 && <span className="w-1.5 h-1.5 rounded-full bg-white/60" />}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Phase encoding parameters */}
          <div className="space-y-3 pt-2 border-t border-elegant-muted/10">
            <div className="space-y-1">
              <label className="text-[11px] font-mono text-elegant-muted font-bold block uppercase">
                {isEn ? 'Spatial Phase Mapping:' : 'Phương Pháp Mã Hóa Pha Không Gian:'}
              </label>
              <div className="grid grid-cols-3 gap-1.5 mt-1">
                {(['radial', 'linear', 'uniform'] as PhaseEncodingMode[]).map(mode => (
                  <button
                    key={`mode-${mode}`}
                    onClick={() => setEncodingMode(mode)}
                    className={`py-1 px-1.5 rounded text-[11px] font-semibold transition border text-center capitalize cursor-pointer ${
                      encodingMode === mode
                        ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35'
                        : 'bg-elegant-bg/50 text-elegant-text/70 border-elegant-muted/10 hover:text-white'
                    }`}
                  >
                    {mode === 'radial'
                      ? isEn
                        ? 'Radial (2D)'
                        : 'Tâm Tròn (2D)'
                      : mode === 'linear'
                      ? isEn
                        ? 'Linear'
                        : 'Tuyến Tính'
                      : isEn
                      ? 'Uniform'
                      : 'Đồng Pha'}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>{isEn ? 'Phase Scale (γ):' : 'Hệ Số Nhân Góc Pha (γ):'}</span>
                <span className="text-elegant-cyan font-bold">{globalPhaseScale.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="2.0"
                step="0.2"
                value={globalPhaseScale}
                onChange={e => setGlobalPhaseScale(parseFloat(e.target.value))}
                className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>{isEn ? 'Sensor Phase Jitter (σ_θ):' : 'Nhiễu Rung Pha Cảm Biến (σ_θ):'}</span>
                <span className="text-rose-400 font-bold">{inputPhaseJitter.toFixed(2)} rad</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.2"
                step="0.1"
                value={inputPhaseJitter}
                onChange={e => setInputPhaseJitter(parseFloat(e.target.value))}
                className="w-full accent-rose-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* COLUMN 2: HARDWARE-AWARE SOFTWARE COMPILER (CHIP EMULATOR) */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span>
                  {isEn
                    ? 'Memristor Crossbar Compiler'
                    : 'Bộ Biên Dịch & Lượng Tử Hóa Crossbar'}
                </span>
              </h3>
              <button
                onClick={() => setNoiseSeed(s => s + 1)}
                className="text-[10px] font-mono text-elegant-cyan hover:underline cursor-pointer"
              >
                {isEn ? 'Resample Chip Noise' : 'Lấy Mẫu Nhiễu Mới'}
              </button>
            </div>

            {/* Precision Bits */}
            <div className="space-y-2 p-3.5 bg-elegant-bg/60 border border-elegant-muted/10 rounded-lg">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white font-bold">
                  {isEn ? 'PCM Conductance Resolution:' : 'Độ Phân Giải Mức Dẫn Nạp PCM:'}
                </span>
                <span className="text-amber-400 font-bold">
                  {hardwareBits === 0
                    ? isEn
                      ? 'FP64 (Ideal)'
                      : 'FP64 (Lý tưởng)'
                    : `${hardwareBits}-bit (${Math.pow(2, hardwareBits)} levels)`}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="8"
                step="1"
                value={hardwareBits}
                onChange={e => setHardwareBits(parseInt(e.target.value))}
                className="w-full accent-amber-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-elegant-muted leading-normal">
                {hardwareBits > 0
                  ? isEn
                    ? `Quantizes W_re and W_im relative to full-scale G_max = ${maxWeightMag.toFixed(3)} S into ${Math.pow(2, hardwareBits)} discrete states.`
                    : `Lượng tử hóa W_re, W_im theo dải chuẩn G_max = ${maxWeightMag.toFixed(3)} S thành ${Math.pow(2, hardwareBits)} mức dẫn nạp rời rạc.`
                  : isEn
                  ? 'Unquantized continuous floating-point weights.'
                  : 'Trọng số liên tục chưa lượng tử hóa.'}
              </p>
            </div>

            {/* Noise Injection */}
            <div className="space-y-2 p-3.5 bg-elegant-bg/60 border border-elegant-muted/10 rounded-lg">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white font-bold">
                  {isEn ? 'Device Write Noise (σ_G / G_max):' : 'Nhiễu Ghi Memristor (σ_G / G_max):'}
                </span>
                <span className="text-rose-400 font-bold">{(writeNoise * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.05"
                value={writeNoise}
                onChange={e => setWriteNoise(parseFloat(e.target.value))}
                className="w-full accent-rose-500 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-elegant-muted leading-normal">
                {isEn
                  ? `Actual RMS Conductance Error on Chip: ${hardwareMetrics.rmsWeightErrorPercent.toFixed(1)}% of G_max.`
                  : `Sai số RMS thực tế trên mảng Crossbar: ${hardwareMetrics.rmsWeightErrorPercent.toFixed(1)}% so với G_max.`}
              </p>
            </div>

            {/* Self correction loop */}
            <div className="p-3.5 bg-elegant-cyan/5 border border-elegant-cyan/20 rounded-xl space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-elegant-cyan font-bold font-display text-xs">
                  <Zap className="w-4 h-4" />
                  <span>
                    {isEn
                      ? 'KAIST On-Chip Write-Verify (2024)'
                      : 'Bù Lỗi On-Chip Write-Verify (KAIST 2024)'}
                  </span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={enableSelfCorrection}
                    onChange={e => setEnableSelfCorrection(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-elegant-bg rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-elegant-muted after:border-elegant-card after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-elegant-cyan peer-checked:after:bg-elegant-bg" />
                </label>
              </div>
              <p className="text-[11px] text-elegant-text/80 leading-relaxed">
                {isEn
                  ? 'Closed-loop write-verify programming suppresses 85% of cycle-to-cycle conductance variation.'
                  : 'Cơ chế ghi-kiểm chứng vòng kín giúp triệt tiêu 85% độ lệch dẫn nạp vật lý khi nạp xung.'}
              </p>
            </div>
          </div>

          {/* Empirical 20-Trial Benchmark Result */}
          <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/15 rounded-xl space-y-2.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-elegant-muted font-mono uppercase font-bold flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                {isEn
                  ? 'Empirical 4-Class Benchmark (20 Trials):'
                  : 'Kiểm Định Thực Nghiệm 4 Lớp (20 Mẫu):'}
              </span>
              <span
                className={`font-mono font-bold text-sm ${
                  hardwareMetrics.empiricalAccuracy >= 0.85
                    ? 'text-emerald-400'
                    : 'text-amber-400'
                }`}
              >
                {(hardwareMetrics.empiricalAccuracy * 100).toFixed(0)}%
              </span>
            </div>

            <div className="h-2 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  hardwareMetrics.empiricalAccuracy >= 0.85
                    ? 'bg-emerald-400'
                    : hardwareMetrics.empiricalAccuracy >= 0.65
                    ? 'bg-amber-400'
                    : 'bg-rose-500'
                }`}
                style={{ width: `${hardwareMetrics.empiricalAccuracy * 100}%` }}
              />
            </div>

            <p className="text-[10px] text-elegant-muted font-mono">
              {isEn
                ? 'Measured by running 20 noisy character patterns (T, L, X, O) through the quantized & noisy crossbar array.'
                : 'Đo trực tiếp bằng cách chạy 20 mẫu ký tự nhiễu (T, L, X, O) qua ma trận trọng số Crossbar đã lượng tử hóa.'}
            </p>
          </div>

          {onNavigateToHardware && (
            <button
              onClick={onNavigateToHardware}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-[#13151A] hover:bg-elegant-bg border border-elegant-cyan/30 hover:border-elegant-cyan text-elegant-cyan rounded-lg transition-all text-xs font-bold font-mono cursor-pointer"
            >
              <Terminal className="w-4 h-4" />
              <span>
                {isEn
                  ? 'Open Neuromorphic Hardware Lab & PyVISA Driver →'
                  : 'Mở Phòng Thí Nghiệm Phần Cứng & Mã Lệnh PyVISA →'}
              </span>
            </button>
          )}
        </div>

        {/* COLUMN 3: REALTIME INFERENCE RESULT & SPIKE RASTER */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
                <Activity className="w-4 h-4 text-emerald-400" />
                <span>
                  {isEn
                    ? 'Live Crossbar Output & Spike Raster'
                    : 'Dòng Ra Crossbar & Chuỗi Xung (15 Steps)'}
                </span>
              </h3>
            </div>

            <div className="space-y-2.5">
              {CLASS_NAMES.map((name, idx) => {
                const isWinner = predictionIdx === idx;
                const spikeRate = simResult?.spikeRates[idx] || 0;
                const coherence = simResult?.phaseCoherence[idx] || 0;
                const currentMag = simResult?.meanCurrentMag[idx] || 0;

                return (
                  <div
                    key={`prob-${name}`}
                    className={`p-3 rounded-lg border transition-all space-y-2 ${
                      isWinner
                        ? 'bg-elegant-cyan/10 border-elegant-cyan/40 shadow-md'
                        : 'bg-elegant-bg/40 border-elegant-muted/10'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-5 h-5 rounded-full flex items-center justify-center font-mono font-bold text-[11px] ${
                            isWinner
                              ? 'bg-elegant-cyan text-elegant-bg'
                              : 'bg-elegant-bg text-elegant-muted'
                          }`}
                        >
                          {name}
                        </span>
                        <span
                          className={`font-semibold ${
                            isWinner ? 'text-white' : 'text-elegant-text/70'
                          }`}
                        >
                          {isEn ? `Class "${name}" Neuron` : `Neuron Chữ "${name}"`}
                        </span>
                      </div>
                      <div className="text-right font-mono text-[10px]">
                        <span
                          className={
                            isWinner ? 'text-elegant-cyan font-bold' : 'text-elegant-muted'
                          }
                        >
                          Rate: {(spikeRate * 100).toFixed(0)}% | |I|: {currentMag.toFixed(2)} | R:{' '}
                          {coherence.toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* 15-step Spike Raster for this class neuron */}
                    <div className="flex items-center gap-1">
                      {simResult?.spikes.map((stepSpk, tIdx) => (
                        <div
                          key={`vspk-${idx}-${tIdx}`}
                          className={`flex-1 h-2.5 rounded-xs ${
                            stepSpk[idx]
                              ? isWinner
                                ? 'bg-elegant-cyan shadow-xs shadow-elegant-cyan'
                                : 'bg-amber-400/80'
                              : 'bg-white/5'
                          }`}
                          title={`t=${tIdx + 1}: ${stepSpk[idx] ? 'Spike Fired' : 'Subthreshold'}`}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {predictionIdx !== -1 ? (
            <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/15 rounded-xl space-y-1.5 text-center shadow-inner">
              <span className="text-[10px] font-mono text-elegant-muted uppercase tracking-wider block">
                {isEn
                  ? 'CONSTRUCTIVE PHASE CLASSIFICATION:'
                  : 'KẾT QUẢ NHẬN DIỆN GIAO THOA PHA:'}
              </span>
              <div className="text-2xl font-black font-display text-elegant-cyan flex items-center justify-center gap-2">
                <Brain className="w-6 h-6 animate-pulse" />
                <span>
                  {isEn
                    ? `Character "${CLASS_NAMES[predictionIdx]}"`
                    : `Ký tự "${CLASS_NAMES[predictionIdx]}"`}
                </span>
              </div>
              <p className="text-[10px] text-elegant-text/70 font-mono">
                {isEn
                  ? `Matched-filter conjugate phase alignment peaked on Neuron ${CLASS_NAMES[predictionIdx]}.`
                  : `Bộ lọc liên hợp pha đạt cộng hưởng cực đại tại Neuron ${CLASS_NAMES[predictionIdx]}.`}
              </p>
            </div>
          ) : (
            <div className="p-4 bg-elegant-bg/50 border border-elegant-muted/10 rounded-xl text-center text-xs text-elegant-muted font-mono">
              {isEn ? 'Draw a pattern on the 8×8 canvas' : 'Hãy vẽ ký tự lên bảng 8×8'}
            </div>
          )}
        </div>
      </div>

      {/* SCIENTIFIC NOTE */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-2 shadow-xl">
        <h4 className="text-sm font-bold text-white flex items-center gap-1.5 font-display">
          <Info className="w-4 h-4 text-elegant-cyan" />
          <span>
            {isEn
              ? 'Algorithmic Fix: Full-Scale Conductance Normalization & Conjugate Phase Matching'
              : 'Chuẩn Hóa Toán Học: Lượng Tử Hóa Theo G_max & Bộ Lọc Phối Hợp Liên Hợp Phức'}
          </span>
        </h4>
        <p className="text-xs text-elegant-text/80 leading-relaxed">
          {isEn
            ? 'Each template class neuron c stores conjugated spatial phases W_ci = |W| e^{-j γ φ_i} for active template pixels and out-of-phase inhibitory weights e^{-j(γ φ_i - π)} for background pixels. When the matching pattern is drawn, all active pixel currents align at 0° phase (constructive interference), while mismatched pixels cancel at 180° (destructive interference).'
            : 'Mỗi neuron lớp c lưu trữ pha liên hợp phức W_ci = |W| e^{-j γ φ_i} cho các pixel thuộc mẫu chuẩn và pha ngược 180° e^{-j(γ φ_i - π)} cho các pixel nền. Khi vẽ đúng ký tự, toàn bộ dòng điện pixel đồng loạt xoay về góc 0° (cộng hưởng cực đại), trong khi các pixel vẽ thừa bị triệt tiêu ngược pha 180°.'}
        </p>
      </div>
    </div>
  );
}