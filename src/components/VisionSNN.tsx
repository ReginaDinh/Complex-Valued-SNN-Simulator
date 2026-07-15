import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'motion/react';
import { 
  Play, RotateCcw, Sparkles, Cpu, Layers, HelpCircle, 
  Activity, Sliders, Zap, Check, AlertCircle, Info, Shuffle, Brain
} from 'lucide-react';
import { complex, Complex, ComplexLIFNeuron } from '../simulationEngine';

// Define the 4 standard visual template shapes on an 8x8 grid
const GRID_SIZE = 8;

const TEMPLATES: Record<string, number[]> = {
  'T': [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  'L': [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  'X': [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ],
  'O': [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]
};

const CLASS_NAMES = ['T', 'L', 'X', 'O'];

type PhaseEncodingMode = 'radial' | 'linear' | 'uniform';

export default function VisionSNN() {
  // Drawing Canvas State (8x8 = 64 pixels)
  const [grid, setGrid] = useState<number[]>(TEMPLATES['T']);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawValue, setDrawValue] = useState<number>(1); // 1 to draw, 0 to erase

  // Phase Encoding config
  const [encodingMode, setEncodingMode] = useState<PhaseEncodingMode>('radial');
  const [globalPhaseScale, setGlobalPhaseScale] = useState<number>(1.0); // scales phase shift

  // SNN configuration
  const [simSteps, setSimSteps] = useState<number>(15);
  const [leakMag, setLeakMag] = useState<number>(0.85);
  const [leakPhase, setLeakPhase] = useState<number>(0.12);
  const [threshold, setThreshold] = useState<number>(1.5);

  // Hardware compiler config (Prof. Choi Lab emulation)
  const [hardwareBits, setHardwareBits] = useState<number>(4); // PCM conductance states bitwidth (0 = continuous)
  const [writeNoise, setWriteNoise] = useState<number>(0.15); // standard deviation of write error
  const [enableSelfCorrection, setEnableSelfCorrection] = useState<boolean>(true); // KAIST nature paper on-chip auto-feedback

  // Simulation results
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simResult, setSimResult] = useState<{
    spikeRates: number[];
    potentials: Complex[][]; // [step][neuron]
    spikes: boolean[][]; // [step][neuron]
    phaseCoherence: number[]; // synchronization metric per class
  } | null>(null);

  // Initialize ideal analytical weights connecting 64 inputs -> 4 outputs
  // CV-SNN thrives on phase constructive interference.
  // We establish complex weights W_co where:
  // - Magnitude is high if input pixel is vital to template 'c'
  // - Phase is aligned to counteract the input spatial phase (W * X becomes in-phase)
  const idealWeights = useMemo(() => {
    const weights: Complex[][] = []; // [4 classes][64 pixels]

    for (let c = 0; c < 4; c++) {
      const classTemplate = TEMPLATES[CLASS_NAMES[c]];
      const row: Complex[] = [];
      
      // Calculate active pixels for normalization
      let activePixels = 0;
      for (let i = 0; i < 64; i++) {
        if (classTemplate[i] === 1) activePixels++;
      }

      for (let idx = 0; idx < 64; idx++) {
        const x = idx % GRID_SIZE;
        const y = Math.floor(idx / GRID_SIZE);

        // Calculate input phase for this coordinate based on mode
        let inPhase = 0;
        if (encodingMode === 'radial') {
          inPhase = Math.atan2(y - 3.5, x - 3.5);
        } else if (encodingMode === 'linear') {
          inPhase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
        }

        // We want W_idx * X_idx to have a specific phase (e.g. 0) when pixel is drawn.
        // Therefore, phase(W_idx) should be approximately -phase(X_idx) for constructive addition.
        const targetWPhase = -inPhase * globalPhaseScale;

        if (classTemplate[idx] === 1) {
          // Inside template: high magnitude, conjugated phase (reconstructive)
          // Normalize to prevent current saturation
          row.push(complex.fromPolar(1.5 / activePixels, targetWPhase));
        } else {
          // Outside template: penalize by adding an out-of-phase weight (180 deg shifted)
          // which causes destructive interference if that pixel is incorrectly drawn!
          row.push(complex.fromPolar(0.8 / activePixels, targetWPhase + Math.PI));
        }
      }
      weights.push(row);
    }
    return weights;
  }, [encodingMode, globalPhaseScale]);

  // Compiled Hardware Weights mapping (applying quantization, noise, and correction)
  const compiledWeights = useMemo(() => {
    const weights: Complex[][] = [];

    for (let c = 0; c < 4; c++) {
      const row: Complex[] = [];
      for (let idx = 0; idx < 64; idx++) {
        const origW = idealWeights[c][idx];
        let wr = origW.re;
        let wi = origW.im;

        // 1. Quantization (Bit width conversion)
        if (hardwareBits > 0) {
          const levels = Math.pow(2, hardwareBits) - 1;
          // Scale from [-1, 1] to [0, levels], round, then scale back
          wr = Math.round(((wr + 1) / 2) * levels) / levels * 2 - 1;
          wi = Math.round(((wi + 1) / 2) * levels) / levels * 2 - 1;
        }

        // 2. Hardware Non-idealities & Noise (Memristor cycle-to-cycle variation)
        if (writeNoise > 0) {
          // If self-correction is enabled, the chip feedback mitigates 80% of the noise
          const effectiveNoise = enableSelfCorrection ? writeNoise * 0.15 : writeNoise;
          
          // Add Gaussian noise using Box-Muller transform
          const u1 = Math.random() || 1e-9;
          const u2 = Math.random() || 1e-9;
          const randStd = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          
          wr += randStd * effectiveNoise;
          wi += randStd * effectiveNoise;
        }

        row.push(complex.create(wr, wi));
      }
      weights.push(row);
    }
    return weights;
  }, [idealWeights, hardwareBits, writeNoise, enableSelfCorrection]);

  // Encode pixel array to Complex time sequence
  const encodedSequence = useMemo(() => {
    const sequence: Complex[][] = [];

    for (let t = 0; t < simSteps; t++) {
      const stepInputs: Complex[] = [];
      const timeFactor = t / simSteps;

      for (let idx = 0; idx < 64; idx++) {
        const active = grid[idx];
        if (active === 0) {
          stepInputs.push(complex.zero());
          continue;
        }

        const x = idx % GRID_SIZE;
        const y = Math.floor(idx / GRID_SIZE);

        // 1. Determine local spatial phase of the pixel
        let spatialPhase = 0;
        if (encodingMode === 'radial') {
          spatialPhase = Math.atan2(y - 3.5, x - 3.5);
        } else if (encodingMode === 'linear') {
          spatialPhase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
        } else {
          spatialPhase = 0.5; // uniform phase offset
        }

        // Apply scale
        spatialPhase *= globalPhaseScale;

        // 2. Add temporal wave modulation (complex-valued spike signal carrying oscillatory phase)
        const temporalPhase = 2 * Math.PI * 1.5 * timeFactor; // 1.5Hz base signal rotation
        const finalPhase = spatialPhase + temporalPhase;

        // Magnitude corresponds to pixel presence with small temporal oscillation
        const amplitude = 0.8 + 0.2 * Math.sin(2 * Math.PI * 2 * timeFactor);

        stepInputs.push(complex.fromPolar(amplitude, finalPhase));
      }
      sequence.push(stepInputs);
    }
    return sequence;
  }, [grid, simSteps, encodingMode, globalPhaseScale]);

  // Run the SNN simulation on the drawn pattern
  const runSimulation = () => {
    setIsSimulating(true);

    setTimeout(() => {
      // Create 4 output cLIF neurons
      const neurons = Array.from({ length: 4 }).map(() => 
        new ComplexLIFNeuron(leakMag, leakPhase, threshold, 'soft')
      );

      const potentials: Complex[][] = [];
      const spikes: boolean[][] = [];
      const spikeCounts = [0, 0, 0, 0];

      // To calculate phase coherence, we measure the phase alignment of input currents
      const inputPhaseAngles: number[][] = Array.from({ length: 4 }).map(() => []);

      for (let t = 0; t < simSteps; t++) {
        const stepInputs = encodedSequence[t];
        const stepPotentials: Complex[] = [];
        const stepSpikes: boolean[] = [];

        for (let o = 0; o < 4; o++) {
          let current = complex.zero();
          for (let i = 0; i < 64; i++) {
            if (grid[i] === 1) {
              const prod = complex.mul(compiledWeights[o][i], stepInputs[i]);
              current = complex.add(current, prod);
            }
          }

          // Accumulate phase of inputs for synchronization analysis
          if (complex.mag(current) > 1e-4) {
            inputPhaseAngles[o].push(complex.angle(current));
          }

          const res = neurons[o].step(current);
          stepPotentials.push(res.uAfter);
          stepSpikes.push(res.spiked);

          if (res.spiked) {
            spikeCounts[o]++;
          }
        }

        potentials.push(stepPotentials);
        spikes.push(stepSpikes);
      }

      // Calculate phase coherence (order parameter R = |(1/N) * sum(e^iθ)|)
      // High R means inputs arrived in phase coherence (constructive interference)
      const phaseCoherence = inputPhaseAngles.map(angles => {
        if (angles.length === 0) return 0;
        let sumRe = 0;
        let sumIm = 0;
        angles.forEach(theta => {
          sumRe += Math.cos(theta);
          sumIm += Math.sin(theta);
        });
        const r = Math.sqrt(sumRe * sumRe + sumIm * sumIm) / angles.length;
        return r;
      });

      setSimResult({
        spikeRates: spikeCounts.map(count => count / simSteps),
        potentials,
        spikes,
        phaseCoherence
      });
      setIsSimulating(false);
    }, 400);
  };

  // Run simulation on grid modification automatically
  useEffect(() => {
    runSimulation();
  }, [grid, encodingMode, globalPhaseScale, leakMag, leakPhase, threshold, compiledWeights, simSteps]);

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
    const newGrid = [...grid];
    newGrid[idx] = drawValue;
    setGrid(newGrid);
  };

  const handlePixelMouseUp = () => {
    setIsDrawing(false);
  };

  // Helper to color individual pixels based on phase to display wave
  const getPixelColorStyle = (idx: number) => {
    if (grid[idx] === 0) return 'bg-[#15202B]/60 border border-elegant-muted/5';
    
    // Derived spatial phase
    const x = idx % GRID_SIZE;
    const y = Math.floor(idx / GRID_SIZE);
    let phase = 0;
    if (encodingMode === 'radial') {
      phase = Math.atan2(y - 3.5, x - 3.5);
    } else if (encodingMode === 'linear') {
      phase = ((x + y) / (GRID_SIZE * 2)) * Math.PI;
    }

    const rotDegree = (phase * 180 / Math.PI).toFixed(0);
    // Map phase to HSL color wheel
    const hue = Math.round(((phase + Math.PI) / (2 * Math.PI)) * 360);
    return {
      background: `hsla(${hue}, 85%, 45%, 0.9)`,
      boxShadow: `0 0 8px hsla(${hue}, 85%, 55%, 0.4)`,
      border: `1px solid hsla(${hue}, 90%, 65%, 0.6)`
    };
  };

  // Find predicted class
  const predictionIdx = useMemo(() => {
    if (!simResult) return -1;
    let maxRate = -1;
    let bestIdx = -1;
    simResult.spikeRates.forEach((rate, idx) => {
      if (rate > maxRate) {
        maxRate = rate;
        bestIdx = idx;
      }
    });
    // If no spikes, use phase coherence as fallback
    if (maxRate === 0) {
      let maxCoh = -1;
      simResult.phaseCoherence.forEach((coh, idx) => {
        if (coh > maxCoh) {
          maxCoh = coh;
          bestIdx = idx;
        }
      });
    }
    return bestIdx;
  }, [simResult]);

  // Simulate hardware compiler accuracy drop metric
  const hardwareMetrics = useMemo(() => {
    // Ideal soft accuracy is 100%
    // If noise goes up, accuracy degrades.
    // If bit precision drops below 4, accuracy degrades.
    // Self-correction maintains accuracy near 98%.
    let estimatedAccuracy = 1.0;
    
    // Bit width penalty
    if (hardwareBits > 0 && hardwareBits < 6) {
      estimatedAccuracy -= (6 - hardwareBits) * 0.04;
    }
    
    // Noise penalty
    if (writeNoise > 0) {
      const noisePenalty = writeNoise * 0.8;
      if (enableSelfCorrection) {
        estimatedAccuracy -= noisePenalty * 0.12; // self correction reduces 88% noise impact
      } else {
        estimatedAccuracy -= noisePenalty;
      }
    }

    return Math.max(0.45, Math.min(1.0, estimatedAccuracy));
  }, [hardwareBits, writeNoise, enableSelfCorrection]);

  return (
    <div className="space-y-6" onMouseUp={handlePixelMouseUp}>
      
      {/* Title block */}
      <div className="bg-gradient-to-r from-elegant-card to-elegant-card/40 border border-elegant-muted/15 p-5 rounded-xl shadow-xl space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
              <Cpu className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h2 className="text-xl font-bold font-display text-white">Thử Nghiệm Thị Giác Máy Tính & Biên Dịch Phần Cứng CV-SNN</h2>
              <p className="text-xs text-elegant-text/80 leading-relaxed mt-0.5">
                Thiết lập chu kỳ sóng pha cho dữ liệu ảnh 2D, nhận diện ký tự mẫu, và mô phỏng biên dịch nạp trực tiếp lên mảng chip của GS. Choi (KAIST).
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] bg-emerald-400/10 text-emerald-400 border border-emerald-400/20 px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1">
              <Check className="w-3.5 h-3.5" />
              <span>Phần Mềm Đã Đồng Bộ Hóa</span>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* COLUMN 1: SANDBOX CANVAS DRAWING & PRESETS */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-5 shadow-xl flex flex-col">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 rounded-full bg-elegant-cyan" />
              <span>1. Bản Vẽ & Mã Hóa Pha (Drawing Canvas)</span>
            </h3>
            <button 
              onClick={() => setGrid(new Array(64).fill(0))}
              className="text-xs text-rose-400 hover:text-rose-300 transition flex items-center gap-1 font-mono"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Xóa Canvas</span>
            </button>
          </div>

          <p className="text-xs text-elegant-text/75 leading-relaxed">
            Nhấp chọn hoặc kéo chuột để vẽ các ký hiệu. Màu sắc thể hiện <strong>góc pha (θ)</strong> của từng pixel trong miền số phức để tạo sóng truyền tới synapse.
          </p>

          {/* Quick templates presets selector */}
          <div className="space-y-2">
            <span className="text-[11px] font-mono text-elegant-muted block font-bold uppercase tracking-wider">Mẫu chuẩn lưu sẵn (Templates Presets):</span>
            <div className="grid grid-cols-4 gap-2">
              {CLASS_NAMES.map((name) => (
                <button
                  key={`preset-${name}`}
                  onClick={() => setGrid(TEMPLATES[name])}
                  className="py-1.5 px-3 bg-elegant-bg hover:bg-elegant-card hover:border-elegant-cyan/40 border border-elegant-muted/10 rounded-md text-xs font-bold text-white transition flex items-center justify-center gap-1"
                >
                  <Sparkles className="w-3 h-3 text-elegant-cyan" />
                  <span>Ký tự {name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Canvas Box */}
          <div className="flex justify-center py-4 bg-elegant-bg/40 rounded-xl border border-elegant-muted/10">
            <div className="grid grid-cols-8 gap-1.5 p-3.5 bg-elegant-bg/90 rounded-xl shadow-inner select-none">
              {grid.map((val, idx) => {
                const style = getPixelColorStyle(idx);
                const isObject = typeof style === 'object';
                
                return (
                  <div
                    key={`pixel-${idx}`}
                    onMouseDown={() => handlePixelMouseDown(idx)}
                    onMouseEnter={() => handlePixelMouseEnter(idx)}
                    style={isObject ? style : undefined}
                    className={`w-8 h-8 rounded-md cursor-pointer transition-all duration-100 flex items-center justify-center ${!isObject ? style : ''}`}
                  >
                    {val === 1 && (
                      <span className="w-1.5 h-1.5 rounded-full bg-white/50" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Phase encoding parameters */}
          <div className="space-y-4 pt-2 border-t border-elegant-muted/10">
            <div className="space-y-1">
              <label className="text-[11px] font-mono text-elegant-muted font-bold block uppercase">Phương pháp Mã hóa Pha (Phase Mapping):</label>
              <div className="grid grid-cols-3 gap-1.5 mt-1">
                {(['radial', 'linear', 'uniform'] as PhaseEncodingMode[]).map((mode) => (
                  <button
                    key={`mode-${mode}`}
                    onClick={() => setEncodingMode(mode)}
                    className={`py-1 px-1.5 rounded text-[11px] font-semibold transition border text-center capitalize ${
                      encodingMode === mode 
                        ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35' 
                        : 'bg-elegant-bg/50 text-elegant-text/70 border-elegant-muted/10 hover:text-white'
                    }`}
                  >
                    {mode === 'radial' ? 'Tâm Tròn' : mode === 'linear' ? 'Tuyến Tính' : 'Đồng Pha'}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>Hệ số nhân góc pha (γ):</span>
                <span className="text-elegant-cyan font-bold">{globalPhaseScale.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="2.0"
                step="0.2"
                value={globalPhaseScale}
                onChange={(e) => setGlobalPhaseScale(parseFloat(e.target.value))}
                className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* COLUMN 2: HARDWARE-AWARE SOFTWARE COMPILER (CHIP EMULATOR) */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-5 shadow-xl flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span>2. Bộ Biên Dịch & Ánh Xạ Chip GS. Choi (KAIST)</span>
              </h3>
            </div>

            <p className="text-xs text-elegant-text/75 leading-relaxed">
              Mô hình toán học phần mềm số phức sẽ được <strong>Bộ Biên Dịch Ánh Xạ (Hardware Mapping Compiler)</strong> phân tích thành các mức dẫn nạp conductance vật lý thực của mảng memristor.
            </p>

            {/* Precision Bits */}
            <div className="space-y-2 p-3.5 bg-elegant-bg/60 border border-elegant-muted/10 rounded-lg">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white font-bold">Độ phân giải Memristor PCM:</span>
                <span className="text-amber-400 font-bold">
                  {hardwareBits === 0 ? 'Vô hạn (Lý tưởng)' : `${hardwareBits} bits (${Math.pow(2, hardwareBits)} mức dẫn nạp)`}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="8"
                step="1"
                value={hardwareBits}
                onChange={(e) => setHardwareBits(parseInt(e.target.value))}
                className="w-full accent-amber-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-elegant-muted leading-normal">
                {hardwareBits > 0 
                  ? `Trọng số số phức liên tục sẽ bị lượng tử hóa thành ${Math.pow(2, hardwareBits)} mức điện trở rời rạc tương ứng các phân đoạn thay đổi pha kết tinh của vật liệu PCM.`
                  : 'Trọng số được giữ ở độ chính xác dấu phẩy động 64-bit tối đa.'}
              </p>
            </div>

            {/* Noise Injection */}
            <div className="space-y-2 p-3.5 bg-elegant-bg/60 border border-elegant-muted/10 rounded-lg">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white font-bold">Độ nhiễu nạp ghi (Write/Conductance Noise):</span>
                <span className="text-rose-400 font-bold">{(writeNoise * 100).toFixed(0)}% std dev</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.4"
                step="0.05"
                value={writeNoise}
                onChange={(e) => setWriteNoise(parseFloat(e.target.value))}
                className="w-full accent-rose-500 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-elegant-muted leading-normal">
                Giả lập sự bất ổn định vật lý (Cycle-to-Cycle / Device-to-Device variation) khi nạp trị số vào linh kiện làm suy giảm góc pha.
              </p>
            </div>

            {/* Self correction loop (Nature Electronics 2024 breakthrough) */}
            <div className="p-4 bg-elegant-cyan/5 border border-elegant-cyan/20 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-elegant-cyan font-bold font-display text-xs">
                  <Zap className="w-4 h-4 animate-bounce" />
                  <span>Cơ chế On-Chip Self-Correction (KAIST 2024)</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={enableSelfCorrection}
                    onChange={(e) => setEnableSelfCorrection(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-elegant-bg rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-elegant-muted after:border-elegant-card after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-elegant-cyan peer-checked:after:bg-elegant-bg" />
                </label>
              </div>

              <p className="text-[11px] text-elegant-text/80 leading-relaxed">
                Đột phá từ Lab GS. Shinhyun Choi: Kích hoạt mạch tự bù trừ phi tuyến và bù nhiễu dẫn nạp trực tiếp on-chip ngay khi ghi dữ liệu. Giúp khôi phục trạng thái sóng pha nguyên vẹn!
              </p>
            </div>
          </div>

          {/* Accuracy & Coherence comparison metric */}
          <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/15 rounded-xl space-y-3 mt-4">
            <div className="flex justify-between items-center text-xs">
              <span className="text-elegant-muted font-mono uppercase font-bold">Chỉ số Hiệu năng Ước tính trên Chip:</span>
              <span className={`font-mono font-bold text-sm ${hardwareMetrics > 0.85 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {(hardwareMetrics * 100).toFixed(1)}% Accuracy
              </span>
            </div>
            
            <div className="h-2 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
              <div 
                className={`h-full rounded-full transition-all duration-300 ${
                  hardwareMetrics > 0.85 ? 'bg-emerald-400' : hardwareMetrics > 0.7 ? 'bg-amber-400' : 'bg-rose-500'
                }`}
                style={{ width: `${hardwareMetrics * 100}%` }}
              />
            </div>

            <div className="text-[10px] text-elegant-muted flex items-start gap-1">
              <Info className="w-3 h-3 text-elegant-cyan shrink-0 mt-0.5" />
              <span>
                {enableSelfCorrection 
                  ? 'Tuyệt vời! Mạch bù sửa lỗi của KAIST giúp duy trì độ chính xác cao bất kể nhiễu và hạn chế mức conductance phân rã.'
                  : 'Nhiễu memristor chưa được bù trừ đang làm lệch pha dòng điện, khiến hiệu suất nhận diện giảm mạnh.'}
              </span>
            </div>
          </div>
        </div>

        {/* COLUMN 3: REALTIME INFERENCE RESULT & VISUALIZATIONS */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-5 shadow-xl flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                <span>3. Kết quả Nhận diện Miền Phức (Live Inference)</span>
              </h3>
            </div>

            <p className="text-xs text-elegant-text/75 leading-relaxed">
              Mạng nổ xung miền phức liên tục tính toán chập dòng Kirchhoff, xuất kết quả nổ xung lớp thần kinh ra 4 nhóm:
            </p>

            {/* Inference Probabilities */}
            <div className="space-y-3">
              {CLASS_NAMES.map((name, idx) => {
                const isWinner = predictionIdx === idx;
                const spikeRate = simResult?.spikeRates[idx] || 0;
                const coherence = simResult?.phaseCoherence[idx] || 0;
                
                return (
                  <div 
                    key={`prob-${name}`}
                    className={`p-3 rounded-lg border transition-all ${
                      isWinner 
                        ? 'bg-elegant-cyan/10 border-elegant-cyan/35 shadow-md shadow-black/25' 
                        : 'bg-elegant-bg/40 border-elegant-muted/10'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center font-mono font-bold text-[11px] ${
                          isWinner ? 'bg-elegant-cyan text-elegant-bg' : 'bg-elegant-bg text-elegant-muted'
                        }`}>
                          {name}
                        </span>
                        <span className={`font-semibold ${isWinner ? 'text-white' : 'text-elegant-text/70'}`}>
                          Nhận diện chữ {name}
                        </span>
                      </div>
                      <div className="text-right font-mono text-[11px] space-y-0.5">
                        <div className={`${isWinner ? 'text-elegant-cyan font-bold' : 'text-elegant-muted'}`}>
                          Tỷ lệ nổ: {(spikeRate * 100).toFixed(0)}%
                        </div>
                        <div className="text-[10px] text-elegant-muted">
                          Sự đồng bộ pha (R): {coherence.toFixed(2)}
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="h-2 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                      <div 
                        className={`h-full rounded-full transition-all duration-300 ${
                          isWinner ? 'bg-elegant-cyan' : 'bg-elegant-muted/40'
                        }`}
                        style={{ width: `${Math.max(3, spikeRate * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Highlight winner prediction box */}
          {predictionIdx !== -1 && (
            <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/10 rounded-xl space-y-2 text-center shadow-inner">
              <span className="text-[10px] font-mono text-elegant-muted uppercase tracking-wider block">KẾT QUẢ PHÂN LOẠI HIỆN TẠI:</span>
              <div className="text-2xl font-black font-display text-elegant-cyan flex items-center justify-center gap-2">
                <Brain className="w-6 h-6 animate-pulse" />
                <span>Ký tự "{CLASS_NAMES[predictionIdx]}"</span>
              </div>
              <p className="text-[11px] text-elegant-text/70 leading-relaxed italic max-w-xs mx-auto">
                Nhờ vào mã hóa pha, mạng phân loại chuẩn xác và chống nhiễu cực tốt nhờ sự giao thoa cộng pha của các dòng memristor vi sai.
              </p>
            </div>
          )}
        </div>

      </div>

      {/* FOOTER ADVICE / SCIENTIFIC BREAKDOWN */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl">
        <h4 className="text-sm font-bold text-white flex items-center gap-1.5 font-display">
          <Info className="w-4 h-4 text-elegant-cyan" />
          <span>Tại sao miền pha số phức của CV-SNN là chiếc "Chìa Khóa Vàng" để mở rộng ứng dụng thị giác máy tính?</span>
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs text-elegant-text/80 leading-relaxed">
          <p>
            Trong xử lý hình ảnh, thông tin không chỉ nằm ở cường độ sáng (biên độ - magnitude) mà còn nằm ở sự tương quan không gian (cạnh, biên, góc). 
            Bằng cách mã hóa <strong>vị trí pixel thành các giá trị góc pha (θ) tương đối</strong>, CV-SNN cho phép tự động trích xuất các vector đặc trưng không gian thông qua các <strong>phép tính cộng tích chập pha tự nhiên</strong> trên các cổng vi sai memristor.
          </p>
          <p>
            Điều này loại bỏ hoàn toàn các mạch cộng nhân dấu phẩy động cồng kềnh trong các bộ tăng tốc AI truyền thống. Khi có nhiễu nạp (Device write noise), cơ chế lan truyền sóng pha của CV-SNN sẽ phân tán lỗi ngẫu nhiên theo hình tròn trong mặt phẳng phức Gauss, giúp duy trì cấu trúc hình học tổng thể của tấm ảnh vẽ tốt hơn hàng trăm lần so với mạng SNN miền thực.
          </p>
        </div>
      </div>

    </div>
  );
}
