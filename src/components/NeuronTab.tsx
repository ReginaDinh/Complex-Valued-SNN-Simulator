import { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'motion/react';
import { Activity, Play, Pause, RefreshCw, Sliders, Info, Zap, Radio } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { complex, ComplexLIFNeuron } from '../simulationEngine';

export default function NeuronTab() {
  const { t, language } = useLanguage();
  const isEn = language === 'en';

  const [isPlaying, setIsPlaying] = useState(true);
  const [leakMag, setLeakMag] = useState(0.85);
  const [leakPhase, setLeakPhase] = useState(0.5);
  const [threshold, setThreshold] = useState(1.2);
  const [resetMode, setResetMode] = useState<'soft' | 'hard'>('soft');

  const [inputType, setInputType] = useState<'ac' | 'dc' | 'pulse' | 'noise'>('ac');
  const [inputAmp, setInputAmp] = useState(0.35);
  const [inputFreq, setInputFreq] = useState(0.08);
  const [inputPhase, setInputPhase] = useState(0.0);

  const [liveStats, setLiveStats] = useState({
    spikeCount: 0,
    lastSpikeAngleDeg: 0,
    instantMag: 0,
  });

  const simTimeRef = useRef(0);
  const totalSpikesRef = useRef(0);
  const neuronRef = useRef<ComplexLIFNeuron | null>(null);

  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const fgCanvasRef = useRef<HTMLCanvasElement>(null);
  const timeCanvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<
    {
      t: number;
      uRe: number;
      uIm: number;
      preRe: number;
      preIm: number;
      spiked: boolean;
    }[]
  >([]);

  // Theoretical resonant frequency f_res = leakPhase / (2 * pi) wrapped into [0.01, 0.25]
  const resonantFreq = useMemo(() => {
    let f = leakPhase / (2 * Math.PI);
    while (f < 0) f += 1.0;
    return Math.max(0.01, Math.min(0.25, f));
  }, [leakPhase]);

  // Compute Frequency Resonance Sweep (Bode Gain + Empirical Spike Rate across f in [0.01, 0.25])
  const resonanceCurve = useMemo(() => {
    const points: {
      freq: number;
      gain: number;
      spikeRate: number;
    }[] = [];

    for (let i = 0; i <= 24; i++) {
      const f = 0.01 + i * 0.01;
      const omega = 2 * Math.PI * f;
      // Transfer function |H(e^{j*omega})| = 1 / |1 - leakMag * e^{j*(leakPhase - omega)}|
      const delta = leakPhase - omega;
      const reDenom = 1 - leakMag * Math.cos(delta);
      const imDenom = -leakMag * Math.sin(delta);
      const gain = 1.0 / Math.sqrt(reDenom * reDenom + imDenom * imDenom + 1e-6);

      // Run 50-step micro-simulation to measure actual firing rate at frequency f
      const testNeuron = new ComplexLIFNeuron(leakMag, leakPhase, threshold, resetMode);
      let spks = 0;
      for (let step = 0; step < 50; step++) {
        const [inRe, inIm] = complex.fromPolar(inputAmp, omega * step + inputPhase);
        if (testNeuron.step(inRe, inIm).spiked) spks++;
      }

      points.push({
        freq: f,
        gain,
        spikeRate: spks / 50,
      });
    }
    return points;
  }, [leakMag, leakPhase, threshold, resetMode, inputAmp, inputPhase]);

  // Initialize neuron
  useEffect(() => {
    neuronRef.current = new ComplexLIFNeuron(leakMag, leakPhase, threshold, resetMode);
  }, [leakMag, leakPhase, threshold, resetMode]);

  // Background Canvas (Static Grid & Threshold Circle)
  useEffect(() => {
    const canvas = bgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 80;

    // Grid circles
    ctx.strokeStyle = 'rgba(69, 162, 158, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (let r = 0.5; r <= 2.0; r += 0.5) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r * scale, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = '#45A29E';
      ctx.font = '9px monospace';
      ctx.fillText(r.toFixed(1), centerX + r * scale + 4, centerY - 4);
    }

    // Axes
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(69, 162, 158, 0.25)';
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.stroke();

    // Threshold Circle
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(centerX, centerY, threshold * scale, 0, 2 * Math.PI);
    ctx.stroke();
  }, [threshold]);

  // Simulation Loop with requestAnimationFrame
  useEffect(() => {
    if (!isPlaying) return;

    let animationFrameId: number;
    let lastTime = performance.now();

    const tick = (time: number) => {
      if (time - lastTime > 100) {
        if (neuronRef.current) {
          const tStep = simTimeRef.current;

          let inRe = 0;
          let inIm = 0;

          if (inputType === 'ac') {
            const currentPhase = 2 * Math.PI * inputFreq * tStep + inputPhase;
            [inRe, inIm] = complex.fromPolar(inputAmp, currentPhase);
          } else if (inputType === 'dc') {
            [inRe, inIm] = complex.fromPolar(inputAmp, inputPhase);
          } else if (inputType === 'pulse') {
            const isPulse = tStep % 8 === 0 && tStep > 0;
            const amp = isPulse ? inputAmp * 2.5 : 0.05;
            [inRe, inIm] = complex.fromPolar(amp, inputPhase);
          } else if (inputType === 'noise') {
            const randAmp = Math.max(0, inputAmp * (0.3 + Math.random() * 1.2));
            const randPhase = inputPhase + (Math.random() - 0.5) * Math.PI * 1.5;
            [inRe, inIm] = complex.fromPolar(randAmp, randPhase);
          }

          const stepRes = neuronRef.current.step(inRe, inIm);
          if (stepRes.spiked) {
            totalSpikesRef.current += 1;
          }

          setLiveStats({
            spikeCount: totalSpikesRef.current,
            lastSpikeAngleDeg: stepRes.spiked
              ? (stepRes.preResetAngle * 180) / Math.PI
              : liveStats.lastSpikeAngleDeg,
            instantMag: stepRes.preResetMag,
          });

          historyRef.current.push({
            t: tStep,
            uRe: neuronRef.current.uRe,
            uIm: neuronRef.current.uIm,
            preRe: neuronRef.current.preResetRe,
            preIm: neuronRef.current.preResetIm,
            spiked: stepRes.spiked,
          });
          if (historyRef.current.length > 60) historyRef.current.shift();

          simTimeRef.current += 1;

          // Render Foreground Trajectory Canvas
          const canvas = fgCanvasRef.current;
          const ctx = canvas?.getContext('2d');
          if (canvas && ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const scale = 80;

            // Draw Trajectory using pre-reset spike points so threshold crossings are accurately shown
            ctx.beginPath();
            ctx.strokeStyle = '#66FCF1';
            ctx.lineWidth = 2;

            for (let i = 0; i < historyRef.current.length; i++) {
              const pt = historyRef.current[i];
              const dispRe = pt.spiked ? pt.preRe : pt.uRe;
              const dispIm = pt.spiked ? pt.preIm : pt.uIm;
              const x = centerX + dispRe * scale;
              const y = centerY - dispIm * scale;
              if (i === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw current point and spikes at exact threshold crossing coordinates
            const currentPt = historyRef.current[historyRef.current.length - 1];
            if (currentPt) {
              const dispRe = currentPt.spiked ? currentPt.preRe : currentPt.uRe;
              const dispIm = currentPt.spiked ? currentPt.preIm : currentPt.uIm;
              const x = centerX + dispRe * scale;
              const y = centerY - dispIm * scale;

              ctx.beginPath();
              ctx.arc(x, y, currentPt.spiked ? 8 : 4.5, 0, 2 * Math.PI);
              ctx.fillStyle = currentPt.spiked ? '#F87171' : '#66FCF1';
              ctx.fill();
              if (currentPt.spiked) {
                ctx.strokeStyle = '#F87171';
                ctx.lineWidth = 2;
                ctx.stroke();
              }
            }
          }

          // Render Time-Series Canvas
          const tCanvas = timeCanvasRef.current;
          const tCtx = tCanvas?.getContext('2d');
          if (tCanvas && tCtx) {
            tCtx.clearRect(0, 0, tCanvas.width, tCanvas.height);
            const w = tCanvas.width;
            const h = tCanvas.height;

            tCtx.strokeStyle = 'rgba(69, 162, 158, 0.12)';
            tCtx.lineWidth = 1;
            tCtx.beginPath();
            tCtx.moveTo(0, h / 2);
            tCtx.lineTo(w, h / 2);
            tCtx.stroke();

            const threshY_top = h / 2 - threshold * (h * 0.35);
            const threshY_bot = h / 2 + threshold * (h * 0.35);
            tCtx.strokeStyle = 'rgba(248, 113, 113, 0.4)';
            tCtx.setLineDash([4, 4]);
            tCtx.beginPath();
            tCtx.moveTo(0, threshY_top);
            tCtx.lineTo(w, threshY_top);
            tCtx.moveTo(0, threshY_bot);
            tCtx.lineTo(w, threshY_bot);
            tCtx.stroke();
            tCtx.setLineDash([]);

            tCtx.fillStyle = '#F87171';
            tCtx.font = '9px monospace';
            tCtx.fillText(`+Vth (${threshold.toFixed(1)})`, 5, threshY_top - 4);
            tCtx.fillText(`-Vth (-${threshold.toFixed(1)})`, 5, threshY_bot + 10);

            const pts = historyRef.current;
            if (pts.length > 1) {
              const dx = w / 60;

              // Real part path (cyan)
              tCtx.beginPath();
              tCtx.strokeStyle = '#66FCF1';
              tCtx.lineWidth = 1.8;
              for (let i = 0; i < pts.length; i++) {
                const x = i * dx;
                const valRe = pts[i].spiked ? pts[i].preRe : pts[i].uRe;
                const y = h / 2 - valRe * (h * 0.35);
                if (i === 0) tCtx.moveTo(x, y);
                else tCtx.lineTo(x, y);
              }
              tCtx.stroke();

              // Imaginary part path (amber)
              tCtx.beginPath();
              tCtx.strokeStyle = '#FBBF24';
              tCtx.lineWidth = 1.8;
              for (let i = 0; i < pts.length; i++) {
                const x = i * dx;
                const valIm = pts[i].spiked ? pts[i].preIm : pts[i].uIm;
                const y = h / 2 - valIm * (h * 0.35);
                if (i === 0) tCtx.moveTo(x, y);
                else tCtx.lineTo(x, y);
              }
              tCtx.stroke();

              // Spikes as vertical red columns
              for (let i = 0; i < pts.length; i++) {
                if (pts[i].spiked) {
                  const x = i * dx;
                  tCtx.fillStyle = 'rgba(248, 113, 113, 0.25)';
                  tCtx.fillRect(x - 2, 5, 4, h - 10);

                  tCtx.strokeStyle = '#F87171';
                  tCtx.lineWidth = 1.5;
                  tCtx.beginPath();
                  tCtx.moveTo(x, 5);
                  tCtx.lineTo(x, h - 5);
                  tCtx.stroke();
                }
              }
            }
          }
        }
        lastTime = time;
      }
      animationFrameId = requestAnimationFrame(tick);
    };

    animationFrameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, inputAmp, inputFreq, inputPhase, inputType]);

  const handleReset = () => {
    if (neuronRef.current) neuronRef.current.reset();
    simTimeRef.current = 0;
    totalSpikesRef.current = 0;
    historyRef.current = [];
    setLiveStats({ spikeCount: 0, lastSpikeAngleDeg: 0, instantMag: 0 });

    const canvas = fgCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
    const tCanvas = timeCanvasRef.current;
    if (tCanvas) {
      const tCtx = tCanvas.getContext('2d');
      tCtx?.clearRect(0, 0, tCanvas.width, tCanvas.height);
    }
  };

  return (
    <motion.div
      key="neuron"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="grid grid-cols-1 lg:grid-cols-3 gap-6"
    >
      {/* Control Panel (1 Column) */}
      <div className="space-y-6">
        {/* Neuron Parameters */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
            <Sliders className="w-4 h-4 text-elegant-cyan" />
            <span>{t('neuron.params')}</span>
          </h3>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{t('neuron.leakMag')}</span>
              <span className="text-elegant-cyan font-bold">{leakMag.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.98"
              step="0.01"
              value={leakMag}
              onChange={e => setLeakMag(parseFloat(e.target.value))}
              className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{t('neuron.leakPhase')}</span>
              <span className="text-elegant-cyan font-bold">
                {leakPhase.toFixed(2)} rad (f₀ ≈ {resonantFreq.toFixed(3)})
              </span>
            </div>
            <input
              type="range"
              min="0.05"
              max="1.57"
              step="0.05"
              value={leakPhase}
              onChange={e => setLeakPhase(parseFloat(e.target.value))}
              className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{t('neuron.threshold')}</span>
              <span className="text-elegant-cyan font-bold">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
            />
          </div>

          <div className="space-y-1.5 pt-2 border-t border-elegant-muted/10">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Reset Mode:' : 'Chế độ Reset:'}</span>
              <span className="text-elegant-cyan font-bold">
                {resetMode === 'soft'
                  ? isEn
                    ? 'SOFT (Subtractive)'
                    : 'SOFT (Trừ ngưỡng)'
                  : isEn
                  ? 'HARD (To Zero)'
                  : 'HARD (Về 0)'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-1">
              <button
                onClick={() => setResetMode('soft')}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg border transition-all cursor-pointer ${
                  resetMode === 'soft'
                    ? 'bg-elegant-cyan/20 border-elegant-cyan text-elegant-cyan'
                    : 'border-elegant-muted/15 text-elegant-muted hover:text-white'
                }`}
              >
                Soft Reset
              </button>
              <button
                onClick={() => setResetMode('hard')}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg border transition-all cursor-pointer ${
                  resetMode === 'hard'
                    ? 'bg-elegant-cyan/20 border-elegant-cyan text-elegant-cyan'
                    : 'border-elegant-muted/15 text-elegant-muted hover:text-white'
                }`}
              >
                Hard Reset
              </button>
            </div>
          </div>
        </div>

        {/* Input Stimuli Parameters */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
            <Activity className="w-4 h-4 text-elegant-cyan" />
            <span>{t('neuron.inputCurrent')}</span>
          </h3>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Input Waveform:' : 'Dạng dòng điện:'}</span>
              <span className="text-elegant-cyan font-bold">{inputType.toUpperCase()}</span>
            </div>
            <div className="grid grid-cols-4 gap-1 mt-1">
              {(['ac', 'dc', 'pulse', 'noise'] as const).map(type => (
                <button
                  key={type}
                  onClick={() => setInputType(type)}
                  className={`py-1 text-[9px] font-bold rounded border transition-all cursor-pointer ${
                    inputType === type
                      ? 'bg-elegant-cyan/20 border-elegant-cyan text-elegant-cyan'
                      : 'border-elegant-muted/15 text-elegant-muted hover:text-white hover:bg-elegant-card/40'
                  }`}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Amplitude:' : 'Biên độ:'}</span>
              <span className="text-elegant-cyan font-bold">{inputAmp.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="1.2"
              step="0.05"
              value={inputAmp}
              onChange={e => setInputAmp(parseFloat(e.target.value))}
              className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
            />
          </div>

          {inputType === 'ac' && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>{isEn ? 'Input AC Frequency (f):' : 'Tần số dòng AC (f):'}</span>
                <span className="text-elegant-cyan font-bold">
                  {inputFreq.toFixed(3)} (ω = {(2 * Math.PI * inputFreq).toFixed(2)} rad)
                </span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.25"
                step="0.005"
                value={inputFreq}
                onChange={e => setInputFreq(parseFloat(e.target.value))}
                className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
              <div className="grid grid-cols-2 gap-1.5 pt-1">
                <button
                  onClick={() => setInputFreq(Number(resonantFreq.toFixed(3)))}
                  className="py-1.5 px-2 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/35 text-emerald-300 rounded text-[10px] font-mono font-bold transition cursor-pointer flex items-center justify-center gap-1"
                >
                  <Zap className="w-3 h-3" />
                  <span>{isEn ? 'Lock Resonance' : 'Khóa Cộng Hưởng'}</span>
                </button>
                <button
                  onClick={() =>
                    setInputFreq(Number(Math.min(0.25, resonantFreq + 0.12).toFixed(3)))
                  }
                  className="py-1.5 px-2 bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/35 text-rose-300 rounded text-[10px] font-mono font-bold transition cursor-pointer"
                >
                  {isEn ? 'Detune (Off-Phase)' : 'Lệch Pha Cưỡng Bức'}
                </button>
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Input Phase (rad):' : 'Pha dòng điện (rad):'}</span>
              <span className="text-elegant-cyan font-bold">{inputPhase.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-3.14"
              max="3.14"
              step="0.1"
              value={inputPhase}
              onChange={e => setInputPhase(parseFloat(e.target.value))}
              className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
            />
          </div>

          {/* Live Readout Telemetry */}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-elegant-muted/10 text-center font-mono">
            <div className="p-2 bg-elegant-bg rounded border border-elegant-muted/10">
              <span className="text-[9px] text-elegant-muted block">SPIKES</span>
              <span className="text-xs font-bold text-rose-400">{liveStats.spikeCount}</span>
            </div>
            <div className="p-2 bg-elegant-bg rounded border border-elegant-muted/10">
              <span className="text-[9px] text-elegant-muted block">|U(t)|</span>
              <span className="text-xs font-bold text-elegant-cyan">
                {liveStats.instantMag.toFixed(2)}
              </span>
            </div>
            <div className="p-2 bg-elegant-bg rounded border border-elegant-muted/10">
              <span className="text-[9px] text-elegant-muted block">∠SPIKE</span>
              <span className="text-xs font-bold text-amber-400">
                {liveStats.lastSpikeAngleDeg.toFixed(0)}°
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Canvas Area (2 Columns) */}
      <div className="lg:col-span-2 space-y-6">
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl shadow-xl flex flex-col">
          <div className="flex justify-between items-center w-full mb-6">
            <div>
              <h3 className="text-base font-bold text-white font-display">
                {isEn
                  ? 'Interactive Complex LIF Simulation'
                  : 'Mô phỏng Neuron LIF Số Phức Tương Tác'}
              </h3>
              <p className="text-[11px] text-elegant-muted mt-0.5">
                {isEn
                  ? 'Pre-reset threshold crossings & phase-preserving vectorial reset trajectory'
                  : 'Hiển thị chính xác điểm vượt ngưỡng trước Reset & quỹ đạo giữ pha'}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-2 bg-elegant-bg rounded-lg hover:bg-elegant-cyan/20 text-elegant-cyan transition-colors border border-elegant-muted/20 cursor-pointer"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              <button
                onClick={handleReset}
                className="p-2 bg-elegant-bg rounded-lg hover:bg-elegant-cyan/20 text-elegant-cyan transition-colors border border-elegant-muted/20 cursor-pointer"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center justify-items-center w-full">
            {/* Trajectory */}
            <div className="flex flex-col items-center w-full">
              <span className="text-xs font-bold text-slate-300 font-mono mb-3">
                {t('neuron.trajectory')}
              </span>
              <div className="relative w-full max-w-[320px] aspect-square rounded-xl overflow-hidden border border-elegant-muted/20 bg-[#0B1120] shadow-2xl">
                <canvas
                  ref={bgCanvasRef}
                  width={400}
                  height={400}
                  className="absolute inset-0 w-full h-full"
                />
                <canvas
                  ref={fgCanvasRef}
                  width={400}
                  height={400}
                  className="absolute inset-0 w-full h-full z-10"
                />
              </div>
            </div>

            {/* Time Series */}
            <div className="flex flex-col items-center w-full">
              <span className="text-xs font-bold text-slate-300 font-mono mb-3">
                {isEn ? 'Membrane Potential Time Series' : 'Điện thế màng theo thời gian'}
              </span>
              <div className="relative w-full max-w-[320px] aspect-square rounded-xl overflow-hidden border border-elegant-muted/20 bg-[#0B1120] shadow-2xl">
                <canvas
                  ref={timeCanvasRef}
                  width={400}
                  height={400}
                  className="absolute inset-0 w-full h-full z-10"
                />
              </div>
            </div>
          </div>

          {/* Legend and Info Box */}
          <div className="mt-6 pt-4 border-t border-elegant-muted/10 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs font-mono">
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 bg-[#66FCF1] inline-block rounded"></span>
                <span className="text-elegant-text">
                  {isEn ? 'Real Potential (V)' : 'Thế Màng Thực (V)'}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 bg-[#FBBF24] inline-block rounded"></span>
                <span className="text-elegant-text">
                  {isEn ? 'Imaginary Potential (Y)' : 'Thế Màng Ảo (Y)'}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 bg-[#F87171] inline-block rounded"></span>
                <span className="text-elegant-text">
                  {isEn ? 'Spike / Threshold' : 'Sự kiện Xung / Ngưỡng'}
                </span>
              </div>
            </div>

            <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/10 flex items-start gap-2">
              <Info className="w-4 h-4 text-elegant-cyan mt-0.5 shrink-0" />
              <p className="text-[10px] text-elegant-text/70 leading-normal">
                {isEn ? (
                  <>
                    <strong>Subthreshold Resonance:</strong> When input AC frequency matches{' '}
                    <code>f₀ = ∠β / 2π</code>, input vectors rotate in lockstep with the membrane
                    leak phase, constructively building amplitude to cross threshold even with weak
                    currents.
                  </>
                ) : (
                  <>
                    <strong>Cộng hưởng dưới ngưỡng:</strong> Khi tần số dòng AC trùng với tần số
                    nội sinh <code>f₀ = ∠β / 2π</code>, dòng kích thích xoay đồng bộ với góc pha rò
                    rỉ của màng, giúp tích lũy cộng hưởng vượt ngưỡng ngay cả khi biên độ dòng vào
                    rất nhỏ.
                  </>
                )}
              </p>
            </div>
          </div>
        </div>

        {/* NEW: Subthreshold Frequency Resonance & selectivity Analyzer */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-elegant-muted/10 pb-2.5">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-elegant-cyan" />
              <h3 className="text-sm font-bold text-white font-display">
                {isEn
                  ? 'Frequency Resonance & Selectivity Spectrum (Bode / Spike-Rate Curve)'
                  : 'Phổ Cộng Hưởng Tần Số Chọn Lọc Của Neuron cLIF (Đường Cong Bode & Tần Suất Nổ Xung)'}
              </h3>
            </div>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
              {isEn
                ? `Peak Resonance f₀ = ${resonantFreq.toFixed(3)}`
                : `Đỉnh Cộng Hưởng f₀ = ${resonantFreq.toFixed(3)}`}
            </span>
          </div>

          <div className="h-44 bg-[#0B1120] rounded-xl border border-elegant-muted/15 p-3 relative">
            <svg viewBox="0 0 500 130" className="w-full h-full overflow-visible">
              {/* Grid lines */}
              <line
                x1="30"
                y1="20"
                x2="485"
                y2="20"
                stroke="rgba(69, 162, 158, 0.1)"
                strokeDasharray="3 3"
              />
              <line
                x1="30"
                y1="65"
                x2="485"
                y2="65"
                stroke="rgba(69, 162, 158, 0.1)"
                strokeDasharray="3 3"
              />
              <line
                x1="30"
                y1="110"
                x2="485"
                y2="110"
                stroke="rgba(69, 162, 158, 0.2)"
              />

              {/* Current selected inputFreq vertical marker */}
              {(() => {
                const xSel = 30 + ((inputFreq - 0.01) / 0.24) * 455;
                return (
                  <g>
                    <line
                      x1={xSel}
                      y1="10"
                      x2={xSel}
                      y2="110"
                      stroke="#F87171"
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                    />
                    <text
                      x={xSel}
                      y="10"
                      fill="#F87171"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      f={inputFreq.toFixed(3)}
                    </text>
                  </g>
                );
              })()}

              {/* Analytical Gain Curve |H(f)| (Cyan) */}
              {(() => {
                const maxGain = Math.max(...resonanceCurve.map(p => p.gain), 2.0);
                const pathD = resonanceCurve
                  .map((p, idx) => {
                    const x = 30 + (idx / (resonanceCurve.length - 1)) * 455;
                    const y = 110 - (p.gain / maxGain) * 85;
                    return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');
                return <path d={pathD} fill="none" stroke="#66FCF1" strokeWidth="2.5" />;
              })()}

              {/* Empirical Firing Rate Bars (Amber) */}
              {resonanceCurve.map((p, idx) => {
                const x = 30 + (idx / (resonanceCurve.length - 1)) * 455;
                const barH = p.spikeRate * 85;
                return (
                  <g key={`res-bar-${idx}`}>
                    <rect
                      x={x - 4}
                      y={110 - barH}
                      width="8"
                      height={Math.max(1, barH)}
                      fill="#FBBF24"
                      opacity="0.45"
                      rx="1"
                    />
                    {idx % 4 === 0 && (
                      <text
                        x={x}
                        y="124"
                        fill="#94a3b8"
                        fontSize="8"
                        fontFamily="monospace"
                        textAnchor="middle"
                      >
                        {p.freq.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="flex flex-wrap justify-between items-center text-[10px] font-mono text-elegant-muted">
            <div className="flex gap-4">
              <span className="text-elegant-cyan">
                ● {isEn ? 'Subthreshold Gain |H(f)|' : 'Độ Lợi Cộng Hưởng Màng |H(f)|'}
              </span>
              <span className="text-amber-400">
                ■ {isEn ? 'Empirical Spike Rate' : 'Tỷ Lệ Nổ Xung Thực Nghiệm'}
              </span>
              <span className="text-rose-400">
                | {isEn ? 'Active Input Frequency (f)' : 'Tần Số Đầu Vào Hiện Tại (f)'}
              </span>
            </div>
            <span>
              {isEn
                ? 'Click "Lock Resonance" on the left to align f with f₀!'
                : 'Nhấn "Khóa Cộng Hưởng" bên trái để khớp f = f₀!'}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
