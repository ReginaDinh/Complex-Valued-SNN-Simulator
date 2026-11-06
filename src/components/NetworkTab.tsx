import { useState, useEffect, useMemo } from 'react';
import { motion } from 'motion/react';
import {
  Activity,
  Play,
  Pause,
  RefreshCw,
  Settings,
  Info,
  Sliders,
  Zap,
  Sparkles,
  CheckCircle2,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import {
  complex,
  TSComplexSNN,
  generateSpectralSignal,
  SignalSample,
  ForwardResult,
  SurrogateType,
} from '../simulationEngine';

interface BenchmarkResult {
  cleanAccComplex: number;
  noisyAccComplex: number;
  phaseOnlyAccComplex: number; // fresh CV-SNN trained on the iso-frequency task
  phaseOnlyAccBlind: number; // identical model/training, phase removed from input (|x| only)
  phaseOnlyAccMain: number; // the user's trained network evaluated on the same task
  isoPhaseShift: number; // Δφ actually used by the iso-frequency test (rad, clamped ≥ ISO_MIN_PHASE_SHIFT)
  confusionMatrix: [[number, number], [number, number]]; // cm[target][pred], Class 1 = positive: [[TN, FP], [FN, TP]]
  macSavedPercent: number;
  totalMacs: number;
}

const ISO_TRAIN_STEPS = 60;
// Below ~0.8 rad the per-channel phase gradient is too small for a 60-step control run to separate
const ISO_MIN_PHASE_SHIFT = 0.8;

// Phase-blind control input: |x| in the real part, imaginary part 0 (all phase information removed)
const toPhaseBlind = (s: SignalSample): SignalSample => ({
  ...s,
  inputSpikeSequence: s.inputSpikeSequence.map(step => {
    const out = new Float32Array(step.length);
    for (let i = 0; i < step.length / 2; i++) {
      out[i * 2] = Math.hypot(step[i * 2], step[i * 2 + 1]);
    }
    return out;
  }),
});

const predictClass = (probs: number[]): number => (probs[1] > probs[0] ? 1 : 0);

export default function NetworkTab() {
  const { language } = useLanguage();
  const isEn = language === 'en';

  const [network, setNetwork] = useState<TSComplexSNN | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [trainingLossHistory, setTrainingLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [trainAccuracy, setTrainAccuracy] = useState<number>(0);
  const [currentSample, setCurrentSample] = useState<SignalSample | null>(null);
  const [sampleOutputs, setSampleOutputs] = useState<ForwardResult | null>(null);
  const [sampleProbs, setSampleProbs] = useState<number[]>([0.5, 0.5]);
  const [weightVersion, setWeightVersion] = useState(0);

  // Hyperparameters
  const [netLeakMag, setNetLeakMag] = useState(0.85);
  const [netLeakPhase, setNetLeakPhase] = useState(0.15);
  const [netThreshold, setNetThreshold] = useState(1.0);
  const [netResetMode, setNetResetMode] = useState<'soft' | 'hard'>('soft');
  const [learningRate, setLearningRate] = useState(0.035);
  const [useKaratsuba, setUseKaratsuba] = useState(true);
  const [surrogateType, setSurrogateType] = useState<SurrogateType>('von_mises');

  // Signal properties config states
  const [class0Freq, setClass0Freq] = useState(1.5);
  const [class1Freq, setClass1Freq] = useState(4.5);
  const [class1PhaseShift, setClass1PhaseShift] = useState(1.57); // ~90 deg
  const [signalNoise, setSignalNoise] = useState(0.15);

  // Benchmark verification state
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);

  const handleResetNetwork = () => {
    const net = new TSComplexSNN(
      16,
      2,
      netLeakMag,
      netLeakPhase,
      netThreshold,
      netResetMode,
      useKaratsuba,
      surrogateType
    );
    setNetwork(net);
    setEpoch(0);
    setTrainingLossHistory([]);
    setAccuracyHistory([]);
    setCurrentLoss(null);
    setTrainAccuracy(0.5);
    setIsTraining(false);
    setBenchmark(null);

    const sample = generateSpectralSignal(
      0,
      30,
      16,
      class0Freq,
      class1Freq,
      class1PhaseShift,
      signalNoise
    );
    setCurrentSample(sample);
    const evalRes = net.evaluateLoss(sample);
    setSampleOutputs(evalRes.forward);
    setSampleProbs(evalRes.predictions);
    setWeightVersion(v => v + 1);
  };

  useEffect(() => {
    handleResetNetwork();
  }, [
    netLeakMag,
    netLeakPhase,
    netThreshold,
    netResetMode,
    class0Freq,
    class1Freq,
    class1PhaseShift,
  ]);

  useEffect(() => {
    if (network) {
      network.useKaratsuba = useKaratsuba;
      network.surrogateType = surrogateType;
      if (currentSample) {
        const evalRes = network.evaluateLoss(currentSample);
        setSampleOutputs(evalRes.forward);
        setSampleProbs(evalRes.predictions);
      }
    }
  }, [useKaratsuba, surrogateType]);

  // Live training loop
  useEffect(() => {
    if (!isTraining || !network) return;

    const interval = setInterval(() => {
      // Train mini-batch of 2 balanced samples (Class 0 + Class 1) for stable gradient descent
      const s0 = generateSpectralSignal(
        0,
        30,
        16,
        class0Freq,
        class1Freq,
        class1PhaseShift,
        signalNoise
      );
      const s1 = generateSpectralSignal(
        1,
        30,
        16,
        class0Freq,
        class1Freq,
        class1PhaseShift,
        signalNoise
      );

      const loss0 = network.trainStep(s0, learningRate);
      const loss1 = network.trainStep(s1, learningRate);
      const avgBatchLoss = (loss0 + loss1) * 0.5;

      // Evaluate on balanced validation set of 12 samples
      let correct = 0;
      const valSize = 12;
      for (let v = 0; v < valSize; v++) {
        const valClass = v % 2;
        const valSample = generateSpectralSignal(
          valClass,
          30,
          16,
          class0Freq,
          class1Freq,
          class1PhaseShift,
          signalNoise
        );
        const evalRes = network.evaluateLoss(valSample);
        const predClass = evalRes.predictions[1] > evalRes.predictions[0] ? 1 : 0;
        if (predClass === valClass) correct++;
      }
      const acc = correct / valSize;

      const displaySample = Math.random() > 0.5 ? s1 : s0;
      const currentEval = network.evaluateLoss(displaySample);

      setEpoch(prev => prev + 1);
      setTrainingLossHistory(prev => {
        const next = [...prev, avgBatchLoss];
        if (next.length > 50) next.shift();
        return next;
      });
      setAccuracyHistory(prev => {
        const next = [...prev, acc];
        if (next.length > 50) next.shift();
        return next;
      });
      setCurrentLoss(avgBatchLoss);
      setTrainAccuracy(prevAcc => prevAcc * 0.7 + acc * 0.3);
      setCurrentSample(displaySample);
      setSampleOutputs(currentEval.forward);
      setSampleProbs(currentEval.predictions);
      setWeightVersion(v => v + 1);
    }, 140);

    return () => clearInterval(interval);
  }, [
    isTraining,
    network,
    learningRate,
    class0Freq,
    class1Freq,
    class1PhaseShift,
    signalNoise,
  ]);

  // Run verifiable benchmark suite (Clean vs High-Noise vs Pure Phase-Shift Iso-Frequency)
  const runVerificationBenchmark = () => {
    if (!network) return;
    const trials = 30;
    let cleanCorrect = 0;
    let noisyCorrect = 0;
    let phaseOnlyComplexCorrect = 0;
    let phaseOnlyBlindCorrect = 0;
    let phaseOnlyMainCorrect = 0;

    // 3. Iso-frequency phase test as a controlled experiment: two fresh models with identical
    // initial weights, hyper-parameters, samples and training steps; only the input differs
    // (complex x vs phase-blind |x|).
    const isoShift = Math.max(ISO_MIN_PHASE_SHIFT, class1PhaseShift);
    const makeIsoSample = (cls: number) =>
      generateSpectralSignal(cls, 30, 16, class0Freq, class0Freq, isoShift, 0.1);
    const isoComplexNet = new TSComplexSNN(16, 2, netLeakMag, netLeakPhase, netThreshold, netResetMode, useKaratsuba, surrogateType);
    const isoBlindNet = new TSComplexSNN(16, 2, netLeakMag, netLeakPhase, netThreshold, netResetMode, useKaratsuba, surrogateType);
    isoBlindNet.weights = isoComplexNet.weights.map(w => w.slice());
    for (let k = 0; k < ISO_TRAIN_STEPS; k++) {
      const s = makeIsoSample(k % 2);
      isoComplexNet.trainStep(s, learningRate, false);
      isoBlindNet.trainStep(toPhaseBlind(s), learningRate, false);
    }

    const cm: [[number, number], [number, number]] = [
      [0, 0],
      [0, 0],
    ];

    for (let k = 0; k < trials; k++) {
      const target = k % 2;
      // 1. Clean evaluation
      const sClean = generateSpectralSignal(
        target,
        30,
        16,
        class0Freq,
        class1Freq,
        class1PhaseShift,
        0.08
      );
      const resClean = network.evaluateLoss(sClean, false);
      const predClean = resClean.predictions[1] > resClean.predictions[0] ? 1 : 0;
      if (predClean === target) cleanCorrect++;
      cm[target][predClean]++;

      // 2. High-noise stress test (SNR degradation: noise = 0.55)
      const sNoisy = generateSpectralSignal(
        target,
        30,
        16,
        class0Freq,
        class1Freq,
        class1PhaseShift,
        0.55
      );
      const resNoisy = network.evaluateLoss(sNoisy, false);
      const predNoisy = resNoisy.predictions[1] > resNoisy.predictions[0] ? 1 : 0;
      if (predNoisy === target) noisyCorrect++;

      // Held-out iso-frequency sample, scored by real model predictions only
      const sPhase = makeIsoSample(target);
      if (predictClass(isoComplexNet.evaluateLoss(sPhase, false).predictions) === target) phaseOnlyComplexCorrect++;
      if (predictClass(isoBlindNet.evaluateLoss(toPhaseBlind(sPhase), false).predictions) === target) phaseOnlyBlindCorrect++;
      if (predictClass(network.evaluateLoss(sPhase, false).predictions) === target) phaseOnlyMainCorrect++;
    }

    const totalMacs = 30 * 16 * 2 * (useKaratsuba ? 3 : 4);
    setBenchmark({
      cleanAccComplex: (cleanCorrect / trials) * 100,
      noisyAccComplex: (noisyCorrect / trials) * 100,
      phaseOnlyAccComplex: (phaseOnlyComplexCorrect / trials) * 100,
      phaseOnlyAccBlind: (phaseOnlyBlindCorrect / trials) * 100,
      phaseOnlyAccMain: (phaseOnlyMainCorrect / trials) * 100,
      isoPhaseShift: isoShift,
      confusionMatrix: cm,
      macSavedPercent: useKaratsuba ? 25 : 0,
      totalMacs,
    });
  };

  const handleInjectTestSample = (targetClass: 0 | 1) => {
    if (!network) return;
    const sample = generateSpectralSignal(
      targetClass,
      30,
      16,
      class0Freq,
      class1Freq,
      class1PhaseShift,
      signalNoise
    );
    const evalRes = network.evaluateLoss(sample);
    setCurrentSample(sample);
    setSampleOutputs(evalRes.forward);
    setSampleProbs(evalRes.predictions);
  };

  const getPhaseColor = (radPhase: number) => {
    const deg = ((radPhase + Math.PI) / (2 * Math.PI)) * 360;
    return `hsl(${deg.toFixed(1)}, 90%, 60%)`;
  };

  const maxLossForChart = useMemo(() => Math.max(1.2, ...trainingLossHistory), [trainingLossHistory]);

  return (
    <motion.div
      key="network"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="grid grid-cols-1 lg:grid-cols-3 gap-6"
    >
      {/* COLUMN 1: SNN TRAINING HYPERPARAMETERS & CONTROLS */}
      <div className="space-y-6">
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
            <Settings className="w-4 h-4 text-elegant-cyan" />
            <span>{isEn ? 'SNN Network & Surrogate Config' : 'Cấu Hình Mạng & Đạo Hàm Thay Thế'}</span>
          </h3>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Learning Rate (η):' : 'Tốc Độ Học (η):'}</span>
              <span className="text-elegant-cyan font-bold">{learningRate.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min="0.005"
              max="0.08"
              step="0.005"
              value={learningRate}
              onChange={e => setLearningRate(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Leak Magnitude |β|:' : 'Biên Độ Rò Rỉ |β|:'}</span>
              <span className="text-elegant-cyan font-bold">{netLeakMag.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.6"
              max="0.95"
              step="0.05"
              value={netLeakMag}
              onChange={e => setNetLeakMag(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Intrinsic Leak Phase ∠β (rad):' : 'Pha Dao Động Nội Sinh ∠β:'}</span>
              <span className="text-elegant-cyan font-bold">{netLeakPhase.toFixed(2)} rad</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="0.4"
              step="0.02"
              value={netLeakPhase}
              onChange={e => setNetLeakPhase(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
            />
          </div>

          {/* Surrogate Gradient Type */}
          <div className="space-y-1.5 pt-1 border-t border-elegant-muted/10">
            <label className="text-xs font-mono text-elegant-muted block">
              {isEn ? 'BPTT Surrogate Gradient:' : 'Hàm Surrogate Gradient (BPTT):'}
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {(
                [
                  { id: 'von_mises', label: 'Von Mises' },
                  { id: 'fast_sigmoid', label: 'Fast Sigmoid' },
                  { id: 'atan', label: 'ArcTan' },
                ] as const
              ).map(item => (
                <button
                  key={item.id}
                  onClick={() => setSurrogateType(item.id)}
                  className={`py-1.5 px-2 rounded-md text-[10px] font-mono font-bold transition-all border ${
                    surrogateType === item.id
                      ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/40'
                      : 'bg-elegant-bg text-elegant-muted border-elegant-muted/10 hover:text-white'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* Karatsuba Multiplication Toggle */}
          <div className="space-y-1.5 pt-1 border-t border-elegant-muted/10">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-elegant-muted">
                {isEn ? 'Synaptic Multiplier Mode:' : 'Chế Độ Nhân Phức Synapse:'}
              </span>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">
                {useKaratsuba
                  ? isEn
                    ? '3 Real Mults (-25% MACs)'
                    : '3 Phép Nhân Thực (-25%)'
                  : isEn
                  ? '4 Real Mults (Standard)'
                  : '4 Phép Nhân Thực'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setUseKaratsuba(true)}
                className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                  useKaratsuba
                    ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/35'
                    : 'bg-elegant-bg text-elegant-muted border-elegant-muted/10'
                }`}
              >
                Karatsuba (3-MAC)
              </button>
              <button
                onClick={() => setUseKaratsuba(false)}
                className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                  !useKaratsuba
                    ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35'
                    : 'bg-elegant-bg text-elegant-muted border-elegant-muted/10'
                }`}
              >
                Standard (4-MAC)
              </button>
            </div>
          </div>

          {/* Reset Mechanism */}
          <div className="space-y-1.5 pt-1 border-t border-elegant-muted/10">
            <label className="text-xs font-mono text-elegant-muted block">
              {isEn ? 'Membrane Reset Mode:' : 'Cơ Chế Reset Điện Thế:'}
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setNetResetMode('soft')}
                className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                  netResetMode === 'soft'
                    ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35'
                    : 'bg-elegant-bg text-elegant-text border-elegant-muted/10'
                }`}
              >
                {isEn ? 'Soft (Phase-Preserving)' : 'Reset Mềm (Giữ Pha)'}
              </button>
              <button
                onClick={() => setNetResetMode('hard')}
                className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                  netResetMode === 'hard'
                    ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35'
                    : 'bg-elegant-bg text-elegant-text border-elegant-muted/10'
                }`}
              >
                {isEn ? 'Hard (Zero)' : 'Reset Cứng (Về 0)'}
              </button>
            </div>
          </div>
        </div>

        {/* Input Signal Configuration */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
            <Sliders className="w-4 h-4 text-elegant-cyan" />
            <span>{isEn ? 'Spectral Signal Generator (Nyquist-Safe)' : 'Bộ Phát Phổ Tín Hiệu (Chuẩn Nyquist)'}</span>
          </h3>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Class 0 Carrier Freq:' : 'Tần Số Sóng Mang Lớp 0:'}</span>
              <span className="text-elegant-cyan font-bold">{class0Freq.toFixed(1)} {isEn ? 'cyc/window' : 'chu kỳ/cửa sổ'} (Δφ = 0°)</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.5"
              value={class0Freq}
              onChange={e => setClass0Freq(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Class 1 Carrier Freq:' : 'Tần Số Sóng Mang Lớp 1:'}</span>
              <span className="text-amber-400 font-bold">{class1Freq.toFixed(1)} {isEn ? 'cyc/window' : 'chu kỳ/cửa sổ'}</span>
            </div>
            <input
              type="range"
              min="3.0"
              max="7.0"
              step="0.5"
              value={class1Freq}
              onChange={e => setClass1Freq(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Class 1 Spatial Phase Shift:' : 'Độ Lệch Pha Không Gian Lớp 1:'}</span>
              <span className="text-amber-400 font-bold">
                {((class1PhaseShift * 180) / Math.PI).toFixed(0)}° ({class1PhaseShift.toFixed(2)} rad)
              </span>
            </div>
            <input
              type="range"
              min="0.2"
              max="3.14"
              step="0.1"
              value={class1PhaseShift}
              onChange={e => setClass1PhaseShift(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-elegant-muted">
              <span>{isEn ? 'Channel Noise (σ):' : 'Nhiễu Kênh Truyền (σ):'}</span>
              <span className="text-rose-400 font-bold">{(signalNoise * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="0.6"
              step="0.05"
              value={signalNoise}
              onChange={e => setSignalNoise(parseFloat(e.target.value))}
              className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-rose-400"
            />
          </div>
        </div>

        {/* Training Actions Card */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2.5 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-0.5">
              <span className="text-[9px] text-elegant-muted font-mono block">EPOCH</span>
              <span className="text-base font-bold text-white font-mono">{epoch}</span>
            </div>
            <div className="p-2.5 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-0.5">
              <span className="text-[9px] text-elegant-muted font-mono block">
                {isEn ? 'VAL ACC' : 'ĐỘ CHÍNH XÁC'}
              </span>
              <span className="text-base font-bold text-emerald-400 font-mono">
                {(trainAccuracy * 100).toFixed(0)}%
              </span>
            </div>
            <div className="p-2.5 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-0.5">
              <span className="text-[9px] text-elegant-muted font-mono block">CE LOSS</span>
              <span className="text-base font-bold text-elegant-cyan font-mono">
                {currentLoss !== null ? currentLoss.toFixed(3) : '0.693'}
              </span>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setIsTraining(!isTraining)}
              className={`flex-1 py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-md ${
                isTraining
                  ? 'bg-rose-600 hover:bg-rose-500 text-white'
                  : 'bg-elegant-cyan hover:bg-[#52ebd3] text-elegant-bg'
              }`}
            >
              {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>
                {isTraining
                  ? isEn
                    ? 'Pause BPTT Training'
                    : 'Tạm Dừng Huấn Luyện'
                  : isEn
                  ? 'Start Live BPTT'
                  : 'Bắt Đầu Huấn Luyện'}
              </span>
            </button>

            <button
              onClick={handleResetNetwork}
              className="p-2.5 bg-elegant-bg border border-elegant-muted/20 hover:border-elegant-cyan/40 rounded-xl transition-all"
              title={isEn ? 'Reset network weights' : 'Khởi tạo lại trọng số'}
            >
              <RefreshCw className="w-4 h-4 text-elegant-muted" />
            </button>
          </div>
        </div>
      </div>

      {/* COLUMN 2: LOSS CURVE & CARRIER-DEMODULATED 16-CHANNEL PHASE MAP */}
      <div className="space-y-6">
        {/* Training Loss & Accuracy Curve Card */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
            <h3 className="text-sm font-bold text-white font-display">
              {isEn ? 'Convergence: Loss & Accuracy' : 'Hội Tụ Huấn Luyện: Độ Lỗi & Chính Xác'}
            </h3>
            <div className="flex items-center gap-3 text-[10px] font-mono">
              <span className="text-elegant-cyan">● Loss</span>
              <span className="text-emerald-400">● Accuracy</span>
            </div>
          </div>

          <div className="h-44 bg-elegant-bg rounded-lg border border-elegant-muted/15 relative p-2 overflow-hidden">
            {trainingLossHistory.length > 1 ? (
              <svg viewBox="0 0 320 150" className="w-full h-full overflow-visible">
                <line
                  x1="0"
                  y1="35"
                  x2="320"
                  y2="35"
                  stroke="rgba(69, 162, 158, 0.1)"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
                <line
                  x1="0"
                  y1="75"
                  x2="320"
                  y2="75"
                  stroke="rgba(69, 162, 158, 0.1)"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
                <line
                  x1="0"
                  y1="115"
                  x2="320"
                  y2="115"
                  stroke="rgba(69, 162, 158, 0.1)"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />

                {/* Accuracy path (Emerald) */}
                <path
                  d={accuracyHistory
                    .map((acc, idx) => {
                      const x = (idx / Math.max(1, accuracyHistory.length - 1)) * 310 + 5;
                      const y = 140 - acc * 125;
                      return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                />

                {/* Loss path (Cyan) */}
                <path
                  d={trainingLossHistory
                    .map((loss, idx) => {
                      const x = (idx / Math.max(1, trainingLossHistory.length - 1)) * 310 + 5;
                      const y = 140 - Math.min(1, loss / maxLossForChart) * 125;
                      return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#66FCF1"
                  strokeWidth="2.5"
                />
              </svg>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-elegant-muted font-mono">
                {isEn
                  ? 'Press "Start Live BPTT" to observe convergence'
                  : 'Nhấn "Bắt Đầu Huấn Luyện" để quan sát hội tụ'}
              </div>
            )}
          </div>
          <div className="flex justify-between text-[10px] text-elegant-muted font-mono">
            <span>
              {isEn ? 'MACs / Sample:' : 'Số phép nhân thực / mẫu:'}{' '}
              <strong className="text-white">{sampleOutputs?.realMultiplications ?? 2880}</strong>
            </span>
            <span>{isEn ? `Step #${epoch}` : `Bước #${epoch}`}</span>
          </div>
        </div>

        {/* Carrier-Demodulated 16-Channel Phase Map */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3 shadow-xl">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
            <div>
              <h3 className="text-sm font-bold text-white font-display flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-elegant-cyan" />
                <span>
                  {isEn
                    ? '16-Channel Baseband Phasor Map (Demodulated)'
                    : 'Bản Đồ Pha 16 Kênh (Giải Điều Chế Sóng Mang)'}
                </span>
              </h3>
              <p className="text-[10px] text-elegant-muted font-mono mt-0.5">
                {currentSample
                  ? isEn
                    ? `Active Sample: Class ${currentSample.target} (fc = ${currentSample.carrierFreq.toFixed(1)} cyc/window, Δφ = ${((currentSample.spatialPhaseShift * 180) / Math.PI).toFixed(0)}°)`
                    : `Mẫu đang thử: Lớp ${currentSample.target} (fc = ${currentSample.carrierFreq.toFixed(1)} chu kỳ/cửa sổ, Δφ = ${((currentSample.spatialPhaseShift * 180) / Math.PI).toFixed(0)}°)`
                  : ''}
              </p>
            </div>
            <div className="flex gap-1">
              <button
                onClick={() => handleInjectTestSample(0)}
                className={`px-2 py-1 rounded text-[10px] font-mono font-bold border transition ${
                  currentSample?.target === 0
                    ? 'bg-elegant-cyan/20 text-elegant-cyan border-elegant-cyan'
                    : 'bg-elegant-bg text-elegant-muted border-elegant-muted/15 hover:text-white'
                }`}
              >
                {isEn ? 'Test C0' : 'Thử Lớp 0'}
              </button>
              <button
                onClick={() => handleInjectTestSample(1)}
                className={`px-2 py-1 rounded text-[10px] font-mono font-bold border transition ${
                  currentSample?.target === 1
                    ? 'bg-amber-400/20 text-amber-400 border-amber-400'
                    : 'bg-elegant-bg text-elegant-muted border-elegant-muted/15 hover:text-white'
                }`}
              >
                {isEn ? 'Test C1' : 'Thử Lớp 1'}
              </button>
            </div>
          </div>

          {currentSample ? (
            <div className="space-y-2.5">
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
                {Array.from({ length: 16 }).map((_, fIdx) => {
                  // Demodulate carrier e^{-j 2*pi*fc*(t/T)} so we recover the true channel spatial phase offset!
                  let demodRe = 0;
                  let demodIm = 0;
                  const steps = currentSample.inputSpikeSequence.length;
                  const fc = currentSample.carrierFreq;

                  for (let t = 0; t < steps; t++) {
                    const tNorm = t / steps;
                    const carrierAngle = -2 * Math.PI * fc * tNorm;
                    const cCos = Math.cos(carrierAngle);
                    const cSin = Math.sin(carrierAngle);

                    const valRe = currentSample.inputSpikeSequence[t][fIdx * 2];
                    const valIm = currentSample.inputSpikeSequence[t][fIdx * 2 + 1];

                    demodRe += valRe * cCos - valIm * cSin;
                    demodIm += valRe * cSin + valIm * cCos;
                  }

                  const avgRe = demodRe / steps;
                  const avgIm = demodIm / steps;
                  const coherentMag = complex.mag(avgRe, avgIm);
                  const phaseRad = complex.angle(avgRe, avgIm);
                  const rotDegree = (phaseRad * 180) / Math.PI;
                  const color = getPhaseColor(phaseRad);
                  const isClutter = fIdx >= 12;

                  return (
                    <div
                      key={fIdx}
                      className={`p-2 rounded-lg border flex flex-col items-center justify-center gap-1 ${
                        isClutter
                          ? 'bg-elegant-bg/40 border-elegant-muted/5 opacity-60'
                          : 'bg-elegant-bg border-elegant-muted/15'
                      }`}
                    >
                      <div
                        className="w-7 h-7 rounded-full border border-elegant-muted/20 relative flex items-center justify-center"
                        style={{ backgroundColor: `${color}18` }}
                      >
                        <div
                          className="h-0.5 rounded-full origin-center"
                          style={{
                            backgroundColor: color,
                            transform: `rotate(${-rotDegree}deg)`,
                            width: `${Math.max(25, Math.min(100, coherentMag * 110))}%`,
                          }}
                        />
                      </div>
                      <span className="text-[9px] font-mono text-white font-bold">
                        {rotDegree.toFixed(0)}°
                      </span>
                      <span className="text-[8px] font-mono text-elegant-muted">
                        {isClutter ? 'Noise' : `Ch ${fIdx}`}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-elegant-muted font-mono leading-relaxed">
                {isEn
                  ? 'Channels 0–11 carry coherent phase-locked signal (demodulated by e^{-j2πf_c t/T}); Channels 12–15 are random-phase clutter.'
                  : 'Kênh 0–11 mang tín hiệu khóa pha (đã giải điều chế sóng mang e^{-j2πf_c t/T}); Kênh 12–15 là nhiễu nền ngẫu nhiên.'}
              </p>
            </div>
          ) : null}
        </div>

        {/* Learned Synaptic Weight Constellation */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3 shadow-xl">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
            <h3 className="text-sm font-bold text-white font-display flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-elegant-cyan" />
              <span>
                {isEn
                  ? 'Learned Complex Weight Constellation (W ∈ ℂ)'
                  : 'Chòm Sao Trọng Số Phức Đã Học (W ∈ ℂ)'}
              </span>
            </h3>
            <span className="text-[10px] font-mono text-elegant-muted">
              v{weightVersion}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
            <div className="flex justify-center">
              <svg
                width="160"
                height="160"
                viewBox="0 0 160 160"
                className="bg-elegant-bg rounded-xl border border-elegant-muted/15"
              >
                <circle
                  cx="80"
                  cy="80"
                  r="60"
                  fill="none"
                  stroke="rgba(69, 162, 158, 0.15)"
                  strokeDasharray="3 3"
                />
                <circle
                  cx="80"
                  cy="80"
                  r="30"
                  fill="none"
                  stroke="rgba(69, 162, 158, 0.1)"
                />
                <line
                  x1="10"
                  y1="80"
                  x2="150"
                  y2="80"
                  stroke="rgba(69, 162, 158, 0.2)"
                />
                <line
                  x1="80"
                  y1="10"
                  x2="80"
                  y2="150"
                  stroke="rgba(69, 162, 158, 0.2)"
                />

                {network &&
                  [0, 1].map(oIdx =>
                    Array.from({ length: 16 }).map((_, iIdx) => {
                      const wr = network.weights[oIdx][iIdx * 2];
                      const wi = network.weights[oIdx][iIdx * 2 + 1];
                      const x = 80 + Math.max(-68, Math.min(68, wr * 95));
                      const y = 80 - Math.max(-68, Math.min(68, wi * 95));
                      const color = oIdx === 0 ? '#66FCF1' : '#FBBF24';
                      return (
                        <g key={`w-${oIdx}-${iIdx}`}>
                          <line
                            x1="80"
                            y1="80"
                            x2={x}
                            y2={y}
                            stroke={color}
                            strokeWidth="1"
                            opacity="0.35"
                          />
                          <circle cx={x} cy={y} r="3" fill={color} opacity="0.9" />
                        </g>
                      );
                    })
                  )}
              </svg>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/10">
                <div className="text-elegant-cyan font-bold">
                  ● {isEn ? 'Neuron 0 (Class 0 Detector)' : 'Neuron 0 (Bộ Lọc Lớp 0)'}
                </div>
                <p className="text-[10px] text-elegant-muted mt-0.5 font-sans">
                  {isEn
                    ? 'Aligns synaptic phases to constructively sum coherent low-frequency inputs.'
                    : 'Tự động đồng bộ góc pha để cộng hưởng với tín hiệu đồng pha Lớp 0.'}
                </p>
              </div>
              <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/10">
                <div className="text-amber-400 font-bold">
                  ● {isEn ? 'Neuron 1 (Class 1 Detector)' : 'Neuron 1 (Bộ Lọc Lớp 1)'}
                </div>
                <p className="text-[10px] text-elegant-muted mt-0.5 font-sans">
                  {isEn
                    ? 'Conjugates the spatial phase gradient (-i·Δφ) to focus high-frequency waves.'
                    : 'Xoay pha liên hợp (-i·Δφ) để giải mã độ lệch pha không gian của Lớp 1.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* COLUMN 3: LIVE SPIKE RASTER, OUTPUT DIAGNOSTICS & VERIFICATION BENCHMARK */}
      <div className="space-y-6">
        {/* Output Neuron Spike Raster & Decision Readout */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
            <h3 className="text-sm font-bold text-white font-display flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-elegant-cyan" />
              <span>
                {isEn
                  ? 'Output cLIF Neurons & Spike Raster'
                  : 'Chuỗi Xung Đầu Ra & Quyết Định Phân Loại'}
              </span>
            </h3>
            {currentSample && (
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                  (sampleProbs[1] > sampleProbs[0] ? 1 : 0) === currentSample.target
                    ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                    : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                }`}
              >
                {(sampleProbs[1] > sampleProbs[0] ? 1 : 0) === currentSample.target
                  ? isEn
                    ? '✓ CORRECT'
                    : '✓ CHÍNH XÁC'
                  : isEn
                  ? '✗ MISCLASSIFIED'
                  : '✗ CHƯA KHỚP'}
              </span>
            )}
          </div>

          {sampleOutputs && (
            <div className="space-y-3">
              {[0, 1].map(nIdx => {
                const prob = (sampleProbs[nIdx] ?? 0.5) * 100;
                const spikes = sampleOutputs.spikeCounts[nIdx] ?? 0;
                const plv = sampleOutputs.phaseCoherence[nIdx] ?? 0;
                const isTarget = currentSample?.target === nIdx;

                return (
                  <div
                    key={`out-neuron-${nIdx}`}
                    className={`p-3 rounded-xl border space-y-2 ${
                      isTarget
                        ? 'bg-elegant-cyan/5 border-elegant-cyan/35'
                        : 'bg-elegant-bg border-elegant-muted/10'
                    }`}
                  >
                    <div className="flex justify-between items-center text-xs font-mono">
                      <span
                        className={`font-bold ${
                          nIdx === 0 ? 'text-elegant-cyan' : 'text-amber-400'
                        }`}
                      >
                        {isEn ? `Output Neuron #${nIdx} (Class ${nIdx})` : `Neuron Đầu Ra #${nIdx} (Lớp ${nIdx})`}
                      </span>
                      <span className="text-white font-bold">
                        {prob.toFixed(1)}% | {spikes} spikes | PLV {plv.toFixed(2)}
                      </span>
                    </div>

                    {/* Probability bar */}
                    <div className="h-2 w-full bg-[#070b10] rounded-full overflow-hidden border border-elegant-muted/10">
                      <div
                        className={`h-full transition-all duration-200 ${
                          nIdx === 0 ? 'bg-elegant-cyan' : 'bg-amber-400'
                        }`}
                        style={{ width: `${prob}%` }}
                      />
                    </div>

                    {/* 30-step Spike Raster */}
                    <div className="flex items-center gap-0.5 pt-1">
                      {sampleOutputs.spikeTrains.map((stepSpk, tIdx) => (
                        <div
                          key={`spk-${nIdx}-${tIdx}`}
                          className={`flex-1 h-3 rounded-xs ${
                            stepSpk[nIdx]
                              ? nIdx === 0
                                ? 'bg-elegant-cyan shadow-xs shadow-elegant-cyan'
                                : 'bg-amber-400 shadow-xs shadow-amber-400'
                              : 'bg-white/5'
                          }`}
                          title={`t=${tIdx}: ${stepSpk[nIdx] ? 'SPIKE' : 'subthreshold'}`}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Interactive Algorithm Verification & Stress-Test Suite */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
            <h3 className="text-sm font-bold text-white font-display flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>
                {isEn
                  ? 'Algorithmic Verification & Stress-Test'
                  : 'Kiểm Định Thuật Toán & Stress-Test'}
              </span>
            </h3>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
              30 Trials
            </span>
          </div>

          <p className="text-xs text-elegant-text/75 leading-relaxed">
            {isEn
              ? 'Verify the trained CV-SNN on unseen test signals, extreme channel noise (σ=55%), and iso-frequency pure phase discrimination against a 1D Real-valued SNN.'
              : 'Kiểm tra thực nghiệm mạng CV-SNN trên tập mẫu mới, nhiễu kênh cực đại (σ=55%), và bài toán phân biệt thuần góc pha (cùng tần số) so với mạng SNN thực 1D.'}
          </p>

          <button
            onClick={runVerificationBenchmark}
            className="w-full py-2.5 px-4 bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/35 rounded-xl text-xs font-mono font-bold flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            <span>
              {isEn
                ? 'Run Automated Verification Suite (30 Test Signals)'
                : 'Chạy Kiểm Định Tự Động (30 Mẫu Kiểm Thử)'}
            </span>
          </button>

          {benchmark && (
            <div className="space-y-3 pt-1 font-mono text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/15">
                  <span className="text-[10px] text-elegant-muted block">
                    {isEn ? 'Clean Test Accuracy' : 'Độ chính xác mẫu sạch'}
                  </span>
                  <span className="text-base font-bold text-emerald-400">
                    {benchmark.cleanAccComplex.toFixed(1)}%
                  </span>
                </div>
                <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/15">
                  <span className="text-[10px] text-elegant-muted block">
                    {isEn ? 'Extreme Noise (σ=55%)' : 'Chống nhiễu cao (σ=55%)'}
                  </span>
                  <span className="text-base font-bold text-elegant-cyan">
                    {benchmark.noisyAccComplex.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Pure Phase Discrimination Comparison */}
              <div className="p-3 bg-elegant-bg rounded-lg border border-elegant-muted/15 space-y-2">
                <div className="text-[10px] text-white font-bold uppercase">
                  {isEn
                    ? 'Iso-Frequency Pure Phase Test (Same Freq, Δφ Only):'
                    : 'Kiểm Định Thuần Pha (Cùng Tần Số, Chỉ Lệch Pha Δφ):'}
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-elegant-cyan">{isEn ? 'CV-SNN (complex input):' : 'CV-SNN (đầu vào phức):'}</span>
                  <span className="font-bold text-emerald-400">{benchmark.phaseOnlyAccComplex.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-rose-400">{isEn ? 'Phase-blind SNN (|x| only):' : 'SNN mù pha (chỉ |x|):'}</span>
                  <span className="font-bold text-rose-400">{benchmark.phaseOnlyAccBlind.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-elegant-muted">{isEn ? 'Your trained network on this test:' : 'Mạng bạn đã train trên bài này:'}</span>
                  <span className="font-bold text-white">{benchmark.phaseOnlyAccMain.toFixed(1)}%</span>
                </div>
                <div className="text-[9px] text-elegant-muted">
                  {isEn
                    ? `Δφ used = ${benchmark.isoPhaseShift.toFixed(2)} rad${benchmark.isoPhaseShift > class1PhaseShift ? ` (raised from ${class1PhaseShift.toFixed(2)} to the ${ISO_MIN_PHASE_SHIFT} minimum)` : ''}. Control models: same initial weights, hyper-parameters and ${ISO_TRAIN_STEPS} training samples; only input phase differs. Chance = 50%.`
                    : `Δφ đã dùng = ${benchmark.isoPhaseShift.toFixed(2)} rad${benchmark.isoPhaseShift > class1PhaseShift ? ` (nâng từ ${class1PhaseShift.toFixed(2)} lên mức tối thiểu ${ISO_MIN_PHASE_SHIFT})` : ''}. Mô hình đối chứng: cùng trọng số khởi tạo, siêu tham số và ${ISO_TRAIN_STEPS} mẫu huấn luyện; chỉ khác có/không có pha. Đoán ngẫu nhiên = 50%.`}
                </div>
              </div>

              {/* Confusion Matrix */}
              <div className="p-3 bg-elegant-bg rounded-lg border border-elegant-muted/15 space-y-1.5">
                <div className="flex justify-between items-center text-[10px] text-elegant-muted">
                  <span>{isEn ? 'CONFUSION MATRIX' : 'MA TRẬN NHẦM LẪN'}</span>
                  <span>
                    {isEn ? 'Hardware MAC Savings:' : 'Tiết kiệm nhân phần cứng:'}{' '}
                    <strong className="text-emerald-400">-{benchmark.macSavedPercent}%</strong>
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-center text-[11px]">
                  <div className="p-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded text-emerald-300">
                    TN (C0→C0): <strong>{benchmark.confusionMatrix[0][0]}</strong>
                  </div>
                  <div className="p-1.5 bg-rose-500/10 border border-rose-500/20 rounded text-rose-300">
                    FP (C0→C1): <strong>{benchmark.confusionMatrix[0][1]}</strong>
                  </div>
                  <div className="p-1.5 bg-rose-500/10 border border-rose-500/20 rounded text-rose-300">
                    FN (C1→C0): <strong>{benchmark.confusionMatrix[1][0]}</strong>
                  </div>
                  <div className="p-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded text-emerald-300">
                    TP (C1→C1): <strong>{benchmark.confusionMatrix[1][1]}</strong>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Mathematical Verification Note */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-4 rounded-xl space-y-2 text-xs text-elegant-text/75 leading-relaxed shadow-xl">
          <div className="flex items-center gap-1.5 text-white font-bold font-display">
            <CheckCircle2 className="w-4 h-4 text-elegant-cyan" />
            <span>
              {isEn
                ? 'Verified Algorithmic Improvements:'
                : 'Các Cải Tiến Thuật Toán Đã Chuẩn Hóa:'}
            </span>
          </div>
          <ul className="list-disc pl-4 space-y-1 text-[11px]">
            <li>
              <strong>{isEn ? 'Nyquist-Normalized Carrier:' : 'Chuẩn hóa Nyquist:'}</strong>{' '}
              {isEn
                ? 'Uses normalized time t/T so integer frequencies never alias to DC.'
                : 'Chuẩn hóa trục thời gian t/T giúp sóng mang không bị gập tần (aliasing) về DC.'}
            </li>
            <li>
              <strong>{isEn ? 'Pre-Reset Readout & Trace:' : 'Đọc thế màng trước Reset:'}</strong>{' '}
              {isEn
                ? 'Evaluates surrogate derivatives on U_pre(t) before reset subtraction.'
                : 'Tính đạo hàm Wirtinger trên điện thế U_pre(t) trước khi trừ ngưỡng reset.'}
            </li>
          </ul>
        </div>
      </div>
    </motion.div>
  );
}