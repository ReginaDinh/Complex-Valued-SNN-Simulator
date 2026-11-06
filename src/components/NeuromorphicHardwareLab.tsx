import { useState } from 'react';
import { motion } from 'motion/react';
import {
  Cpu,
  Layers,
  Sparkles,
  Zap,
  Terminal,
  Settings,
  HardDrive,
  ShieldCheck,
  Check,
  Copy,
  Sliders,
  Activity,
  ArrowRight,
  AlertCircle,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { InlineMath, BlockMath } from 'react-katex';
import { useCopyStatus } from '../utils/useCopyStatus';

export default function NeuromorphicHardwareLab() {
  const { language } = useLanguage();
  const isEn = language === 'en';

  const [activeSubTab, setActiveSubTab] = useState<'choi' | 'schematic' | 'driver'>('choi');

  // --- STATE FOR LIVE KIRCHHOFF CROSSBAR VERIFIER (CHOI LAB) ---
  const [choiWeightMag, setChoiWeightMag] = useState<number>(0.8);
  const [choiWeightPhase, setChoiWeightPhase] = useState<number>(0.785); // 45 deg
  const [inputVoltMag, setInputVoltMag] = useState<number>(1.0);
  const [inputVoltPhase, setInputVoltPhase] = useState<number>(0.52); // ~30 deg
  const [pcmNoise, setPcmNoise] = useState<number>(0.15);
  const [writeVerifyEnabled, setWriteVerifyEnabled] = useState<boolean>(true);

  // --- STATE FOR DRAGGABLE SCHEMATIC ---
  const [activeStep, setActiveStep] = useState<number>(0);
  const [pos, setPos] = useState({
    input: { x: 20, y: 200 },
    dac: { x: 180, y: 120 },
    cbRe: { x: 280, y: 70 },
    cbIm: { x: 280, y: 260 },
    tia: { x: 560, y: 200 },
    clif: { x: 680, y: 200 },
  });

  // --- STATE FOR CODE COPY ---
  const { status: copyStatus, copy } = useCopyStatus();

  const onDrag = (key: keyof typeof pos) => (_e: any, info: any) => {
    setPos(p => ({
      ...p,
      [key]: {
        x: p[key].x + info.delta.x,
        y: p[key].y + info.delta.y,
      },
    }));
  };

  const drawPath = (x1: number, y1: number, x2: number, y2: number) => {
    const cp = Math.max(Math.abs(x2 - x1) / 2, 20);
    return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
  };

  // Complex multiplication computations for Kirchhoff Sandbox
  const wRe = choiWeightMag * Math.cos(choiWeightPhase);
  const wIm = choiWeightMag * Math.sin(choiWeightPhase);
  const vRe = inputVoltMag * Math.cos(inputVoltPhase);
  const vIm = inputVoltMag * Math.sin(inputVoltPhase);

  // Ideal complex multiplication: (wRe + j wIm) * (vRe + j vIm)
  const idealRe = wRe * vRe - wIm * vIm;
  const idealIm = wRe * vIm + wIm * vRe;

  // 4 Differential Memristor Conductance Values (Positive G in mS)
  const gBias = 0.5;
  const gRePos = gBias + Math.max(0, wRe);
  const gReNeg = gBias + Math.max(0, -wRe);
  const gImPos = gBias + Math.max(0, wIm);
  const gImNeg = gBias + Math.max(0, -wIm);

  // Physical noise injection (reduced by write-verify compensation)
  const noiseScale = writeVerifyEnabled ? pcmNoise * 0.15 : pcmNoise;
  const measRePos = gRePos + (Math.sin(choiWeightPhase * 3.7) * 0.5) * noiseScale;
  const measReNeg = gReNeg + (Math.cos(choiWeightPhase * 2.1) * 0.5) * noiseScale;
  const measImPos = gImPos + (Math.sin(inputVoltPhase * 4.3) * 0.5) * noiseScale;
  const measImNeg = gImNeg + (Math.cos(inputVoltPhase * 1.9) * 0.5) * noiseScale;

  const effWRe = measRePos - measReNeg;
  const effWIm = measImPos - measImNeg;
  const physRe = effWRe * vRe - effWIm * vIm;
  const physIm = effWRe * vIm + effWIm * vRe;

  const errorRe = Math.abs(physRe - idealRe);
  const errorIm = Math.abs(physIm - idealIm);
  const totalRmsError = Math.sqrt((errorRe * errorRe + errorIm * errorIm) / 2);

  const steps = [
    {
      title: isEn ? '1. Software Training' : '1. Huấn luyện Phần Mềm',
      desc: isEn ? (
        <>
          PyTorch/SNNTorch training with Wirtinger gradients to obtain ideal Complex Weights (
          <InlineMath math="W = W_{re} + jW_{im}" />).
        </>
      ) : (
        <>
          Huấn luyện trên PyTorch với đạo hàm Wirtinger để thu được trọng số phức lý tưởng (
          <InlineMath math="W = W_{re} + jW_{im}" />).
        </>
      ),
      icon: <Terminal className="w-4 h-4" />,
    },
    {
      title: isEn ? '2. Differential Matrix Splitting' : '2. Phân Tách Ma Trận Vi Sai',
      desc: isEn ? (
        <>
          Decompose complex weights into 4 positive physical conductance matrices (
          <InlineMath math="G_{re}^+, G_{re}^-, G_{im}^+, G_{im}^-" />).
        </>
      ) : (
        <>
          Phân tách trọng số phức thành 4 ma trận dẫn nạp thực dương (
          <InlineMath math="G_{re}^+, G_{re}^-, G_{im}^+, G_{im}^-" />).
        </>
      ),
      icon: <Layers className="w-4 h-4" />,
    },
    {
      title: isEn ? '3. Hardware Write-Verify Mapping' : '3. Nạp Trọng Số & Kiểm Chứng',
      desc: isEn
        ? 'Closed-loop DAC write pulses alter Phase-Change Memory (PCM) crystalline state with direct error mitigation.'
        : 'Xung nạp DAC vòng kín thay đổi cấu trúc tinh thể PCM với cơ chế tự sửa sai trực tiếp.',
      icon: <Cpu className="w-4 h-4" />,
    },
    {
      title: isEn ? '4. Analog Kirchhoff Inference' : '4. Suy Luận Tương Tự Kirchhoff',
      desc: isEn
        ? 'Input phasors apply as voltage pulses. Differential currents sum naturally in O(1) time.'
        : 'Tín hiệu điện áp pha kích hoạt tức thời. Dòng điện tự tổng hợp theo định luật Kirchhoff với độ trễ O(1).',
      icon: <Zap className="w-4 h-4" />,
    },
  ];

  const pythonDriverCode = `# Neuromorphic Complex-Valued Crossbar Driver (PyVISA / SCPI Protocol)
import numpy as np
import time
import pyvisa

rm = pyvisa.ResourceManager()
smu = rm.open_resource('USB0::0x0957::0x1736::MY51140135::0::INSTR')
smu.write("*RST")

def compile_complex_weights(w_complex: np.ndarray, g_bias: float = 0.5):
    """Decomposes W into 4 positive differential conductance matrices (units: mS)"""
    w_re = np.real(w_complex)
    w_im = np.imag(w_complex)
    
    g_re_pos = g_bias + np.maximum(0, w_re)
    g_re_neg = g_bias + np.maximum(0, -w_re)
    g_im_pos = g_bias + np.maximum(0, w_im)
    g_im_neg = g_bias + np.maximum(0, -w_im)
    
    return g_re_pos, g_re_neg, g_im_pos, g_im_neg

V_READ = 0.1        # V  - low, non-disturbing read voltage
V_SET = 1.0         # V  - base SET (potentiation) pulse amplitude
V_RESET = -1.0      # V  - base RESET (depression) pulse amplitude, opposite polarity
V_LIMIT = 1.5       # V  - absolute safety limit for any programming pulse
PULSE_WIDTH = 10e-6 # s  - nominal; use the SMU pulse mode for real us-level timing

def read_conductance_mS(channel: int) -> float:
    """Measure conductance at V_READ so the read does not disturb the cell."""
    smu.write(f"ROUT:CLOS (@{channel})")
    smu.write(f"SOUR:VOLT:LEV {V_READ}")
    smu.write(":OUTP ON")
    i_read = float(smu.query("MEAS:CURR?"))
    smu.write("OUTP OFF")
    return i_read / V_READ * 1e3  # S -> mS

def apply_pulse(channel: int, volt: float) -> None:
    """Apply one programming pulse, then always restore the read level."""
    volt = max(-V_LIMIT, min(V_LIMIT, volt))
    smu.write(f"ROUT:CLOS (@{channel})")
    smu.write(f"SOUR:VOLT:LEV {volt:.3f}")
    smu.write(":OUTP ON")
    time.sleep(PULSE_WIDTH)
    smu.write("OUTP OFF")
    smu.write(f"SOUR:VOLT:LEV {V_READ}")

def write_verify_pulse(channel: int, target_mS: float, tol: float = 0.05, max_iter: int = 20) -> float:
    """KAIST-style closed-loop write-verify: SET when below target, RESET when above."""
    g_meas = read_conductance_mS(channel)
    for step in range(max_iter):
        err = target_mS - g_meas
        if abs(err) <= tol * target_mS:
            break
        # Pulse amplitude grows with relative error, capped at V_LIMIT
        rel = min(1.0, abs(err) / max(target_mS, 1e-9))
        volt = V_SET + 0.3 * rel if err > 0 else V_RESET - 0.3 * rel
        apply_pulse(channel, volt)
        g_meas = read_conductance_mS(channel)
    return g_meas

print("Ready: 4-Quadrant differential crossbar calibrated with Write-Verify.")`;

  const handleCopyCode = () => {
    void copy(pythonDriverCode);
  };

  return (
    <div className="space-y-6">
      {/* SUB-NAVIGATION BAR (Full-width segmented controller matching components below) */}
      <div className="w-full bg-elegant-card/80 p-1.5 rounded-xl border border-slate-800/80 shadow-xs">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 w-full">
          <button
            onClick={() => setActiveSubTab('choi')}
            className={`flex items-center justify-center gap-2.5 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all cursor-pointer text-center ${
              activeSubTab === 'choi'
                ? 'bg-elegant-cyan/20 text-white border border-elegant-cyan/40 shadow-xs'
                : 'text-elegant-muted hover:text-white hover:bg-slate-800/40 border border-transparent'
            }`}
          >
            <Sparkles className="w-4 h-4 text-elegant-cyan shrink-0" />
            <span>
              {isEn
                ? 'KAIST SOTA Research & Kirchhoff Verifier'
                : 'Đột Phá KAIST & Kiểm Chứng Kirchhoff'}
            </span>
          </button>

          <button
            onClick={() => setActiveSubTab('schematic')}
            className={`flex items-center justify-center gap-2.5 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all cursor-pointer text-center ${
              activeSubTab === 'schematic'
                ? 'bg-elegant-cyan/20 text-white border border-elegant-cyan/40 shadow-xs'
                : 'text-elegant-muted hover:text-white hover:bg-slate-800/40 border border-transparent'
            }`}
          >
            <Layers className="w-4 h-4 text-elegant-cyan shrink-0" />
            <span>
              {isEn ? 'Physical Architecture Schematic' : 'Sơ Đồ Kiến Trúc Vật Lý'}
            </span>
          </button>

          <button
            onClick={() => setActiveSubTab('driver')}
            className={`flex items-center justify-center gap-2.5 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all cursor-pointer text-center ${
              activeSubTab === 'driver'
                ? 'bg-elegant-cyan/20 text-white border border-elegant-cyan/40 shadow-xs'
                : 'text-elegant-muted hover:text-white hover:bg-slate-800/40 border border-transparent'
            }`}
          >
            <Terminal className="w-4 h-4 text-elegant-cyan shrink-0" />
            <span>{isEn ? 'Hardware SCPI/PyVISA Driver' : 'Mã Lệnh Điều Khiển PyVISA'}</span>
          </button>
        </div>
      </div>

      {/* VIEW 1: KAIST RESEARCH HIGHLIGHTS & LIVE KIRCHHOFF CROSSBAR VERIFIER */}
      {activeSubTab === 'choi' && (
        <motion.div
          key="choi"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15 }}
          className="space-y-6"
        >
          {/* Nature Research Highlights Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-elegant-card border border-slate-800/80 p-4 rounded-xl space-y-2.5">
              <div className="flex items-center gap-2 text-elegant-cyan">
                <Sparkles className="w-4 h-4" />
                <span className="text-[10px] font-bold font-mono tracking-wider uppercase">
                  NATURE ELECTRONICS (2024)
                </span>
              </div>
              <h3 className="text-sm font-bold text-white font-display">
                {isEn
                  ? 'On-Chip Self-Correcting Memristor'
                  : 'Chip Tự Học & Sửa Sai On-Chip'}
              </h3>
              <p className="text-xs text-elegant-text/75 leading-relaxed">
                {isEn
                  ? 'Selector-less crossbar computing chip developed by Prof. Shinhyun Choi automatically corrects nonlinear conductance drift and sneak-path currents directly in hardware.'
                  : 'Chip memristor không cần bộ định tuyến (selector-less) do GS. Shinhyun Choi phát minh tự động triệt tiêu sai số phi tuyến và dòng rò ngay tại phần cứng.'}
              </p>
            </div>

            <div className="bg-elegant-card border border-slate-800/80 p-4 rounded-xl space-y-2.5">
              <div className="flex items-center gap-2 text-elegant-cyan">
                <Cpu className="w-4 h-4" />
                <span className="text-[10px] font-bold font-mono tracking-wider uppercase">
                  NATURE (2024 Breakthrough)
                </span>
              </div>
              <h3 className="text-sm font-bold text-white font-display">
                {isEn ? 'PCM Nanofilament Ultra-low Power' : 'PCM Nanofilament Tiêu Thụ Siêu Thấp'}
              </h3>
              <p className="text-xs text-elegant-text/75 leading-relaxed">
                {isEn
                  ? 'Ultra-small PCM nanofilament reduces phase transition volume to nanometer scale, lowering programming reset energy thousands of times below conventional DRAM.'
                  : 'Cấu trúc nanofilament siêu nhỏ giảm thể tích chuyển pha xuống cấp nanomet, giảm năng lượng reset hàng nghìn lần so với DRAM/NAND truyền thống.'}
              </p>
            </div>

            <div className="bg-elegant-card border border-slate-800/80 p-4 rounded-xl space-y-2.5">
              <div className="flex items-center gap-2 text-elegant-cyan">
                <Layers className="w-4 h-4" />
                <span className="text-[10px] font-bold font-mono tracking-wider uppercase">
                  NATURE (2020-2021)
                </span>
              </div>
              <h3 className="text-sm font-bold text-white font-display">
                {isEn ? 'Alloy Homogeneous Memristor' : 'Memristor Hợp Kim Đồng Nhất'}
              </h3>
              <p className="text-xs text-elegant-text/75 leading-relaxed">
                {isEn
                  ? 'Single-crystal silicon alloy devices eliminate spatial and temporal conductance fluctuations, providing ideal linear synaptic weight adjustment.'
                  : 'Hợp kim đơn tinh thể silicon kiểm soát nghiêm ngặt giúp triệt tiêu bất định không gian và thời gian, cho khả năng tinh chỉnh trọng số tuyến tính hoàn hảo.'}
              </p>
            </div>
          </div>

          {/* Interactive Kirchhoff Complex Multiplier Verification Sandbox */}
          <div className="bg-elegant-card border border-slate-800/80 p-5 rounded-xl space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
              <h3 className="text-sm font-bold text-white font-display flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>
                  {isEn
                    ? 'Live Kirchhoff Crossbar Complex Multiplier Sandbox'
                    : 'Bàn Thử Nghiệm Kiểm Chứng Mạch Nhân Phức Kirchhoff Trên Mảng Memristor'}
                </span>
              </h3>
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-elegant-muted">Write-Verify:</span>
                <button
                  onClick={() => setWriteVerifyEnabled(!writeVerifyEnabled)}
                  className={`px-2.5 py-0.5 rounded font-bold transition-colors cursor-pointer ${
                    writeVerifyEnabled
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                      : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                  }`}
                >
                  {writeVerifyEnabled
                    ? isEn
                      ? 'Calibrated (Active)'
                      : 'Đã Bật Hiệu Chuẩn'
                    : isEn
                      ? 'Raw Drift (Off)'
                      : 'Tắt Bù Sai Số'}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Sliders Box */}
              <div className="lg:col-span-5 space-y-3.5 bg-elegant-bg/60 p-4 rounded-xl border border-slate-800/80">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Synapse Weight |W|:' : 'Biên độ trọng số |W|:'}
                    </span>
                    <span className="text-elegant-cyan font-bold">
                      {choiWeightMag.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={choiWeightMag}
                    onChange={e => setChoiWeightMag(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Synapse Phase ∠W:' : 'Góc pha trọng số ∠W:'}
                    </span>
                    <span className="text-elegant-cyan font-bold">
                      {((choiWeightPhase * 180) / Math.PI).toFixed(0)}° (
                      {choiWeightPhase.toFixed(2)} rad)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="6.28"
                    step="0.05"
                    value={choiWeightPhase}
                    onChange={e => setChoiWeightPhase(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Input Voltage Phase ∠V:' : 'Góc pha điện áp vào ∠V:'}
                    </span>
                    <span className="text-amber-400 font-bold">
                      {((inputVoltPhase * 180) / Math.PI).toFixed(0)}° (
                      {inputVoltPhase.toFixed(2)} rad)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="6.28"
                    step="0.05"
                    value={inputVoltPhase}
                    onChange={e => setInputVoltPhase(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Hardware PCM Thermal Noise:' : 'Nhiễu nhiệt vật lý PCM:'}
                    </span>
                    <span className="text-rose-400 font-bold">
                      {(pcmNoise * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="0.4"
                    step="0.02"
                    value={pcmNoise}
                    onChange={e => setPcmNoise(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>

              {/* Differential 4-Memristor Real-time Readout */}
              <div className="lg:col-span-7 space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-elegant-bg/70 border border-slate-800/80 space-y-1">
                    <span className="text-[10px] text-slate-400">G_re⁺ (Pos)</span>
                    <div className="text-sm font-bold text-emerald-400">
                      {measRePos.toFixed(3)} mS
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-elegant-bg/70 border border-slate-800/80 space-y-1">
                    <span className="text-[10px] text-slate-400">G_re⁻ (Neg)</span>
                    <div className="text-sm font-bold text-rose-400">
                      {measReNeg.toFixed(3)} mS
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-elegant-bg/70 border border-slate-800/80 space-y-1">
                    <span className="text-[10px] text-slate-400">G_im⁺ (Pos)</span>
                    <div className="text-sm font-bold text-emerald-400">
                      {measImPos.toFixed(3)} mS
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-elegant-bg/70 border border-slate-800/80 space-y-1">
                    <span className="text-[10px] text-slate-400">G_im⁻ (Neg)</span>
                    <div className="text-sm font-bold text-rose-400">
                      {measImNeg.toFixed(3)} mS
                    </div>
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-elegant-bg/80 border border-slate-800/80 space-y-2">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Ideal Software Output W·V:' : 'Tích phức lý thuyết W·V:'}
                    </span>
                    <span className="text-white font-bold">
                      {idealRe.toFixed(3)} + j({idealIm.toFixed(3)})
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300">
                      {isEn ? 'Measured Crossbar Output:' : 'Dòng Kirchhoff thực tế:'}
                    </span>
                    <span className="text-elegant-cyan font-bold">
                      {physRe.toFixed(3)} + j({physIm.toFixed(3)})
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-xs font-mono pt-2 border-t border-slate-800/70">
                    <span className="text-slate-400">
                      {isEn ? 'Crossbar RMS Error:' : 'Sai số RMS phần cứng:'}
                    </span>
                    <span
                      className={`font-bold ${
                        totalRmsError < 0.08
                          ? 'text-emerald-400'
                          : totalRmsError < 0.2
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {totalRmsError.toFixed(4)} (
                      {((totalRmsError / Math.max(0.1, choiWeightMag)) * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-elegant-text/80 font-mono bg-elegant-cyan/5 p-3 rounded-lg border border-elegant-cyan/20">
                  <InlineMath math="W = (G_{re}^+ - G_{re}^-) + j(G_{im}^+ - G_{im}^-)" />
                  <span className="block mt-1 text-[10.5px] text-elegant-muted">
                    {isEn
                      ? 'The differential 4-memristor topology perfectly cancels common-mode thermal drift and leakages.'
                      : 'Kiến trúc vi sai 4-memristor tự triệt tiêu dòng rò và trôi nhiệt đồng pha (common-mode cancellation).'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* VIEW 2: DRAGGABLE SCHEMATIC & PIPELINE STEPS */}
      {activeSubTab === 'schematic' && (
        <motion.div
          key="schematic"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15 }}
          className="grid grid-cols-1 lg:grid-cols-12 gap-6"
        >
          {/* Interactive SVG Schematic */}
          <div className="lg:col-span-8 bg-[#0b1016] border border-slate-800/80 rounded-xl overflow-hidden flex flex-col">
            <div className="p-3.5 border-b border-slate-800/80 bg-elegant-card flex justify-between items-center">
              <h3 className="text-xs font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
                <Layers className="w-4 h-4 text-elegant-cyan" />
                <span>{isEn ? 'Interactive Physical Architecture' : 'Sơ Đồ Mạch Vật Lý'}</span>
              </h3>
              <span className="text-[10px] font-mono text-elegant-muted">
                {isEn ? 'Drag blocks to reposition' : 'Kéo thả khối mạch để định vị'}
              </span>
            </div>

            <div className="p-6 flex-1 flex items-center justify-center relative overflow-hidden min-h-[420px]">
              <div
                className="absolute inset-0"
                style={{
                  backgroundImage: 'radial-gradient(#1e293b 1px, transparent 1px)',
                  backgroundSize: '24px 24px',
                  opacity: 0.4,
                }}
              />

              <div className="relative z-10 w-full max-w-2xl h-full flex items-center justify-center">
                <svg viewBox="0 0 850 480" className="w-full h-[460px] overflow-visible">
                  <defs>
                    <marker id="arrCyan" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#06b6d4" />
                    </marker>
                    <marker id="arrSlate" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#94a3b8" />
                    </marker>
                  </defs>

                  <path d={drawPath(pos.input.x + 120, pos.input.y + 40, pos.dac.x, pos.dac.y + 120)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrSlate)" />
                  <path d={drawPath(pos.dac.x + 60, pos.dac.y + 40, pos.cbRe.x, pos.cbRe.y + 75)} stroke="#06b6d4" strokeWidth="2" fill="none" markerEnd="url(#arrCyan)" />
                  <path d={drawPath(pos.dac.x + 60, pos.dac.y + 200, pos.cbIm.x, pos.cbIm.y + 75)} stroke="#06b6d4" strokeWidth="2" fill="none" markerEnd="url(#arrCyan)" />
                  <path d={drawPath(pos.cbRe.x + 180, pos.cbRe.y + 75, pos.tia.x, pos.tia.y + 30)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrSlate)" />
                  <path d={drawPath(pos.cbIm.x + 180, pos.cbIm.y + 75, pos.tia.x, pos.tia.y + 50)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrSlate)" />
                  <path d={drawPath(pos.tia.x + 80, pos.tia.y + 40, pos.clif.x, pos.clif.y + 40)} stroke="#06b6d4" strokeWidth="2" fill="none" markerEnd="url(#arrCyan)" />

                  {/* Input Block */}
                  <motion.g initial={{ x: pos.input.x, y: pos.input.y }} drag dragMomentum={false} onDrag={onDrag('input')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="120" height="75" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2" />
                    <text x="60" y="32" fill="white" fontSize="13" fontWeight="bold" textAnchor="middle">Input Signal</text>
                    <text x="60" y="50" fill="#94a3b8" fontSize="11" textAnchor="middle">Phase Encoder</text>
                  </motion.g>

                  {/* DAC Array */}
                  <motion.g initial={{ x: pos.dac.x, y: pos.dac.y }} drag dragMomentum={false} onDrag={onDrag('dac')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="60" height="240" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="2" />
                    <g transform="translate(30, 120) rotate(-90)">
                      <text x="0" y="-3" fill="#06b6d4" fontSize="13" fontWeight="bold" textAnchor="middle">DAC Pulses</text>
                      <text x="0" y="12" fill="#94a3b8" fontSize="10" textAnchor="middle">V_set / V_reset</text>
                    </g>
                  </motion.g>

                  {/* Crossbar Re */}
                  <motion.g initial={{ x: pos.cbRe.x, y: pos.cbRe.y }} drag dragMomentum={false} onDrag={onDrag('cbRe')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="180" height="140" rx="8" fill="#1e293b" stroke="#06b6d4" strokeWidth="2" />
                    <text x="90" y="26" fill="white" fontSize="13" fontWeight="bold" textAnchor="middle">Crossbar G_re</text>
                    <text x="90" y="44" fill="#94a3b8" fontSize="10" textAnchor="middle">G_re⁺ & G_re⁻ arrays</text>
                    {[1, 2, 3].map(i => <line key={`h${i}`} x1="20" y1={45 + i * 22} x2="160" y2={45 + i * 22} stroke="#475569" strokeWidth="1.5" />)}
                    {[1, 2, 3, 4].map(j => <line key={`v${j}`} x1={25 + j * 26} y1="50" x2={25 + j * 26} y2="125" stroke="#475569" strokeWidth="1.5" />)}
                  </motion.g>

                  {/* Crossbar Im */}
                  <motion.g initial={{ x: pos.cbIm.x, y: pos.cbIm.y }} drag dragMomentum={false} onDrag={onDrag('cbIm')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="180" height="140" rx="8" fill="#1e293b" stroke="#f59e0b" strokeWidth="2" />
                    <text x="90" y="26" fill="white" fontSize="13" fontWeight="bold" textAnchor="middle">Crossbar G_im</text>
                    <text x="90" y="44" fill="#94a3b8" fontSize="10" textAnchor="middle">G_im⁺ & G_im⁻ arrays</text>
                    {[1, 2, 3].map(i => <line key={`h2${i}`} x1="20" y1={45 + i * 22} x2="160" y2={45 + i * 22} stroke="#475569" strokeWidth="1.5" />)}
                    {[1, 2, 3, 4].map(j => <line key={`v2${j}`} x1={25 + j * 26} y1="50" x2={25 + j * 26} y2="125" stroke="#475569" strokeWidth="1.5" />)}
                  </motion.g>

                  {/* TIA */}
                  <motion.g initial={{ x: pos.tia.x, y: pos.tia.y }} drag dragMomentum={false} onDrag={onDrag('tia')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="80" height="80" rx="8" fill="#1e293b" stroke="#a855f7" strokeWidth="2" />
                    <text x="40" y="34" fill="#a855f7" fontSize="12" fontWeight="bold" textAnchor="middle">TIA &</text>
                    <text x="40" y="52" fill="#a855f7" fontSize="12" fontWeight="bold" textAnchor="middle">Diff Amp</text>
                  </motion.g>

                  {/* cLIF Comparator */}
                  <motion.g initial={{ x: pos.clif.x, y: pos.clif.y }} drag dragMomentum={false} onDrag={onDrag('clif')} style={{ cursor: 'grab' }}>
                    <rect x="0" y="0" width="100" height="80" rx="8" fill="#1e293b" stroke="#f43f5e" strokeWidth="2" />
                    <text x="50" y="36" fill="white" fontSize="13" fontWeight="bold" textAnchor="middle">cLIF Soma</text>
                    <text x="50" y="54" fill="#f43f5e" fontSize="11" textAnchor="middle">Spike Reset</text>
                  </motion.g>
                </svg>
              </div>
            </div>
          </div>

          {/* Deployment Workflow Side Steps */}
          <div className="lg:col-span-4 space-y-3">
            <div className="bg-elegant-card border border-slate-800/80 p-4 rounded-xl space-y-2.5">
              <h3 className="text-xs font-bold text-white uppercase font-mono flex items-center gap-1.5">
                <Settings className="w-3.5 h-3.5 text-elegant-cyan" />
                <span>{isEn ? 'Hardware Deployment Flow' : 'Quy Trình Triển Khai'}</span>
              </h3>
              <div className="space-y-2">
                {steps.map((step, idx) => (
                  <button
                    key={idx}
                    onClick={() => setActiveStep(idx)}
                    className={`w-full text-left p-2.5 rounded-lg border text-xs font-mono transition-colors cursor-pointer ${
                      activeStep === idx
                        ? 'bg-elegant-cyan/15 border-elegant-cyan text-white'
                        : 'bg-elegant-bg border-slate-800/80 text-elegant-muted hover:text-white'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className="p-1 rounded bg-elegant-card text-elegant-cyan shrink-0">
                        {step.icon}
                      </div>
                      <span className="font-bold">{step.title}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-elegant-card border border-slate-800/80 p-4 rounded-xl space-y-2">
              <h4 className="text-[10px] font-bold text-elegant-cyan uppercase font-mono">
                {steps[activeStep].title}
              </h4>
              <div className="text-xs text-elegant-text/80 leading-relaxed font-mono">
                {steps[activeStep].desc}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* VIEW 3: PYTHON PYVISA / SCPI HARDWARE DRIVER CODE */}
      {activeSubTab === 'driver' && (
        <motion.div
          key="driver"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15 }}
          className="bg-elegant-card border border-slate-800/80 rounded-xl overflow-hidden shadow-xl"
        >
          <div className="p-3.5 bg-elegant-bg border-b border-slate-800/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-elegant-cyan" />
              <span className="text-xs font-mono font-bold text-white">
                memristor_crossbar_driver.py (PyVISA / SCPI API)
              </span>
            </div>
            <button
              onClick={handleCopyCode}
              className="flex items-center gap-1.5 px-3 py-1 bg-elegant-card hover:border-elegant-cyan text-elegant-cyan border border-slate-700/80 rounded-md text-xs font-mono transition-colors cursor-pointer"
            >
              {copyStatus === 'copied' ? (
                <Check className="w-3.5 h-3.5" />
              ) : copyStatus === 'failed' ? (
                <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span>
                {copyStatus === 'copied'
                  ? isEn ? 'Copied' : 'Đã Chép'
                  : copyStatus === 'failed'
                    ? isEn ? 'Copy failed: select manually' : 'Chép thất bại: hãy bôi đen để chép'
                    : isEn ? 'Copy Script' : 'Sao Chép'}
              </span>
            </button>
          </div>

          <pre className="p-4 text-xs font-mono text-slate-300 leading-relaxed overflow-x-auto bg-[#07090e]">
            <code>{pythonDriverCode}</code>
          </pre>
        </motion.div>
      )}
    </div>
  );
}
