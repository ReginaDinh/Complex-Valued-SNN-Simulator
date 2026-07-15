import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  GitCompare, Zap, Sliders, Info, HelpCircle, ArrowRight,
  TrendingDown, Check, Sparkles, RefreshCw, AlertCircle, Eye, Activity
} from 'lucide-react';

export default function SNNComparison() {
  // --- STATE FOR INTERFERENCE DEMO ---
  const [magA, setMagA] = useState<number>(0.8);
  const [phaseA, setPhaseA] = useState<number>(0); // in degrees: 0 - 360
  const [magB, setMagB] = useState<number>(0.8);
  const [phaseB, setPhaseB] = useState<number>(180); // in degrees: 0 - 360

  // --- STATE FOR LEAKAGE SIMULATOR ---
  const [leakMag, setLeakMag] = useState<number>(0.85);
  const [leakPhase, setLeakPhase] = useState<number>(30); // decay rotation in degrees
  const [simSteps, setSimSteps] = useState<number>(20);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isAutoPlay, setIsAutoPlay] = useState<boolean>(true);

  // Auto-play loop for the leakage simulation spiral
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoPlay) {
      interval = setInterval(() => {
        setCurrentStep((prev) => (prev + 1) % (simSteps + 1));
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isAutoPlay, simSteps]);

  // Calculations for Interference Demo
  const radiansA = (phaseA * Math.PI) / 180;
  const radiansB = (phaseB * Math.PI) / 180;

  // Complex components
  const complexA = useMemo(() => ({
    re: magA * Math.cos(radiansA),
    im: magA * Math.sin(radiansA)
  }), [magA, radiansA]);

  const complexB = useMemo(() => ({
    re: magB * Math.cos(radiansB),
    im: magB * Math.sin(radiansB)
  }), [magB, radiansB]);

  const complexSum = useMemo(() => ({
    re: complexA.re + complexB.re,
    im: complexA.im + complexB.im
  }), [complexA, complexB]);

  const sumMag = useMemo(() => {
    return Math.sqrt(complexSum.re * complexSum.re + complexSum.im * complexSum.im);
  }, [complexSum]);

  const sumPhase = useMemo(() => {
    let angle = Math.atan2(complexSum.im, complexSum.re) * (180 / Math.PI);
    if (angle < 0) angle += 360;
    return angle;
  }, [complexSum]);

  // Traditional SNN summation (strictly scalar addition of magnitudes / rate coding)
  const traditionalSum = magA + magB;

  // Calculate coordinates for SVG visualization of vector addition
  // Center is (150, 150), scale factor is 80 (since max mag is 1.5)
  const SVG_CENTER = 150;
  const SVG_SCALE = 70;

  const getCoords = (mag: number, rad: number) => {
    return {
      x: SVG_CENTER + mag * Math.cos(rad) * SVG_SCALE,
      y: SVG_CENTER - mag * Math.sin(rad) * SVG_SCALE // flip Y for SVG coordinates
    };
  };

  const coordA = getCoords(magA, radiansA);
  const coordB = getCoords(magB, radiansB);
  
  // Vector B starting from tip of Vector A (constructive head-to-tail addition)
  const coordB_fromA = {
    x: coordA.x + (coordB.x - SVG_CENTER),
    y: coordA.y + (coordB.y - SVG_CENTER)
  };

  const coordSum = {
    x: SVG_CENTER + complexSum.re * SVG_SCALE,
    y: SVG_CENTER - complexSum.im * SVG_SCALE
  };

  // Calculations for Leakage Demo Spiral
  // Traditional SNN decays simply: V(t) = V(0) * (leakMag)^t
  // Complex SNN decays and rotates: U(t) = U(0) * (leakMag e^{j * leakPhase})^t
  const leakageData = useMemo(() => {
    const data: { step: number; s_re: number; s_im: number; s_mag: number; c_re: number; c_im: number; c_mag: number }[] = [];
    const initialMag = 1.2;
    const initialPhaseRad = (45 * Math.PI) / 180; // 45 degrees start

    for (let t = 0; t <= simSteps; t++) {
      // Traditional decay (strictly real axis, 1D)
      const s_mag = initialMag * Math.pow(leakMag, t);
      const s_re = s_mag * Math.cos(initialPhaseRad); // keep on same angle for 1D visual comparison, or just real line
      const s_im = 0; // 1D line

      // Complex decay (spiral, magnitude decay + phase rotation)
      const c_mag = initialMag * Math.pow(leakMag, t);
      const totalRotRad = initialPhaseRad + (t * leakPhase * Math.PI) / 180;
      const c_re = c_mag * Math.cos(totalRotRad);
      const c_im = c_mag * Math.sin(totalRotRad);

      data.push({
        step: t,
        s_re,
        s_im,
        s_mag,
        c_re,
        c_im,
        c_mag
      });
    }
    return data;
  }, [leakMag, leakPhase, simSteps]);

  return (
    <div className="space-y-8">
      {/* HEADER SECTION */}
      <div className="bg-gradient-to-r from-elegant-card to-elegant-card/40 border border-elegant-muted/15 p-6 rounded-xl shadow-xl space-y-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
            <GitCompare className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold font-display text-white">So Sánh Kiến Trúc & Hoạt Động: SNN vs Complex-valued SNN (cLIF)</h2>
            <p className="text-xs text-elegant-text/80 leading-relaxed mt-0.5">
              Khám phá sự khác biệt cốt lõi giữa Mạng nổ xung miền thực truyền thống (Traditional SNN) và Mạng nổ xung miền số phức (Complex-valued SNN).
            </p>
          </div>
        </div>
      </div>

      {/* CORE CONCEPT CARDS (Real vs Complex) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* CARD 1: TRADITIONAL SNN */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-rose-500/5 rounded-full blur-2xl" />
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
            <h3 className="text-sm font-bold text-rose-400 flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
              <span>Mạng Nổ Xung Miền Thực (Traditional SNN)</span>
            </h3>
            <span className="text-[10px] font-mono bg-rose-500/10 text-rose-400 border border-rose-500/20 px-2 py-0.5 rounded uppercase">
              Miền Thực ({"$\\mathbb{R}$"})
            </span>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-elegant-text/90">
            <p>
              Mạng nổ xung truyền thống mô phỏng sinh học bằng các biến trạng thái là các số thực đơn thuần. Thông tin được mã hóa dựa trên **tần số nổ xung (Rate coding)** hoặc **thời điểm nổ xung đơn lẻ (Temporal coding)**.
            </p>
            <ul className="space-y-2.5 font-mono text-[11px] text-elegant-text/80">
              <li className="flex items-start gap-2">
                <span className="text-rose-400 font-bold shrink-0">■ Điện thế màng:</span>
                <span>{"$V(t) \\in \\mathbb{R}$"} — Một giá trị vô hướng (scalar) 1 chiều biểu diễn mức tích lũy năng lượng.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-rose-400 font-bold shrink-0">■ Trọng số Synapse:</span>
                <span>{"$W \\in \\mathbb{R}$"} — Chỉ biểu diễn cường độ kích thích {"(W > 0)"} hoặc ức chế {"(W < 0)"}.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-rose-400 font-bold shrink-0">■ Tích lũy dòng điện:</span>
                <span>Phép cộng đại số trực tiếp các xung kích hoạt đi vào. Không có khái niệm lệch pha hay đồng bộ hóa.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-rose-400 font-bold shrink-0">■ Suy hao rò rỉ:</span>
                <span>Decay theo hàm mũ thẳng về trạng thái nghỉ 0 theo một chiều duy nhất.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* CARD 2: COMPLEX-VALUED SNN */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-elegant-cyan/5 rounded-full blur-2xl" />
          <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-3">
            <h3 className="text-sm font-bold text-elegant-cyan flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 rounded-full bg-elegant-cyan" />
              <span>Mạng Nổ Xung Miền Phức (Complex-valued cLIF)</span>
            </h3>
            <span className="text-[10px] font-mono bg-elegant-cyan/10 text-elegant-cyan border border-elegant-cyan/20 px-2 py-0.5 rounded uppercase animate-pulse">
              Miền Phức ({"$\\mathbb{C}$"})
            </span>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-elegant-text/90">
            <p>
              CV-SNN mở rộng trạng thái nổ xung sang miền phức để mô phỏng **đồng bộ hóa pha và nhịp sinh học tự nhiên**. Thông tin mang cả biên độ (năng lượng) lẫn góc pha (thời gian tương đối).
            </p>
            <ul className="space-y-2.5 font-mono text-[11px] text-elegant-text/80">
              <li className="flex items-start gap-2">
                <span className="text-elegant-cyan font-bold shrink-0">■ Điện thế màng:</span>
                <span>{"$U(t) = V(t) + j \\cdot Y(t) \\in \\mathbb{C}$"} — Đại lượng 2 chiều mang cả biên độ {"$|U(t)|$"} và pha góc {"$\\theta(t)$"}.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-elegant-cyan font-bold shrink-0">■ Trọng số Synapse:</span>
                <span>{"$W = M \\cdot e^{j\\phi} \\in \\mathbb{C}$"} — Vừa điều khiển biên độ truyền {"$M$"}, vừa làm **lệch pha dòng điện** một góc {"$\\phi$"}.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-elegant-cyan font-bold shrink-0">■ Tích lũy dòng điện:</span>
                <span>Phép cộng vector số phức, cho phép **Giao thoa cộng pha** (constructive) hoặc **Giao thoa triệt tiêu** (destructive).</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-elegant-cyan font-bold shrink-0">■ Suy hao rò rỉ:</span>
                <span>Decay biên độ đồng thời xoay pha liên tục, vẽ nên một quỹ đạo xoắn ốc (spiral trajectory) tuyệt đẹp.</span>
              </li>
            </ul>
          </div>
        </div>

      </div>

      {/* DEMO 1: INTERACTIVE WAVE INTERFERENCE DEMO */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-6 shadow-xl">
        <div className="border-b border-elegant-muted/10 pb-3 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
            <span className="w-2.5 h-2.5 rounded-full bg-elegant-cyan" />
            <span>Mô Phỏng 1: Sự Kỳ Diệu Của Giao Thoa Pha (Phase Interference)</span>
          </h3>
          <span className="text-xs text-elegant-muted font-mono">Cơ chế cộng hợp vector vi sai</span>
        </div>

        <p className="text-xs text-elegant-text/80 leading-relaxed">
          Hãy điều chỉnh pha góc ($\theta$) và biên độ của hai dòng tín hiệu đầu vào **Kênh A (Đỏ)** và **Kênh B (Xanh dương)**. 
          Quan sát xem chúng triệt tiêu hay cộng hợp lại với nhau thế nào trong miền số phức so với phép cộng tuyến tính miền thực!
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          
          {/* SLIDERS BOX (4 Cols) */}
          <div className="lg:col-span-4 space-y-5 bg-elegant-bg/40 p-4 rounded-xl border border-elegant-muted/10">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-elegant-muted block border-b border-elegant-muted/5 pb-1.5">Tín hiệu Kênh A</span>
            
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-rose-400 font-bold">Biên độ Kênh A:</span>
                <span className="text-white">{magA.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.2"
                step="0.1"
                value={magA}
                onChange={(e) => setMagA(parseFloat(e.target.value))}
                className="w-full accent-rose-500 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-rose-400 font-bold">Góc Pha Kênh A:</span>
                <span className="text-white">{phaseA}°</span>
              </div>
              <input
                type="range"
                min="0"
                max="360"
                step="15"
                value={phaseA}
                onChange={(e) => setPhaseA(parseInt(e.target.value))}
                className="w-full accent-rose-500 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-elegant-muted block border-b border-elegant-muted/5 pt-3 pb-1.5">Tín hiệu Kênh B</span>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-indigo-400 font-bold">Biên độ Kênh B:</span>
                <span className="text-white">{magB.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.2"
                step="0.1"
                value={magB}
                onChange={(e) => setMagB(parseFloat(e.target.value))}
                className="w-full accent-indigo-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-indigo-400 font-bold">Góc Pha Kênh B:</span>
                <span className="text-white">{phaseB}°</span>
              </div>
              <input
                type="range"
                min="0"
                max="360"
                step="15"
                value={phaseB}
                onChange={(e) => setPhaseB(parseInt(e.target.value))}
                className="w-full accent-indigo-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>
          </div>

          {/* COMPLEX PLANE VISUALIZER (4 Cols) */}
          <div className="lg:col-span-4 flex flex-col items-center justify-center bg-elegant-bg/70 p-4 rounded-xl border border-elegant-muted/10 h-[320px]">
            <span className="text-[10px] font-mono text-elegant-muted uppercase font-bold mb-2">Đồ thị Mặt phẳng Phức Gauss</span>
            
            <svg width="260" height="260" className="bg-[#0b1016]/90 border border-elegant-muted/5 rounded-xl">
              {/* Grid axes */}
              <line x1="130" y1="10" x2="130" y2="250" stroke="#1d2836" strokeWidth="1" strokeDasharray="3 3" />
              <line x1="10" y1="130" x2="250" y2="130" stroke="#1d2836" strokeWidth="1" strokeDasharray="3 3" />
              
              {/* Unit circle */}
              <circle cx="130" cy="130" r={SVG_SCALE} fill="none" stroke="#253545" strokeWidth="1" />
              <text x="135" y="24" fill="#5a7187" className="text-[9px] font-mono">Im (Trực giao)</text>
              <text x="190" y="125" fill="#5a7187" className="text-[9px] font-mono">Re (Đồng pha)</text>

              {/* Vector A (Red) */}
              <line 
                x1="130" y1="130" 
                x2={coordA.x} y2={coordA.y} 
                stroke="#ef4444" strokeWidth="2.5" 
                strokeLinecap="round"
              />
              <circle cx={coordA.x} cy={coordA.y} r="4" fill="#ef4444" />

              {/* Vector B (Blue) represented head-to-tail for summing visual */}
              <line 
                x1={coordA.x} y1={coordA.y} 
                x2={coordB_fromA.x} y2={coordB_fromA.y} 
                stroke="#6366f1" strokeWidth="2" 
                strokeLinecap="round"
                strokeDasharray="2 2"
              />
              {/* Vector B original for display */}
              <line 
                x1="130" y1="130" 
                x2={coordB.x} y2={coordB.y} 
                stroke="#6366f1" strokeWidth="2" 
                strokeLinecap="round"
                opacity="0.5"
              />
              <circle cx={coordB.x} cy={coordB.y} r="3" fill="#6366f1" opacity="0.6" />

              {/* Vector Sum (Cyan) */}
              <line 
                x1="130" y1="130" 
                x2={coordSum.x} y2={coordSum.y} 
                stroke="#00f2fe" strokeWidth="3" 
                strokeLinecap="round"
              />
              <circle cx={coordSum.x} cy={coordSum.y} r="5" fill="#00f2fe" className="animate-pulse" />
            </svg>

            <div className="flex gap-4 text-[10px] font-mono mt-3">
              <span className="text-rose-400">■ Kênh A</span>
              <span className="text-indigo-400">■ Kênh B</span>
              <span className="text-elegant-cyan">■ Cộng Complex</span>
            </div>
          </div>

          {/* COMPARATIVE METRICS RESULT (4 Cols) */}
          <div className="lg:col-span-4 space-y-4">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-elegant-muted block">Kết quả so sánh cộng luồng</span>
            
            {/* TRADITIONAL SNN SUM BOX */}
            <div className="p-3.5 bg-rose-500/5 border border-rose-500/15 rounded-xl space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="text-rose-400 font-bold font-display uppercase tracking-wide">Phép cộng SNN miền thực:</span>
                <span className="font-mono text-sm font-black text-rose-400">{traditionalSum.toFixed(2)}</span>
              </div>
              <p className="text-[10px] text-elegant-text/70 leading-normal">
                Không quan tâm pha lệch, dòng điện tổng hợp đơn giản bằng **tổng trị số tuyệt đối** ($A + B$). SNN hoàn toàn mù thông tin thời gian/lệch pha!
              </p>
            </div>

            {/* COMPLEX CV-SNN SUM BOX */}
            <div className="p-3.5 bg-elegant-cyan/5 border border-elegant-cyan/20 rounded-xl space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="text-elegant-cyan font-bold font-display uppercase tracking-wide">Phép cộng CV-SNN (Complex):</span>
                <span className="font-mono text-sm font-black text-elegant-cyan">{sumMag.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-[10px] font-mono text-elegant-muted">
                <span>Góc pha tổng: {sumPhase.toFixed(0)}°</span>
                <span>
                  {Math.abs(phaseA - phaseB) === 180 
                    ? '⚠️ Triệt tiêu hoàn toàn' 
                    : Math.abs(phaseA - phaseB) === 0 
                      ? '⚡ Đồng pha tối đa' 
                      : '🔄 Giao thoa bán phần'}
                </span>
              </div>
              <p className="text-[10px] text-elegant-text/70 leading-normal">
                Tổng hợp thông qua **phép cộng vector**. Nếu lệch pha 180°, chúng triệt tiêu nhau về 0. Nếu đồng pha, chúng cộng hưởng mạnh mẽ!
              </p>
            </div>

            <div className="p-3.5 bg-elegant-bg border border-elegant-muted/10 rounded-xl">
              <p className="text-[10.5px] leading-relaxed text-elegant-text/80">
                <strong className="text-elegant-cyan">Ứng dụng Biometric (Vân tay / Mống mắt):</strong> Lượng thông tin góc pha giúp mô phỏng cấu trúc vân lồi lõm cực nhạy. Nhiễu môi trường ngẫu nhiên sẽ tự triệt tiêu lẫn nhau (destructive interference), trong khi các đặc điểm sinh trắc học chuẩn xác được cộng pha tự nhiên!
              </p>
            </div>

          </div>

        </div>
      </div>

      {/* DEMO 2: MEMBRANE POTENTIAL LEAKAGE (Linear vs Spiral Spiral) */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-6 shadow-xl">
        <div className="border-b border-elegant-muted/10 pb-3 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            <span>Mô Phỏng 2: Sự Rò Rỉ Đặc Trưng (Leakage - Real Decay vs Complex Rotation)</span>
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsAutoPlay(!isAutoPlay)}
              className={`px-2.5 py-1 text-[10px] font-mono rounded font-bold transition flex items-center gap-1 border ${
                isAutoPlay 
                  ? 'bg-amber-400/10 text-amber-400 border-amber-400/20' 
                  : 'bg-elegant-bg text-elegant-muted border-elegant-muted/10'
              }`}
            >
              <RefreshCw className={`w-3 h-3 ${isAutoPlay ? 'animate-spin' : ''}`} />
              <span>{isAutoPlay ? 'Tự Động Chạy' : 'Tạm Dừng'}</span>
            </button>
            <button
              onClick={() => {
                setCurrentStep(0);
                setIsAutoPlay(false);
              }}
              className="px-2 py-1 text-[10px] font-mono rounded bg-elegant-bg border border-elegant-muted/10 text-rose-400 font-bold hover:bg-elegant-card transition"
            >
              Reset
            </button>
          </div>
        </div>

        <p className="text-xs text-elegant-text/80 leading-relaxed">
          Sự rò rỉ (leakage) của neuron sinh học xảy ra liên tục qua màng lipid. Trong khi SNN truyền thống chỉ rò rỉ **giảm dần biên độ**, thì CV-SNN (cLIF) rò rỉ bằng cách **xoay pha góc đồng thời giảm dần biên độ**, tạo nên quỹ đạo xoắn ốc hoàn hảo.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          
          {/* Controls Box (3 Cols) */}
          <div className="lg:col-span-3 space-y-4 bg-elegant-bg/40 p-4 rounded-xl border border-elegant-muted/10">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-elegant-muted block">Hệ số rò rỉ màng</span>
            
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white">Hệ số suy hao ($\lambda$):</span>
                <span className="text-amber-400 font-bold">{leakMag.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.75"
                max="0.98"
                step="0.01"
                value={leakMag}
                onChange={(e) => setLeakMag(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-white">Xoay pha rò rỉ ($\omega$):</span>
                <span className="text-amber-400 font-bold">{leakPhase}°</span>
              </div>
              <input
                type="range"
                min="5"
                max="60"
                step="5"
                value={leakPhase}
                onChange={(e) => setLeakPhase(parseInt(e.target.value))}
                className="w-full accent-amber-400 bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="pt-2 border-t border-elegant-muted/10 text-center font-mono">
              <span className="text-[10px] text-elegant-muted">Bước thời gian hiện tại (t):</span>
              <div className="text-xl font-bold text-amber-400 mt-1">{currentStep} / {simSteps}</div>
            </div>
          </div>

          {/* SVG SPIRAL GRAPH (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col items-center justify-center bg-elegant-bg/70 p-4 rounded-xl border border-elegant-muted/10 h-[300px]">
            <span className="text-[10px] font-mono text-elegant-muted uppercase font-bold mb-2">Quỹ Đạo Phân Rã Điện Thế Màng</span>
            
            <svg width="240" height="240" className="bg-[#0b1016]/90 border border-elegant-muted/5 rounded-xl">
              {/* Grid axes */}
              <line x1="120" y1="10" x2="120" y2="230" stroke="#1d2836" strokeWidth="1" strokeDasharray="2 2" />
              <line x1="10" y1="120" x2="230" y2="120" stroke="#1d2836" strokeWidth="1" strokeDasharray="2 2" />
              
              {/* Traditional 1D linear decay line (Red dots along the 45-degree ray, but strictly real-axis decays) */}
              {leakageData.map((d, idx) => {
                const x_trad = 120 + d.s_re * 70;
                const y_trad = 120 - d.s_im * 70; // 1D along real line
                const isCurrent = idx === currentStep;

                return (
                  <circle 
                    key={`trad-dot-${idx}`}
                    cx={x_trad} 
                    cy="120" 
                    r={isCurrent ? "4.5" : "1.5"} 
                    fill={isCurrent ? "#ef4444" : "#f87171"} 
                    opacity={idx <= currentStep ? 0.9 : 0.15}
                    className={isCurrent ? "animate-pulse" : ""}
                  />
                );
              })}

              {/* Complex 2D spiral decay path (Cyan dots) */}
              {/* Draw spiral connector lines */}
              {leakageData.map((d, idx) => {
                if (idx === 0) return null;
                const prev = leakageData[idx - 1];
                const x1 = 120 + prev.c_re * 70;
                const y1 = 120 - prev.c_im * 70;
                const x2 = 120 + d.c_re * 70;
                const y2 = 120 - d.c_im * 70;

                return (
                  <line
                    key={`spiral-line-${idx}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="#00f2fe"
                    strokeWidth="1"
                    opacity={idx <= currentStep ? 0.5 : 0.05}
                  />
                );
              })}

              {leakageData.map((d, idx) => {
                const x_comp = 120 + d.c_re * 70;
                const y_comp = 120 - d.c_im * 70;
                const isCurrent = idx === currentStep;

                return (
                  <circle 
                    key={`comp-dot-${idx}`}
                    cx={x_comp} 
                    cy={y_comp} 
                    r={isCurrent ? "5" : "2"} 
                    fill={isCurrent ? "#00f2fe" : "#22d3ee"} 
                    opacity={idx <= currentStep ? 1.0 : 0.15}
                    className={isCurrent ? "animate-pulse" : ""}
                  />
                );
              })}
            </svg>

            <div className="flex gap-4 text-[9px] font-mono mt-3">
              <span className="text-rose-400">● Tuyến tính 1D (SNN thường)</span>
              <span className="text-elegant-cyan">● Xoắn ốc 2D (CV-SNN cLIF)</span>
            </div>
          </div>

          {/* Educational Insights (4 Cols) */}
          <div className="lg:col-span-4 space-y-3.5">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-elegant-muted block">Ý nghĩa vật lý học</span>
            
            <div className="p-3 bg-[#0b1016]/40 border border-elegant-muted/10 rounded-xl text-xs space-y-2">
              <p className="leading-relaxed">
                <strong className="text-rose-400">Phân rã tuyến tính (SNN):</strong> Khi không có kích thích, điện thế màng rò rỉ thẳng tuột về 0. Năng lượng mất đi nhưng không giữ lại bất cứ dấu vết chu kỳ thời gian nào của xung trước đó.
              </p>
              <p className="leading-relaxed">
                <strong className="text-elegant-cyan">Quỹ đạo xoắn ốc (CV-SNN):</strong> Khi rò rỉ, cLIF neuron tiếp tục **xoay pha**. Điều này đóng vai trò giống như một **bộ nhớ pha thời gian ngắn (short-term phase memory)**. Neuron vẫn ghi nhớ thời gian tương đối của xung gần nhất, giúp liên kết hoàn hảo với xung tiếp theo!
              </p>
            </div>

            <div className="p-3 bg-elegant-cyan/5 border border-elegant-cyan/15 rounded-xl text-[10px] leading-relaxed italic text-elegant-text/80">
              "Xoay pha trong phân rã rò rỉ hoạt động tương tự như một mạch dao động RLC (điện cảm - điện dung - điện trở) thu nhỏ ngay trên màng tế bào số phức, tối ưu hóa băng thông thời gian cực lớn."
            </div>
          </div>

        </div>
      </div>

      {/* INTERACTIVE MODEL ARCHITECTURE SECTION */}
      <InteractiveArchitecture />

      {/* BENTO-GRID COMPARISON MATRIX */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
        <div className="border-b border-elegant-muted/10 pb-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
            <span className="w-2.5 h-2.5 rounded-full bg-elegant-cyan" />
            <span>Bảng So Sánh Đặc Tính Kỹ Thuật Chi Tiết</span>
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-elegant-text/90 border-collapse">
            <thead>
              <tr className="border-b border-elegant-muted/15 font-mono text-elegant-muted text-[10px] uppercase font-bold bg-elegant-bg/40">
                <th className="py-3 px-4">Đặc tính So Sánh</th>
                <th className="py-3 px-4 text-rose-400">SNN Miền Thực (Traditional)</th>
                <th className="py-3 px-4 text-elegant-cyan">Complex-valued SNN (cLIF)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-elegant-muted/10">
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Biến Điện Thế Màng ($U_m$)</td>
                <td className="py-3.5 px-4 text-rose-400/90 font-mono">Số thực ({"$V \\in \\mathbb{R}$"})</td>
                <td className="py-3.5 px-4 text-elegant-cyan font-mono">Số phức ({"$U = V + jY \\in \\mathbb{C}$"})</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Cách mã hóa thông tin</td>
                <td className="py-3.5 px-4">Mã hóa tần số (Rate) hoặc mốc thời gian tuyệt đối.</td>
                <td className="py-3.5 px-4 text-elegant-cyan/90 font-semibold">Mã hóa cả Biên độ (Amplitude) và Pha tương đối (Phase Angle).</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Phép toán tại Synapse</td>
                <td className="py-3.5 px-4">Phép nhân vô hướng thực: $W \cdot S(t)$</td>
                <td className="py-3.5 px-4 text-elegant-cyan/90 font-mono">Phép nhân phức (chuyển đổi pha tự nhiên): {"$W \\cdot S(t) \\in \\mathbb{C}$"}</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Hiện tượng tích tụ điện thế</td>
                <td className="py-3.5 px-4">Tích lũy tịnh tiến (đơn thuần tăng/giảm biên độ).</td>
                <td className="py-3.5 px-4 text-elegant-cyan/90 font-semibold">Giao thoa sóng pha (Cộng hưởng hoặc Triệt tiêu vi sai).</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Độ bền chống nhiễu dẫn nạp</td>
                <td className="py-3.5 px-4">Kém. Sai lệch conductance vật lý làm lệch hướng tính toán 1D trực tiếp.</td>
                <td className="py-3.5 px-4 text-emerald-400 font-semibold">Cực tốt. Sai lệch dẫn nạp memristor PCM phân tán đều 360°, tự triệt tiêu ngẫu nhiên.</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold font-display">Khả năng ánh xạ memristor</td>
                <td className="py-3.5 px-4">Cần nhiều linh kiện vi sai ghép cặp phức tạp để giả lập số âm.</td>
                <td className="py-3.5 px-4 text-emerald-400 font-semibold">Tự nhiên. Ánh xạ trực tiếp góc dẫn nạp vật lý thành Pha và Biên độ của mảng PCM.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

function InteractiveArchitecture() {
  const [selectedArch, setSelectedArch] = useState<'traditional' | 'complex'>('complex');
  const [selectedNode, setSelectedNode] = useState<string>('soma');
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // node details for Complex-valued SNN
  const complexNodeDetails: Record<string, { title: string; subtitle: string; formula: string; explanation: string; hardware: string; icon: React.ReactNode }> = {
    inputs: {
      title: "Luồng Xung Đầu Vào (Input Spikes - s_i(t))",
      subtitle: "Sự kiện nhị phân kết hợp mã hóa thời gian vi mô",
      formula: "s_i(t) \\in \\{0, 1\\} \\quad \\text{hoặc} \\quad s_i(t) = e^{j\\theta_i}",
      explanation: "Trong mạng nổ xung số phức nâng cao, bản thân xung có thể mang một góc pha góc riêng biệt đại diện cho mốc thời gian vi mô (sub-millisecond timing) hoặc chu kỳ nhịp sinh học tương đối, giúp tăng mật độ biểu diễn thông tin lên nhiều lần so với chỉ đếm số lượng xung.",
      hardware: "Tín hiệu điện áp xoay chiều kích hoạt ngắn hạn, được đồng bộ hóa pha nhờ mạch dao động biên trên chip neuromorphic.",
      icon: <Activity className="w-5 h-5 text-elegant-cyan" />
    },
    weights: {
      title: "Trọng Số Synapse Số Phức (Complex Weights - W_i)",
      subtitle: "Điều biến biên độ và xoay pha tức thời",
      formula: "W_i = M_i \\cdot e^{j\\phi_i} = W_{re} + j W_{im}",
      explanation: "Trọng số khớp thần kinh (Synapse) là một đại lượng số phức. Khi xung điện đi qua, cường độ của nó được co giãn theo biên độ M_i, đồng thời mốc thời gian xung bị dịch đi một góc pha \\phi_i (làm trễ hoặc đẩy sớm dòng điện đầu vào). Đây là cơ chế mô phỏng hoàn hảo tính chất trễ pha tự nhiên của synapse sinh học.",
      hardware: "Ánh xạ trực tiếp lên hai phần tử memristor PCM (Phase-Change Memory) mắc song song biểu diễn phần Thực và phần Ảo (Re & Im), hoặc điều biến góc dẫn nạp vật lý trong thiết bị nhớ đa mức.",
      icon: <Sliders className="w-5 h-5 text-amber-400" />
    },
    dendrite: {
      title: "Tích Hợp Nhánh Cây (Dendritic Summation - \\Sigma)",
      subtitle: "Giao thoa sóng pha vi sai trên mạng tản nhiệt",
      formula: "I_{syn}(t) = \\sum_{i} W_i \\cdot s_i(t) \\in \\mathbb{C}",
      explanation: "Tại nhánh cây (Dendrite), tất cả các luồng dòng điện phức được gom và cộng lại với nhau dưới dạng vector 2D trên mặt phẳng số phức. Nếu hai luồng tín hiệu đầu vào đồng pha (in-phase), chúng sẽ cộng hưởng mạnh mẽ làm điện thế màng tăng vọt. Ngược lại, nếu lệch pha 180° (out-of-phase), chúng sẽ tự triệt tiêu lẫn nhau.",
      hardware: "Phép cộng dòng điện Kirchhoff (Kirchhoff's Current Law) diễn ra tự nhiên tại điểm nút dây dẫn chung của phần Thực và phần Ảo.",
      icon: <GitCompare className="w-5 h-5 text-elegant-cyan" />
    },
    soma: {
      title: "Điện Thế Màng cLIF (Membrane Potential - U(t))",
      subtitle: "Hai chiều tích lũy năng lượng & nhịp dao động",
      formula: "U(t) = U(t-1) \\cdot \\lambda e^{j\\omega} + I_{syn}(t)",
      explanation: "Điện thế màng của neuron cLIF là một số phức U(t) = V(t) + j Y(t). Biến này tích lũy cả phần thực (năng lượng điện tích lũy) lẫn phần ảo (pha dao động nội sinh). Khi rò rỉ (leak), điện thế màng sẽ tự động rò rỉ thu hẹp biên độ đồng thời xoay pha liên tục góc \\omega tạo thành quỹ đạo xoắn ốc tuyệt đẹp.",
      hardware: "Mạch tích hợp tương tự (Analog) sử dụng tụ điện cho phần tích lũy thực/ảo và cuộn cảm/mạch tích hợp hồi tiếp để duy trì pha dao động.",
      icon: <Zap className="w-5 h-5 text-indigo-400" />
    },
    threshold: {
      title: "Bộ So Ngưỡng Vòng Tròn (Circular Thresholding - |U| \\ge \\theta)",
      subtitle: "Nổ xung hai chiều và hồi phục trạng thái",
      formula: "|U(t)| \\ge \\theta_{thres} \\implies S_{out}(t) = 1, \\ U(t) \\leftarrow 0",
      explanation: "Khác hoàn toàn với SNN truyền thống, cLIF neuron thiết lập một đường biên tròn bán kính \\theta trên mặt phẳng số phức Gauss. Khi biên độ điện thế màng |U(t)| vượt khỏi vòng tròn an toàn này, neuron lập tiếp phát xung phát xạ (spike) ra sợi trục Axon.",
      hardware: "Bộ so sánh dòng rò siêu dẫn tương tự (Analog Comparator) cực nhạy kích hoạt khi biên độ vector vượt quá ngưỡng phân cực điện áp tròn.",
      icon: <Sparkles className="w-5 h-5 text-emerald-400" />
    }
  };

  // node details for Traditional SNN
  const traditionalNodeDetails: Record<string, { title: string; subtitle: string; formula: string; explanation: string; hardware: string; icon: React.ReactNode }> = {
    inputs: {
      title: "Luồng Xung Đầu Vào (Input Spikes - s_i(t))",
      subtitle: "Sự kiện nhị phân đơn giản 1D",
      formula: "s_i(t) \\in \\{0, 1\\}",
      explanation: "Trong SNN truyền thống, đầu vào là một chuỗi xung nhị phân diễn ra theo thời gian. Trạng thái tại thời điểm t chỉ là 0 (không có xung) hoặc 1 (có xung). Giá trị này hoàn toàn không chứa pha góc hay thông tin xoay vòng thời gian vi mô.",
      hardware: "Các xung điện áp DC ngắn hạn đơn giản được phát ra từ nguồn kích hoạt cảm biến.",
      icon: <Activity className="w-5 h-5 text-rose-400" />
    },
    weights: {
      title: "Trọng Số Synapse Miền Thực (Real Weights - W_i)",
      subtitle: "Co giãn biên độ tuyến tính vô hướng",
      formula: "W_i \\in \\mathbb{R}",
      explanation: "Trọng số khớp thần kinh là một số thực vô hướng. Khi một xung nhị phân đi qua, nó chỉ đơn thuần nhân biên độ xung với giá trị thực W_i. Giá trị dương thể hiện kích thích thần kinh, giá trị âm thể hiện ức chế thần kinh. Không có khả năng dịch chuyển mốc thời gian hay lệch pha.",
      hardware: "Được cấu hình bằng một transistor đơn lẻ hoặc một thiết bị memristor đơn mức điện trở biểu diễn độ dẫn (conductance).",
      icon: <Sliders className="w-5 h-5 text-amber-400" />
    },
    dendrite: {
      title: "Tích Hợp Nhánh Cây (Dendritic Summation - \\Sigma)",
      subtitle: "Phép cộng số thực vô hướng trực tiếp",
      formula: "I_{syn}(t) = \\sum_{i} W_i \\cdot s_i(t) \\in \\mathbb{R}",
      explanation: "Tại nhánh cây, các xung tích hợp được cộng trực tiếp với nhau dưới dạng số đại số vô hướng 1 chiều. Không có khái niệm giao thoa sóng pha, do đó không thể tự động triệt tiêu các tín hiệu lệch pha hay lọc nhiễu nền vi mô bằng phương pháp vi sai tự nhiên.",
      hardware: "Dòng điện được gom tự nhiên tại nút dẫn thông qua định luật Kirchhoff, nhưng chỉ thao tác trên một kênh dòng thực duy nhất.",
      icon: <GitCompare className="w-5 h-5 text-elegant-cyan" />
    },
    soma: {
      title: "Điện Thế Màng LIF (Membrane Potential - V(t))",
      subtitle: "Tích lũy năng lượng 1D tuyến tính",
      formula: "V(t) = V(t-1) \\cdot \\lambda + I_{syn}(t)",
      explanation: "Điện thế màng của neuron LIF truyền thống là một số thực 1 chiều V(t). Khi không có xung điện kích thích bên ngoài đi vào, năng lượng tích tụ sẽ tự động rò rỉ tuyến tính lùi dần về 0 theo thời gian dựa vào hệ số rò rỉ \\lambda. Neuron hoàn toàn không có trạng thái tự dao động pha.",
      hardware: "Mạch tích tụ đơn giản sử dụng một tụ điện mắc song song với một điện trở rò rỉ (RC circuit) về đất.",
      icon: <Zap className="w-5 h-5 text-indigo-400" />
    },
    threshold: {
      title: "Bộ So Ngưỡng Tuyến Tính (Linear Thresholding - V \\ge \\theta)",
      subtitle: "So sánh một chiều và xả điện thế",
      formula: "V(t) \\ge \\theta_{thres} \\implies S_{out}(t) = 1, \\ V(t) \\leftarrow V_{reset}",
      explanation: "Cơ chế so ngưỡng diễn ra trên trục số thực 1D. Khi điện thế tích tụ V(t) vượt ngưỡng \\theta_{thres}, neuron lập tức kích hoạt phát xung điện ra ngoài qua sợi trục Axon và xả cạn điện thế màng về mức nghỉ V_reset (thường là 0).",
      hardware: "Sử dụng một bộ so sánh điện áp cơ bản (Voltage Comparator) kích hoạt ở mức điện thế chuẩn cố định.",
      icon: <Sparkles className="w-5 h-5 text-emerald-400" />
    }
  };

  const currentDetails = selectedArch === 'complex' ? complexNodeDetails : traditionalNodeDetails;
  const selectedData = currentDetails[selectedNode] || currentDetails['soma'];

  // Style attributes based on selected architecture
  const themeColor = selectedArch === 'complex' ? '#00f2fe' : '#ef4444';
  const themeGlow = selectedArch === 'complex' ? 'url(#cyan-glow)' : 'url(#rose-glow)';
  const themeSomaStroke = selectedArch === 'complex' ? '#6366f1' : '#f43f5e';
  const themeSomaText = selectedArch === 'complex' ? '#a5b4fc' : '#fca5a5';

  return (
    <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-6 shadow-xl">
      <div className="border-b border-elegant-muted/10 pb-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2 font-display">
            <span className={`w-2.5 h-2.5 rounded-full animate-pulse`} style={{ backgroundColor: themeColor }} />
            <span>Sơ Đồ Kiến Trúc Neuron Tương Tác: SNN vs cLIF</span>
          </h3>
          <span className="text-xs text-elegant-muted font-mono">Bản đồ cấu trúc và luồng tích tụ phần tử</span>
        </div>
        
        {/* Architecture Switcher */}
        <div className="bg-elegant-bg/85 p-1 rounded-lg border border-elegant-muted/15 flex gap-1 self-stretch sm:self-auto">
          <button
            onClick={() => {
              setSelectedArch('traditional');
              // trigger refresh of selected details
            }}
            className={`px-3 py-1 text-xs font-bold rounded-md transition-all duration-200 flex-1 sm:flex-none ${
              selectedArch === 'traditional'
                ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                : 'text-elegant-muted hover:text-elegant-text border border-transparent'
            }`}
          >
            SNN Thường (1D LIF)
          </button>
          <button
            onClick={() => {
              setSelectedArch('complex');
            }}
            className={`px-3 py-1 text-xs font-bold rounded-md transition-all duration-200 flex-1 sm:flex-none ${
              selectedArch === 'complex'
                ? 'bg-elegant-cyan/10 text-elegant-cyan border border-elegant-cyan/20'
                : 'text-elegant-muted hover:text-elegant-text border border-transparent'
            }`}
          >
            SNN Số Phức (2D cLIF)
          </button>
        </div>
      </div>

      <p className="text-xs text-elegant-text/80 leading-relaxed">
        Nhấp chuột vào bất kỳ bộ phận nào của mô hình neuron dưới đây để so sánh chi tiết toán học giải tích, nguyên lý hoạt động sinh học và cách thức thiết kế trên phần cứng chip vi mạch (memristor) của từng loại!
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* INTERACTIVE DIAGRAM SIDE (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col justify-center bg-elegant-bg/50 p-4 rounded-xl border border-elegant-muted/10 relative overflow-hidden min-h-[350px]">
          {/* Subtle backgrounds */}
          <div className="absolute inset-0 bg-radial-gradient from-elegant-cyan/2 to-transparent pointer-events-none" />
          
          <div className="w-full flex justify-center">
            <svg width="100%" height="320" viewBox="0 0 540 320" className="max-w-[540px] select-none">
              
              {/* Definition of glow filters for interactive highlights */}
              <defs>
                <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="rose-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="indigo-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Connections/Lines with active pulses */}
              <g stroke="#1e293b" strokeWidth="2">
                {/* Inputs to Weights */}
                <line x1="60" y1="60" x2="150" y2="60" strokeDasharray="3 3" />
                <line x1="60" y1="160" x2="150" y2="160" strokeDasharray="3 3" />
                <line x1="60" y1="260" x2="150" y2="260" strokeDasharray="3 3" />

                {/* Weights to Dendrite */}
                <path d="M 190 60 Q 255 60 255 160" stroke={selectedArch === 'complex' ? '#ef4444' : '#f43f5e'} strokeWidth="1.5" fill="none" opacity="0.6" />
                <line x1="190" y1="160" x2="240" y2="160" stroke="#a1a1aa" strokeWidth="1.5" />
                <path d="M 190 260 Q 255 260 255 160" stroke={selectedArch === 'complex' ? '#6366f1' : '#f43f5e'} strokeWidth="1.5" fill="none" opacity="0.6" />

                {/* Dendrite to Soma */}
                <line x1="280" y1="160" x2="350" y2="160" stroke={themeColor} strokeWidth="2" className="animate-pulse" />

                {/* Soma to Threshold */}
                <line x1="430" y1="160" x2="480" y2="160" stroke={themeColor} strokeWidth="2" />
              </g>

              {/* Animated particles flowing */}
              <circle r="3" fill={themeColor}>
                <animateMotion 
                  path="M 280 160 L 350 160" 
                  dur="2s" 
                  repeatCount="indefinite" 
                />
              </circle>
              <circle r="2.5" fill={selectedArch === 'complex' ? '#ef4444' : '#f43f5e'}>
                <animateMotion 
                  path="M 190 60 Q 255 60 255 160" 
                  dur="2.5s" 
                  repeatCount="indefinite" 
                />
              </circle>
              <circle r="2.5" fill={selectedArch === 'complex' ? '#6366f1' : '#f43f5e'}>
                <animateMotion 
                  path="M 190 260 Q 255 260 255 160" 
                  dur="1.8s" 
                  repeatCount="indefinite" 
                />
              </circle>

              {/* SECTION: INPUT SPIKES (Left Nodes) */}
              <g 
                className="cursor-pointer group" 
                onClick={() => setSelectedNode('inputs')}
                onMouseEnter={() => setHoveredNode('inputs')}
                onMouseLeave={() => setHoveredNode(null)}
              >
                {/* Interactivity Indicator Backing */}
                <circle cx="60" cy="160" r="45" fill="none" stroke={selectedNode === 'inputs' ? '#f43f5e' : 'transparent'} strokeWidth="1.5" strokeDasharray="3 3" />
                
                {/* 3 Input Nodes */}
                <circle cx="60" cy="60" r="14" fill="#1e1b4b" stroke="#f43f5e" strokeWidth="2" className="transition-all duration-200 group-hover:scale-105" />
                <text x="60" y="64" fill="#f43f5e" className="text-[10px] font-mono font-bold text-center" textAnchor="middle">s₁</text>

                <circle cx="60" cy="160" r="14" fill="#1e1b4b" stroke="#f43f5e" strokeWidth="2" className="transition-all duration-200 group-hover:scale-105" />
                <text x="60" y="164" fill="#f43f5e" className="text-[10px] font-mono font-bold text-center" textAnchor="middle">s₂</text>

                <circle cx="60" cy="260" r="14" fill="#1e1b4b" stroke="#f43f5e" strokeWidth="2" className="transition-all duration-200 group-hover:scale-105" />
                <text x="60" y="264" fill="#f43f5e" className="text-[10px] font-mono font-bold text-center" textAnchor="middle">s₃</text>

                <text x="60" y="30" fill="#f43f5e" className="text-[10px] font-mono font-bold uppercase tracking-wider" textAnchor="middle">
                  {selectedArch === 'complex' ? "Xung Vào" : "Xung 1D"}
                </text>
              </g>

              {/* SECTION: SYNAPSE WEIGHTS (W_i) */}
              <g 
                className="cursor-pointer group" 
                onClick={() => setSelectedNode('weights')}
                onMouseEnter={() => setHoveredNode('weights')}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <rect x="130" y="30" width="80" height="260" rx="6" fill="none" stroke={selectedNode === 'weights' ? '#f59e0b' : 'transparent'} strokeWidth="1.5" strokeDasharray="3 3" />

                {/* Synaptic multiplier boxes */}
                <rect x="145" y="45" width="40" height="30" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1.5" className="transition-all duration-200 group-hover:fill-slate-800" />
                <text x="165" y="63" fill="#f59e0b" className="text-[10px] font-mono" textAnchor="middle">W₁</text>

                <rect x="145" y="145" width="40" height="30" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1.5" className="transition-all duration-200 group-hover:fill-slate-800" />
                <text x="165" y="163" fill="#f59e0b" className="text-[10px] font-mono" textAnchor="middle">W₂</text>

                <rect x="145" y="245" width="40" height="30" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1.5" className="transition-all duration-200 group-hover:fill-slate-800" />
                <text x="165" y="263" fill="#f59e0b" className="text-[10px] font-mono" textAnchor="middle">W₃</text>

                <text x="165" y="25" fill="#f59e0b" className="text-[10px] font-mono font-bold uppercase tracking-wider" textAnchor="middle">
                  {selectedArch === 'complex' ? "Trọng Số" : "Trọng Số Thực"}
                </text>
              </g>

              {/* SECTION: DENDRITE SUMMATION */}
              <g 
                className="cursor-pointer group" 
                onClick={() => setSelectedNode('dendrite')}
                onMouseEnter={() => setHoveredNode('dendrite')}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <circle cx="260" cy="160" r="28" fill={selectedArch === 'complex' ? "#09333f" : "#2a1515"} stroke={themeColor} strokeWidth="2.5" 
                  filter={selectedNode === 'dendrite' || hoveredNode === 'dendrite' ? themeGlow : 'none'}
                  className="transition-all duration-200" 
                />
                <text x="260" y="165" fill={themeColor} className="text-base font-bold text-center" textAnchor="middle">∑</text>
                <text x="260" y="120" fill={themeColor} className="text-[10px] font-mono font-bold uppercase tracking-wider" textAnchor="middle">Cộng ∑</text>
              </g>

              {/* SECTION: SOMA (cLIF MEMBRANE) */}
              <g 
                className="cursor-pointer group" 
                onClick={() => setSelectedNode('soma')}
                onMouseEnter={() => setHoveredNode('soma')}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <rect x="340" y="110" width="80" height="100" rx="10" fill={selectedArch === 'complex' ? "#1a1c3d" : "#1a1217"} stroke={themeSomaStroke} strokeWidth="2.5"
                  filter={selectedNode === 'soma' || hoveredNode === 'soma' ? 'url(#indigo-glow)' : 'none'}
                  className="transition-all duration-200" 
                />
                <text x="380" y="150" fill={themeSomaText} className="text-xs font-bold font-display" textAnchor="middle">
                  {selectedArch === 'complex' ? "cLIF SOMA" : "LIF SOMA"}
                </text>
                <text x="380" y="172" fill={selectedArch === 'complex' ? "#818cf8" : "#f87171"} className="text-[9px] font-mono font-semibold" textAnchor="middle">
                  {selectedArch === 'complex' ? "U(t) = V+jY" : "V(t) ∈ ℝ"}
                </text>
                
                {/* Spiral miniature icon inside soma (only for complex) */}
                {selectedArch === 'complex' ? (
                  <path d="M 370 185 Q 380 185 380 180 Q 380 176 374 176 Q 370 176 371 181 Q 372 184 376 183" fill="none" stroke="#6366f1" strokeWidth="1" />
                ) : (
                  <line x1="365" y1="185" x2="395" y2="185" stroke="#f43f5e" strokeWidth="1" strokeDasharray="2 1" />
                )}

                <text x="380" y="98" fill={themeSomaStroke} className="text-[10px] font-mono font-bold uppercase tracking-wider" textAnchor="middle">
                  {selectedArch === 'complex' ? "Màng cLIF" : "Màng LIF 1D"}
                </text>
              </g>

              {/* SECTION: THRESHOLD COMPARATOR */}
              <g 
                className="cursor-pointer group" 
                onClick={() => setSelectedNode('threshold')}
                onMouseEnter={() => setHoveredNode('threshold')}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <circle cx="475" cy="160" r="22" fill={selectedArch === 'complex' ? "#064e3b" : "#451a03"} stroke={selectedArch === 'complex' ? "#10b981" : "#f43f5e"} strokeWidth="2"
                  filter={selectedNode === 'threshold' || hoveredNode === 'threshold' ? themeGlow : 'none'}
                  className="transition-all duration-200" 
                />
                <text x="475" y="164" fill={selectedArch === 'complex' ? "#34d399" : "#fca5a5"} className="text-[10px] font-bold font-mono" textAnchor="middle">
                  {selectedArch === 'complex' ? "|U| ≥ θ" : "V ≥ θ"}
                </text>
                <text x="475" y="125" fill={selectedArch === 'complex' ? "#10b981" : "#f43f5e"} className="text-[10px] font-mono font-bold uppercase tracking-wider" textAnchor="middle">
                  {selectedArch === 'complex' ? "Ngưỡng Tròn" : "Ngưỡng 1D"}
                </text>

                {/* Final Spike Out Axon */}
                <line x1="497" y1="160" x2="530" y2="160" stroke={selectedArch === 'complex' ? "#10b981" : "#f43f5e"} strokeWidth="2.5" />
                <polygon points="527,157 534,160 527,163" fill={selectedArch === 'complex' ? "#10b981" : "#f43f5e"} />
                <text x="525" y="180" fill={selectedArch === 'complex' ? "#34d399" : "#fca5a5"} className="text-[9px] font-mono">S_out</text>
              </g>

            </svg>
          </div>

          <div className="flex justify-between items-center px-4 pt-1 text-[10px] text-elegant-muted border-t border-elegant-muted/5 font-mono">
            <span>💡 Nhấp vào các bộ phận để xem chi tiết</span>
            <span className="capitalize font-bold" style={{ color: themeColor }}>Bộ phận: {hoveredNode || selectedNode}</span>
          </div>
        </div>

        {/* DETAILED EDUCATIONAL CARD SIDE (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col justify-between bg-elegant-bg/80 border border-elegant-muted/15 p-5 rounded-xl space-y-4">
          <div className="space-y-3.5">
            <div className="flex items-center gap-2.5 pb-2.5 border-b border-elegant-muted/10">
              <div className="p-1.5 bg-elegant-bg border border-elegant-muted/15 rounded-lg">
                {selectedData.icon}
              </div>
              <div>
                <h4 className="text-sm font-bold text-white font-display leading-tight">{selectedData.title}</h4>
                <p className="text-[10.5px] text-elegant-muted font-mono leading-none mt-1">{selectedData.subtitle}</p>
              </div>
            </div>

            {/* LaTeX Math block */}
            <div className="p-2.5 bg-[#070b10] border border-elegant-muted/10 rounded-lg text-center">
              <span className="text-[11px] font-mono font-semibold block" style={{ color: themeColor }}>Công thức Toán giải tích:</span>
              <div className="text-xs font-mono text-white/95 mt-1.5 font-bold">
                {"$$ " + selectedData.formula + " $$"}
              </div>
            </div>

            {/* Explanations */}
            <div className="space-y-2 text-xs">
              <span className="text-[11px] font-mono font-bold text-elegant-muted block uppercase tracking-wider">Cơ chế hoạt động:</span>
              <p className="text-elegant-text/90 leading-relaxed text-[11.5px]">
                {selectedData.explanation}
              </p>
            </div>
          </div>

          {/* Hardware Memristor Mapping */}
          <div className="border p-3 rounded-lg space-y-1" style={{ backgroundColor: `${themeColor}05`, borderColor: `${themeColor}20` }}>
            <span className="text-[10px] font-mono font-black uppercase tracking-wider block" style={{ color: themeColor }}>Tương thích Phần cứng Neuromorphic:</span>
            <p className="text-[11px] text-elegant-text/80 leading-normal">
              {selectedData.hardware}
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
