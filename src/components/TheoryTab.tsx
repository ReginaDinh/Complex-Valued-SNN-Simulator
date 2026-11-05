import { useState } from 'react';
import { motion } from 'motion/react';
import { 
  Layers, Zap, BookOpen, AlertTriangle, Info, Sparkles
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { InlineMath, BlockMath } from 'react-katex';

export default function TheoryTab() {
  const { language } = useLanguage();
  const isEn = language === 'en';

  // Interactive Phasor Multiplier State
  const [inPhase, setInPhase] = useState<number>(0.5); // rad
  const [wPhase, setWPhase] = useState<number>(1.2);   // rad
  const [inAmp, setInAmp] = useState<number>(1.0);
  const [wAmp, setWAmp] = useState<number>(0.8);

  // Math output
  const resPhase = inPhase + wPhase;
  const resAmp = inAmp * wAmp;
  const cosDiff = Math.cos(resPhase);

  // Determine interference type
  let interferenceLabel = "";
  let interferenceColor = "";
  if (cosDiff > 0.3) {
    interferenceLabel = isEn ? "CONSTRUCTIVE (Excitation)" : "GIAO THOA ĐỒNG PHA (Kích thích)";
    interferenceColor = "text-emerald-400";
  } else if (cosDiff < -0.3) {
    interferenceLabel = isEn ? "DESTRUCTIVE (Inhibition)" : "GIAO THOA NGƯỢC PHA (Triệt tiêu)";
    interferenceColor = "text-rose-400";
  } else {
    interferenceLabel = isEn ? "ORTHOGONAL (Neutral / Quadrature)" : "TRỰC GIAO (Trung tính)";
    interferenceColor = "text-amber-400";
  }

  // Trigonometric coordinates for SVG rendering (center = 100, 100)
  const cx = 100;
  const cy = 100;
  const scale = 75; // max radius for amp=1.0

  const inX = cx + inAmp * scale * Math.cos(inPhase);
  const inY = cy - inAmp * scale * Math.sin(inPhase);

  const wX = cx + wAmp * scale * Math.cos(wPhase);
  const wY = cy - wAmp * scale * Math.sin(wPhase);

  const resX = cx + resAmp * scale * Math.cos(resPhase);
  const resY = cy - resAmp * scale * Math.sin(resPhase);

  return (
    <motion.div
      key="theory"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="grid grid-cols-1 lg:grid-cols-3 gap-6"
    >
      {/* Left & Middle Column: Detailed analysis */}
      <div className="lg:col-span-2 space-y-6">
        
        {/* Introduction section */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
          <div className="flex items-center gap-2 text-elegant-cyan">
            <BookOpen className="w-5 h-5" />
            <h2 className="text-lg font-bold font-display">{isEn ? 'Complex-valued SNN Model' : 'Mô hình SNN trong Miền Phức (Complex-valued SNN)'}</h2>
          </div>
          
          <p className="text-sm text-elegant-text/80 leading-relaxed">
            {isEn ? "Traditional Spiking Neural Networks (SNNs) typically use real values for synaptic weights and membrane potentials. The state-of-the-art Complex-Valued SNN introduces " : "Mạng nơ-ron xung (SNN) truyền thống thường sử dụng các giá trị thực cho trọng số synapse và điện thế màng. Các nghiên cứu SOTA (State-of-the-art) gần đây giới thiệu "}
            <strong className="text-elegant-cyan font-mono">{isEn ? <>Complex numbers (<InlineMath math="U = V + jY \in \mathbb{C}" />)</> : <>Số phức (<InlineMath math="U = V + jY \in \mathbb{C}" />)</>}</strong>
            {isEn ? " to concurrently encode spike timing (Phase) and spike magnitude (Amplitude) directly in a single pass." : " để mã hóa đồng thời độ trễ thời gian (Pha) và biên độ xung (Amplitude) trực tiếp ngay trong quá trình tính toán."}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-elegant-muted/10">
            {/* Column 1: Math properties */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-rose-400 font-bold text-sm border-b border-rose-400/15 pb-1.5">
                <AlertTriangle className="w-4 h-4" />
                <span>{isEn ? "Challenges in Traditional SNNs" : "Thách Thức SNN Hiện Tại"}</span>
              </div>
              <div className="space-y-3.5">
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">1. {isEn ? "Rate-coding is Inefficient" : "Mã hóa tần số (Rate-coding) chậm"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Real-valued networks need multiple time steps to accumulate information, increasing latency and power consumption." : "Các mạng giá trị thực cần rất nhiều bước thời gian (time-steps) để tích lũy thông tin, gây tốn năng lượng và tăng độ trễ."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">2. {isEn ? "Mathematical Instability" : "Bất Ổn Cấu Trúc Toán Học"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Backpropagation through time (BPTT) with complex variables often violates Cauchy-Riemann equations, making Surrogate Gradients unstable." : "Việc đạo hàm Backprop qua miền phức (BPTT) thường xuyên vi phạm phương trình vi phân (Cauchy-Riemann), làm cho việc tính Gradient thay thế (Surrogate Gradient) gặp nhiều bất ổn."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">3. {isEn ? "Hardware Compatibility Gap" : "Thiếu Tương Thích Phần Cứng (Hardware Gap)"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Current commercial neuromorphic chips (Loihi, TrueNorth) only support real/binary signals, lacking native complex processing units." : "Các chip Neuromorphic thương mại hiện tại (Loihi, TrueNorth) chỉ hỗ trợ tín hiệu thực hoặc nhị phân, chưa hỗ trợ trực tiếp đơn vị xử lý số phức nguyên bản."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">4. {isEn ? "Complex Input Encoding" : "Phức Tạp Trong Mã Hóa Đầu Vào"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Transforming continuous physical signals (audio, light) to complex spikes requires delicate Analog-to-Spike encoders." : "Việc biến đổi tín hiệu vật lý liên tục (âm thanh, ánh sáng) sang chuỗi xung phức yêu cầu các bộ mã hóa Analog-to-Spike phức tạp và nhạy cảm với nhiễu biên."}
                  </p>
                </div>
              </div>
            </div>

            {/* Column 3: Improvement Proposals */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-elegant-cyan font-bold text-sm border-b border-elegant-cyan/15 pb-1.5">
                <Zap className="w-4 h-4" />
                <span>{isEn ? "SOTA Improvement Proposals" : "Đề Xuất Cải Tiến SOTA"}</span>
              </div>
              <div className="space-y-3.5">
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">1. {isEn ? "Decoupled SNN Architecture" : "Kiến Trúc Tách Luồng (Decoupled SNN)"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Designing networks that separately process Phase and Amplitude on parallel real streams to run on current neuromorphic chips." : "Thiết kế mạng thần kinh tách biệt xử lý Pha và Biên độ riêng lẻ trên hai luồng thực song song, cho phép chạy trực tiếp trên các chip neuromorphic thương mại hiện nay."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">2. {isEn ? "Karatsuba Multiplication Algorithm" : "Thuật Toán Nhân Nhanh Karatsuba"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Applying the Karatsuba method simplifies complex Synaptic multiplication from 4 down to 3 real multiplications, instantly saving 25% power consumption." : "Áp dụng phương pháp Karatsuba giúp tối giản hóa phép nhân Synaptic phức từ 4 xuống còn 3 phép nhân thực, tiết kiệm lập tức 25% điện năng tiêu thụ ở cấp độ transistor."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">3. {isEn ? "Von Mises Adaptive Surrogate Gradient" : "Gradient Thay Thế Thích Ứng Von Mises"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Using the Von Mises probability distribution (specific for circular phase data) as a Surrogate Gradient ensures mathematical stability." : "Ứng dụng phân phối xác suất Von Mises (đặc thù cho dữ liệu vòng tuần hoàn pha) làm hàm Surrogate Gradient, đảm bảo ổn định toán học khi lan truyền ngược qua thời gian (BPTT)."}
                  </p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-200">4. {isEn ? "Phase-Attention Mechanism" : "Bộ Tự Chú Ý Đồng Bộ Pha (Phase-Attention)"}</h4>
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn ? "Integrating an Attention mechanism based on phase synchronization allows neurons to actively filter and lock onto crucial frequencies." : "Tích hợp cơ chế Attention dựa trên mức độ đồng bộ pha, giúp neuron chủ động lọc và khóa các tần số mang thông tin quan trọng, nâng cao hiệu năng phân loại phổ lên mức tối đa."}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* INTERACTIVE PHASE INTERFERENCE PLAYGROUND CARD */}
        <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-6 shadow-xl">
          <div className="flex items-center gap-2 text-elegant-cyan border-b border-elegant-muted/10 pb-3">
            <Sparkles className="w-5 h-5 text-elegant-cyan" />
            <div>
              <h3 className="text-base font-bold text-white font-display">
                {isEn ? "Interactive Synaptic Phase Multiplier" : "Bộ Nhân Pha Synaptic Tương Tác"}
              </h3>
              <p className="text-[11px] text-elegant-text/60 mt-0.5">
                {isEn ? "Discover how synaptic weight phase shifts input phase to trigger constructive (excitation) or destructive (inhibition) signals." : "Khám phá cách pha của trọng số xoay pha đầu vào nhằm tạo ra sự cộng hưởng đồng pha hoặc triệt tiêu ngược pha."}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            
            {/* SVG Vector Visualizer */}
            <div className="flex flex-col items-center">
              <div className="relative bg-[#070b12] rounded-xl p-4 border border-elegant-muted/10 w-full max-w-[240px] aspect-square flex items-center justify-center shadow-inner">
                <svg width="200" height="200" className="overflow-visible">
                  {/* Grid circle scales */}
                  <circle cx={cx} cy={cy} r={scale * 0.5} fill="none" stroke="rgba(69, 162, 158, 0.1)" strokeWidth="1" strokeDasharray="3 3" />
                  <circle cx={cx} cy={cy} r={scale} fill="none" stroke="rgba(69, 162, 158, 0.15)" strokeWidth="1" />
                  
                  {/* Axis lines */}
                  <line x1="20" y1={cy} x2="180" y2={cy} stroke="rgba(69, 162, 158, 0.15)" strokeWidth="1" />
                  <line x1={cx} y1="20" x2={cx} y2="180" stroke="rgba(69, 162, 158, 0.15)" strokeWidth="1" />
                  
                  <text x="175" y={cy - 4} fill="rgba(69, 162, 158, 0.6)" fontSize="8" fontFamily="monospace">Re</text>
                  <text x={cx + 4} y="25" fill="rgba(69, 162, 158, 0.6)" fontSize="8" fontFamily="monospace">Im</text>

                  {/* Input Phasor (Cyan) */}
                  <line x1={cx} y1={cy} x2={inX} y2={inY} stroke="#06b6d4" strokeWidth="2.5" strokeLinecap="round" />
                  <circle cx={inX} cy={inY} r="4" fill="#06b6d4" />

                  {/* Weight Phasor (Amber) */}
                  <line x1={cx} y1={cy} x2={wX} y2={wY} stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" />
                  <circle cx={wX} cy={wY} r="4" fill="#f59e0b" />

                  {/* Combined Output Phasor (Purple) */}
                  <line x1={cx} y1={cy} x2={resX} y2={resY} stroke="#a855f7" strokeWidth="3" strokeLinecap="round" strokeDasharray="none" />
                  <circle cx={resX} cy={resY} r="5" fill="#a855f7" />
                </svg>

                {/* Badges for Legend inside visualizer */}
                <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[8px] font-mono text-slate-400 bg-slate-900/60 p-1.5 rounded border border-slate-800">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 bg-cyan-500 rounded-full"></span>{isEn ? "Input" : "Vào"}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 bg-amber-500 rounded-full"></span>{isEn ? "Weight" : "Trọng số"}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 bg-purple-500 rounded-full"></span>{isEn ? "Output" : "Kết quả"}</span>
                </div>
              </div>
            </div>

            {/* Sliders and Math Output */}
            <div className="space-y-4">
              <div className="bg-[#101622] p-4 rounded-xl border border-elegant-muted/10 space-y-3">
                <div className="flex justify-between items-center border-b border-elegant-muted/10 pb-2">
                  <span className="text-xs font-mono text-elegant-muted">{isEn ? "Interference State:" : "Trạng thái giao thoa:"}</span>
                  <span className={`text-xs font-extrabold tracking-wide ${interferenceColor}`}>{interferenceLabel}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div>
                    <span className="text-slate-400">{isEn ? "Output Amplitude:" : "Biên độ đầu ra:"}</span>
                    <p className="text-white font-bold text-sm mt-0.5">{resAmp.toFixed(2)}</p>
                  </div>
                  <div>
                    <span className="text-slate-400">{isEn ? "Alignment Score (cos):" : "Mức đồng bộ (cos):"}</span>
                    <p className={`font-bold text-sm mt-0.5 ${interferenceColor}`}>{cosDiff.toFixed(2)}</p>
                  </div>
                </div>
              </div>

              {/* Sliders */}
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-cyan-400">{isEn ? "Input Phase (rad):" : "Pha tín hiệu vào:"}</span>
                    <span className="text-white font-bold">{inPhase.toFixed(2)} rad</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="6.28" 
                    step="0.05" 
                    value={inPhase} 
                    onChange={e => setInPhase(parseFloat(e.target.value))} 
                    className="w-full accent-cyan-400 bg-elegant-bg h-1 rounded cursor-pointer" 
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-amber-400">{isEn ? "Weight Synaptic Phase (rad):" : "Pha của trọng số:"}</span>
                    <span className="text-white font-bold">{wPhase.toFixed(2)} rad</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="6.28" 
                    step="0.05" 
                    value={wPhase} 
                    onChange={e => setWPhase(parseFloat(e.target.value))} 
                    className="w-full accent-amber-500 bg-elegant-bg h-1 rounded cursor-pointer" 
                  />
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-slate-400">{isEn ? "Input Amp:" : "B.độ vào:"}</span>
                      <span className="text-white font-bold">{inAmp.toFixed(1)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.1" 
                      max="1.0" 
                      step="0.05" 
                      value={inAmp} 
                      onChange={e => setInAmp(parseFloat(e.target.value))} 
                      className="w-full accent-cyan-400 bg-elegant-bg h-1 rounded cursor-pointer" 
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-slate-400">{isEn ? "Weight Amp:" : "B.độ t.số:"}</span>
                      <span className="text-white font-bold">{wAmp.toFixed(1)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.1" 
                      max="1.0" 
                      step="0.05" 
                      value={wAmp} 
                      onChange={e => setWAmp(parseFloat(e.target.value))} 
                      className="w-full accent-amber-500 bg-elegant-bg h-1 rounded cursor-pointer" 
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/10 flex items-start gap-2">
            <Info className="w-4 h-4 text-elegant-cyan mt-0.5 shrink-0" />
            <p className="text-[10px] text-elegant-text/70 leading-normal">
              {isEn ? (
                <><strong>Mathematical Insight:</strong> Synaptic multiplication in CV-SNN behaves as vector rotation: <InlineMath math="U_{out} = W \cdot X" />. When the phase of the weight and the phase of the input sum up close to <InlineMath math="2\pi n" /> (cos closer to <InlineMath math="+1.0" />), they trigger massive constructive charge accumulation, generating immediate action potential (spikes) without any latency.</>
              ) : (
                <><strong>Nguyên lý toán học:</strong> Phép nhân synapse trong CV-SNN đóng vai trò xoay véc-tơ: <InlineMath math="U_{out} = W \cdot X" />. Khi tổng pha của trọng số và pha tín hiệu vào trùng khớp hướng về phía thực dương (cos tiến tới <InlineMath math="+1.0" />), thế màng sẽ cộng hưởng mạnh mẽ, kích hoạt xung đột biến tức thì mà không cần tích lũy mỏi mệt nhiều chu kỳ.</>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Right Column: Mathematical Foundations & Wirtinger Calculus */}
      <div className="space-y-6">
        {/* cLIF State & Vectorial Leakage Card */}
        <div className="bg-elegant-card border border-slate-800/80 p-5 rounded-xl space-y-3 shadow-xl">
          <h3 className="text-xs font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
            <span className="w-2 h-2 rounded-xs bg-elegant-cyan inline-block" />
            <span>{isEn ? 'cLIF State-Space Formulation' : 'Hệ Phương Trình Trạng Thái cLIF'}</span>
          </h3>
          <div className="p-3 bg-[#07090e] rounded-lg border border-slate-800/80 text-xs font-mono">
            <BlockMath math="U(t) = \beta \cdot U(t-1) + \sum_{i} W_i S_i(t)" />
            <div className="text-[10px] text-slate-400 mt-2">
              <InlineMath math="\beta = \lambda e^{j\omega} \in \mathbb{C}" /> · {isEn ? 'Complex decay rotating by angle ω per timestep' : 'Hệ số suy hao xoay pha góc ω mỗi bước'}
            </div>
          </div>
          <p className="text-xs text-elegant-text/75 leading-relaxed">
            {isEn
              ? 'Membrane potential retains short-term temporal memory as an intrinsic limit cycle oscillator, multiplying information bandwidth.'
              : 'Điện thế màng lưu giữ bộ nhớ thời gian tương đối như một máy dao động điều hòa tự nhiên, nhân rộng băng thông thông tin.'}
          </p>
        </div>

        {/* Wirtinger Gradient Calculus Card */}
        <div className="bg-elegant-card border border-slate-800/80 p-5 rounded-xl space-y-3 shadow-xl">
          <h3 className="text-xs font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
            <span className="w-2 h-2 rounded-xs bg-amber-400 inline-block" />
            <span>{isEn ? 'Wirtinger Surrogate Gradient Calculus' : 'Giải Tích Đạo Hàm Wirtinger'}</span>
          </h3>
          <div className="p-3 bg-[#07090e] rounded-lg border border-slate-800/80 text-xs font-mono">
            <BlockMath math="\frac{\partial \mathcal{L}}{\partial W^*} = \frac{1}{2}\left(\frac{\partial \mathcal{L}}{\partial W_{re}} + j\frac{\partial \mathcal{L}}{\partial W_{im}}\right)" />
          </div>
          <p className="text-xs text-elegant-text/75 leading-relaxed">
            {isEn
              ? 'Overcomes Cauchy-Riemann non-holomorphism in complex BPTT by treating W and its complex conjugate W* as independent variables, ensuring smooth gradient descent.'
              : 'Khắc phục điều kiện Cauchy-Riemann không chỉnh hình trong BPTT bằng cách xem W và liên hợp W* là các biến độc lập, giúp gradient descent hội tụ ổn định.'}
          </p>
        </div>

        {/* Circular Threshold & Soft Reset Card */}
        <div className="bg-elegant-card border border-slate-800/80 p-5 rounded-xl space-y-3 shadow-xl">
          <h3 className="text-xs font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
            <span className="w-2 h-2 rounded-xs bg-emerald-400 inline-block" />
            <span>{isEn ? 'Phase-Preserving Vectorial Reset' : 'Cơ Chế Reset Vector Giữ Pha'}</span>
          </h3>
          <div className="p-3 bg-[#07090e] rounded-lg border border-slate-800/80 text-xs font-mono">
            <BlockMath math="|U(t)| \ge \theta \implies U(t) \leftarrow U(t) - \theta \cdot e^{j \arg U(t)}" />
          </div>
          <p className="text-xs text-elegant-text/75 leading-relaxed">
            {isEn
              ? 'Subtracts threshold along the exact radial direction, preserving instantaneous spike timing and residual harmonic phase across time horizons.'
              : 'Trừ ngưỡng chính xác theo phương góc pha tức thời, bảo toàn nhịp nổ xung và pha sóng hài dư qua các bước thời gian.'}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

