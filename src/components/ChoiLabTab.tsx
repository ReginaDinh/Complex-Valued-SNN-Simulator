import { useState } from 'react';
import { motion } from 'motion/react';
import {
  Brain,
  Cpu,
  Layers,
  Activity,
  Settings,
  Zap,
  Sparkles,
  ShieldCheck,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export default function ChoiLabTab() {
  const { language } = useLanguage();
  const isEn = language === 'en';

  const [choiWeightMag, setChoiWeightMag] = useState(0.8);
  const [choiWeightPhase, setChoiWeightPhase] = useState(0.785); // 45 deg
  const [inputVoltMag, setInputVoltMag] = useState(1.0);
  const [inputVoltPhase, setInputVoltPhase] = useState(0.52); // ~30 deg
  const [pcmNoise, setPcmNoise] = useState(0.18);
  const [writeVerifyEnabled, setWriteVerifyEnabled] = useState(true);

  return (
    <motion.div
      key="choi-lab"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Header Title */}
      <div className="bg-gradient-to-r from-elegant-card to-elegant-card/40 border border-elegant-muted/15 p-6 rounded-xl shadow-xl space-y-2">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
            <Brain className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h2 className="text-xl font-bold font-display text-white">
              {isEn
                ? 'SOTA Research Prof. Shinhyun Choi (KAIST) & CV-SNN Integration'
                : 'Nghiên Cứu SOTA GS. Shinhyun Choi (KAIST) & Tích Hợp CV-SNN'}
            </h2>
            <p className="text-xs text-elegant-muted font-mono mt-0.5">
              {isEn
                ? 'Neuromorphic Memristor Array hardware breakthrough combined with Complex-Valued Spiking Neural Network algorithms'
                : 'Đột phá phần cứng Neuromorphic Memristor Array kết hợp với thuật toán Mạng thần kinh nổ xung miền phức'}
            </p>
          </div>
        </div>
      </div>

      {/* SECTION 1: GS SHINHYUN CHOI SOTA HIGHLIGHTS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
          <div className="flex items-center gap-2 text-elegant-cyan">
            <Sparkles className="w-5 h-5" />
            <span className="text-xs font-bold font-mono tracking-wider uppercase">
              NATURE ELECTRONICS (2024)
            </span>
          </div>
          <h3 className="text-md font-bold text-white font-display">
            {isEn
              ? 'On-Chip Self-Learning & Self-Correcting Chip'
              : 'Chip Tự Học & Sửa Sai On-Chip (Self-Learning & Self-Correcting)'}
          </h3>
          <p className="text-xs text-elegant-text/70 leading-relaxed">
            {isEn
              ? 'Co-developed by Prof. Shinhyun Choi & Prof. Young-Gyu Yoon (KAIST). The selector-less memristor computing chip is capable of '
              : 'Đồng phát triển bởi GS. Shinhyun Choi & GS. Young-Gyu Yoon (KAIST). Chip điện toán memristor không cần bộ định tuyến (selector-less) có khả năng '}
            <strong>
              {isEn
                ? 'automatically correcting nonlinear errors and electrical leakage'
                : 'tự động sửa lỗi phi tuyến và rò điện'}
            </strong>
            {isEn
              ? ' right on hardware. The design has demonstrated outstanding real-world performance in foreground-background video separation with accuracy comparable to ideal software simulations.'
              : ' ngay tại phần cứng. Thiết kế đã chứng minh hiệu năng thực tế xuất sắc trong nhiệm vụ tách biên tách hình động với độ chính xác tương đương mô phỏng phần mềm lý tưởng.'}
          </p>
        </div>

        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
          <div className="flex items-center gap-2 text-elegant-cyan">
            <Cpu className="w-5 h-5" />
            <span className="text-xs font-bold font-mono tracking-wider uppercase">
              NATURE (2024 Breakthrough)
            </span>
          </div>
          <h3 className="text-md font-bold text-white font-display">
            {isEn
              ? 'PCM Nanofilament Ultra-low Power'
              : 'PCM Nanofilament Siêu Tiêu Thụ Điện Thấp (Ultra-low Power)'}
          </h3>
          <p className="text-xs text-elegant-text/70 leading-relaxed">
            {isEn
              ? "Traditional Phase Change Memory (PCM) suffers from massive reset thermal energy barriers. Prof. Choi's group invented an ultra-small PCM nano-filament structure, reducing phase transition volume to nanometer scale. The result achieves "
              : 'Bộ nhớ thay đổi pha (PCM) truyền thống gặp rào cản năng lượng nhiệt reset cực lớn. Nhóm của GS. Choi đã phát minh cấu trúc PCM nano-filament siêu nhỏ, giảm thể tích vùng chuyển pha xuống mức nanomet. Kết quả đạt mức '}
            <strong>
              {isEn
                ? 'extremely low power consumption (thousands of times lower than DRAM/NAND)'
                : 'tiêu thụ năng lượng cực thấp (thấp hơn DRAM/NAND hàng nghìn lần)'}
            </strong>
            {isEn
              ? ', ready for ultra-high-density integration of millions of complex synapses.'
              : ', sẵn sàng cho việc tích hợp hàng triệu synapse phức mật độ siêu cao.'}
          </p>
        </div>

        <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
          <div className="flex items-center gap-2 text-elegant-cyan">
            <Layers className="w-5 h-5" />
            <span className="text-xs font-bold font-mono tracking-wider uppercase">
              NATURE (2020-2021)
            </span>
          </div>
          <h3 className="text-md font-bold text-white font-display">
            {isEn
              ? 'Alloy-based Homogeneous Memristor'
              : 'Memristor Hợp Kim Đồng Nhất Siêu Ổn Định (Alloy-based)'}
          </h3>
          <p className="text-xs text-elegant-text/70 leading-relaxed">
            {isEn
              ? "Thoroughly resolves temporal and spatial fluctuation of resistive memories. Using strictly controlled silicon single-crystal alloys, the lab's synapse devices achieve superior linearity, enabling smooth and symmetric weight tuning during hardware machine learning."
              : 'Giải quyết triệt để sự bất định thời gian (temporal) và không gian (spatial fluctuation) của các bộ nhớ điện trở. Sử dụng hợp kim đơn tinh thể silicon được kiểm soát nghiêm ngặt, thiết bị synapse của phòng thí nghiệm đạt độ tuyến tính vượt trội.'}
          </p>
        </div>
      </div>

      {/* SECTION 2 & 3 GRID: INTEGRATION BLUEPRINT & LIVE HARDWARE SANDBOX */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: INTEGRATION BLUEPRINT PROPOSAL (2/3 cols) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-5 shadow-xl">
            <div className="flex items-center gap-2 text-elegant-cyan border-b border-elegant-muted/10 pb-2.5">
              <Zap className="w-5 h-5" />
              <h3 className="text-md font-bold font-display text-white">
                {isEn
                  ? 'Proposed Architecture to Integrate CV-SNN on Memristor Chip'
                  : 'Đề Xuất Kiến Trúc Tích Hợp CV-SNN trên Chip Memristor'}
              </h3>
            </div>

            <p className="text-xs text-elegant-text/80 leading-relaxed">
              {isEn ? 'To operate the ' : 'Để vận hành mô hình '}
              <strong>Complex-Valued SNN (CV-SNN)</strong>
              {isEn
                ? ' model with complex signals and weights on memristor hardware arrays that only support physical positive real conductance values (G ≥ 0), we implement a 3-component integrated architecture:'
                : ' với các tín hiệu và trọng số phức trên mảng phần cứng memristor vốn chỉ hỗ trợ các giá trị dẫn nạp thực dương vật lý (G ≥ 0), chúng tôi triển khai kiến trúc 3 thành phần tích hợp:'}
            </p>

            <div className="space-y-4">
              <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-elegant-cyan">
                    {isEn
                      ? '1. Differential 4-Memristor Synapse'
                      : '1. Tầng Trọng Số Phức Vi Sai (Differential 4-Memristor Synapse)'}
                  </span>
                  <span className="text-[10px] font-mono text-elegant-muted font-bold">
                    Synaptic Core
                  </span>
                </div>
                <p className="text-xs text-elegant-text/70 leading-relaxed">
                  {isEn
                    ? 'Each complex weight W = W_r + j W_i is separated and represented through 4 differential memristor crossbar nodes with baseline conductance G_bias:'
                    : 'Mỗi trọng số phức W = W_r + j W_i được phân tách và biểu diễn thông qua 4 điểm nối memristor chéo (crossbar nodes) vi sai có mức nền G_bias:'}
                  <code className="text-rose-300 font-mono text-[10.5px] ml-1">
                    W = (G_re⁺ - G_re⁻) + j(G_im⁺ - G_im⁻)
                  </code>
                  .
                </p>
              </div>

              <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-elegant-cyan">
                    {isEn
                      ? '2. Phase-to-Latency Encoder'
                      : '2. Bộ Chuyển Đổi Pha sang Độ Trễ Kích Xung (Phase-to-Latency Encoder)'}
                  </span>
                  <span className="text-[10px] font-mono text-elegant-muted font-bold">
                    Spike Interface
                  </span>
                </div>
                <p className="text-xs text-elegant-text/70 leading-relaxed">
                  {isEn
                    ? 'The complex phase of membrane potential θ = arg(U) is directly mapped into temporal spike latency of the emitted nerve spike.'
                    : 'Pha phức của điện thế màng θ = arg(U) được ánh xạ trực tiếp thành độ trễ thời gian tuyệt đối (temporal spike latency) của xung thần kinh phát ra.'}
                </p>
              </div>

              <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-elegant-cyan">
                    {isEn
                      ? '3. On-Chip Nonlinear Direct Compensation (Write-Verify)'
                      : '3. Thuật Toán Bù Trực Tiếp Phi Tuyến On-Chip (Write-Verify)'}
                  </span>
                  <span className="text-[10px] font-mono text-elegant-muted font-bold">
                    Error Mitigation
                  </span>
                </div>
                <p className="text-xs text-elegant-text/70 leading-relaxed">
                  {isEn
                    ? 'Closed-loop write-verify pulses calibrate G_re and G_im against thermal drift and sneak-path leakage.'
                    : 'Xung ghi-kiểm chứng vòng kín hiệu chuẩn G_re và G_im chống lại trôi nhiệt và dòng rò sneak-path.'}
                </p>
              </div>
            </div>
          </div>

          {/* Interactive Kirchhoff Complex Multiplier Verification Panel */}
          <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-elegant-muted/10 pb-2.5">
              <h3 className="text-sm font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>
                  {isEn
                    ? 'Live Kirchhoff Crossbar Complex Multiplier Verifier'
                    : 'Bộ Kiểm Chứng Phép Nhân Phức Vật Lý Kirchhoff Trên Crossbar'}
                </span>
              </h3>
              <label className="flex items-center gap-2 text-xs font-mono text-elegant-cyan cursor-pointer">
                <input
                  type="checkbox"
                  checked={writeVerifyEnabled}
                  onChange={e => setWriteVerifyEnabled(e.target.checked)}
                  className="accent-elegant-cyan"
                />
                <span>{isEn ? 'KAIST Write-Verify Calibration' : 'Bật Bù Lỗi Write-Verify KAIST'}</span>
              </label>
            </div>

            {(() => {
              const Wr = choiWeightMag * Math.cos(choiWeightPhase);
              const Wi = choiWeightMag * Math.sin(choiWeightPhase);
              const Vr = inputVoltMag * Math.cos(inputVoltPhase);
              const Vi = inputVoltMag * Math.sin(inputVoltPhase);

              // Ideal complex product I_ideal = W * V
              const idealRe = Wr * Vr - Wi * Vi;
              const idealIm = Wr * Vi + Wi * Vr;
              const idealMag = Math.sqrt(idealRe * idealRe + idealIm * idealIm);
              const idealPhaseDeg = (Math.atan2(idealIm, idealRe) * 180) / Math.PI;

              // Physical 4-memristor conductances with deterministic thermal/write variation
              const bias = 0.1;
              const effNoise = writeVerifyEnabled ? pcmNoise * 0.12 : pcmNoise;
              const gRePlus = Math.max(0.01, bias + Math.max(0, Wr) + effNoise * 0.18);
              const gReMinus = Math.max(0.01, bias + Math.max(0, -Wr) - effNoise * 0.14);
              const gImPlus = Math.max(0.01, bias + Math.max(0, Wi) - effNoise * 0.16);
              const gImMinus = Math.max(0.01, bias + Math.max(0, -Wi) + effNoise * 0.15);

              // Kirchhoff differential current summation:
              // I_re = V_r*(G_re+ - G_re-) - V_i*(G_im+ - G_im-)
              // I_im = V_r*(G_im+ - G_im-) + V_i*(G_re+ - G_re-)
              const netGre = gRePlus - gReMinus;
              const netGim = gImPlus - gImMinus;
              const hwRe = Vr * netGre - Vi * netGim;
              const hwIm = Vr * netGim + Vi * netGre;
              const hwMag = Math.sqrt(hwRe * hwRe + hwIm * hwIm);
              const hwPhaseDeg = (Math.atan2(hwIm, hwRe) * 180) / Math.PI;

              const phaseErr = Math.abs(idealPhaseDeg - hwPhaseDeg);
              const rmse = Math.sqrt(
                (idealRe - hwRe) * (idealRe - hwRe) + (idealIm - hwIm) * (idealIm - hwIm)
              );

              return (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div className="p-3 bg-elegant-bg rounded-lg border border-elegant-muted/15 space-y-1.5">
                      <div className="flex justify-between text-[11px] font-mono text-elegant-muted">
                        <span>{isEn ? 'Input |V_in|:' : 'Biên độ điện áp |V_in|:'}</span>
                        <span className="text-elegant-cyan font-bold">
                          {inputVoltMag.toFixed(2)} V
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.2"
                        max="1.5"
                        step="0.05"
                        value={inputVoltMag}
                        onChange={e => setInputVoltMag(parseFloat(e.target.value))}
                        className="w-full accent-elegant-cyan bg-elegant-card h-1 rounded cursor-pointer"
                      />
                    </div>

                    <div className="p-3 bg-elegant-bg rounded-lg border border-elegant-muted/15 space-y-1.5">
                      <div className="flex justify-between text-[11px] font-mono text-elegant-muted">
                        <span>{isEn ? 'Input Phase ∠V:' : 'Pha điện áp ∠V:'}</span>
                        <span className="text-elegant-cyan font-bold">
                          {((inputVoltPhase * 180) / Math.PI).toFixed(0)}°
                        </span>
                      </div>
                      <input
                        type="range"
                        min="-3.14"
                        max="3.14"
                        step="0.05"
                        value={inputVoltPhase}
                        onChange={e => setInputVoltPhase(parseFloat(e.target.value))}
                        className="w-full accent-elegant-cyan bg-elegant-card h-1 rounded cursor-pointer"
                      />
                    </div>

                    <div className="p-3 bg-elegant-bg rounded-lg border border-elegant-muted/15 space-y-1.5">
                      <div className="flex justify-between text-[11px] font-mono text-elegant-muted">
                        <span>{isEn ? 'PCM Device Noise:' : 'Nhiễu Vật Liệu PCM:'}</span>
                        <span className="text-rose-400 font-bold">
                          {(pcmNoise * 100).toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="0.5"
                        step="0.02"
                        value={pcmNoise}
                        onChange={e => setPcmNoise(parseFloat(e.target.value))}
                        className="w-full accent-rose-400 bg-elegant-card h-1 rounded cursor-pointer"
                      />
                    </div>
                  </div>

                  {/* Comparison Cards: Ideal Math vs Physical Crossbar Kirchhoff Current */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
                    <div className="p-3.5 bg-elegant-bg/80 rounded-xl border border-elegant-cyan/25 space-y-1">
                      <span className="text-[10px] text-elegant-muted uppercase block">
                        {isEn ? 'Ideal Math (W · V_in)' : 'Tích Số Phức Lý Tưởng (W · V_in)'}
                      </span>
                      <div className="text-sm font-bold text-white">
                        {idealRe.toFixed(3)} {idealIm >= 0 ? '+' : '-'} {Math.abs(idealIm).toFixed(3)}j mA
                      </div>
                      <div className="text-[10px] text-elegant-cyan">
                        |I| = {idealMag.toFixed(3)} mA | ∠I = {idealPhaseDeg.toFixed(1)}°
                      </div>
                    </div>

                    <div className="p-3.5 bg-elegant-bg/80 rounded-xl border border-amber-400/25 space-y-1">
                      <span className="text-[10px] text-elegant-muted uppercase block">
                        {isEn ? 'Physical Kirchhoff Crossbar' : 'Dòng Thực Tế Từ 4-Memristor'}
                      </span>
                      <div className="text-sm font-bold text-white">
                        {hwRe.toFixed(3)} {hwIm >= 0 ? '+' : '-'} {Math.abs(hwIm).toFixed(3)}j mA
                      </div>
                      <div className="text-[10px] text-amber-400">
                        |I_hw| = {hwMag.toFixed(3)} mA | ∠I_hw = {hwPhaseDeg.toFixed(1)}°
                      </div>
                    </div>

                    <div className="p-3.5 bg-elegant-bg/80 rounded-xl border border-emerald-400/25 space-y-1">
                      <span className="text-[10px] text-elegant-muted uppercase block">
                        {isEn ? 'Hardware Fidelity' : 'Độ Chính Xác Vật Lý'}
                      </span>
                      <div className="text-sm font-bold text-emerald-400">
                        Δθ = {phaseErr.toFixed(2)}°
                      </div>
                      <div className="text-[10px] text-elegant-muted">
                        RMSE: {rmse.toFixed(4)} mA
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>

        {/* RIGHT: LIVE HARDWARE MAPPING SANDBOX (1/3 cols) */}
        <div className="space-y-6">
          <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
            <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
              <Settings className="w-4 h-4 text-elegant-cyan" />
              <span>
                {isEn ? 'W Complex Weight Configuration' : 'Cấu Hình Trọng Số Số Phức W'}
              </span>
            </h3>

            <p className="text-xs text-elegant-text/70 leading-relaxed">
              {isEn
                ? "Drag the sliders to change Amplitude and Phase of the complex weight W. The system auto-calculates how to program 4 physical memristors simulating Prof. Choi's platform!"
                : 'Kéo chỉnh thanh trượt để thay đổi Biên độ và Pha của trọng số phức W. Hệ thống tự động tính toán cách lập trình 4 thiết bị memristor thực tế!'}
            </p>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>{isEn ? 'Weight amplitude |W|:' : 'Biên độ trọng số |W|:'}</span>
                <span className="text-elegant-cyan font-bold">{choiWeightMag.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={choiWeightMag}
                onChange={e => setChoiWeightMag(parseFloat(e.target.value))}
                className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono text-elegant-muted">
                <span>{isEn ? 'Phase angle θ_W (radians):' : 'Góc pha θ_W (radians):'}</span>
                <span className="text-elegant-cyan font-bold">
                  {choiWeightPhase.toFixed(2)} rad ({((choiWeightPhase * 180) / Math.PI).toFixed(0)}
                  °)
                </span>
              </div>
              <input
                type="range"
                min="-3.1415"
                max="3.1415"
                step="0.05"
                value={choiWeightPhase}
                onChange={e => setChoiWeightPhase(parseFloat(e.target.value))}
                className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
              />
            </div>

            <div className="p-3.5 bg-elegant-bg rounded-lg border border-elegant-muted/10 text-xs font-mono space-y-1.5 text-center shadow-inner">
              <div className="text-elegant-muted text-[11px] uppercase tracking-wider">
                {isEn ? 'Complex weights:' : 'Trọng số dạng phức:'}
              </div>
              <div className="text-sm font-bold text-white">
                W = {(choiWeightMag * Math.cos(choiWeightPhase)).toFixed(3)}{' '}
                {choiWeightMag * Math.sin(choiWeightPhase) >= 0 ? '+' : '-'}{' '}
                {Math.abs(choiWeightMag * Math.sin(choiWeightPhase)).toFixed(3)}i
              </div>
              <div className="text-[10px] text-elegant-cyan">
                {isEn ? 'Exponential:' : 'Dạng mũ:'} {choiWeightMag.toFixed(2)} · e^(
                {choiWeightPhase >= 0 ? '+' : ''}
                {choiWeightPhase.toFixed(2)}i)
              </div>
            </div>
          </div>

          {/* Physical Conductance Mapping meters */}
          <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
            <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
              <Activity className="w-4 h-4 text-elegant-cyan animate-pulse" />
              <span>
                {isEn ? 'Conductance Memristor Programming' : 'Conductance Lập Trình Memristor'}
              </span>
            </h3>

            {(() => {
              const Wr = choiWeightMag * Math.cos(choiWeightPhase);
              const Wi = choiWeightMag * Math.sin(choiWeightPhase);
              const bias = 0.1;
              const gRePlus = bias + Math.max(0, Wr);
              const gReMinus = bias + Math.max(0, -Wr);
              const gImPlus = bias + Math.max(0, Wi);
              const gImMinus = bias + Math.max(0, -Wi);
              const scaleMax = 1.6;

              return (
                <div className="space-y-4">
                  <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                    {isEn
                      ? 'Actual Conductance (G) programmed into 4 differential memristor crossbars:'
                      : 'Mức dẫn nạp (Conductance G) thực tế được ghi nạp vào 4 thanh memristor vi sai chéo:'}
                  </p>

                  <div className="space-y-3 font-mono">
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-elegant-cyan font-bold">
                          {isEn ? 'G_re⁺ (Positive real):' : 'G_re⁺ (Dẫn thực dương):'}
                        </span>
                        <span className="text-white">{gRePlus.toFixed(3)} mS</span>
                      </div>
                      <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                        <div
                          className="h-full bg-gradient-to-r from-[#45A29E] to-elegant-cyan rounded-full transition-all duration-150"
                          style={{ width: `${(gRePlus / scaleMax) * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-rose-400 font-bold">
                          {isEn ? 'G_re⁻ (Negative real):' : 'G_re⁻ (Dẫn thực âm):'}
                        </span>
                        <span className="text-white">{gReMinus.toFixed(3)} mS</span>
                      </div>
                      <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                        <div
                          className="h-full bg-gradient-to-r from-rose-600 to-rose-400 rounded-full transition-all duration-150"
                          style={{ width: `${(gReMinus / scaleMax) * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-amber-400 font-bold">
                          {isEn ? 'G_im⁺ (Positive imag):' : 'G_im⁺ (Dẫn ảo dương):'}
                        </span>
                        <span className="text-white">{gImPlus.toFixed(3)} mS</span>
                      </div>
                      <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                        <div
                          className="h-full bg-gradient-to-r from-amber-600 to-amber-400 rounded-full transition-all duration-150"
                          style={{ width: `${(gImPlus / scaleMax) * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-indigo-400 font-bold">
                          {isEn ? 'G_im⁻ (Negative imag):' : 'G_im⁻ (Dẫn ảo âm):'}
                        </span>
                        <span className="text-white">{gImMinus.toFixed(3)} mS</span>
                      </div>
                      <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full transition-all duration-150"
                          style={{ width: `${(gImMinus / scaleMax) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-3 bg-elegant-cyan/5 border border-elegant-cyan/25 rounded-lg text-[11px] text-elegant-text/90 leading-relaxed italic">
                    <strong className="text-white">
                      {isEn ? 'Differential principle:' : 'Nguyên lý vi sai:'}
                    </strong>{' '}
                    {isEn ? 'Net real conductance ' : 'Độ dẫn thực thuần '}
                    <span className="text-elegant-cyan font-bold font-mono">
                      G_re⁺ - G_re⁻ = {Wr.toFixed(3)} mS
                    </span>
                    {isEn ? ', net imaginary ' : ', độ dẫn ảo thuần '}
                    <span className="text-amber-400 font-bold font-mono">
                      G_im⁺ - G_im⁻ = {Wi.toFixed(3)} mS
                    </span>
                    .
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </motion.div>
  );
}