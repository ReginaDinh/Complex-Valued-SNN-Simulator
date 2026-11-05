import React, { useState } from 'react';
import { motion } from 'motion/react';
import { 
  Cpu, Layers, Zap, Activity, ArrowRight, Server, Terminal, Settings, Database, HardDrive, CheckCircle2, Play
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { InlineMath } from 'react-katex';

export default function HardwareBlueprintTab() {
  const { language } = useLanguage();
  const isEn = language === 'en';


  const [activeStep, setActiveStep] = useState(0);

  const [pos, setPos] = useState({
    input: { x: 20, y: 200 },
    dac: { x: 180, y: 120 },
    cbRe: { x: 280, y: 70 },
    cbIm: { x: 280, y: 260 },
    tia: { x: 560, y: 200 },
    clif: { x: 680, y: 200 }
  });

  const onDrag = (key: keyof typeof pos) => (e: any, info: any) => {
    setPos(p => ({
      ...p,
      [key]: {
        x: p[key].x + info.delta.x,
        y: p[key].y + info.delta.y
      }
    }));
  };

  const drawPath = (x1: number, y1: number, x2: number, y2: number) => {
    const cp = Math.max(Math.abs(x2 - x1) / 2, 20);
    return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
  };


  const steps = [
    {
      title: isEn ? '1. Software Training' : '1. Huấn luyện Phần Mềm',
      desc: isEn ? <>PyTorch/SNNTorch training to obtain ideal Complex Weights (<InlineMath math="W = W_{re} + jW_{im}" />).</> : <>Huấn luyện trên PyTorch/SNNTorch để thu được trọng số phức lý tưởng (<InlineMath math="W = W_{re} + jW_{im}" />).</>,
      icon: <Terminal className="w-5 h-5" />
    },
    {
      title: isEn ? '2. Matrix Splitting' : '2. Phân Tách Ma Trận',
      desc: isEn ? <>Convert complex weights into 4 positive real matrices (<InlineMath math="G_{re+}, G_{re-}, G_{im+}, G_{im-}" />).</> : <>Chuyển đổi trọng số phức thành 4 ma trận thực dương (<InlineMath math="G_{re+}, G_{re-}, G_{im+}, G_{im-}" />).</>,
      icon: <Layers className="w-5 h-5" />
    },
    {
      title: isEn ? '3. Hardware Mapping' : '3. Nạp Trọng Số (Mapping)',
      desc: isEn ? 'Program DAC to send Write Pulses, altering Phase-Change Memory (PCM) crystalline structure.' : 'Lập trình DAC gửi Xung Ghi, thay đổi cấu trúc tinh thể của vật liệu PCM.',
      icon: <Database className="w-5 h-5" />
    },
    {
      title: isEn ? '4. Analog Inference' : '4. Suy Luận Tương Tự',
      desc: isEn ? 'Convert inputs to delayed spikes. Currents sum naturally via Kirchhoff laws on crossbar.' : 'Mã hóa đầu vào thành xung trễ. Dòng điện tự tổng hợp qua định luật Kirchhoff trên Crossbar.',
      icon: <Cpu className="w-5 h-5" />
    }
  ];

  return (
    <motion.div
      key="blueprint"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl shadow-xl space-y-2 relative overflow-hidden">
        
        <div className="flex items-center gap-3 relative z-10">
          <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
            <HardDrive className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold font-display text-white">{isEn ? 'Hardware Blueprint & Implementation' : 'Sơ đồ Thiết kế & Triển khai Phần cứng'}</h2>
            <p className="text-xs text-elegant-muted font-mono mt-0.5">
              {isEn ? 'End-to-end architecture from algorithm to physical memristor crossbar array deployment' : 'Kiến trúc toàn trình từ thuật toán đến triển khai vật lý trên mảng Crossbar Memristor'}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Schematic SVG - Left side */}
        <div className="lg:col-span-8 bg-[#0b1016] border border-elegant-muted/15 rounded-xl shadow-xl overflow-hidden flex flex-col">
          <div className="p-4 border-b border-elegant-muted/10 bg-elegant-card flex justify-between items-center">
            <h3 className="text-sm font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
              <Layers className="w-4 h-4 text-elegant-cyan" />
              {isEn ? 'Physical Architecture Schematic' : 'Sơ Đồ Kiến Trúc Vật Lý'}
            </h3>
          </div>
          
          <div className="p-6 flex-1 flex items-center justify-center relative overflow-hidden min-h-[400px]">
             {/* Decorative Grid */}
             <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(#1e293b 1px, transparent 1px)', backgroundSize: '24px 24px', opacity: 0.4 }}></div>
             

             {/* Schematic Graphic SVG */}
             <div className="relative z-10 w-full max-w-2xl h-full flex items-center justify-center">
               <svg viewBox="0 0 850 500" className="w-full h-[500px] drop-shadow-2xl overflow-visible">
                  <defs>
                    <marker id="arrowHead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                    </marker>
                    <marker id="arrowHeadCyan" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#06b6d4" />
                    </marker>
                    <marker id="arrowHeadPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
                    </marker>
                  </defs>

                  {/* Connections */}
                  <path d={drawPath(pos.input.x + 120, pos.input.y + 40, pos.dac.x, pos.dac.y + 120)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrowHead)" />
                  
                  <path d={drawPath(pos.dac.x + 60, pos.dac.y + 40, pos.cbRe.x, pos.cbRe.y + 90)} stroke="#06b6d4" strokeWidth="2" fill="none" markerEnd="url(#arrowHeadCyan)" />
                  <path d={drawPath(pos.dac.x + 60, pos.dac.y + 200, pos.cbIm.x, pos.cbIm.y + 60)} stroke="#06b6d4" strokeWidth="2" fill="none" markerEnd="url(#arrowHeadCyan)" />
                  
                  <path d={drawPath(pos.cbRe.x + 180, pos.cbRe.y + 75, pos.tia.x, pos.tia.y + 30)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrowHead)" />
                  <path d={drawPath(pos.cbIm.x + 180, pos.cbIm.y + 75, pos.tia.x, pos.tia.y + 50)} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrowHead)" />
                  
                  <path d={drawPath(pos.tia.x + 80, pos.tia.y + 40, pos.clif.x, pos.clif.y + 40)} stroke="#a855f7" strokeWidth="2" fill="none" markerEnd="url(#arrowHeadPurple)" />
                  
                  <path d={drawPath(pos.clif.x + 100, pos.clif.y + 40, pos.clif.x + 140, pos.clif.y + 40)} stroke="#f43f5e" strokeWidth="2" fill="none" markerEnd="url(#arrowHead)" />

                  {/* 1. Input Module */}
                  <motion.g initial={{ x: pos.input.x, y: pos.input.y }} drag dragMomentum={false} onDrag={onDrag('input')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="120" height="80" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2" />
                    <text x="60" y="30" fill="white" fontSize="14" fontWeight="bold" textAnchor="middle">Signal/Image</text>
                    <text x="60" y="50" fill="#94a3b8" fontSize="12" textAnchor="middle">Phase Encoder</text>
                    <path d="M 20 65 L 40 65 L 40 55 L 50 55 L 50 65 L 100 65" stroke="#f43f5e" strokeWidth="2" fill="none" />
                  </motion.g>

                  {/* 2. DAC Array */}
                  <motion.g initial={{ x: pos.dac.x, y: pos.dac.y }} drag dragMomentum={false} onDrag={onDrag('dac')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="60" height="240" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="2" />
                    <g transform="translate(30, 120) rotate(-90)">
                      <text x="0" y="-5" fill="#06b6d4" fontSize="14" fontWeight="bold" textAnchor="middle">DAC Array</text>
                      <text x="0" y="10" fill="#94a3b8" fontSize="10" textAnchor="middle">(Voltage Pulses)</text>
                    </g>
                  </motion.g>

                  {/* 3. Memristor Crossbar Re */}
                  <motion.g initial={{ x: pos.cbRe.x, y: pos.cbRe.y }} drag dragMomentum={false} onDrag={onDrag('cbRe')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="180" height="150" rx="8" fill="#1e293b" stroke="#64748b" strokeWidth="2" />
                    <text x="90" y="25" fill="white" fontSize="14" fontWeight="bold" textAnchor="middle">Crossbar G_re</text>
                    <text x="90" y="45" fill="#94a3b8" fontSize="11" textAnchor="middle">Re+ and Re- arrays</text>
                    {[1,2,3].map(i => <line key={`h${i}`} x1="20" y1={50 + i*20} x2="160" y2={50 + i*20} stroke="#475569" strokeWidth="2" />)}
                    {[1,2,3,4].map(i => <line key={`v${i}`} x1={30 + i*25} y1="50" x2={30 + i*25} y2="130" stroke="#475569" strokeWidth="2" />)}
                    {[1,2,3].map(i => [1,2,3,4].map(j => <circle key={`d${i}${j}`} cx={30 + j*25} cy={50 + i*20} r="4" fill="#38bdf8" />))}
                  </motion.g>

                  {/* 4. Memristor Crossbar Im */}
                  <motion.g initial={{ x: pos.cbIm.x, y: pos.cbIm.y }} drag dragMomentum={false} onDrag={onDrag('cbIm')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="180" height="150" rx="8" fill="#1e293b" stroke="#64748b" strokeWidth="2" />
                    <text x="90" y="25" fill="white" fontSize="14" fontWeight="bold" textAnchor="middle">Crossbar G_im</text>
                    <text x="90" y="45" fill="#94a3b8" fontSize="11" textAnchor="middle">Im+ and Im- arrays</text>
                    {[1,2,3].map(i => <line key={`h${i}`} x1="20" y1={50 + i*20} x2="160" y2={50 + i*20} stroke="#475569" strokeWidth="2" />)}
                    {[1,2,3,4].map(i => <line key={`v${i}`} x1={30 + i*25} y1="50" x2={30 + i*25} y2="130" stroke="#475569" strokeWidth="2" />)}
                    {[1,2,3].map(i => [1,2,3,4].map(j => <circle key={`d${i}${j}`} cx={30 + j*25} cy={50 + i*20} r="4" fill="#f59e0b" />))}
                  </motion.g>

                  {/* 5. Subtractor & Amplifier */}
                  <motion.g initial={{ x: pos.tia.x, y: pos.tia.y }} drag dragMomentum={false} onDrag={onDrag('tia')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="80" height="80" rx="8" fill="#1e293b" stroke="#a855f7" strokeWidth="2" />
                    <text x="40" y="30" fill="#a855f7" fontSize="14" fontWeight="bold" textAnchor="middle">TIA &</text>
                    <text x="40" y="48" fill="#a855f7" fontSize="14" fontWeight="bold" textAnchor="middle">Diff Amp</text>
                    <text x="40" y="65" fill="#94a3b8" fontSize="10" textAnchor="middle">(I to V)</text>
                  </motion.g>

                  {/* 6. Analog Comparator / Spike Gen */}
                  <motion.g initial={{ x: pos.clif.x, y: pos.clif.y }} drag dragMomentum={false} onDrag={onDrag('clif')} style={{ cursor: 'grab' }} whileDrag={{ scale: 1.05, cursor: 'grabbing' }}>
                    <rect x="0" y="0" width="100" height="80" rx="8" fill="#1e293b" stroke="#f43f5e" strokeWidth="2" />
                    <text x="50" y="35" fill="white" fontSize="14" fontWeight="bold" textAnchor="middle">cLIF</text>
                    <text x="50" y="55" fill="#94a3b8" fontSize="11" textAnchor="middle">Comparator</text>
                  </motion.g>

               </svg>
             </div>
          </div>
          
          <div className="p-4 bg-elegant-card/50 border-t border-elegant-muted/10 text-xs text-elegant-text/80 leading-relaxed font-mono">
             {isEn ? <>Complex Matrix is separated: <InlineMath math="W_{complex} = (G_{re+} - G_{re-}) + j(G_{im+} - G_{im-})" />. The crossbars sum currents naturally, Transimpedance Amplifiers (TIA) convert current back to voltage for the cLIF comparators.</> : <>Ma trận phức tách biệt: <InlineMath math="W_{complex} = (G_{re+} - G_{re-}) + j(G_{im+} - G_{im-})" />. Các Crossbar cộng dòng tự nhiên theo Kirchhoff, các bộ Khuếch đại TIA chuyển dòng điện về điện áp cho bộ so sánh cLIF.</>}
          </div>
        </div>

        {/* Workflow & Steps - Right side */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl shadow-xl space-y-4">
             <h3 className="text-sm font-bold text-white tracking-wider uppercase font-mono flex items-center gap-2">
                <Settings className="w-4 h-4 text-elegant-cyan" />
                {isEn ? 'Hardware Deployment Flow' : 'Quy trình Triển khai'}
             </h3>
             
             <div className="space-y-3">
               {steps.map((step, idx) => (
                 <button 
                   key={idx}
                   onClick={() => setActiveStep(idx)}
                   className={`w-full text-left p-3 rounded-lg border transition-all duration-300 ${
                     activeStep === idx 
                       ? 'bg-elegant-cyan/10 border-elegant-cyan shadow-sm' 
                       : 'bg-[#0b1016] border-elegant-muted/20 hover:border-elegant-cyan/50'
                   }`}
                 >
                    <div className="flex items-center gap-3">
                      <div className={`p-1.5 rounded-md ${activeStep === idx ? 'bg-elegant-cyan text-white' : 'bg-elegant-card text-elegant-muted'}`}>
                        {step.icon}
                      </div>
                      <div>
                        <div className={`text-sm font-bold ${activeStep === idx ? 'text-elegant-cyan' : 'text-slate-300'}`}>
                          {step.title}
                        </div>
                      </div>
                    </div>
                 </button>
               ))}
             </div>
          </div>

          {/* Step Details Pane */}
          <div className="flex-1 bg-gradient-to-b from-[#0b1016] to-[#070b10] border border-elegant-muted/15 p-5 rounded-xl shadow-xl flex flex-col">
             <h4 className="text-xs font-bold text-elegant-muted tracking-wide font-mono mb-3">
               {isEn ? 'STEP DETAILS' : 'CHI TIẾT BƯỚC'}
             </h4>
             <div className="text-sm text-slate-300 leading-relaxed font-mono">
               <span className="text-elegant-cyan font-bold block mb-2">{steps[activeStep].title}</span>
               {steps[activeStep].desc}
             </div>
             
             <div className="mt-auto pt-6">
                {activeStep === 0 && (
                  <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/20">
                     <span className="text-[10px] text-emerald-400 font-bold block">Python / PyTorch</span>
                     <code className="text-[11px] text-slate-300 mt-1 block whitespace-pre-wrap">
                        {`optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\nloss = criterion(out, target)\nloss.backward()`}
                     </code>
                  </div>
                )}
                {activeStep === 1 && (
                  <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/20">
                     <span className="text-[10px] text-emerald-400 font-bold block">Digital Mapping</span>
                     <code className="text-[11px] text-slate-300 mt-1 block whitespace-pre-wrap">
                        {`W_re = W.real; W_im = W.imag\nG_re_plus = torch.clamp(W_re, min=0)\nG_re_minus = torch.clamp(-W_re, min=0)`}
                     </code>
                  </div>
                )}
                {activeStep === 2 && (
                  <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/20">
                     <span className="text-[10px] text-orange-400 font-bold block">Analog Pulse / SCPI</span>
                     <code className="text-[11px] text-slate-300 mt-1 block whitespace-pre-wrap">
                        {`visa.write("SOUR:VOLT:PULS 2.5V, 50ns")\n# Crystallizing Phase Change Mat.`}
                     </code>
                  </div>
                )}
                {activeStep === 3 && (
                  <div className="bg-[#121b26] p-3 rounded-lg border border-elegant-muted/20">
                     <span className="text-[10px] text-rose-400 font-bold block">Physics Math (Kirchhoff)</span>
                     <code className="text-[11px] text-slate-300 mt-1 block whitespace-pre-wrap">
                        {`I_sum = Σ(V_in * G_memristor)\nV_out = TIA_Gain * I_sum`}
                     </code>
                  </div>
                )}
             </div>
          </div>
        </div>

      </div>

      {/* Hardware Components Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
         <ComponentCard 
           title={isEn ? "Input Phase Encoder" : "Bộ Mã Hóa Pha Đầu Vào"}
           desc={isEn ? <>Converts temporal spike timings into continuous phase angles mapping to <InlineMath math="[0, 2\pi]" />.</> : <>Chuyển đổi thời gian phát xung thành các góc pha liên tục trong khoảng <InlineMath math="[0, 2\pi]" />.</>}
           icon={<Activity />}
         />
         <ComponentCard 
           title={isEn ? "Memristor Crossbar" : "Mảng Crossbar Memristor"}
           desc={isEn ? "Analog memory matrices (PCM/RRAM) performing massively parallel MAC operations." : "Ma trận bộ nhớ tương tự (PCM/RRAM) thực hiện tính toán MAC song song khối lượng lớn."}
           icon={<Layers />}
         />
         <ComponentCard 
           title={isEn ? "Transimpedance Amp" : "Bộ Khuếch Đại TIA"}
           desc={isEn ? "Converts summed output currents back into voltage differentials for processing." : "Chuyển đổi dòng điện tổng từ crossbar trở lại thành vi sai điện áp để xử lý tiếp."}
           icon={<Zap />}
         />
         <ComponentCard 
           title={isEn ? "cLIF Comparator" : "Bộ So Sánh cLIF"}
           desc={isEn ? "Fires a real electrical spike if voltage magnitude exceeds the circular threshold boundary." : "Phát một xung điện thực tế nếu biên độ điện áp vượt qua ranh giới ngưỡng tròn."}
           icon={<Cpu />}
         />
      </div>
    </motion.div>
  );
}

function ComponentCard({ title, desc, icon }: { title: string, desc: React.ReactNode, icon: React.ReactNode }) {
  return (
    <div className="bg-elegant-card border border-elegant-muted/15 p-4 rounded-xl flex flex-col gap-3 hover:border-elegant-cyan/30 transition-all">
      <div className="w-8 h-8 rounded-lg bg-[#0b1016] border border-elegant-muted/20 flex items-center justify-center text-elegant-cyan">
        {React.cloneElement(icon as React.ReactElement<any>, { className: "w-4 h-4" })}
      </div>
      <div>
        <h4 className="text-sm font-bold text-white mb-1 font-mono">{title}</h4>
        <div className="text-[11px] text-slate-400 leading-relaxed">{desc}</div>
      </div>
    </div>
  );
}
