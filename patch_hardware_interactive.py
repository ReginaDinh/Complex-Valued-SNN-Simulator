import re

with open('src/components/HardwareBlueprintTab.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# Add motion values/state
state_code = """
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
"""

code = code.replace('  const [activeStep, setActiveStep] = useState(0);', state_code)

svg_content = """
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
                    <text x="30" y="120" fill="#06b6d4" fontSize="14" fontWeight="bold" textAnchor="middle" transform="rotate(-90 30 120)">DAC Array</text>
                    <text x="30" y="150" fill="#94a3b8" fontSize="10" textAnchor="middle" transform="rotate(-90 30 150)">(Voltage Pulses)</text>
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
"""

start_idx = code.find('             {/* Schematic Graphic SVG */}')
end_idx = code.find('          </div>\n          \n          <div className="p-4 bg-elegant-card/50')

code = code[:start_idx] + svg_content + code[end_idx:]

with open('src/components/HardwareBlueprintTab.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

