import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { Terminal, Copy, Check } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { pytorchCodeString } from '../pytorchCode';
import { copyText } from '../utils/clipboard';

export default function PytorchTab() {
  const [copied, setCopied] = useState(false);
  const { language } = useLanguage();
  const isEn = language === 'en';

  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
  }, []);

  const handleCopyCode = async () => {
    const ok = await copyText(pytorchCodeString);
    if (!ok) {
      console.warn('Clipboard copy failed');
      return;
    }
    setCopied(true);
    if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      key="pytorch"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold flex items-center gap-2 font-display">
              <Terminal className="text-elegant-cyan w-5 h-5" />
              <span>{isEn ? "SOTA PyTorch Code for Complex-Valued SNN (cLIF)" : "Mã Nguồn PyTorch SOTA Complex-Valued SNN (cLIF)"}</span>
            </h2>
            <p className="text-xs text-elegant-muted mt-1 font-mono">
              {isEn ? "Standardized SOTA source code with Karatsuba Algorithm (Decoupled SNN), Von Mises Surrogate Gradient, and Phase-Attention." : "Mã nguồn chuẩn hóa SOTA với Thuật toán Karatsuba (Decoupled SNN), Von Mises Surrogate Gradient và Phase-Attention."}
            </p>
          </div>
          
          <button
            onClick={handleCopyCode}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-elegant-cyan hover:bg-[#52ebd3] text-elegant-bg rounded-md text-xs font-bold transition-all font-mono self-start shadow-md shadow-elegant-cyan/15"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? (isEn ? 'Copied!' : 'Đã sao chép!') : (isEn ? 'Copy PyTorch Code' : 'Sao chép PyTorch Code')}</span>
          </button>
        </div>
        <div className="p-4 bg-elegant-bg rounded-lg border border-elegant-muted/10 max-h-[500px] overflow-y-auto font-mono text-xs text-elegant-text leading-relaxed scrollbar-thin scrollbar-thumb-elegant-card">
          <pre className="whitespace-pre">{pytorchCodeString}</pre>
        </div>
        
        <div className="p-4 bg-elegant-cyan/5 border border-elegant-cyan/25 rounded-lg space-y-2">
          <h4 className="text-xs font-bold text-elegant-cyan uppercase tracking-wider font-mono">{isEn ? "SNNTorch and Norse Integration Guide:" : "Hướng Dẫn Tích Hợp SNNTorch và Norse:"}</h4>
          <p className="text-xs text-elegant-text leading-relaxed">
            {isEn ? "This SOTA code integrates the latest advancements: " : "Đoạn code SOTA tích hợp các cải tiến mới nhất: "}
            <strong>{isEn ? "1) Karatsuba Complex Linear" : "1) Karatsuba Complex Linear"}</strong> 
            {isEn ? " minimizes hardware computation, " : " giảm thiểu tính toán phần cứng, "}
            <strong>{isEn ? "2) Von Mises Surrogate" : "2) Von Mises Surrogate"}</strong> 
            {isEn ? " enhances BPTT derivative stability, and " : " tăng cường ổn định đạo hàm BPTT, và "}
            <strong>{isEn ? "3) Phase Attention" : "3) Phase Attention"}</strong> 
            {isEn ? " automatically locks frequencies. You can replace snnTorch's default LIF layer with the " : " tự động khóa tần số. Bạn có thể thay thế tầng LIF mặc định của snnTorch bằng lớp "}
            <span className="font-mono text-elegant-cyan">ComplexLIF</span> 
            {isEn ? " class and use " : " và sử dụng "}
            <span className="font-mono text-elegant-cyan">KaratsubaComplexLinear</span> 
            {isEn ? " for Fully Connected layers." : " cho các tầng Fully Connected."}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
