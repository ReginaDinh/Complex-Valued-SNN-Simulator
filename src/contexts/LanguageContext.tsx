import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export type Language = 'vi' | 'en';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, defaultText?: string) => string;
}

const translations: Record<string, Record<Language, string>> = {
  // App.tsx
  "app.title": { vi: "Mạng Nơ-ron Xung Số Phức", en: "Complex-Valued Spiking Neural Network" },
  "app.tab.theory": { vi: "Lý Thuyết", en: "Theory" },
  "app.tab.comparison": { vi: "So Sánh SNN", en: "SNN Comparison" },
  "app.tab.pytorch": { vi: "PyTorch Code", en: "PyTorch Code" },
  "app.tab.neuron": { vi: "Neuron Sim", en: "Neuron Sim" },
  "app.tab.network": { vi: "Network Train", en: "Network Train" },
  "app.tab.vision": { vi: "Vision SNN", en: "Vision SNN" },
  "app.tab.choilab": { vi: "Choi Lab HW", en: "Choi Lab HW" },
  "app.footer": { vi: "Mô hình toán học và mã nguồn dựa trên các ấn bản SOTA IEEE & arXiv năm 2022-2026.", en: "Mathematical models and source code based on SOTA IEEE & arXiv publications 2022-2026." },
  
  // NeuronTab.tsx
  "neuron.params": { vi: "Thông số Neuron", en: "Neuron Parameters" },
  "neuron.leakMag": { vi: "Biên độ rò rỉ (Leak Magnitude):", en: "Leak Magnitude:" },
  "neuron.leakPhase": { vi: "Pha rò rỉ (Leak Phase, rad):", en: "Leak Phase (rad):" },
  "neuron.threshold": { vi: "Ngưỡng (Threshold):", en: "Threshold:" },
  "neuron.inputCurrent": { vi: "Dòng điện đầu vào (Phức)", en: "Input Current (Complex)" },
  "neuron.amplitude": { vi: "Biên độ:", en: "Amplitude:" },
  "neuron.frequency": { vi: "Tần số:", en: "Frequency:" },
  "neuron.trajectory": { vi: "Quỹ đạo mặt phẳng phức", en: "Complex Plane Trajectory" },
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>(() => {
    const saved = localStorage.getItem('app_language');
    return (saved === 'en' || saved === 'vi') ? saved : 'vi';
  });

  useEffect(() => {
    localStorage.setItem('app_language', language);
  }, [language]);

  const t = (key: string, defaultText?: string): string => {
    return translations[key]?.[language] || defaultText || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

// Additional keys can be added here
const moreTranslations = {
  "theory.title": { vi: "Mô hình SNN trong Miền Phức (Complex-valued SNN)", en: "Complex-valued Spiking Neural Networks" },
  "theory.subtitle": { vi: "Đột Phá Toán Học và Phần Cứng Mới", en: "Mathematical and Hardware Breakthroughs" },
};
Object.assign(translations, moreTranslations);
