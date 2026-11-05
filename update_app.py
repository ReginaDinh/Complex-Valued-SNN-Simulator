import re

with open('src/main.tsx', 'r') as f:
    main_code = f.read()

if "LanguageProvider" not in main_code:
    main_code = main_code.replace("import App from './App'", "import App from './App'\nimport { LanguageProvider } from './contexts/LanguageContext'")
    main_code = main_code.replace("<App />", "<LanguageProvider><App /></LanguageProvider>")
    with open('src/main.tsx', 'w') as f:
        f.write(main_code)

with open('src/App.tsx', 'r') as f:
    app_code = f.read()

if "useLanguage" not in app_code:
    app_code = app_code.replace("import { useState } from 'react';", "import { useState } from 'react';\nimport { useLanguage } from './contexts/LanguageContext';")
    
    app_code = app_code.replace("export default function App() {", "export default function App() {\n  const { t, language, setLanguage } = useLanguage();")
    
    app_code = app_code.replace("Mạng Nơ-ron Xung Số Phức", "{t('app.title')}")
    app_code = app_code.replace("Mô hình toán học và mã nguồn dựa trên các ấn bản SOTA IEEE & arXiv năm 2022-2026. Phát triển bởi Google AI Studio Build.", "{t('app.footer')}")
    
    # Replace the tabs definition to use translation
    tabs_replacement = """[
              { id: 'theory', icon: BookOpen, label: t('app.tab.theory') },
              { id: 'comparison', icon: GitCompare, label: t('app.tab.comparison') },
              { id: 'pytorch', icon: Terminal, label: t('app.tab.pytorch') },
              { id: 'neuron', icon: Activity, label: t('app.tab.neuron') },
              { id: 'network', icon: Layers, label: t('app.tab.network') },
              { id: 'vision', icon: Eye, label: t('app.tab.vision') },
              { id: 'choi-lab', icon: Cpu, label: t('app.tab.choilab') }
            ]"""
    app_code = re.sub(r'\[\s*\{\s*id:\s*\'theory\'.*?\]', tabs_replacement, app_code, flags=re.DOTALL)
    
    # Add a language switcher button next to the title
    switcher_jsx = """
            </div>
            <div className="ml-4 flex items-center bg-elegant-card/50 rounded-lg p-1 border border-elegant-muted/20">
              <button 
                onClick={() => setLanguage('vi')} 
                className={`px-2 py-1 text-xs font-bold rounded-md transition-colors ${language === 'vi' ? 'bg-elegant-cyan text-elegant-bg' : 'text-elegant-muted hover:text-white'}`}
              >
                VI
              </button>
              <button 
                onClick={() => setLanguage('en')} 
                className={`px-2 py-1 text-xs font-bold rounded-md transition-colors ${language === 'en' ? 'bg-elegant-cyan text-elegant-bg' : 'text-elegant-muted hover:text-white'}`}
              >
                EN
              </button>
            </div>
          </div>
"""
    app_code = app_code.replace("            </div>\n          </div>", switcher_jsx)
    
    with open('src/App.tsx', 'w') as f:
        f.write(app_code)
