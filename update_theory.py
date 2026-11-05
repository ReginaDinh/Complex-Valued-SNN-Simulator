with open('src/components/TheoryTab.tsx', 'r') as f:
    code = f.read()

if "useLanguage" not in code:
    code = code.replace("from 'lucide-react';", "from 'lucide-react';\nimport { useLanguage } from '../contexts/LanguageContext';")
    code = code.replace("export default function TheoryTab() {", "export default function TheoryTab() {\n  const { t } = useLanguage();")
    
    code = code.replace("Mô hình SNN trong Miền Phức (Complex-valued SNN)", "{t('theory.title', 'Mô hình SNN trong Miền Phức (Complex-valued SNN)')}")
    
    with open('src/components/TheoryTab.tsx', 'w') as f:
        f.write(code)
