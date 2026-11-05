with open('src/main.tsx', 'r') as f:
    code = f.read()

if "LanguageProvider" not in code:
    pass # Wait, it is in code but the import is missing.
    
code = code.replace("import App from './App.tsx';", "import App from './App.tsx';\nimport { LanguageProvider } from './contexts/LanguageContext';")

with open('src/main.tsx', 'w') as f:
    f.write(code)
