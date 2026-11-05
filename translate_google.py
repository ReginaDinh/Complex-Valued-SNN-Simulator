import json
import urllib.request
import urllib.parse
import time
import os

def translate_text(text):
    if not text.strip(): return text
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=vi&tl=en&dt=t&q=" + urllib.parse.quote(text)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode('utf-8'))
            return "".join([x[0] for x in res[0]])
    except Exception as e:
        print(f"Error translating '{text[:10]}': {e}")
        return text

with open('translations_todo.json', 'r') as f:
    todo = json.load(f)

print(f"Translating {len(todo)} items...")

translations = {}
for i, vi in enumerate(todo.keys()):
    en = translate_text(vi)
    translations[vi] = en
    if i % 10 == 0:
        print(f"Progress: {i}/{len(todo)}")
        time.sleep(1) # avoid rate limit

with open('translations.json', 'w') as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

files = [
    'src/components/SNNComparison.tsx', 
    'src/components/ChoiLabTab.tsx', 
    'src/components/VisionSNN.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/TheoryTab.tsx',
    'src/components/NeuronTab.tsx',
    'src/components/PytorchTab.tsx'
]

for filename in files:
    with open(filename, 'r') as f:
        code = f.read()
        
    for vi, en in translations.items():
        if vi in code and en:
            en_safe = en.replace("'", "\\'").replace("\n", " ").replace("{", "").replace("}", "")
            code = code.replace(f">{vi}<", f">{{isEn ? '{en_safe}' : '{vi}'}}")
            code = code.replace(f'"{vi}"', f"{{isEn ? '{en_safe}' : '{vi}'}}")

    with open(filename, 'w') as f:
        f.write(code)

print("Done.")
