const fs = require('fs');

async function translate(text) {
  if (!text.trim()) return text;
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=vi&tl=en&dt=t&q=${encodeURIComponent(text)}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    return data[0].map(x => x[0]).join('');
  } catch (e) {
    console.error("Error:", text.substring(0,10));
    return text;
  }
}

async function run() {
  const todo = JSON.parse(fs.readFileSync('translations_todo.json', 'utf8'));
  const keys = Object.keys(todo);
  const translations = {};
  
  for (let i = 0; i < keys.length; i += 5) {
    const chunk = keys.slice(i, i + 5);
    console.log(`Translating chunk ${i} to ${i+5}...`);
    const results = await Promise.all(chunk.map(k => translate(k)));
    chunk.forEach((k, idx) => {
      translations[k] = results[idx];
    });
    // Wait 500ms
    await new Promise(r => setTimeout(r, 500));
  }
  
  fs.writeFileSync('translations.json', JSON.stringify(translations, null, 2));
  
  const files = [
    'src/components/SNNComparison.tsx', 
    'src/components/ChoiLabTab.tsx', 
    'src/components/VisionSNN.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/TheoryTab.tsx',
    'src/components/NeuronTab.tsx',
    'src/components/PytorchTab.tsx'
  ];

  for (const file of files) {
    let code = fs.readFileSync(file, 'utf8');
    for (const [vi, en] of Object.entries(translations)) {
      if (!en) continue;
      const en_safe = en.replace(/'/g, "\\'").replace(/\n/g, " ").replace(/[{}]/g, "");
      code = code.replace(new RegExp(`>${vi.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}<`, 'g'), `>{isEn ? '${en_safe}' : '${vi}'}<`);
      code = code.replace(new RegExp(`"${vi.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g'), `"{isEn ? '${en_safe}' : '${vi}'}"`);
    }
    fs.writeFileSync(file, code);
  }
  console.log("Done.");
}

run();
