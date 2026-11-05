import re
import json

def has_vietnamese(text):
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    return any(c in text for c in vi_chars)

with open('src/components/SNNComparison.tsx', 'r') as f:
    code = f.read()

# find all strings between > and <
matches = re.findall(r'>([^<]+)<', code)
results = [m.strip() for m in matches if has_vietnamese(m.strip())]

print(json.dumps(results[:30], ensure_ascii=False, indent=2))
