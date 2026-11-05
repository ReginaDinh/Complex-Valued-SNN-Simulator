with open('src/components/NetworkTab.tsx', 'r') as f:
    code = f.read()

code = code.replace("{isEn ? 'SNN Network Configuration' : {isEn ? 'SNN Network Configuration' : 'Cài Đặt Cấu Hình Mạng SNN'}}/span>", "{isEn ? 'SNN Network Configuration' : 'Cài Đặt Cấu Hình Mạng SNN'}</span>")
code = code.replace("{isEn ? 'Learning Rate:' : {isEn ? 'Learning Rate:' : 'Tốc độ học (Learning Rate):'}}/span>", "{isEn ? 'Learning Rate:' : 'Tốc độ học (Learning Rate):'}</span>")
code = code.replace("{isEn ? 'Class 0 Freq (Low):' : {isEn ? 'Class 0 Freq (Low):' : 'Tần số Lớp 0 (Thấp):'}}/span>", "{isEn ? 'Class 0 Freq (Low):' : 'Tần số Lớp 0 (Thấp):'}</span>")
code = code.replace("{isEn ? 'Class 1 Freq (High):' : {isEn ? 'Class 1 Freq (High):' : 'Tần số Lớp 1 (Cao):'}}/span>", "{isEn ? 'Class 1 Freq (High):' : 'Tần số Lớp 1 (Cao):'}</span>")
code = code.replace("{isEn ? 'No data' : {isEn ? 'No data' : 'Chưa có dữ liệu'}}", "{isEn ? 'No data' : 'Chưa có dữ liệu'}")
code = code.replace("{isEn ? 'Loss' : 'Loss'}", "Loss")

with open('src/components/NetworkTab.tsx', 'w') as f:
    f.write(code)
