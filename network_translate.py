with open('src/components/NetworkTab.tsx', 'r') as f:
    code = f.read()

if "useLanguage" not in code:
    code = code.replace("from 'lucide-react';", "from 'lucide-react';\nimport { useLanguage } from '../contexts/LanguageContext';")
    code = code.replace("export default function NetworkTab() {", "export default function NetworkTab() {\n  const { language } = useLanguage();\n  const isEn = language === 'en';")

replaces = {
    "Cài Đặt Cấu Hình Mạng SNN": "SNN Network Configuration",
    "Tốc độ học (Learning Rate):": "Learning Rate:",
    "Tần số Lớp 0 (Thấp):": "Class 0 Freq (Low):",
    "Tần số Lớp 1 (Cao):": "Class 1 Freq (High):",
    "Pha Lớp 1 (rad):": "Class 1 Phase (rad):",
    "Huấn Luyện (Training)": "Training Control",
    "Chưa có dữ liệu": "No data",
    "Tạm Dừng": "Pause",
    "Bắt Đầu BPTT": "Start BPTT",
    "Reset Mạng": "Reset Network",
    "Chỉ số Giám sát": "Monitoring Metrics",
    "Epoch": "Epoch",
    "Độ chính xác": "Accuracy",
    "Loss": "Loss",
    "Đồ thị Loss BPTT": "BPTT Loss Graph",
    "Lịch sử hội tụ Surrogate Gradient": "Surrogate Gradient convergence history",
    "Thời gian (Epoch)": "Time (Epoch)",
    "Dữ liệu Lớp": "Class Data",
    "Mẫu hiện tại:": "Current Sample:",
    "Xung Thời Gian & Pha Đầu Vào": "Input Spikes & Phase",
    "t = 30 steps": "t = 30 steps",
    "Tín Hiệu Gốc (Tần số & Pha)": "Original Signal (Freq & Phase)",
    "Luồng Xung Đầu Vào theo thời gian": "Input Spike Stream over time",
    "Tỷ lệ Kích hoạt Đầu ra (Output Firing Rate)": "Output Firing Rate",
    "So khớp tín hiệu với Lớp mục tiêu": "Signal matching with Target Class",
    "Lớp 0 (Trầm)": "Class 0 (Low)",
    "Lớp 1 (Cao)": "Class 1 (High)",
    "đang chờ": "waiting",
    "đang huấn luyện...": "training...",
}

for vi, en in replaces.items():
    code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
    code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")
    code = code.replace(f"'{vi}'", f"{{isEn ? '{en}' : '{vi}'}}")
    # For standalone text nodes
    code = code.replace(f"<span>{vi}</span>", f"<span>{{isEn ? '{en}' : '{vi}'}}</span>")
    code = code.replace(f" {vi} ", f" {{isEn ? '{en}' : '{vi}'}} ")

with open('src/components/NetworkTab.tsx', 'w') as f:
    f.write(code)

