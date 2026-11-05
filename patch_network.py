import re

with open('src/components/NetworkTab.tsx', 'r') as f:
    code = f.read()

replaces = {
    "Mô-đun rò rỉ |β|:": "Leak Magnitude |β|:",
    "Pha rò rỉ ∠β (rad):": "Leak Phase ∠β (rad):",
    "Ngưỡng kích hoạt (Threshold):": "Activation Threshold:",
    "Cơ chế Reset:": "Reset Mechanism:",
    "Cấu Hình Tín Hiệu Đầu Vào": "Input Signal Configuration",
    "Lệch Pha Lớp 1 (Kênh):": "Class 1 Phase Shift (Channel):",
    "EPOCH / SỐ BƯỚC:": "EPOCH / STEPS:",
    "ĐỘ CHÍNH XÁC:": "ACCURACY:",
    "LOSS TRÊN MẪU HIỆN TẠI (ĐỘ LỖI):": "CURRENT SAMPLE LOSS:",
    "Nhiệm vụ Phân loại Phổ Tần Số:": "Frequency Spectrum Classification Task:",
    "Lớp 0:": "Class 0:",
    "Tín hiệu tần số thấp (1.0Hz) đồng pha.": "Low frequency (1.0Hz) in-phase signal.",
    "Lớp 1:": "Class 1:",
    "Tín hiệu tần số cao (4.5Hz) bị lệch pha góc 90° giữa các kênh.": "High frequency (4.5Hz) signal with 90° phase shift between channels.",
    "Trọng số phức được tối ưu hóa liên tục để phát hiện đồng thời tần số và phân rã pha của các xung điện.": "Complex weights are continuously optimized to simultaneously detect frequency and phase decay of electrical spikes.",
    "Biểu Đồ Độ Lỗi Loss": "Loss Chart",
    "Bắt đầu huấn luyện để vẽ đồ thị": "Start training to plot",
    "Cũ nhất": "Oldest",
    "Epoch hiện tại ({epoch})": "Current Epoch ({epoch})",
    "Phân Tích 16 Kênh Tín Hiệu Đầu Vào (Input Phase Map)": "16-Channel Input Signal Analysis (Phase Map)",
    "Luồng Xung Kênh Đặc Trưng (Kênh 0-15)": "Feature Channel Spike Stream (Channels 0-15)",
    "Thời Gian (steps)": "Time (steps)",
    "Luồng Xung 16 Kênh Đầu Vào": "16-Channel Input Spike Stream",
}

for vi, en in replaces.items():
    code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
    code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")

# Special handling for JS expressions containing Vietnamese
code = code.replace("{isTraining ? 'Tạm Dừng Huấn Luyện' : 'Bắt Đầu Huấn Luyện'}", "{isTraining ? (isEn ? 'Pause Training' : 'Tạm Dừng Huấn Luyện') : (isEn ? 'Start Training' : 'Bắt Đầu Huấn Luyện')}")
code = code.replace("({isTraining ? 'Tạm Dừng' : 'Bắt Đầu BPTT'})", "({isTraining ? (isEn ? 'Pause' : 'Tạm Dừng') : (isEn ? 'Start BPTT' : 'Bắt Đầu BPTT')})")

with open('src/components/NetworkTab.tsx', 'w') as f:
    f.write(code)

