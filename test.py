# ====================================================================
# ĐỒ ÁN MÔN HỌC - FILE 3: KIỂM THỬ KỊCH BẢN TỰ ĐỘNG (SCENARIO TESTING)
# Mục đích: Test độ hiệu quả của model AI
# ====================================================================

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

def run_scenario_tests():
    print("="*65)
    print("CHƯƠNG TRÌNH KIỂM THỬ MÔ HÌNH (10 KỊCH BẢN THỰC TẾ)")
    print("="*65)

    try:
        model_data = joblib.load('stress_model.pkl')
    except FileNotFoundError:
        print("[ LỖI ] Không tìm thấy 'stress_model.pkl'. Hãy chạy train_model.py trước!")
        return

    model = model_data['model']
    feature_columns = model_data['feature_columns']
    median_values = model_data['median_values']

    # 10 kịch bản ngoại suy
    scenarios = [
        {"name": "1. Nguoi cao tuoi nghi huu (Khong ap luc)", "age": 65, "gender": "Nam", "sleep": 8.0, "reels": 15, "work": 0, "min": 0, "max": 10, "desc": "< 10 diem"},
        {"name": "2. Sinh vien 'Chua lanh' (Song chill, it lam)", "age": 21, "gender": "Nu", "sleep": 8.0, "reels": 45, "work": 15, "min": 10, "max": 16, "desc": "10 - 16 diem"},
        {"name": "3. Nguoi choi he ky luat (Ngu du, it MXH, lam viec chuan)", "age": 30, "gender": "Nam", "sleep": 7.5, "reels": 20, "work": 40, "min": 14, "max": 20, "desc": "14 - 20 diem"},
        {"name": "4. Nguoi di lam tieu chuan (Can bang work-life)", "age": 26, "gender": "Nu", "sleep": 7.0, "reels": 90, "work": 40, "min": 18, "max": 25, "desc": "18 - 25 diem"},
        {"name": "5. Nguoi that nghiep nghien MXH (Khong lam, MXH nhieu)", "age": 24, "gender": "Nam", "sleep": 6.0, "reels": 300, "work": 0, "min": 25, "max": 33, "desc": "25 - 33 diem"},
        {"name": "6. Sinh vien mua thi (Thuc khuya, ap luc bai vo)", "age": 20, "gender": "Nu", "sleep": 4.5, "reels": 60, "work": 50, "min": 25, "max": 32, "desc": "25 - 32 diem"},
        {"name": "7. IT Coder chay Deadline (Thieu ngu, lam nhieu)", "age": 25, "gender": "Nam", "sleep": 5.0, "reels": 120, "work": 60, "min": 27, "max": 34, "desc": "27 - 34 diem"},
        {"name": "8. Doanh nhan khoi nghiep (Lam cuc nhieu, it MXH)", "age": 32, "gender": "Nu", "sleep": 4.0, "reels": 20, "work": 80, "min": 31, "max": 37, "desc": "31 - 37 diem"},
        {"name": "9. Me bim sua (Mat ngu tram trong, viec ngap dau)", "age": 29, "gender": "Nu", "sleep": 3.0, "reels": 150, "work": 70, "min": 35, "max": 40, "desc": "35 - 40 diem"},
        {"name": "10. Ca Dot Bien (Ban mang lam viec, thuc trang luot MXH)","age": 28, "gender": "Nam", "sleep": 2.0, "reels": 240, "work": 90, "min": 38, "max": 40, "desc": "38 - 40 diem"}
    ]

    print(f"-> Đang chạy kiểm thử tự động cho {len(scenarios)} kịch bản...\n")
    print("-" * 65)

    passed_count = 0

    for i, s in enumerate(scenarios):
        user_data = pd.DataFrame([median_values])
        
        for col in user_data.columns:
            col_name = str(col).lower().strip()
            if 'age' in col_name or 'tuổi' in col_name: user_data[col] = s['age']
            elif 'sleep' in col_name or 'ngủ' in col_name: user_data[col] = s['sleep']
            elif 'reels' in col_name or 'instagram' in col_name: user_data[col] = s['reels']
            elif 'work' in col_name or 'làm' in col_name: user_data[col] = s['work']
            elif 'gender' in col_name or 'giới' in col_name: user_data[col] = 0
            
        for col in user_data.columns:
            col_name = str(col).lower().strip()
            if s['gender'] == 'Nam' and ('nam' in col_name or ('male' in col_name and 'female' not in col_name)):
                user_data[col] = 1
            elif (s['gender'] == 'Nữ' or s['gender'] == 'Nu') and ('nu' in col_name or 'nữ' in col_name or 'female' in col_name):
                user_data[col] = 1
                
        user_data = user_data[feature_columns]
        
        # Dự đoán
        raw_score = model.predict(user_data)[0]
        final_score = np.clip(raw_score, 0, 40)
        
        # Chấm điểm (Cho phép sai số 1.5 điểm vì bản chất của ML là xấp xỉ)
        is_passed = (s['min'] - 1.5) <= final_score <= (s['max'] + 1.5)
        
        if is_passed:
            status_text = "[ ĐẠT ]"
            passed_count += 1
        else:
            status_text = "[ KHÔNG ĐẠT ]"

        print(f"TEST {i+1}: {s['name']}")
        print(f"  - Đầu vào : T={s['age']} | Ngủ: {s['sleep']}h | Reels: {s['reels']}p | Làm: {s['work']}h")
        print(f"  - Kỳ vọng : {s['desc']}")
        print(f"  - Thực tế : {final_score:.1f} điểm -> {status_text}")
        print("-" * 65)

    # Đánh giá theo chuẩn nghiệp vụ thực tế
    print("\n[ KẾT LUẬN TỪ HỆ THỐNG ĐÁNH GIÁ CHUYÊN MÔN ]")
    print(f"Tỷ lệ vượt qua (Pass Rate): {passed_count}/{len(scenarios)}")

    if passed_count >= 9:
        print("Trạng thái: ĐẠT CHUẨN XUẤT SẮC (Sẵn sàng Deploy)")
        print("Mô hình bao quát được hầu hết các trường hợp ngoại suy phức tạp.")
    elif passed_count >= 7:
        print("Trạng thái: ĐẠT (Mức Khá/Giỏi)")
        print("Không có AI nào hoàn hảo 100%. Mức Pass 70-80% kịch bản góc (Edge cases)")
        print("đã là một thành công lớn cho mô hình Linear Regression cơ bản.")
    else:
        print("Trạng thái: CHƯA ĐẠT (Cần huấn luyện lại)")
        print("Mô hình đang dự đoán sai logic tâm lý quá nhiều. Cần kiểm tra lại dữ liệu.")
    print("="*65)

if __name__ == "__main__":
    run_scenario_tests()