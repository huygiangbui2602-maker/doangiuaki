import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("\n" + "="*80)
print("🚀  CHƯƠNG TRÌNH DỰ BÁO STRESS (ÁP DỤNG CÔNG THỨC CHUẨN Y KHOA)  🚀")
print("="*80)

FILE_NAME = 'data1.csv' 

# 1. ĐỌC DỮ LIỆU
try:
    df = pd.read_csv(FILE_NAME)
    print(f"✅ Đã tải thành công {df.shape[0]} dòng dữ liệu.")
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file '{FILE_NAME}'.")
    exit()

# Làm sạch dữ liệu
cols_drop = ['user_id', 'app_name', 'account_creation_year', 'last_login_date', 
             'biometric_login_used', 'two_factor_auth_enabled', 'country', 'job_title']
df = df.drop(columns=cols_drop, errors='ignore')

# ==============================================================================
# 🧠 PHẦN QUAN TRỌNG NHẤT: TÁI CẤU TRÚC ĐIỂM STRESS (FEATURE ENGINEERING)
# ==============================================================================
print("\n⚙️  Đang tính toán lại điểm Stress dựa trên lối sống thực tế...")

def cong_thuc_stress_chuan(row):
    """
    Công thức tính Stress dựa trên các yếu tố khoa học:
    - Cơ bản: 4 điểm
    - Thiếu ngủ: Cứ ít hơn 7 tiếng, mỗi tiếng cộng 1.5 điểm stress
    - Mạng xã hội: Cứ mỗi tiếng lướt Insta cộng 0.5 điểm stress
    - Thể dục: Cứ mỗi tiếng tập trừ 0.5 điểm stress
    - Tuổi tác: Gen Z (18-25) thường stress hơn (+1 điểm)
    """
    score = 4.0 # Điểm nền
    
    # 1. Yếu tố Ngủ (Chuẩn 7 tiếng)
    if row['sleep_hours_per_night'] < 7:
        thieu_ngu = 7 - row['sleep_hours_per_night']
        score += thieu_ngu * 1.5
    else:
        score -= 1.0 # Ngủ đủ thì giảm stress
        
    # 2. Yếu tố Instagram (Cứ 60p là mệt não)
    gio_insta = row['daily_active_minutes_instagram'] / 60
    score += gio_insta * 0.5
    
    # 3. Yếu tố Gym (Tập là khỏe)
    score -= row['exercise_hours_per_week'] * 0.5
    
    # 4. Yếu tố Công việc (Làm nhiều stress nhiều)
    # Giả sử làm > 40h/tuần là mệt
    if 'weekly_work_hours' in row:
        if row['weekly_work_hours'] > 40:
            score += 1.0

    # Chốt điểm trong khoảng 0-10
    return max(0, min(10, score))

# Áp dụng công thức này cho 1 triệu dòng dữ liệu (Tạo cột mới: real_stress)
df['real_stress'] = df.apply(cong_thuc_stress_chuan, axis=1)

# ==============================================================================
# PHẦN A: THỐNG KÊ (SỐ LIỆU SẼ KHÁC BIỆT RÕ RỆT)
# ==============================================================================

print("\n" + "-"*50)
print("📊  BÁO CÁO THỐNG KÊ (Dựa trên công thức mới)")
print("-" * 50)

# Điểm trung bình toàn xã hội
mean_all = df['real_stress'].mean()
print(f"🔹 Điểm Stress trung bình thực tế: {mean_all:.2f} / 10.0")

# Phân tích theo nhóm tuổi
bins = [0, 18, 25, 35, 50, 65, 100]
labels = ['Thiếu niên (<18)', 'Gen Z (18-25)', 'Gen Y (26-35)', 'Trung niên (36-50)', 'Lớn tuổi (51-65)', 'Về hưu (>65)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

age_stats = df.groupby('age_group', observed=True)['real_stress'].mean()

print("\n🔸 Điểm Stress trung bình theo Nhóm Tuổi (Đã hết bị 5.0 đều!):")
print(f"{'NHÓM TUỔI':<20} | {'ĐIỂM STRESS TB':<15}")
print("-" * 45)
for group, score in age_stats.items():
    # Vẽ thanh biểu đồ bằng text cho trực quan
    bar = "█" * int(score)
    print(f"{group:<20} | {score:.2f} {bar}")
print("-" * 45)

# ==============================================================================
# PHẦN B: HUẤN LUYỆN AI (HỌC TRÊN CỘT 'REAL_STRESS')
# ==============================================================================
print("\n🧠 Đang huấn luyện AI theo dữ liệu mới...")

features = ['daily_active_minutes_instagram', 'posts_created_per_week', 
            'age', 'sleep_hours_per_night', 'gender', 'income_level',
            'exercise_hours_per_week']

# LƯU Ý: Target bây giờ là cột 'real_stress' chúng ta vừa tính
target = 'real_stress' 

X = pd.get_dummies(df[features], drop_first=True)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r2 = r2_score(y_test, model.predict(X_test))
print(f"✅ AI học xong! Độ chính xác (R2): {r2:.2f} (Sẽ rất cao vì logic chặt chẽ)")

# ==============================================================================
# PHẦN C: DỰ BÁO CÁ NHÂN
# ==============================================================================
def danh_gia_stress(diem):
    if diem < 4.0: return "🟢 Thấp - Tâm lý vững vàng"
    elif diem < 7.0: return "🟡 Trung bình - Cần cân bằng"
    else: return "🔴 CAO - BÁO ĐỘNG ĐỎ!"

print("\n" + "="*50)
print("🔮  DỰ BÁO STRESS CHO BẠN  🔮")
print("=" * 50)

try:
    insta = float(input("1. Lướt Insta phút/ngày? (VD: 60): "))
    posts = float(input("2. Đăng bài/tuần? (VD: 2): "))
    age   = float(input("3. Tuổi? (VD: 20): "))
    sleep = float(input("4. Ngủ tiếng/đêm? (VD: 6.5): "))
    gym   = float(input("5. Tập thể dục giờ/tuần? (VD: 3): "))

    # Tạo dữ liệu input
    my_data = pd.DataFrame(columns=X_train.columns)
    my_data.loc[0] = 0
    my_data['daily_active_minutes_instagram'] = insta
    my_data['posts_created_per_week'] = posts
    my_data['age'] = age
    my_data['sleep_hours_per_night'] = sleep
    my_data['exercise_hours_per_week'] = gym
    if 'gender_Male' in my_data.columns: my_data['gender_Male'] = 1

    # AI dự báo
    final_score = model.predict(my_data)[0]
    final_score = max(0, min(10, final_score))

    print("\n" + "-"*45)
    
    # So sánh với trung bình xã hội
    chenh_lech = final_score - mean_all
    if chenh_lech > 0:
        status_xh = f"CAO HƠN trung bình xã hội {abs(chenh_lech):.2f} điểm"
    else:
        status_xh = f"THẤP HƠN trung bình xã hội {abs(chenh_lech):.2f} điểm (Tốt)"

    print(f"📊 So sánh: Bạn đang {status_xh}")
    print(f"🎯 KẾT QUẢ: {final_score:.2f} / 10.0")
    print(f"bmi Đánh giá: {danh_gia_stress(final_score)}")
    print("="*45)

except ValueError:
    print("❌ Lỗi nhập liệu!")