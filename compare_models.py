# ===============================================================================
# ĐỒ ÁN MÔN HỌC - FILE: SO SÁNH CÁC MÔ HÌNH, TỐC ĐỘ VÀ SAI SỐ
# Mục đích: Đánh giá R-squared, MSE (Mean Squared Error) và Thời gian huấn luyện
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import gc
import time

# Tắt cảnh báo để màn hình hiển thị sạch đẹp
warnings.filterwarnings('ignore')

FILE_NAME = 'instagram_usage_lifestyle.csv'

def get_clean_data(file_path):
    """Tiền xử lý dữ liệu y hệt file train_model.py"""
    if not os.path.exists(file_path):
        print(f"[ LỖI ] Không tìm thấy {file_path}. Vui lòng tải dữ liệu trước.")
        return None, None

    df = pd.read_csv(file_path, low_memory=False)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Lấy 50k dòng để Random Forest chạy nhanh, dễ so sánh
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    error_mask = (
        ((df['age'] < 18) & (df['employment_status'].isin(['Retired', 'Full-time employed']))) |
        ((df['employment_status'] == 'Retired') & (df['weekly_work_hours'] > 10)) |
        ((df['daily_active_minutes_instagram'] < 10) & (df['user_engagement_score'] > 5.0)) |
        ((df['sleep_hours_per_night'] < 3) | (df['sleep_hours_per_night'] > 15))
    )
    df_clean = df[~error_mask].copy()
    
    del df
    gc.collect() # Giải phóng RAM

    df_clean['perceived_stress_score'] = (
        28.0 
        - df_clean['sleep_hours_per_night'] * 2.8 
        + df_clean['weekly_work_hours'] * 0.18 
        + df_clean['time_on_reels_per_day'] * 0.04 
        + np.random.normal(0, 1.5, len(df_clean))
    )
    df_clean['perceived_stress_score'] = np.clip(df_clean['perceived_stress_score'], 0, 40)

    X = df_clean.drop(columns=['perceived_stress_score', 'user_id', 'id'], errors='ignore')
    y = df_clean['perceived_stress_score'].astype('float32')

    X_encoded = pd.get_dummies(X, drop_first=True, dtype='int8')
    for col in X_encoded.select_dtypes(include=['float64']).columns:
        X_encoded[col] = X_encoded[col].astype('float32')

    return train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def compare_models():
    data = get_clean_data(FILE_NAME)
    if data[0] is None: return
    X_train, X_test, y_train, y_test = data

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=42)
    }

    results = {}

    print("--- ĐANG HUẤN LUYỆN VÀ SO SÁNH CÁC MÔ HÌNH ---")
    for name, model in models.items():
        start_time = time.time() # Bắt đầu bấm giờ
        
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        end_time = time.time() # Kết thúc bấm giờ
        
        duration = end_time - start_time
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred) # Tính Sai số MSE
        
        results[name] = {'R-squared (↑)': r2, 'MSE (↓)': mse, 'Thời gian (giây) (↓)': duration}
        print(f"-> Đã chạy xong: {name}")

    # ==========================================================
    # 1. TẠO BẢNG SO SÁNH TRÊN TERMINAL
    # ==========================================================
    # Chuyển dữ liệu thành dạng bảng (DataFrame) để in ra cho đẹp
    results_df = pd.DataFrame(results).T
    print("\n" + "="*65)
    print(" BẢNG TỔNG KẾT ĐÁNH GIÁ MÔ HÌNH ".center(65, "*"))
    print("="*65)
    # Dùng to_markdown() hoặc to_string() để in bảng cực xịn
    print(results_df.round(4).to_string())
    print("="*65)
    print("Ghi chú: (↑) Càng cao càng tốt | (↓) Càng thấp càng tốt\n")

    # ==========================================================
    # 2. VẼ BIỂU ĐỒ 3 CỘT (R2, MSE, TIME)
    # ==========================================================
    names = list(results.keys())
    r2_scores = [results[n]['R-squared (↑)'] for n in names]
    mse_scores = [results[n]['MSE (↓)'] for n in names]
    times = [results[n]['Thời gian (giây) (↓)'] for n in names]

    # Tạo khung 1 hàng, 3 cột
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cột 1: R-squared (Cao là tốt)
    sns.barplot(x=names, y=r2_scores, ax=axes[0], palette='viridis')
    axes[0].set_title("1. ĐỘ CHÍNH XÁC (R-squared)\n[Càng CAO càng tốt]", fontweight='bold')
    axes[0].set_ylim(0, 1)
    for i, score in enumerate(r2_scores):
        axes[0].text(i, score + 0.02, f"{score:.4f}", ha='center', fontweight='bold')

    # Cột 2: MSE (Thấp là tốt)
    sns.barplot(x=names, y=mse_scores, ax=axes[1], palette='flare')
    axes[1].set_title("2. SAI SỐ TRUNG BÌNH (MSE)\n[Càng THẤP càng tốt]", fontweight='bold')
    for i, score in enumerate(mse_scores):
        axes[1].text(i, score + (max(mse_scores)*0.02), f"{score:.2f}", ha='center', fontweight='bold', color='black')

    # Cột 3: Tốc độ (Thấp là tốt)
    sns.barplot(x=names, y=times, ax=axes[2], palette='magma')
    axes[2].set_title("3. TỐC ĐỘ XỬ LÝ (Giây)\n[Càng THẤP càng tốt]", fontweight='bold')
    for i, t in enumerate(times):
        axes[2].text(i, t + (max(times)*0.02), f"{t:.3f}s", ha='center', fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    print("-> Đã xuất ảnh biểu đồ 3 trục ra file 'model_comparison.png'")

if __name__ == "__main__":
    compare_models()