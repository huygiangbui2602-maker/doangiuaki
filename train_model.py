# =====================================================================================================
# ĐỒ ÁN MÔN HỌC - FILE: HUẤN LUYỆN VÀ LƯU MÔ HÌNH
# Mục đích: Làm mô hình học máy vẽ biểu đồ (Tích hợp tự động cài đầu vào từ Google Drive nếu không có)
# Mô hình: Hồi quy tuyến tính (Có Chuẩn hóa dữ liệu StandardScaler)
# =====================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib   # Đóng gói và lưu mô hình AI (file .pkl) xuống ổ cứng để tái sử dụng mà không cần train lại
import warnings # Ẩn các cảnh báo vặt (không phải lỗi) của thư viện để màn hình Terminal sạch sẽ, chuyên nghiệp
import gc       # Garbage Collector: Ép dọn rác, xả RAM ngay lập tức để chống treo máy khi xử lý data lớn
import gdown    # Thư viện tải file từ Drive
import os       # Thư viện kiểm tra file trong thư mục

# --- CẤU HÌNH TẢI FILE (THÊM MỚI) ---
FILE_ID = '1_5A5yfMU9ywFO5z-OtEOegKPh0SlmkYo' 
FILE_NAME = 'instagram_usage_lifestyle.csv'

def download_data():
    """Hàm tự động tải file nếu máy chưa có (THÊM MỚI)"""
    if not os.path.exists(FILE_NAME):
        print(f"[ THÔNG BÁO ] Không tìm thấy {FILE_NAME}. Đang tải từ Google Drive...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            gdown.download(url, FILE_NAME, quiet=False)
            print("[ THÀNH CÔNG ] Đã tải xong dữ liệu.")
        except Exception as e:
            print(f"[ LỖI ] Không thể tải file: {e}")
    else:
        print(f"[ OK ] File {FILE_NAME} đã sẵn sàng.")

warnings.filterwarnings('ignore') 

def preprocess_data(file_path):
    print("[ 1/5 ] Đang tải dữ liệu gốc...")
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'user_id' in df.columns:
        df.drop(columns=['user_id'], inplace=True)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
        
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    print("[ 2/5 ] Đang tối ưu hóa dung lượng (Sampling & Filtering)...")
    if len(df) > 200000:
        df = df.sample(n=200000, random_state=42)
        
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 10:
            df.drop(columns=[col], inplace=True)

    print("[ 3/5 ] Đang dọn dẹp dữ liệu rác (Data Cleaning)...")
    initial_rows = len(df)
    error_mask = (
        ((df['age'] < 18) & (df['employment_status'].isin(['Retired', 'Full-time employed']))) |
        ((df['employment_status'] == 'Retired') & (df['weekly_work_hours'] > 10)) |
        ((df['daily_active_minutes_instagram'] < 10) & (df['user_engagement_score'] > 5.0)) |
        ((df['sleep_hours_per_night'] < 3) | (df['sleep_hours_per_night'] > 15))
    )
    df_clean = df[~error_mask].copy()
    
    del df 
    gc.collect() 
    print(f"        -> Đã xóa bỏ {initial_rows - len(df_clean):,} dòng dữ liệu phi logic.")

    # FEATURE ENGINEERING 
    df_clean['perceived_stress_score'] = (
        28.0 
        - df_clean['sleep_hours_per_night'] * 2.8 
        + df_clean['weekly_work_hours'] * 0.18 
        + df_clean['time_on_reels_per_day'] * 0.04 
        + np.random.normal(0, 1.5, len(df_clean))
    )
    
    df_clean['perceived_stress_score'] = np.clip(df_clean['perceived_stress_score'], 0, 40)
 
    X = df_clean.drop(columns=['perceived_stress_score'])
    y = df_clean['perceived_stress_score'].astype('float32') 

    X_encoded = pd.get_dummies(X, drop_first=True, dtype='int8')
    for col in X_encoded.select_dtypes(include=['float64']).columns:
        X_encoded[col] = X_encoded[col].astype('float32')

    feature_columns = X_encoded.columns 
    median_values = X_encoded.median()
    
    del X
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    del X_encoded, y
    gc.collect()

    return X_train, X_test, y_train, y_test, df_clean, feature_columns, median_values

def train(X_train, X_test, y_train, y_test):
    print("[ 4/5 ] Đang huấn luyện AI (StandardScaler + Linear Regression)...")
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    print(f"        -> Hoàn thành! ")
    return model

def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr(numeric_only=True)[['perceived_stress_score']].sort_values(by='perceived_stress_score', ascending=False)
    corr_matrix = corr_matrix.drop('perceived_stress_score') 
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0, fmt=".2f")
    plt.title("NGUYÊN NHÂN GÂY STRESS", fontweight='bold')
    plt.tight_layout()
    plt.savefig("buoc4_heatmap.png")
    plt.close()

def main():

    download_data()
    
    try:
        X_train, X_test, y_train, y_test, df_clean, feature_columns, median_values = preprocess_data(FILE_NAME)
        model = train(X_train, X_test, y_train, y_test)
        
        print("[ 5/5 ] Đang đóng gói bộ não AI và vẽ biểu đồ...")
        plot_heatmap(df_clean)
        
        y_all_scores = df_clean['perceived_stress_score'].copy()
        del df_clean
        gc.collect()
        
        model_data = {
            'model': model,
            'feature_columns': feature_columns,
            'median_values': median_values,
            'y_all': y_all_scores
        }
        joblib.dump(model_data, 'stress_model.pkl')
        print("======== HOÀN TẤT! ĐÃ LƯU FILE 'stress_model.pkl' ========")
        
    except FileNotFoundError:
        print(f"[ LỖI ] Không tìm thấy file '{FILE_NAME}'.")

if __name__ == "__main__":
    main()