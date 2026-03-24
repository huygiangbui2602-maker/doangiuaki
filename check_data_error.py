# ==============================================================================
# KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY CHECK)
# Mục đích: Phát hiện các điểm bất thường (Outliers/Anomalies) và lỗi logic.
# ==============================================================================

import os        # Giao tiếp hệ điều hành (Kiểm tra file)
import gc        # Garbage Collector: Ép dọn rác, xả RAM
import warnings  # Ẩn các cảnh báo vặt
import pandas as pd

warnings.filterwarnings('ignore')

def main():
    file_path = 'instagram_usage_lifestyle.csv'
    
    if not os.path.exists(file_path):
        print(f"[ LỖI ] Không tìm thấy '{file_path}'. Hãy chạy 'train_model.py' để tải dữ liệu trước nhé!")
        return

    try:
        df = pd.read_csv(file_path, low_memory=False)
        total_rows = len(df)
        print("======== BÁO CÁO KIỂM TRA LOGIC DỮ LIỆU ========\n")

        # 1. Tuổi và Tình trạng việc làm
        mask1 = (df['age'] < 18) & (df['employment_status'].isin(['Retired', 'Full-time employed']))
        error1 = df[mask1]
        print(f"[1] Số dòng lỗi Trẻ em đi làm/nghỉ hưu: {len(error1):,}")
        if len(error1) > 0:
            print("    -> Ví dụ minh chứng (3 dòng đầu):")
            # In ra 3 dòng đầu tiên của các cột liên quan để xem thử
            print(error1[['age', 'employment_status']].head(3).to_string(), "\n")

        # 2. Người nghỉ hưu
        mask2 = (df['employment_status'] == 'Retired') & (df['weekly_work_hours'] > 10)
        error2 = df[mask2]
        print(f"[2] Số dòng lỗi Người nghỉ hưu cày cuốc (>10h/tuần): {len(error2):,}")
        if len(error2) > 0:
            print("    -> Ví dụ minh chứng (3 dòng đầu):")
            print(error2[['age', 'employment_status', 'weekly_work_hours']].head(3).to_string(), "\n")

        # 3. Tương tác mạng xã hội
        mask3 = (df['daily_active_minutes_instagram'] < 10) & (df['user_engagement_score'] > 5.0)
        error3 = df[mask3]
        print(f"[3] Số dòng lỗi Ít dùng IG nhưng điểm tương tác cao (>5.0): {len(error3):,}")
        if len(error3) > 0:
            print("    -> Ví dụ minh chứng (3 dòng đầu):")
            print(error3[['daily_active_minutes_instagram', 'user_engagement_score']].head(3).to_string(), "\n")

        # ==========================================================
        # TỔNG KẾT (DATA QUALITY SUMMARY)
        # ==========================================================
        total_error_mask = mask1 | mask2 | mask3 
        total_error_rows = total_error_mask.sum()
        
        clean_rows = total_rows - total_error_rows
        error_percent = (total_error_rows / total_rows) * 100

        print("="*55)
        print(" 📊 BẢNG TỔNG KẾT CHẤT LƯỢNG DỮ LIỆU ".center(55, "*"))
        print("="*55)
        print(f"🔹 Tổng số dòng dữ liệu gốc   : {total_rows:,} dòng")
        print(f"❌ Tổng số dòng chứa dữ liệu rác: {total_error_rows:,} dòng ({error_percent:.2f}%)")
        print(f"✅ Số dòng dữ liệu sạch còn lại : {clean_rows:,} dòng")
        print("="*55)
        
        if total_error_rows > 0:
            print("-> KẾT LUẬN: CẦN LỌC BỎ CÁC DÒNG RÁC NÀY TRƯỚC KHI TRAIN AI ĐỂ TRÁNH NHIỄU MÔ HÌNH!\n")
        else:
            print("-> KẾT LUẬN: DỮ LIỆU RẤT SẠCH, CÓ THỂ ĐƯA VÀO HUẤN LUYỆN NGAY!\n")
            
        del df
        gc.collect()

    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc dữ liệu: {e}")

if __name__ == "__main__":
    main()