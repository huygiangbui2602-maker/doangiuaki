import pandas as pd

def inspect_and_show_errors(file_path):
    print("="*65)
    print("HỆ THỐNG KIỂM TRA CHẤT LƯỢNG DATA ")
    print("="*65)
    
    print(f"Đang tải file {file_path}...\n")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("[ LỖI ] Không tìm thấy file CSV!")
        return

    # 1. Định nghĩa các quy tắc bắt lỗi logic
    error_age_work = df[(df['age'] < 18) & (df['employment_status'].isin(['Retired', 'Full-time employed']))]
    error_retired_working = df[(df['employment_status'] == 'Retired') & (df['weekly_work_hours'] > 10)]
    error_fake_engagement = df[(df['daily_active_minutes_instagram'] < 10) & (df['user_engagement_score'] > 5.0)]

    # 2. Hàm hỗ trợ in ra các dòng lỗi cụ thể
    def print_sample(error_df, title, columns_to_show):
        print(f"[ CẢNH BÁO ] {title}")
        print(f"Tổng số ca vi phạm: {len(error_df):,} dòng")
        
        if not error_df.empty:
            print("   -> TRÍCH XUẤT 3 DÒNG LỖI ĐIỂN HÌNH:")
            # In ra 3 dòng đầu tiên, lấy các cột cần thiết, kèm số thứ tự dòng (Index)
            print(error_df[columns_to_show].head(3).to_string(index=True))
        else:
            print("   -> Dữ liệu sạch: Không phát hiện lỗi này.")
        print("-" * 65)

    # 3. Tiến hành in dữ liệu lỗi
    
    # Lỗi 1: Trẻ em nhưng đi làm Full-time hoặc Nghỉ hưu
    print_sample(
        error_age_work, 
        "LỖI 1: Trẻ em (< 18 tuổi) khai báo đi làm Full-time hoặc Nghỉ hưu",
        ['age', 'employment_status', 'weekly_work_hours']
    )

    # Lỗi 2: Đã nghỉ hưu nhưng làm việc nhiều giờ
    print_sample(
        error_retired_working, 
        "LỖI 2: Khai báo Đã nghỉ hưu (Retired) nhưng làm việc > 10 tiếng/tuần",
        ['age', 'employment_status', 'weekly_work_hours']
    )

    # Lỗi 3: Tương tác ảo (Không mở app nhưng điểm tương tác cao)
    print_sample(
        error_fake_engagement, 
        "LỖI 3: Mở app < 10 phút/ngày nhưng điểm tương tác > 5.0",
        ['daily_active_minutes_instagram', 'user_engagement_score']
    )

if __name__ == "__main__":
    inspect_and_show_errors('instagram_usage_lifestyle.csv')