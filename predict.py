# ==============================================================================
# ĐỒ ÁN MÔN HỌC - FILE 2: HỆ THỐNG DỰ ĐOÁN
# Tương tác với người dùng qua Terminal
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import joblib  
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Đang khởi động hệ thống AI...")
    try:
        model_data = joblib.load('stress_model.pkl')
    except FileNotFoundError:
        print("[ LỖI ] Không tìm thấy 'stress_model.pkl'. Hãy chạy file train_model.py trước!")
        return

    model = model_data['model']
    feature_columns = model_data['feature_columns']
    median_values = model_data['median_values']
    y_all = model_data['y_all']

    while True:
        print("\n" + "="*60)
        print("HỆ THỐNG DỰ ĐOÁN MỨC ĐỘ STRESS")
        print("="*60)
        
        # 1. Nhập và kiểm tra TUỔI
        while True:
            try:
                age = float(input("- Nhập tuổi của bạn: "))
                if age <= 0 or age > 100:
                    print("  [ LỖI ] Tuổi không hợp lý. Vui lòng nhập lại!")
                    continue
                break
            except ValueError:
                print("  [ LỖI ] Vui lòng nhập bằng số!")

        # 2. Nhập và kiểm tra GIỚI TÍNH
        while True:
            gender = input("- Giới tính (Nam/Nữ): ").strip().capitalize()
            if gender in ['Nam', 'Nữ', 'Nu']:
                break
            print("  [ LỖI ] Bạn chỉ được nhập 'Nam' hoặc 'Nữ'!")

        # 3. Nhập các thói quen
        while True:
            try:
                sleep = float(input("- Số giờ ngủ trung bình mỗi đêm (vd: 6.5): "))
                reels = float(input("- Số phút lướt Reels trên Instagram mỗi ngày (vd: 120): "))
                work = float(input("- Số giờ làm việc/học tập mỗi tuần (vd: 40): "))
                break
            except ValueError:
                print("  [ LỖI ] Vui lòng nhập bằng số!")

        # Tiền xử lý dữ liệu người dùng
        user_data = pd.DataFrame([median_values])
        
        for col in user_data.columns:
            col_name = str(col).lower().strip()
            if 'age' in col_name or 'tuổi' in col_name:
                user_data[col] = age
            elif 'sleep' in col_name or 'ngủ' in col_name:
                user_data[col] = sleep
            elif 'reels' in col_name or 'instagram' in col_name:
                user_data[col] = reels 
            elif 'work' in col_name or 'làm' in col_name:
                user_data[col] = work
            elif 'gender' in col_name or 'giới' in col_name:
                user_data[col] = 0 
                
        for col in user_data.columns:
            col_name = str(col).lower().strip()
            if gender == 'Nam' and ('nam' in col_name or ('male' in col_name and 'female' not in col_name)):
                user_data[col] = 1
            elif (gender == 'Nữ' or gender == 'Nu') and ('nu' in col_name or 'nữ' in col_name or 'female' in col_name):
                user_data[col] = 1
                
        user_data = user_data[feature_columns]
        
        # Dự đoán
        print("\n-> AI đang phân tích dữ liệu...")
        raw_score = model.predict(user_data)[0]
        user_stress_score = np.clip(raw_score, 0, 40)
        percentile = (y_all < user_stress_score).mean() * 100
        
        print("\n" + "-"*60)
        print("KET QUA DU DOAN: {:.1f} / 40 diem".format(user_stress_score))
        print("Ban dang co muc do stress nang hon {:.1f}% cong dong!".format(percentile))
        print("-"*60)
        
        # Vẽ biểu đồ chuông
        print("-> Đang xuất biểu đồ so sánh...")
        


        plt.figure(figsize=(10, 6))
        num_bins = 30
        sns.histplot(y_all, stat='count', bins=num_bins, color='skyblue', alpha=0.4, label='Phan bo cong dong')
        
        mu, std = norm.fit(y_all)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        
        bin_width = (y_all.max() - y_all.min()) / num_bins
        p_scaled = p * len(y_all) * bin_width
        
        plt.plot(x, p_scaled, 'k', linewidth=2.5, label='Duong cong phan bo chuan')
        plt.axvline(user_stress_score, color='red', linestyle='dashed', linewidth=2.5, 
                    label=f'Diem cua ban: {user_stress_score:.1f}')
        
        plt.title("SO SANH MUC DO STRESS CUA BAN VOI CONG DONG", fontweight='bold', fontsize=14)
        plt.xlabel("Diem Stress (PSS-10)")
        plt.ylabel("So nguoi")
        plt.legend()
        plt.tight_layout()
        plt.savefig("buoc5_bieu_do_chuong.png")
        plt.close() 
        print("-> Đã lưu file 'buoc5_bieu_do_chuong.png'.")

        tiep_tuc = input("\nBạn có muốn thử với thông tin khác không? (y/n): ")
        if tiep_tuc.lower() != 'y':
            print("Cảm ơn bạn đã sử dụng hệ thống. Tạm biệt!")
            break

if __name__ == "__main__":
    main()