
﻿ĐỒ ÁN: HỆ THỐNG DỰ ĐOÁN MỨC ĐỘ STRESS 
 
Hệ thống sử dụng mô hình Hồi quy tuyến tính (Linear Regression) để dự đoán điểm Stress dựa trên thói quen sinh hoạt và sử dụng mạng xã hội.

Bước 1: Tải mã nguồn về máy (Clone)

Mở Terminal (CMD/PowerShell) và gõ lệnh:
```bash
git clone [https://github.com/huygiangbui2602-maker/doangiaki.git](https://github.com/huygiangbui2602-maker/doangiaki.git)
cd doangiaki
```
Bước 2: Chạy bằng Docker
Để đảm bảo hệ thống chạy ổn định và không lỗi thư viện, vui lòng sử dụng Docker theo các bước sau:
1. Đóng gói (Build Image)
Lệnh này dùng để cài đặt môi trường và các thư viện cần thiết (pandas, scikit-learn, seaborn...).
```bash
docker build -t do-an-ai .
```
2. Huấn luyện mô hình (Train)
Chạy lệnh này để AI học từ dữ liệu CSV, tạo ra file não bộ .pkl và sơ đồ nhiệt buoc4_heatmap.png.
```bash
docker run --rm -v "%cd%:/app" do-an-ai python train_model.py
```
3. Kiểm thử kịch bản (Test)
Chạy 10 kịch bản thực tế để đánh giá độ chính xác của AI.
```bash
docker run --rm -v "%cd%:/app" do-an-ai python test.py
```
4. Dự đoán cá nhân (Predict)
Nhập dữ liệu của bạn để AI dự đoán và xuất biểu đồ chuông so sánh buoc5_bieu_do_chuong.png.
```bash
docker run --rm -it -v "%cd%:/app" do-an-ai python predict.py
```
