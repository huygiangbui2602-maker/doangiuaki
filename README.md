<<<<<<< HEAD
﻿ĐỒ ÁN: HỆ THỐNG DỰ ĐOÁN MỨC ĐỘ STRESS 
Hệ thống sử dụng mô hình Hồi quy tuyến tính (Linear Regression) để dự đoán điểm Stress dựa trên thói quen sinh hoạt và sử dụng mạng xã hội.
Hướng dẫn chạy bằng Docker
Để đảm bảo hệ thống chạy ổn định và không lỗi thư viện, vui lòng sử dụng Docker theo các bước sau:
1. Đóng gói (Build Image)
Lệnh này dùng để cài đặt môi trường và các thư viện cần thiết (pandas, scikit-learn, seaborn...).
Bash
docker build -t do-an-ai .
2. Huấn luyện mô hình (Train)
Chạy lệnh này để AI học từ dữ liệu CSV, tạo ra file não bộ .pkl và sơ đồ nhiệt buoc4_heatmap.png.
Bash
docker run --rm -v "%cd%:/app" do-an-ai python train_model.py
3. Kiểm thử kịch bản (Test)
Chạy 10 kịch bản thực tế để đánh giá độ chính xác của AI.
Bash
docker run --rm -v "%cd%:/app" do-an-ai python test.py
4. Dự đoán cá nhân (Predict)
Nhập dữ liệu của bạn để AI dự đoán và xuất biểu đồ chuông so sánh buoc5_bieu_do_chuong.png.
Bash
docker run --rm -it -v "%cd%:/app" do-an-ai python predict.py



=======
# doangiuaki
Dự toán mức độ stress dựa trên thói quen sử dụng mạng xã hội và lối sống - Đồ án Nhập môn Khoa học dữ liệu
>>>>>>> 57b5513faf34c0f0ed9b4a8fb663bfd17fb7338f
