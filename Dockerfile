# 1. Chọn hệ điều hành cơ bản có cài sẵn Python 
FROM python:3.9-slim

# 2. Tạo một thư mục làm việc 
WORKDIR /app

# 3. Copy file danh sách 
COPY requirements.txt .

# 4. Ra lệnh cài những thư viện trong file danh sách
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code (4 file .py) 
COPY check_data_error.py .
COPY train_model.py .
COPY predict.py .
COPY test.py .

# 6. Mặc định mở file kiểm tra dữ liệu đầu tiên
CMD ["python", "check_data_error.py"]