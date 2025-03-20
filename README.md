BÁO CÁO HỆ THỐNG QUẢN LÝ NGƯỜI RA VÀO PHÒNG
![image](https://github.com/user-attachments/assets/c09b333d-d821-4e4c-8f09-d292b14c1ff6)

Giám sát trong không gian làm việc, trường học hay khu vực an ninh cao luôn cần được giám sát chặt chẽ. 
Việc kiểm soát người ra vào không chỉ giúp đảm bảo an ninh mà còn hỗ trợ trong việc quản lý số lượng người tại mỗi thời điểm. 
Đề tài này tập trung vào việc xây dựng hệ thống giám sát tự động, có khả năng nhận diện người và ghi nhận thông tin ra vào theo thời gian thực.

📌 Giới thiệu hệ thống

Hệ thống quản lý người ra vào phòng sử dụng công nghệ xử lý ảnh và trí tuệ nhân tạo (AI). Hệ thống có khả năng:
📸 Nhận diện người ra vào
🔍 Ghi lại thời gian ra vào từng cá nhân
📊 Lưu trữ dữ liệu vào file csv và app telegram
📤 Xuất báo cáo chi tiết về số lượng người ra vào

🏗️ Cấu trúc hệ thống

Hệ thống bao gồm các thành phần chính:

📹 Camera giám sát: Ghi lại hình ảnh người ra vào.

🖥️ Xử lý ảnh & AI: Nhận diện khuôn mặt và xác định thông tin cá nhân.

💾 Cơ sở dữ liệu: Lưu thông tin người ra vào và hỗ trợ truy vấn.
<img width="965" alt="Screenshot 2025-03-20 at 00 17 51" src="https://github.com/user-attachments/assets/3978345e-4dbc-47a7-8173-ca833cf029a0" />


🛠️ Công cụ sử dụng và các thư viện cần thiết 

Python OpenCV
Thư viện hỗ trợ: os,cv2,threading,time,collections, pygame,pandas, numpy, datetime, gTTS, YOLO, cvzone
Cam IP
App Telegram

🚀 Hướng dẫn cài đặt và chạy
1. Cài đặt thư viện
Yêu cầu Python 3.7+

Cài đặt các thư viện:
pip install các thư viện tôi gửi phía trên 

2.Lấy địa chỉ IP,port,username,passord của cam
Kết nối cam với chương trình qua rtsp_url = "rtsp://username:password@ip_address:port/stream"

3. Tạo môi trường ảo (tùy chọn)

python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
.\venv\Scripts\activate  # Trên Windows

4. Chạy hệ thống

python main.py

📖 Hướng dẫn sử dụng

Mở giao diện và bắt đầu giám sát.

Khi có người ra vào, hệ thống sẽ tự động nhận diện và lưu dữ liệu.

Kiểm tra kết quả trên màn hình hoặc xuất báo cáo dưới dạng CSV và thông báo lên telegram .

⚙️ Cấu hình & Ghi chú

Cấu hình đường dẫn camera trong main.py.
Cấu hình telegram bằng câu lệnh 
# Telegram Bot
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

📰 Báo cáo dữ liệu ra vào

Hệ thống lưu thông tin người ra vào vào file csv. 

Poster của nhóm 
<img width="596" alt="Screenshot 2025-03-20 at 01 01 50" src="https://github.com/user-attachments/assets/a89ab7bb-8391-4117-bed5-527367743f6b" />


🤝 Đóng góp

Dự án được phát triển bởi nhóm 1:

Nguyễn Thuý Hằng (Nhóm trưởng)
Lê Bá Hoan
Nguyễn Đức Anh

© 2025 NHÓM 1, CNTT16-04, TRƯỜNG ĐẠI HỌC ĐẠI NAM

