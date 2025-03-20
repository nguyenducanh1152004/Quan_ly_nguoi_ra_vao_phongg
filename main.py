import os
import cv2
import pandas as pd
import numpy as np
import threading
import time
import collections
import requests
import pygame
from datetime import datetime
from gtts import gTTS
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# ✅ Khởi tạo YOLO
model = YOLO("yolov8n.pt")

# 🔗 Camera IP hoặc Video test
#camera_url = "test_1.mp4"
camera_url = "http://192.168.137.94:4747/video"
#rtsp_url = "rtsp://admin:La12345678@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0"

# 🚀 Biến toàn cục
frame_latest = None
frame_lock = threading.Lock()
fps_history = collections.deque(maxlen=10)
tracker = Tracker(max_distance=30, max_disappeared=50, min_box_size=40)

# 📄 File CSV lưu trữ
csv_file = "people_count_log.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Timestamp", "In", "Out", "Total"]).to_csv(csv_file, index=False)

# 🏷️ Đọc danh sách nhãn COCO
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# 🔊 Khởi tạo pygame để phát âm thanh
pygame.mixer.init()
alert_sound_path = "alert.mp3"
last_alert_time = 0

# 📢 Biến đếm người
in_count, out_count = 0, 0
previous_positions = {}

# 📲 Telegram Bot
BOT_TOKEN = "7635413575:AAEO4yyTnuYHMtdZlWtKaf3wb2vIk8y-N5A"
CHAT_ID = "6980175456"

def send_telegram_message(text):
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": text})

def send_telegram_photo(image_path):
    with open(image_path, 'rb') as photo:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", files={'photo': photo}, data={'chat_id': CHAT_ID})
    os.remove(image_path)

# 🎥 Luồng lấy hình ảnh từ camera
def capture_camera():
    global frame_latest
    cap = cv2.VideoCapture(camera_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame_latest = cv2.resize(frame, (960, 540))
        time.sleep(0.05)  # Giảm tốc độ xử lý

t = threading.Thread(target=capture_camera, daemon=True)
t.start()

# 🚨 Phát cảnh báo
def play_alert(text, frame=None):
    global last_alert_time
    if time.time() - last_alert_time < 5:
        return
    last_alert_time = time.time()
    
    tts = gTTS(text, lang="vi")
    tts.save(alert_sound_path)
    pygame.mixer.Sound(alert_sound_path).play()
    send_telegram_message(text)
    
    if frame is not None:
        image_path = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(image_path, frame)
        threading.Thread(target=send_telegram_photo, args=(image_path,)).start()

# 🔥 Nhận diện & đếm người
while True:
    start_time = time.time()
    
    with frame_lock:
        if frame_latest is None:
            continue
        frame = frame_latest.copy()

    # 🔍 Dự đoán đối tượng
    results = model.predict(frame, conf=0.3, iou=0.5, augment=True)
    px = pd.DataFrame(results[0].boxes.data).astype("float")

    list_rect = [
        list(map(int, row[:4])) for _, row in px.iterrows()
        if class_list[int(row[5])] == 'person'
    ]

    # 🚶‍♂️ Cập nhật tracker
    tracked_objects = tracker.update(list_rect)
    
    # 📏 Đường ranh giới
    line_y = 270
    cv2.line(frame, (0, line_y), (960, line_y), (0, 0, 255), 2)

    for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
        current_center_y = (y1 + y2) // 2  
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f"ID: {obj_id}", (x1, y1 - 10), 1, 1, colorR=(0, 255, 0))

        if obj_id in previous_positions:
            prev_y = previous_positions[obj_id]
            if prev_y < line_y and current_center_y >= line_y:
                in_count += 1
                play_alert("Có người vào phòng!", frame.copy())
            elif prev_y > line_y and current_center_y <= line_y:
                out_count += 1
                play_alert("Có người ra khỏi phòng!", frame.copy())
        previous_positions[obj_id] = current_center_y

    out_count = min(out_count, in_count)
    total_people = max(in_count - out_count, 0)

    if total_people == 1 and time.time() - last_alert_time >= 30:
        play_alert("Cảnh báo còn 1 người trong phòng!", frame.copy())

    fps_current = 1 / max(time.time() - start_time, 0.01)
    fps_history.append(fps_current)
    fps_avg = int(sum(fps_history) / len(fps_history))

    cvzone.putTextRect(frame, f'FPS: {fps_avg}', (50, 150), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'In: {in_count}', (50, 50), 2, 2)
    cvzone.putTextRect(frame, f'Out: {out_count}', (50, 80), 2, 2)
    cvzone.putTextRect(frame, f'Total: {total_people}', (50, 110), 2, 2)

    if int(time.time()) % 5 == 0:
        pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), in_count, out_count, total_people]],
                     columns=["Timestamp", "In", "Out", "Total"]).to_csv(csv_file, mode='a', header=False, index=False)

    cv2.imshow("Camera Live", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
