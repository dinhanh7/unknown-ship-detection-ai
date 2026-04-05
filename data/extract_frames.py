import cv2
import os

video_path = '/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/data/origin_video.mp4'
output_dir = '/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/data/frames'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Lỗi: Không thể mở được video.")
    exit(1)

# Lấy FPS của video để báo cáo thêm thông tin
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS gốc của video: {fps}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Tính toán giây và số thứ tự frame trong giây (video 25 FPS)
    second = (frame_idx // 25) + 1
    frame_in_second = (frame_idx % 25) + 1
    
    # Tạo tên thư mục second01_frame1.jpg
    filename = f"second{second:02d}_frame{frame_in_second}.jpg"
    out_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(out_path, frame)
    frame_idx += 1

cap.release()
print(f"Hoàn thành trích xuất! Tổng số frames: {frame_idx}")
