import cv2
import os

video_path = '/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/data/origin_video.mp4'
output_dir = '/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/data/images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Lỗi: Không thể mở được video output_padded.mp4.")
    exit(1)

frame_idx = 0
saved_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Cứ 5 frame thì cắt và lưu lại 1 frame ảnh (vd: frame 0, 5, 10...)
    if frame_idx % 4 == 0:
        filename = f"frame_{saved_idx:04d}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, frame)
        saved_idx += 1
        
    frame_idx += 1

cap.release()
print(f"Hoàn thành trích xuất! Đã lưu {saved_idx} frames vào {output_dir}.")
