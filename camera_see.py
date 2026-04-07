import cv2
import time

# ================= 摄像头配置 =================
# 摄像头索引 (L7)
CAMERA_INDEX = 0
# 窗口名称 (L9)
WINDOW_NAME = 'Camera Stream'
# 文本颜色 (L11)
TEXT_COLOR = (0, 255, 0)
# 文本缩放 (L13)
TEXT_SCALE = 1.0
# =============================================

def main():
    # 1. 打开摄像头 (L18)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return

    frame_count = 0
    start_time = time.time()
    
    print(f"Starting camera stream. Press 'q' to quit.")

    try:
        while True:
            # 2. 读取一帧 (L29)
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1
            
            # 3. 计算实时 FPS (L36)
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # 4. 在画面上绘制 FPS 和总帧数 (L41)
            info_text = f"FPS: {fps:.2f} | Total Frames: {frame_count}"
            cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        TEXT_SCALE, TEXT_COLOR, 2)

            # 5. 显示画面 (L46)
            cv2.imshow(WINDOW_NAME, frame)

            # 6. 退出机制 (按 'q' 退出) (L49)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stream stopped by user.")
    finally:
        # 7. 释放资源 (L56)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stream ended. Total frames processed: {frame_count}")

if __name__ == '__main__':
    main()
