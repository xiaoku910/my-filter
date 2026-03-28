import cv2
import numpy as np
import os

# --- 1. 解決中文路徑讀取問題 ---
def imread_unicode(path):
    # 使用 numpy 讀取二進制數據，再解碼，這樣中文路徑就不會噴錯
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None

# 定位檔案
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mask_path = os.path.join(BASE_DIR, "mask.png")

# 載入濾鏡
overlay = imread_unicode(mask_path)

if overlay is None:
    print(f"❌ 錯誤：找不到或無法開啟 {mask_path}")
    print("請確認 mask.png 是否真的在該資料夾下面。")
    exit()

# --- 2. 啟動相機 ---
# 嘗試多個後端啟動相機，防止 FFMPEG 報錯
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("❌ 錯誤：找不到攝影機，請檢查是否被其他程式佔用。")
    exit()

print("✅ 成功啟動！正在預覽中... 按下鍵盤上的 'q' 鍵可結束。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mask_resized = cv2.resize(overlay, (w, h))

    # --- 3. 疊加濾鏡 ---
    if mask_resized.shape[2] == 4: # PNG 有透明層
        alpha = mask_resized[:, :, 3] / 255.0
        overlay_color = mask_resized[:, :, :3]
        # 進行 BGR 疊加
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * (1 - alpha) + overlay_color[:, :, c] * alpha).astype(np.uint8)
    else: # 沒有透明層
        frame = cv2.addWeighted(frame, 0.6, mask_resized[:, :, :3], 0.4, 0)

    # --- 4. 顯示視窗 ---
    cv2.imshow('Art Filter', frame)

    # 偵測按鍵
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()