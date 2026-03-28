import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import os

# 1. 頁面標題
st.set_page_config(page_title="大師濾鏡-手機同步版")

# --- 2. 保留你最成功的圖片讀取邏輯 ---
def imread_unicode(path):
    try:
        # 雲端環境也適用這種讀取方式，最保險
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mask_path = os.path.join(BASE_DIR, "mask.png")
overlay = imread_unicode(mask_path)

if overlay is None:
    st.error(f"❌ 找不到濾鏡檔案 mask.png")
    st.stop()

# --- 3. 將原本的 while 迴圈邏輯包裝進 callback ---
def video_frame_callback(frame):
    # 將相機每一幀轉成 numpy 陣列
    img = frame.to_ndarray(format="bgr24")
    
    h, w, _ = img.shape
    mask_resized = cv2.resize(overlay, (w, h))

    # --- 這裡完全搬移你原本成功的疊加邏輯 ---
    if mask_resized.shape[2] == 4: # PNG 有透明層
        alpha = mask_resized[:, :, 3] / 255.0
        overlay_color = mask_resized[:, :, :3]
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1 - alpha) + overlay_color[:, :, c] * alpha).astype(np.uint8)
    else: # 沒有透明層 (JPG)
        img = cv2.addWeighted(img, 0.6, mask_resized[:, :, :3], 0.4, 0)

    # 回傳處理好的畫面
    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. 啟動網頁串流 (取代 cv2.imshow) ---
webrtc_streamer(
    key="munch-filter",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    # 手機連線必要的伺服器設定
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"facingMode": "user"}, # 預設前鏡頭
        "audio": False
    },
    async_processing=True, # 防止畫面卡住
)