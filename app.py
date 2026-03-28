import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import os

st.set_page_config(page_title="大師濾鏡-雲端成功版")

# 1. 保留你最成功的讀取邏輯
def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mask_path = os.path.join(BASE_DIR, "mask.png")

@st.cache_resource
def get_final_mask():
    return imread_unicode(mask_path)

overlay = get_final_mask()

# 2. 把 while 迴圈改成這個「加工廠」函式
def video_frame_callback(frame):
    # 這一行等於原本的 ret, frame = cap.read()
    img = frame.to_ndarray(format="bgr24") 
    
    if overlay is not None:
        h, w = img.shape[:2]
        mask_resized = cv2.resize(overlay, (w, h))

        # --- 完全搬移你成功的疊加邏輯 ---
        if mask_resized.shape[2] == 4: # PNG 有透明層
            alpha = mask_resized[:, :, 3] / 255.0
            overlay_color = mask_resized[:, :, :3]
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * (1 - alpha) + overlay_color[:, :, c] * alpha).astype(np.uint8)
        else: # 沒有透明層
            img = cv2.addWeighted(img, 0.6, mask_resized[:, :, :3], 0.4, 0)

    # 這一行等於原本的 cv2.imshow，把結果傳回手機
    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

# 3. 啟動網頁串流
webrtc_streamer(
    key="munch-filter-ok",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # 👈 讓畫面動起來的關鍵！
)