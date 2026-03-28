import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import os

st.set_page_config(page_title="大師濾鏡-移植版", layout="centered")

# --- 1. 完全保留你成功的讀取邏輯 ---
# 加入 @st.cache_resource 是為了防止雲端主機每次都重新讀取圖片導致崩潰
@st.cache_resource
def load_overlay():
    def imread_unicode(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            return img
        except Exception as e:
            print(f"讀取錯誤: {e}")
            return None

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mask_path = os.path.join(BASE_DIR, "mask.png")
    return imread_unicode(mask_path)

overlay = load_overlay()

if overlay is None:
    st.error("❌ 錯誤：找不到 mask.png！請確認檔案有跟 app.py 放在一起。")
    st.stop()
else:
    st.success("✅ 濾鏡載入成功！請點擊下方 START 啟動相機。")

# --- 2. 把你的 while 迴圈內容，變成網頁專用的加工廠 ---
def video_frame_callback(frame):
    # 這裡相當於原本的 ret, frame = cap.read()
    img = frame.to_ndarray(format="bgr24")

    h, w, _ = img.shape
    mask_resized = cv2.resize(overlay, (w, h))

    # --- 3. 完全照搬你成功的疊加邏輯 ---
    if mask_resized.shape[2] == 4: # PNG 有透明層
        alpha = mask_resized[:, :, 3] / 255.0
        overlay_color = mask_resized[:, :, :3]
        # 進行 BGR 疊加
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1 - alpha) + overlay_color[:, :, c] * alpha).astype(np.uint8)
    else: # 沒有透明層
        img = cv2.addWeighted(img, 0.6, mask_resized[:, :, :3], 0.4, 0)

    # 回傳處理好的畫面 (相當於原本的 cv2.imshow)
    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. 啟動網頁攝影機 (取代 cv2.VideoCapture) ---
webrtc_streamer(
    key="munch-web",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    # 設定穿透伺服器，讓手機連得上
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    # 稍微限制解析度，避免雲端免費主機資源超載 (Over memory limit)
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True, # 必須開啟，防止畫面卡在第一幀
)