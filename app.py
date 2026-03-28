import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import os

st.set_page_config(page_title="最終完成版", layout="centered")

# --- 1. 確保圖片能被讀取 (使用緩存避免重複讀取) ---
@st.cache_resource
def get_mask():
    # 雲端環境路徑最穩定的寫法
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "mask.png")
    
    if not os.path.exists(path):
        return None
        
    # 用 OpenCV 讀取
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return mask

MASK_IMG = get_mask()

if MASK_IMG is None:
    st.error("❌ 找不到 mask.png，請確認檔案已上傳至 GitHub 根目錄。")
else:
    st.success("✅ 濾鏡素材載入成功！")

# --- 2. 影像處理函式 (優化速度) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    if MASK_IMG is not None:
        h, w = img.shape[:2]
        # 縮放濾鏡
        overlay = cv2.resize(MASK_IMG, (w, h))
        
        if overlay.shape[2] == 4: # PNG 透明通道處理
            alpha = (overlay[:, :, 3] / 255.0)[:, :, np.newaxis]
            img = (img * (1.0 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)
        else: # 如果是 JPG 或沒透明層
            img = cv2.addWeighted(img, 0.5, overlay[:, :, :3], 0.5, 0)
            
    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. WebRTC 設定 (關鍵：解決卡頓) ---
webrtc_streamer(
    key="munch-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},  # 調低解析度保證順暢
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True, # 必須開啟，否則畫面會卡在第一幀
)