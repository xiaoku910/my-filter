import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

# 1. 頁面設定
st.set_page_config(page_title="大師濾鏡雲端版", layout="centered")


# 2. 讀取素材 (雲端環境直接讀取同資料夾檔案)
@st.cache_resource
def load_mask():
    # 雲端環境不支援 cv2.imread 中文路徑，所以直接讀取
    mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
    return mask


mask_img = load_mask()

if mask_img is None:
    st.error("找不到 mask.png，請確認檔案已上傳至 GitHub。")
    st.stop()


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    overlay = cv2.resize(mask_img, (w, h))


    if overlay.shape[2] == 4:
        alpha = (overlay[:, :, 3] / 255.0)[:, :, np.newaxis]
        img = (img * (1.0 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)
    else:
        img = cv2.addWeighted(img, 0.5, overlay[:, :, :3], 0.5, 0)

    return cv2.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="cloud-filter",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    async_processing=True,
)