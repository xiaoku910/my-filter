import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import os

# 1. 基本設定
st.set_page_config(page_title="孟克濾鏡-雲端流暢版")
st.title("🎨 藝術濾鏡測試中")

# 2. 強化版圖片讀取 (加入錯誤檢查提示)
@st.cache_resource
def load_filter_image():
    # 這裡直接讀取，不處理中文路徑，因為 GitHub 伺服器路徑是英文的
    path = "mask.png" 
    if not os.path.exists(path):
        return None
    # 讀取並強制轉換為帶 Alpha 通道的內容
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

overlay = load_filter_image()

if overlay is None:
    st.error("❌ 找不到 mask.png！請確認檔案有跟 app.py 放在 GitHub 同一個資料夾。")
else:
    st.success("✅ 濾鏡素材讀取成功，請按下方 Start。")

# 3. 影像處理邏輯 (優化效能，防止卡在第一幀)
def transform(frame):
    img = frame.to_ndarray(format="bgr24")
    
    if overlay is not None:
        # 取得目前相機畫面的寬高
        h, w = img.shape[:2]
        # 縮放濾鏡以匹配相機
        mask_res = cv2.resize(overlay, (w, h))

        if mask_res.shape[2] == 4: # PNG 有透明層
            alpha = mask_res[:, :, 3] / 255.0
            overlay_rgb = mask_res[:, :, :3]
            # 使用快速疊加法，避免逐像素迴圈
            img = (img * (1.0 - alpha[:, :, np.newaxis]) + 
                   overlay_rgb * alpha[:, :, np.newaxis]).astype(np.uint8)
        else:
            # 如果是 JPG
            img = cv2.addWeighted(img, 0.5, mask_res[:, :, :3], 0.5, 0)

    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

# 4. 啟動 WebRTC (關鍵參數調整)
webrtc_streamer(
    key="munch-cloud-final",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=transform,
    # 加入 STUN 伺服器
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    # 限制解析度以提升順暢度 (非常重要！)
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    # 讓影像處理跟顯示分開，防止卡死
    async_processing=True,
)