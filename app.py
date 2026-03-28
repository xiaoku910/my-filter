import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

# 這裡放入你原本成功的 imread_unicode 邏輯
@st.cache_resource
def get_mask():
    # 讀取 mask.png ... (略)
    return mask

overlay = get_mask()

# 這個函式會「自動被呼叫」，相機傳來一張，它就跑一次，不需要 while 迴圈
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24") # 取得目前畫面
    
    # --- 這裡貼上你原本成功的 if mask_resized.shape[2] == 4 疊加邏輯 ---
    # ... (處理 img) ...

    # 處理完後，直接 return 回去給手機看
    return cv2.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="munch-test",
    video_frame_callback=video_frame_callback, # 指定加工廠
    async_processing=True, # 👈 這是讓畫面動起來的關鍵！
    # ... 其他設定
)