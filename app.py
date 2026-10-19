import streamlit as st
import cv2
import numpy as np
import os
import io
import zipfile

# 網頁基本設定
st.set_page_config(page_title="批次圖片切片工具", page_icon="✂️", layout="wide")
st.title("✂️ 批量不規則圖片自動切割工具")
st.write("上傳多張相片，自動偵測物件輪廓、按位置排序裁切，並匯出指定編號格式的獨立圖片！")

# 側邊欄：調節參數
st.sidebar.header("⚙️ 辨識與裁切設定")
bg_thresh = st.sidebar.slider("背景純白靈敏度", 200, 255, 245, help="數值越高越接近純白")
min_area = st.sidebar.slider("最小物件面積 (過濾雜訊)", 500, 20000, 4000, step=500)
padding = st.sidebar.slider("外圍留白像素 (Padding)", 0, 50, 10)
export_format = st.sidebar.selectbox("匯出格式", ["JPG", "PNG"], index=0)

def process_image(image_bytes, filename, bg_thresh, min_area, padding, export_format):
    # 讀取圖片數據
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（白底變黑，物件變白）
    _, thresh = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)

    # 閉運算將物件內部空隙連成整體
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 尋找輪廓並排序（由上至下、由左至右）
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
    valid_boxes = sorted(valid_boxes, key=lambda b: (b[1] // 80, b[0]))

    # 提取原檔名（去掉副檔名，例如 dog.jpg -> dog）
    base_name = os.path.splitext(filename)[0]
    cropped_results = []
    ext = ".jpg" if export_format == "JPG" else ".png"

    for idx, (x, y, bw, bh) in enumerate(valid_boxes, start=1):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        cropped = img[y1:y2, x1:x2]

        # 檔名格式：原檔名_a001.jpg, 原檔名_a002.jpg ...
        item_name = f"{base_name}_a{idx:03d}{ext}"

        # 編碼並儲存
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if export_format == "JPG" else []
        is_success, buffer = cv2.imencode(ext, cropped, encode_param)
        if is_success:
            cropped_results.append((item_name, buffer.tobytes()))

    return cropped_results

# 批次檔案上傳（支援按住 Ctrl / Shift 多選）
uploaded_files = st.file_uploader(
    "請選擇或拖曳圖片到此（支援 PNG, JPG, JPEG, WEBP）", 
    type=["png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"已選取 {len(uploaded_files)} 張圖片")
    
    if st.button("🚀 開始批次切割所有圖片", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"正在處理 ({idx+1}/{len(uploaded_files)}): {file.name} ...")
            sub_results = process_image(
                file.read(), file.name, bg_thresh, min_area, padding, export_format
            )
            all_results.extend(sub_results)
            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("✅ 全部處理完畢！")
        st.success(f"共成功切出 {len(all_results)} 張獨立圖案！")

        # 打包成 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item_name, img_data in all_results:
                zip_file.writestr(item_name, img_data)
        zip_buffer.seek(0)

        # 下載按鈕
        st.download_button(
            label="📦 一鍵下載全部切片 ZIP 壓縮包",
            data=zip_buffer,
            file_name="all_cropped_images.zip",
            mime="application/zip",
            type="primary"
        )

        # 預覽部分成品
        st.subheader("👀 成果預覽（前 12 張）")
        cols = st.columns(6)
        for i, (name, data) in enumerate(all_results[:12]):
            with cols[i % 6]:
                st.image(data, caption=name, use_container_width=True)
