import streamlit as st
import cv2
import numpy as np
import os
import io
import zipfile
from PIL import Image

# 設置頁面標題與佈局
st.set_page_config(page_title="PRO 圖片批量裁切與高清增強工具", page_icon="✂️", layout="wide")
st.title("✂️ 專業級不規則圖片自動切割與畫質優化工具")
st.caption("支援抗陰影、防白毛切碎、正方形置中補齊、透明底及高解析度印刷輸出！")

# ================= 安全讀圖函數 =================
def load_image_safely(file_bytes):
    """安全讀取圖片，自動將透明背景轉換為白底，徹底解決黑底無法辨識問題"""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        # 檢查是否有透明通道
        if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
            pil_img = pil_img.convert('RGBA')
            # 建立一張純白背景
            bg = Image.new('RGBA', pil_img.size, (255, 255, 255, 255))
            # 將原圖貼上白底（透明部分會變成白色）
            bg.paste(pil_img, mask=pil_img)
            pil_img = bg.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')
            
        # 轉為 OpenCV 的 BGR 格式
        cv_img = np.array(pil_img)
        return cv_img[:, :, ::-1].copy()
    except Exception as e:
        return None

# ================= 側邊欄：功能開關與視覺圖解 =================
st.sidebar.header("🛠️ 裁切與畫質設定")

with st.sidebar.expander("📖 參數調節圖解指引 (點擊查看說明)", expanded=False):
    st.markdown("""
    **1. 留白距離 (Padding)**
    * 作用：控制紅框與圖案之間的「安全邊距」。
    * 調整：如果紅線太貼近貼紙邊緣或文字，請**調大 Padding**（例如 35~60px）。
    
    ```
    ┌─────────────────────────┐  <--- 紅色裁切框 (加了 Padding)
    │        Padding 留白      │
    │      ┌───────────┐      │
    │      │  貼紙圖案  │      │
    │      └───────────┘      │
    │                         │
    └─────────────────────────┘
    ```

    **2. 背景純白靈敏度**
    * 數值高 (245~255)：只認純白底。
    * 數值低 (210~235)：抗陰影能力強，能過濾貼紙周圍的淺灰陰影或污漬。

    **3. 防切碎保護強度**
    * 數值高 (30~55)：將圖案內部白色區域（如白毛、白色英文字空隙）強制縫合，防止被掏空。
    """)

# 功能開關
make_square = st.sidebar.checkbox("1. 強制擴展為 1:1 正方形（圖案置中）", value=True)
make_transparent = st.sidebar.checkbox("2. 一鍵去背（匯出透明底 PNG）", value=False)

if make_transparent:
    st.sidebar.info("💡 已開啟去背，自動鎖定為 PNG 格式匯出。")
    export_format = "PNG"
else:
    export_format = st.sidebar.selectbox("3. 匯出格式", ["JPG", "PNG"], index=0)

target_size_option = st.sidebar.selectbox(
    "4. 匯出最小尺寸 (Export Size)",
    ["1200 px (推薦打印)", "1600 px (高精細)", "2000 px (超高清)", "2400 px (大型噴繪)", "保持原圖尺寸 (不放大)"],
    index=0
)

size_map = {
    "1200 px (推薦打印)": 1200,
    "1600 px (高精細)": 1600,
    "2000 px (超高清)": 2000,
    "2400 px (大型噴繪)": 2400,
    "保持原圖尺寸 (不放大)": 0
}
target_size = size_map[target_size_option]

st.sidebar.divider()
st.sidebar.subheader("🎛️ 邊界與靈敏度調節")
padding = st.sidebar.slider("外圍留白距離 (Padding px)", 0, 100, 35, help="數值越大，紅框與貼紙之間的留白距離就越寬。")
bg_thresh = st.sidebar.slider("背景純白靈敏度", 180, 255, 235, help="數值調低可有效過濾灰色陰影/污漬；調高適合極純白底。")
protect_white = st.sidebar.slider("防切碎保護強度", 5, 55, 31, step=2, help="數值越高越能保證白色字體或寵物內部不被切開。")
min_area = st.sidebar.slider("最小物件面積 (過濾小雜點)", 500, 50000, 5000, step=500)

prefix_char = st.sidebar.text_input("檔名編號前綴字元", value="a", help="例如輸入 'a'，產出檔名為 dog_a001.jpg")

# ================= 核心圖像處理邏輯 =================
def detect_boxes(img, bg_thresh, protect_white, min_area):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)

    k_size = (protect_white, protect_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            hull = cv2.convexHull(c)
            valid_boxes.append(cv2.boundingRect(hull))

    valid_boxes = sorted(valid_boxes, key=lambda b: (b[1] // 100, b[0]))
    return valid_boxes, gray

def enhance_and_crop(img, box, gray, padding, make_square, target_size, make_transparent, bg_thresh):
    h, w = img.shape[:2]
    x, y, bw, bh = box

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    cropped = img[y1:y2, x1:x2]
    ch, cw = cropped.shape[:2]

    if make_transparent:
        cropped_gray = gray[y1:y2, x1:x2]
        _, mask = cv2.threshold(cropped_gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        b, g, r = cv2.split(cropped)
        cropped = cv2.merge([b, g, r, mask], 4)

    if make_square:
        max_dim = max(ch, cw)
        top = (max_dim - ch) // 2
        bottom = max_dim - ch - top
        left = (max_dim - cw) // 2
        right = max_dim - cw - left

        if make_transparent:
            border_val = [0, 0, 0, 0]
        else:
            border_val = [255, 255, 255]

        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_val)
        ch, cw = cropped.shape[:2]

    if target_size > 0:
        if ch < target_size or cw < target_size:
            scale = max(target_size / ch, target_size / cw)
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return cropped

# ================= 網頁主介面 =================
uploaded_files = st.file_uploader(
    "請選擇或批量拖曳圖片到此（支援 PNG, JPG, JPEG, WEBP）", 
    type=["png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    col_info, col_prev = st.columns([2, 1])
    with col_info:
        st.info(f"📁 已選取 **{len(uploaded_files)}** 張圖片")
    with col_prev:
        btn_preview = st.button("🔍 預覽第一張圖的裁切邊界 (Preview)")

    if btn_preview:
        first_file = uploaded_files[0]
        # 使用安全讀取機制，自動把透明底變白底
        img_preview = load_image_safely(first_file.read())
        first_file.seek(0)

        if img_preview is None:
            st.error("❌ 圖片讀取失敗，請確認檔案格式是否正確。")
        else:
            h_img, w_img = img_preview.shape[:2]
            boxes, gray = detect_boxes(img_preview, bg_thresh, protect_white, min_area)
            
            debug_img = img_preview.copy()
            for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w_img, x + bw + padding)
                y2 = min(h_img, y + bh + padding)

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(debug_img, f"#{idx}", (x1 + 8, max(30, y1 + 35)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            st.subheader("👀 實際裁切紅框預覽（紅線已包含留白 Padding）：")
            st.caption("提示：調整側邊欄的「外圍留白距離 (Padding)」，可即時改變紅框範圍。")
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            if boxes:
                st.write("---")
                st.subheader("單張成品效果預覽（第一隻圖案）：")
                sample_crop = enhance_and_crop(
                    img_preview, boxes[0], gray, padding, make_square, target_size, make_transparent, bg_thresh
                )
                h_out, w_out = sample_crop.shape[:2]
                st.caption(f"📐 輸出解析度：**{w_out} x {h_out} px**")
                if make_transparent:
                    st.image(cv2.cvtColor(sample_crop, cv2.COLOR_BGRA2RGBA), width=320)
                else:
                    st.image(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB), width=320)

    st.write("---")
    if st.button("🚀 開始批量切割與優化所有圖片", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        ext = ".png" if (make_transparent or export_format == "PNG") else ".jpg"

        for f_idx, file in enumerate(uploaded_files):
            status_text.text(f"正在處理 ({f_idx+1}/{len(uploaded_files)}): {file.name} ...")
            
            img = load_image_safely(file.read())
            
            if img is None:
                continue

            boxes, gray = detect_boxes(img, bg_thresh, protect_white, min_area)
            base_name = os.path.splitext(file.name)[0]

            for idx, box in enumerate(boxes, start=1):
                cropped = enhance_and_crop(
                    img, box, gray, padding, make_square, target_size, make_transparent, bg_thresh
                )
                
                item_name = f"{base_name}_{prefix_char}{idx:03d}{ext}"

                if ext == ".jpg":
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
                else:
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

                is_success, buffer = cv2.imencode(ext, cropped, encode_param)
                if is_success:
                    all_results.append((item_name, buffer.tobytes(), ext))

            progress_bar.progress((f_idx + 1) / len(uploaded_files))

        status_text.text("✅ 全部處理完畢！")
        st.success(f"🎉 成功完成！共產出 {len(all_results)} 張高解析度獨立圖片！")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item_name, img_data, _ in all_results:
                zip_file.writestr(item_name, img_data)
        zip_buffer.seek(0)

        st.download_button(
            label=f"📦 一鍵下載全部切片 ZIP 壓縮包 ({len(all_results)} 張)",
            data=zip_buffer,
            file_name="all_enhanced_sprites.zip",
            mime="application/zip",
            type="primary"
        )

        st.subheader("👀 切割成品成果預覽（前 12 張）")
        cols = st.columns(6)
        for i, (name, data, cur_ext) in enumerate(all_results[:12]):
            with cols[i % 6]:
                img_pil = Image.open(io.BytesIO(data))
                st.image(img_pil, caption=f"{name}\n({img_pil.width}x{img_pil.height})", use_container_width=True)
