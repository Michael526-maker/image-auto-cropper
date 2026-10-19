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
st.caption("支援抗陰影、防白毛切碎、正方形置中補齊、透明底及 1200x1200px 印刷級高解析度輸出！")

# ================= 側邊欄：功能開關與參數微調 =================
st.sidebar.header("🛠️ 裁切與畫質設定")

# 功能開關
make_square = st.sidebar.checkbox("1. 強制擴展為 1:1 正方形（圖案置中）", value=True)
upscale_1200 = st.sidebar.checkbox("2. 清晰度增強：保證最小 1200 x 1200 px（適合打印）", value=True)
make_transparent = st.sidebar.checkbox("3. 一鍵去背（匯出透明底 PNG）", value=False)
export_format = st.sidebar.selectbox("4. 匯出格式", ["PNG" if make_transparent else "JPG", "PNG", "JPG"])

st.sidebar.divider()
st.sidebar.subheader("🎛️ 辨識精度微調")
bg_thresh = st.sidebar.slider("背景純白靈敏度", 180, 255, 240, help="數值調低可有效過濾灰色陰影/污漬；調高適合極純白底。")
protect_white = st.sidebar.slider("白色毛髮防切碎保護強度", 5, 45, 25, step=2, help="數值越高越能保證白色寵物內部不被掏空。")
min_area = st.sidebar.slider("最小物件面積 (過濾小雜點)", 500, 30000, 3500, step=500)
padding = st.sidebar.slider("外圍留白 (Padding px)", 0, 80, 15)

# 自訂字母代號
prefix_char = st.sidebar.text_input("檔名編號前綴字元", value="a", help="例如輸入 'a'，產出檔名為 dog_a001.jpg")

# ================= 核心圖像處理邏輯 =================
def detect_boxes(img, bg_thresh, protect_white, min_area):
    """偵測圖片中的物件邊界，並防範白色毛髮切碎"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化抗陰影
    _, thresh = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)

    # 形態學閉運算（將白狗、白貓內部空洞完全縫合）
    k_size = (protect_white, protect_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 尋找外部最外層輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            # 使用 Convex Hull 確保邊界平滑包裹
            hull = cv2.convexHull(c)
            valid_boxes.append(cv2.boundingRect(hull))

    # 按行優先排序（由上至下、由左至右）
    valid_boxes = sorted(valid_boxes, key=lambda b: (b[1] // 80, b[0]))
    return valid_boxes, gray

def enhance_and_crop(img, box, gray, padding, make_square, upscale_1200, make_transparent, bg_thresh):
    """執行裁切、補齊正方形、去背與 1200px 超採樣"""
    h, w = img.shape[:2]
    x, y, bw, bh = box

    # 邊界擴展
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    cropped = img[y1:y2, x1:x2]
    ch, cw = cropped.shape[:2]

    # 去背處理（如果需要）
    if make_transparent:
        cropped_gray = gray[y1:y2, x1:x2]
        _, mask = cv2.threshold(cropped_gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        b, g, r = cv2.split(cropped)
        cropped = cv2.merge([b, g, r, mask], 4)

    # 1:1 正方形補齊（圖案置中）
    if make_square:
        max_dim = max(ch, cw)
        top = (max_dim - ch) // 2
        bottom = max_dim - ch - top
        left = (max_dim - cw) // 2
        right = max_dim - cw - left

        if make_transparent:
            # 透明填充 [B, G, R, A]
            border_val = [0, 0, 0, 0]
        else:
            # 純白填充 [B, G, R]
            border_val = [255, 255, 255]

        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_val)
        ch, cw = cropped.shape[:2]

    # 至少 1200 x 1200 px 畫質優化（高精度 Lanczos 插值）
    if upscale_1200:
        target_size = 1200
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

    # 即時預覽模式
    if btn_preview:
        first_file = uploaded_files[0]
        file_bytes = np.asarray(bytearray(first_file.read()), dtype=np.uint8)
        img_preview = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        first_file.seek(0) # 重置讀取指針

        boxes, gray = detect_boxes(img_preview, bg_thresh, protect_white, min_area)
        
        # 在預覽圖上標示紅色識別框
        debug_img = img_preview.copy()
        for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(debug_img, f"#{idx}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        st.subheader("👀 識別紅框預覽（請確認是否有圖案漏檢或被切破）：")
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        if boxes:
            st.write("單張成品效果預覽（第一隻圖案）：")
            sample_crop = enhance_and_crop(
                img_preview, boxes[0], gray, padding, make_square, upscale_1200, make_transparent, bg_thresh
            )
            h_out, w_out = sample_crop.shape[:2]
            st.caption(f"成品解析度：{w_out} x {h_out} px")
            if make_transparent:
                st.image(cv2.cvtColor(sample_crop, cv2.COLOR_BGRA2RGBA), width=280)
            else:
                st.image(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB), width=280)

    st.write("---")
    # 批量處理按鈕
    if st.button("🚀 開始批量切割與優化所有圖片", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        ext = ".png" if (make_transparent or export_format == "PNG") else ".jpg"

        for f_idx, file in enumerate(uploaded_files):
            status_text.text(f"正在處理 ({f_idx+1}/{len(uploaded_files)}): {file.name} ...")
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            boxes, gray = detect_boxes(img, bg_thresh, protect_white, min_area)
            base_name = os.path.splitext(file.name)[0]

            for idx, box in enumerate(boxes, start=1):
                cropped = enhance_and_crop(
                    img, box, gray, padding, make_square, upscale_1200, make_transparent, bg_thresh
                )
                
                # 命名規則：原檔名_a001.jpg
                item_name = f"{base_name}_{prefix_char}{idx:03d}{ext}"

                # 高質量編碼輸出
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

        # 打包 ZIP
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

        # 預覽前 12 張
        st.subheader("👀 切割成品成果預覽（前 12 張）")
        cols = st.columns(6)
        for i, (name, data, cur_ext) in enumerate(all_results[:12]):
            with cols[i % 6]:
                img_pil = Image.open(io.BytesIO(data))
                st.image(img_pil, caption=f"{name}\n({img_pil.width}x{img_pil.height})", use_container_width=True)
