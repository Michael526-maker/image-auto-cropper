import streamlit as st
import io
import zipfile
import os
from PIL import Image, ImageEnhance

# 設置頁面標題與佈局
st.set_page_config(page_title="PRO 圖片批量縮放與增強工具", page_icon="🖼️", layout="wide")
st.title("🖼️ 專業級圖片批量縮放與畫質增強工具")
st.caption("支援批量 Resize、色彩鮮明度、對比度、銳利度微調，並一鍵打包下載！")

# ================= 側邊欄：功能開關與設定 =================
st.sidebar.header("🛠️ 影像與尺寸設定")

# 1. 尺寸設定 (Resize)
target_size_option = st.sidebar.selectbox(
    "1. 匯出最大邊長尺寸 (Resize)",
    ["1200 px (推薦打印)", "1600 px (高精細)", "2000 px (超高清)", "2400 px (大型噴繪)", "保持原圖尺寸 (不縮放)"],
    index=1
)

size_map = {
    "1200 px (推薦打印)": 1200,
    "1600 px (高精細)": 1600,
    "2000 px (超高清)": 2000,
    "2400 px (大型噴繪)": 2400,
    "保持原圖尺寸 (不縮放)": 0
}
target_size = size_map[target_size_option]

st.sidebar.divider()
st.sidebar.subheader("🎨 色彩與畫質微調")
st.sidebar.caption("提示：數值為 1.0 代表原圖不變。")

# 2. 影像增強滑桿
color_factor = st.sidebar.slider("🌈 色彩鮮明度 (Saturation)", 0.0, 3.0, 1.2, step=0.1, help="> 1.0 顏色更鮮豔，< 1.0 變黑白。")
contrast_factor = st.sidebar.slider("🌓 對比度 (Contrast)", 0.5, 2.5, 1.1, step=0.1, help="> 1.0 亮暗對比更強烈。")
sharpness_factor = st.sidebar.slider("🔪 銳利度 (Sharpness)", 0.0, 3.0, 1.5, step=0.1, help="> 1.0 邊緣更清晰銳利。")

st.sidebar.divider()
st.sidebar.subheader("💾 輸出設定")
export_format = st.sidebar.selectbox("匯出格式", ["JPG", "PNG"], index=0)
prefix_char = st.sidebar.text_input("自訂檔名後綴 (後綴字元)", value="HD", help="例如輸入 'HD'，產出檔名為 原名_HD.jpg")

# ================= 核心圖像處理邏輯 =================
def process_image(img, target_size, color, contrast, sharpness):
    """執行 Resize 與影像增強"""
    # 1. 影像增強處理
    if color != 1.0:
        img = ImageEnhance.Color(img).enhance(color)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
    # 2. 縮放處理 (保持長寬比，以最長邊為基準)
    if target_size > 0:
        w, h = img.size
        # 計算縮放比例
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 LANCZOS 演算法保證縮放不失真
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    return img

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
        btn_preview = st.button("🔍 預覽第一張圖 (Preview)")

    # 即時預覽模式 (對比原圖與修改後的效果)
    if btn_preview:
        first_file = uploaded_files[0]
        try:
            img_original = Image.open(io.BytesIO(first_file.read())).convert("RGB")
            first_file.seek(0)
            
            st.write("---")
            st.subheader("👀 效果預覽對比 (Preview)")
            
            # 處理預覽圖
            img_preview = process_image(img_original.copy(), target_size, color_factor, contrast_factor, sharpness_factor)
            
            # 雙欄並排顯示對比
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**修改前 (Original)** - {img_original.width} x {img_original.height} px")
                st.image(img_original, use_container_width=True)
            with col2:
                st.caption(f"**修改後 (Enhanced)** - {img_preview.width} x {img_preview.height} px")
                st.image(img_preview, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ 圖片讀取失敗: {e}")

    st.write("---")
    # 批量處理按鈕
    if st.button("🚀 開始批量轉換所有圖片", type="primary"):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        ext = ".png" if export_format == "PNG" else ".jpg"

        for f_idx, file in enumerate(uploaded_files):
            status_text.text(f"正在處理 ({f_idx+1}/{len(uploaded_files)}): {file.name} ...")
            
            try:
                # 讀取圖片並確保轉換為 RGB (避免透明底報錯)
                img = Image.open(io.BytesIO(file.read()))
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    bg.paste(img, mask=img)
                    img = bg.convert('RGB')
                else:
                    img = img.convert('RGB')
                
                # 處理圖片
                processed_img = process_image(img, target_size, color_factor, contrast_factor, sharpness_factor)
                
                # 命名規則：原檔名_後綴.jpg (例如：dog_HD.jpg)
                base_name = os.path.splitext(file.name)[0]
                item_name = f"{base_name}_{prefix_char}{ext}"

                # 儲存到記憶體
                img_byte_arr = io.BytesIO()
                if ext == ".jpg":
                    processed_img.save(img_byte_arr, format='JPEG', quality=98)
                else:
                    processed_img.save(img_byte_arr, format='PNG', compress_level=3)
                
                all_results.append((item_name, img_byte_arr.getvalue()))
                
            except Exception as e:
                st.error(f"處理圖片 {file.name} 時發生錯誤: {e}")

            progress_bar.progress((f_idx + 1) / len(uploaded_files))

        status_text.text("✅ 全部處理完畢！")
        st.success(f"🎉 成功完成！共產出 {len(all_results)} 張優化圖片！")

        # 打包 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item_name, img_data in all_results:
                zip_file.writestr(item_name, img_data)
        zip_buffer.seek(0)

        # 下載按鈕
        st.download_button(
            label=f"📦 一鍵下載全部 ZIP 壓縮包 ({len(all_results)} 張)",
            data=zip_buffer,
            file_name="batch_resized_images.zip",
            mime="application/zip",
            type="primary"
        )
