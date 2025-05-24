import streamlit as st
import pandas as pd
import os
import io
import zipfile
import altair as alt
import requests

# --- CẤU HÌNH TRANG VÀ BIẾN TOÀN CỤC ---
st.set_page_config(
    page_title="Garen God-King: Dữ Liệu Voice Cloning",
    page_icon="👑",
    layout="wide"
)

base_dir = os.getcwd()

# Đường dẫn đến thư mục chứa dataset và file metadata
BASE_DATA_DIR  = os.path.join(base_dir, "DataGaren","wavs")
METADATA_FILE = os.path.join(base_dir, "DataGaren", "wavs","metadata.csv")
# AUDIO_FILES_SUBDIR = "" # Nếu audio nằm trong thư mục con của BASE_DATA_DIR
                         # Trong trường hợp này, audio và metadata.csv cùng cấp trong BASE_DATA_DIR

@st.cache_data # Cache để không phải đọc file CSV mỗi lần vào
def load_metadata():
    """Tải metadata từ file CSV và kiểm tra sự tồn tại của file audio."""
    if not os.path.exists(METADATA_FILE):
        st.error(f"Lỗi: Không tìm thấy file metadata tại '{METADATA_FILE}'.")
        st.info(f"Vui lòng chạy script crawl dữ liệu trước để tạo ra file này và các file audio trong thư mục '{BASE_DATA_DIR}'.")
        return pd.DataFrame(columns=['filename', 'transcript', 'audio_path', 'audio_exists'])

    try:
        df = pd.read_csv(METADATA_FILE)
        if 'filename' not in df.columns or 'transcript' not in df.columns:
            st.error("File metadata.csv phải có ít nhất 2 cột: 'filename' và 'transcript'.")
            return pd.DataFrame(columns=['filename', 'transcript', 'audio_path', 'audio_exists'])

        # Tạo đường dẫn đầy đủ và kiểm tra file audio (định dạng .wav theo script crawl)
        df['audio_path'] = df['filename'].apply(
            lambda x: os.path.join(BASE_DATA_DIR, str(x)) if pd.notna(x) else None
        )
        df['audio_exists'] = df['audio_path'].apply(lambda p: os.path.exists(p) if p else False)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file metadata.csv: {e}")
        return pd.DataFrame(columns=['filename', 'transcript', 'audio_path', 'audio_exists'])

def display_data_table(df_display):
    """Hiển thị bảng dữ liệu với tên file, nút phát audio, và transcript."""
    if df_display.empty:
        st.info("Không có dữ liệu nào phù hợp với bộ lọc của bạn.")
        return

    st.write(f"Hiển thị {len(df_display)} bản ghi:")
    header_cols = st.columns([0.3, 0.15, 0.55]) # Tên file, Nút phát, Transcript
    with header_cols[0]: st.markdown("**Tên File Audio (.wav)**")
    with header_cols[1]: st.markdown("**Phát Audio**")
    with header_cols[2]: st.markdown("**Transcript**")
    st.divider()

    for _, row in df_display.iterrows(): # Dùng iterrows() cho DataFrame
        cols = st.columns([0.3, 0.15, 0.55])
        with cols[0]:
            st.code(row["filename"], language=None) # Dùng st.code cho tên file
        with cols[1]:
            if row["audio_exists"] and row["audio_path"]:
                try:
                    st.audio(row["audio_path"], format="audio/wav") # Chỉ định format
                except Exception:
                    st.error(f"Lỗi phát: {row['filename']}")
            elif row["audio_path"]:
                 st.caption("File không tìm thấy")
            else:
                 st.caption("-")
        with cols[2]:
            st.write(row["transcript"])
        st.divider()

# --- TẢI DỮ LIỆU BAN ĐẦU ---
data_df = load_metadata()

# --- SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("👑 Garen God-King")
st.sidebar.markdown("Dữ liệu cho AI Voice Cloning")
st.sidebar.markdown("---")

page_options = ["Trang Chủ", "Xem Dữ Liệu", "Thống Kê Dữ Liệu", "Tải Dữ Liệu", "Tạo Audio"]
selected_page = st.sidebar.radio("Điều hướng:", page_options)
st.sidebar.markdown("---")

if data_df is not None and not data_df.empty:
    st.sidebar.info(f"Tổng số bản ghi: {len(data_df)}")
    if 'audio_exists' in data_df.columns:
        valid_audio_count = data_df['audio_exists'].sum()
        st.sidebar.success(f"File audio có sẵn: {valid_audio_count}")
        if len(data_df) - valid_audio_count > 0 :
            st.sidebar.warning(f"Thiếu file audio: {len(data_df) - valid_audio_count}")
else:
    st.sidebar.warning("Chưa có dữ liệu. Hãy chạy script crawl.")


# --- NỘI DUNG CÁC TRANG ---
if selected_page == "Trang Chủ":
    st.title("Bộ Dữ Liệu Giọng Nói Garen God-King (LMHT)")
    st.markdown(f"""
    Chào mừng! Đây là giao diện khám phá bộ dữ liệu giọng nói của Garen God-King từ League of Legends,
    được thu thập tự động từ nhóm em qua trang (https://wiki.leagueoflegends.com/en-us/League_of_Legends_Wiki).

    **Mục đích:**
    * Cung cấp bộ dữ liệu gồm các file âm thanh (`.wav`) và transcript tương ứng.
    * Dữ liệu này có thể được sử dụng cho các bài toán AI Voice Cloning, nghiên cứu về giọng nói.

    **Quy trình Thu thập Dữ liệu (Crawl):**
    * Dữ liệu được crawl từ trang audio của Garen trên League of Legends Wiki.
    * Script sử dụng thư viện `requests` để tải nội dung HTML và `BeautifulSoup4` để phân tích, trích xuất link audio và transcript.
    * Các file audio `.wav` và metadata (tên file, transcript) được lưu trữ cục bộ trong thư mục `{BASE_DATA_DIR}`.
    * Script crawl được thiết kế để chỉ lấy các audio liên quan đến "God-King Garen".

    **Hướng dẫn sử dụng giao diện:**
    * **Xem Dữ Liệu:** Duyệt, tìm kiếm và nghe các mẫu audio.
    * **Thống Kê Dữ Liệu:** Xem thông tin tổng quan về bộ dữ liệu (số lượng, độ dài transcript,...).
    * **Tải Dữ Liệu:** Tải xuống file metadata (CSV) hoặc toàn bộ file audio (ZIP).

    Giao diện được xây dựng bằng **Streamlit**.
    """)
    st.image("https://wiki.leagueoflegends.com/en-us/images/Garen_God-KingSkin.jpg?f7032", # Garen God-King Splash
             caption="Garen, God-King", use_container_width=True)

elif selected_page == "Xem Dữ Liệu":
    st.title("🔍 Khám Phá Dữ Liệu Audio & Transcript")
    if data_df is not None and not data_df.empty:
        # --- BỘ LỌC VÀ TÌM KIẾM ---
        st.subheader("Bộ lọc và Tìm kiếm")
        search_col, filter_col = st.columns([3,2])
        with search_col:
            search_term = st.text_input("Tìm kiếm trong transcript:", key="search_transcript")
        with filter_col:
            show_only_existing_audio = st.checkbox("Chỉ hiển thị bản ghi có file audio", value=True, key="filter_audio")

        filtered_df = data_df.copy()
        if show_only_existing_audio:
            filtered_df = filtered_df[filtered_df['audio_exists'] == True]
        if search_term:
            filtered_df = filtered_df[filtered_df['transcript'].astype(str).str.contains(search_term, case=False, na=False)]

        # --- PHÂN TRANG ---
        if not filtered_df.empty:
            items_per_page = st.selectbox("Số mục mỗi trang:", [5, 10, 20, 50], index=1, key="items_page")
            total_items = len(filtered_df)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            page_num_col, _ = st.columns([1,3])
            with page_num_col:
                page_number = st.number_input(f"Trang (1-{total_pages}):", min_value=1, max_value=total_pages, step=1, value=1, key="page_num")

            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            paginated_df = filtered_df.iloc[start_idx:end_idx]
            display_data_table(paginated_df)
        else:
            st.info("Không có dữ liệu nào khớp với tìm kiếm/bộ lọc của bạn.")
    else:
        st.info(f"Chưa có dữ liệu. Hãy chạy script crawl để tạo file `{METADATA_FILE}` và các file audio.")

elif selected_page == "Thống Kê Dữ Liệu":
    st.title("📊 Thống Kê Bộ Dữ Liệu Garen God-King")
    if data_df is not None and not data_df.empty and data_df['audio_exists'].sum() > 0:
        # Chỉ thực hiện thống kê trên các bản ghi có file audio tồn tại
        stats_df = data_df[data_df['audio_exists']].copy() # Tạo bản sao để tránh SettingWithCopyWarning

        st.subheader("I. Thông Tin Chung")
        num_valid_records = len(stats_df)
        st.metric("Tổng số cặp Audio-Transcript hợp lệ (có file audio)", num_valid_records)

        st.subheader("II. Phân Tích Transcript")
        if 'transcript' in stats_df.columns:
            stats_df.loc[:, 'transcript_length_chars'] = stats_df['transcript'].astype(str).apply(len)
            stats_df.loc[:, 'transcript_length_words'] = stats_df['transcript'].astype(str).apply(lambda x: len(x.split()))

            avg_chars = stats_df['transcript_length_chars'].mean()
            avg_words = stats_df['transcript_length_words'].mean()
            st.write(f"- Độ dài transcript trung bình: **{avg_chars:.2f} ký tự** / **{avg_words:.2f} từ**.")

            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                st.markdown("#### Phân bố độ dài (ký tự)")
                chart_chars = alt.Chart(stats_df[['transcript_length_chars']]).mark_bar().encode(
                    alt.X("transcript_length_chars:Q", bin=alt.Bin(maxbins=20), title="Số ký tự"),
                    alt.Y("count()", title="Số lượng")
                ).properties(height=300)
                st.altair_chart(chart_chars, use_container_width=True)
            with col_hist2:
                st.markdown("#### Phân bố độ dài (số từ)")
                chart_words = alt.Chart(stats_df[['transcript_length_words']]).mark_bar().encode(
                    alt.X("transcript_length_words:Q", bin=alt.Bin(maxbins=20), title="Số từ"),
                    alt.Y("count()", title="Số lượng")
                ).properties(height=300)
                st.altair_chart(chart_words, use_container_width=True)
        else:
            st.warning("Không tìm thấy cột 'transcript' để phân tích.")



    elif data_df is not None and data_df['audio_exists'].sum() == 0 :
         st.warning("Không có file audio nào được tìm thấy trong thư mục dữ liệu để thực hiện thống kê.")
    else:
        st.info(f"Chưa có dữ liệu. Hãy chạy script crawl để tạo file `{METADATA_FILE}`.")


elif selected_page == "Tải Dữ Liệu":
    st.title("📥 Tải Xuống Dữ Liệu")
    if data_df is not None and not data_df.empty:
        st.subheader("1. Tải File Metadata (CSV)")
        csv_export_df = data_df[data_df['audio_exists']][['filename', 'transcript']] # Chỉ xuất metadata của file audio tồn tại
        if not csv_export_df.empty:
            csv_data = csv_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải Metadata (CSV)",
                data=csv_data,
                file_name="Garen_godking_metadata.csv",
                mime="text/csv",
            )
        else:
            st.info("Không có metadata nào (cho các file audio tồn tại) để tải.")

        st.subheader("2. Tải Tất Cả File Audio (.wav) Hiện Có (ZIP)")
        audio_files_to_zip = data_df[data_df['audio_exists'] == True]['audio_path'].tolist()
        if audio_files_to_zip:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_f:
                for file_path in audio_files_to_zip:
                    zip_f.write(file_path, arcname=os.path.basename(file_path))
            zip_buffer.seek(0)
            st.download_button(
                label=f"Tải {len(audio_files_to_zip)} file audio (ZIP)",
                data=zip_buffer,
                file_name="Garen_godking_audio_files.zip",
                mime="application/zip"
            )
        else:
            st.info("Không có file audio nào tồn tại để tạo file ZIP.")
    else:
        st.info(f"Chưa có dữ liệu. Hãy chạy script crawl để tạo file `{METADATA_FILE}`.")
elif selected_page == "Tạo Audio":
    st.title("🎤 Tạo Audio")
    st.markdown("""
    Chào mừng đến với trang tạo audio sử dụng Garen God-King!  
    Nhập văn bản bên dưới và nhấn "Tạo Audio" để nghe giọng nói được tạo bởi mô hình StyleTTS2.
    """)

    # Text input field
    text_input = st.text_area("Nhập văn bản để tạo giọng nói:", "I am Garen, the God-King!", height=100)

    # Generate audio button
    if st.button("Tạo Audio"):
        if text_input.strip():
            try:
                # Send request to local Flask API (tts.py)
                with st.spinner("Đang tạo audio..."):
                    response = requests.post(
                        "http://127.0.0.1:5000/generate_wav",
                        json={"text": text_input},
                        timeout=60  # Increased timeout for GPU processing
                    )

                if response.status_code == 200:
                    # Display success message
                    st.success("Đã tạo audio thành công!")

                    # Display audio player
                    audio_data = io.BytesIO(response.content)
                    st.audio(audio_data, format="audio/wav", start_time=0)
                else:
                    error_msg = response.json().get("error", "Lỗi không xác định từ API.")
                    st.error(f"Lỗi khi tạo audio: {error_msg}")
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi kết nối đến API TTS: {str(e)}")
                st.warning("Hãy đảm bảo rằng Flask API (`tts.py`) đang chạy trên máy của bạn tại `http://localhost:5000`. Mở một terminal và chạy lệnh `python tts.py`.")
        else:
            st.warning("Vui lòng nhập văn bản trước khi tạo audio.")
# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"© {pd.Timestamp('today').year} - Báo cáo cuối kì KTLT&PTDL 2025")
st.sidebar.markdown("SVTH: Trương Quốc Khánh - Nguyễn Trọng Tín - Nguyễn Đức Quang")