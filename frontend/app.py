# app.py
import streamlit as st
import tempfile
import os

# Import your pipelines
from src.mini_pipeline.video2face_embedding_pipeline import video2face_embedding
from src.mini_pipeline.card_process_pipeline import card_process
from src.mini_pipeline.card2face_embedding_pipeline import card2face_embedding
from src.retrieval.face_retrieval_verify import get_user_embeddings, verify_face
from src.mini_pipeline.card2text_pipeline import card2text
from src.utils.image_processing import read_image
from src.text_processing.text_recognition import load_vietocr_model
from src.utils.utils import load_YOLO
from src.utils.config import CARD_DETECT_MODEL , FACE_CARD_DETECT_MODEL , HEAD_DETECT_MODEL , TEXT_DETECT_MODEL, TEXT_RECOG_MODEL

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="🪪 Card Face Verification & OCR", layout="centered")

vietocr_model = load_vietocr_model()

card_detect_model = load_YOLO(CARD_DETECT_MODEL)
face_card_detect_model = load_YOLO(FACE_CARD_DETECT_MODEL)
head_detect_model = load_YOLO(HEAD_DETECT_MODEL)
text_detect_model = load_YOLO(TEXT_DETECT_MODEL)

# ==================== INITIAL SESSION STATES ====================
if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "user_enrolled" not in st.session_state:
    st.session_state.user_enrolled = False

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

if "face_verified" not in st.session_state:
    st.session_state.face_verified = False

# ==================== PAGE TITLE ====================
st.title("🪪 Face ID Verification & Card OCR Extraction")

st.markdown("---")

# ==================== KHỐI 1: FACE ID SELECTION ====================
st.header("Step 1: Face ID Enrollment")

face_id_option = st.selectbox(
    "Do you already have a Face ID profile?",
    ("⬜ Select an option", "No, I need to create one", "Yes, I already have Face ID")
)

if face_id_option == "⬜ Select an option":
    st.warning("⚠️ Please select an option first.")
    st.stop()

st.markdown("---")

# ==================== BRANCHING FLOW ====================
if face_id_option == "Yes, I already have Face ID":
    st.subheader("🔎 Check your Face ID")

    user_name_input = st.text_input("Enter your User Name")

    check_user_btn = st.button("🔍 Check User", use_container_width=True)

    if check_user_btn:
        if user_name_input:
            # Thực sự check trong db
            try:
                embeddings = get_user_embeddings(user_name_input)
                if embeddings.shape[0] > 0:
                    st.success(f"✅ Welcome back, {user_name_input}!")
                    st.session_state.user_name = user_name_input
                    st.session_state.user_enrolled = True
                else:
                    st.error(f"❌ User '{user_name_input}' not found. Please enroll Face ID.")
                    st.session_state.user_name = user_name_input
                    st.session_state.user_enrolled = False
            except Exception as e:
                st.error(f"❌ Error checking user: {e}")
                st.session_state.user_name = user_name_input
                st.session_state.user_enrolled = False
        else:
            st.warning("⚠️ Please enter your User Name.")

else:  # No, need to create
    st.subheader("📹 Create Your Face ID")

    col1, col2, col3, col4 = st.columns([3, 3, 2, 1])

    with col1:
        user_name_input = st.text_input("Enter your User Name")

    with col2:
        uploaded_video = st.file_uploader("Upload Face Video", type=["mp4", "avi", "mov"])

    @st.dialog("Hướng dẫn tạo Face ID")
    def show_faceid_tutorial():
        st.markdown("""
        **Hướng dẫn tạo Face ID:**
        - Nhìn thẳng vào camera.
        - Quay video khoảng 5-10 giây.
        - Tránh đeo kính hoặc khẩu trang.
        - Đảm bảo ánh sáng tốt.
        """)
        if st.button("Đóng"):
            st.rerun()

    with col3:
        if st.button("❓ Tutorial", use_container_width=True):
            show_faceid_tutorial()

    with col4:
        start_enroll = st.button("🎥", help="Process video", use_container_width=True)

    if start_enroll:
        if user_name_input:
            if uploaded_video:
                with st.spinner("⏳ Processing video..."):
                    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_video_path.write(uploaded_video.read())
                    temp_video_path.close()
                    video2face_embedding(temp_video_path.name, user_id=user_name_input)  # Pass user_name
                    st.success(f"✅ Welcome, {user_name_input}! Your Face ID has been enrolled.")
                    st.session_state.user_name = user_name_input
                    st.session_state.user_enrolled = True
                    os.unlink(temp_video_path.name)
            else:
                st.warning("⚠️ Please upload your face video first.")
        else:
            st.warning("⚠️ Please enter your User Name.")

st.markdown("---")

# ==================== KHỐI 2: UPLOAD CARD + VERIFY OCR ====================
if st.session_state.user_enrolled:
    st.header("Step 2: Card Verification & OCR")

    col5, col6 = st.columns([5, 1])

    with col5:
        uploaded_card = st.file_uploader("Upload Card Image", type=["jpg", "jpeg", "png"], key="card_upload")

    @st.dialog("Hướng dẫn tải lên CCCD")
    def show_cccd_tutorial():
        st.markdown("""
        **Hướng dẫn tải lên CCCD:**
        - Tải lên hình ảnh rõ ràng của CCCD.
        - Tránh mờ, lóa sáng hoặc bị cắt xén.
        """)
        if st.button("Đóng"):
            st.rerun()

    with col6:
        if st.button("📜 Tutorial", use_container_width=True):
            show_cccd_tutorial()

    verify_btn = st.button("🛡️", help="Verify and Extract", use_container_width=True)

    st.markdown("---")

    # ==================== KHỐI 3: LIVE STATUS PLACEHOLDER ====================
    status_placeholder = st.empty()

    if verify_btn:
        if uploaded_card:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_card_file:
                temp_card_file.write(uploaded_card.read())
                card_path = temp_card_file.name

            # Step 1: Detecting
            status_placeholder.markdown("🔍 Detecting Card...")
            transformed_card = card_process(card_path, card_detect_model)

            # Step 2: Transforming
            status_placeholder.markdown("🛠 Transforming Card...")
            # (Card đã transform ở trên rồi)

            # Step 3: Verifying Face
            status_placeholder.markdown("🛡️ Verifying Face...")
            face_card_embedding = card2face_embedding(transformed_card, face_card_detect_model, head_detect_model)
            db_embeddings= get_user_embeddings(st.session_state.user_name)
            verified = verify_face(face_card_embedding, db_embeddings)

            if verified:
                st.success("✅ Face Verified Successfully!")

                # Step 4: Extracting Text
                status_placeholder.markdown("📝 Extracting Text...")
                extracted_text = card2text(transformed_card, vietocr_model, text_detect_model)

                # Step 5: Completed
                status_placeholder.markdown("✅ Completed! See extracted text below.")
                st.subheader("📝 Extracted Information:")
                for field, value in extracted_text.items():
                    st.write(f"**{field.upper()}**: {value}")

            else:
                st.error("❌ Face verification failed!")

            os.unlink(card_path)

        else:
            st.warning("⚠️ Please upload your Card Image first.")

