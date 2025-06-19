import streamlit as st
import tempfile
from datetime import timedelta
from config import BUCKET_NAME
from pipeline import run_rag_pipeline  # You must have this pipeline implemented as before

st.set_page_config(page_title="PDF Q&A with Image Retrieval", layout="wide")
st.title("PDF Q&A and Visual Retrieval")

with st.sidebar:
    st.header("Upload & Ask")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    user_query = st.text_input("Your question", value="explain Scaled Dot-Product Attention")
    search_button = st.button("Search")

def show_images(image_urls):
    st.subheader("🖼 Related Images")
    cols = st.columns(min(3, len(image_urls)))
    for idx, url in enumerate(image_urls):
        with cols[idx % len(cols)]:
            st.image(url, use_column_width=True)
            st.caption(f"Image {idx+1}")

if uploaded_pdf and search_button:
    with st.spinner("Processing..."):
        # Save the uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_pdf_path = tmp_file.name
        try:
            result = run_rag_pipeline(tmp_pdf_path, user_query)
            answer = result.get("answer", "No answer generated.")
            matched_images = result.get("matched_images", [])
            st.success("Results ready!")
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            st.stop()

        st.subheader("🧠 GPT-4o Answer")
        st.write(answer)
        if matched_images:
            show_images(matched_images)
        else:
            st.info("No related images found.")

else:
    st.info("Upload a PDF and enter your question, then click Search.")
