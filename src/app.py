import streamlit as st
from PIL import Image

st.title("Image Upload")

uploaded_files = st.file_uploader(
    "Choose images", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, width='stretch')