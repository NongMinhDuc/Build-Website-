import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model

# Mã hóa ảnh base64
def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def display_history():

    st.sidebar.markdown("""
        <div style='display: flex; align-items: center; background-color: #8f1f29; padding: 10px;'>
            <img src="data:image/png;base64,{}" style="width: 50px; height: 50px; margin-right: 10px;">
            <h2 style='color: white; text-align: center; margin: 0;'>Hệ Thống Phân Loại Hình Ảnh</h2>
        </div>
    """.format(image_to_base64('icon.png')), unsafe_allow_html=True)

    st.sidebar.markdown("""
        <hr style='border: 1px solid #ccc;'>
    """, unsafe_allow_html=True)

## title
st.markdown("""
        <div style='background-color: #2c2c69; padding: 10px; border-radius: 15px;'>
            <h2 style='color: white; text-align: center;'>Website Hỗ Trợ Phân Loại Hình Ảnh Ung Thư Cổ Tử Cung</h2>
        </div>
    """, unsafe_allow_html=True)


if 'history' not in st.session_state:
    st.session_state.history = []

upload_file = st.file_uploader("Chọn ảnh tải lên", type=["jpg", "png"])

# Load model
model = load_model("best_model.h5")

def print_label(y):
    if y == 0:
        return "ASC_H"
    if y == 1:
        return "ASC_US"
    if y == 2:
        return "HSIL"
    if y == 3:
        return "LSIL"
    if y == 4:
        return "SCC"


if upload_file is not None:
    image_show = Image.open(upload_file)
    st.image(image_show, caption='Ảnh đã tải lên.', width=150)
    new_size = (224,224)

    if image_show.mode != 'RGB':
        image_show = image_show.convert('RGB')
        img_resize = image_show.resize(new_size)
        img_arr = np.array(img_resize).astype("float32") /255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        pred = model.predict(img_arr)
        label = np.argmax(pred)
        label = print_label(label)

if st.button('Dự đoán'):
        st.session_state.label = print_label(label)
        st.write(f"Ảnh thuộc lớp: {st.session_state.label}")

note = st.text_area("Ghi chú của bạn:")

if st.button('Lưu kết quả'):
    st.session_state.history.append([upload_file.name,label,note])
        
col1, col2 = st.columns([1,1])

with col1:
    if st.button('Xem lịch sử'):
        df = pd.DataFrame(st.session_state.history, columns=["Tên tệp", "Nhãn", "Ghi chú"])
        df.index = df.index + 1
        st.dataframe(df)
with col2:
    if st.button('Chỉ Dẫn'):
        st.markdown("""
                <div style='background-color: #f0f0f5; padding: 10px; border-radius: 10px;'>
                    <h3>Hướng dẫn sử dụng:</h3>
                    <p>1. Tải lên hình ảnh có định dạng JPG hoặc PNG.</p>
                    <p>2. Xem kết quả phân loại của ảnh tải lên.</p>
                    <p>3. Ghi chú nếu cần và lưu lại kết quả.</p>
                    <p>4. Xem lịch sử các hình ảnh đã kiểm tra trong phần 'Lịch Sử Tra Cứu Hình Ảnh' ở thanh bên.</p>
                </div>
            """, unsafe_allow_html=True)


display_history()

