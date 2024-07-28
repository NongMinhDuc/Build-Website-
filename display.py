import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load model
model_load = tf.keras.models.load_model('Xception.keras')

# Ensure session state initialization
if 'history' not in st.session_state:
    st.session_state.history = []

# Mapping labels
label_map = {
    0: "ASC_H",
    1: "ASC_US",
    2: "HSIL",
    3: "LSIL",
    4: "SCC"
}

# Mã hóa ảnh base64
def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Function to predict image class
def predict_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    predictions = model_load.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Save history
def save_to_history(image, predicted_class, note):
    st.session_state.history.append({'Ảnh': image.name, 'Kết quả': predicted_class, 'Ghi chú': note})

# Function to display history in sidebar
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

    st.sidebar.markdown("""
        <div style='background-color: #e3803d; padding: 10px; border-radius: 5px;'>
            <h2 style='color: #333300; text-align: center;'>Lịch Sử Tra Cứu Hình Ảnh</h2>
        </div>
    """, unsafe_allow_html=True)
    history = st.session_state.get('history', [])
    for idx, entry in enumerate(reversed(history[-10:])):
        label = label_map.get(entry['Kết quả'], 'Unknown')
        st.sidebar.markdown(f"**[+] Ảnh số : {idx+1}**")
        st.sidebar.write(f"Tên ảnh: {entry['Ảnh']}")
        st.sidebar.write(f"Kết quả: {label}")
        st.sidebar.write(f"Ghi chú: {entry['Ghi chú']}")
        st.sidebar.markdown("---")

# Main Streamlit app

## image background
background_image = image_to_base64('image_background_2.png')  
st.markdown(f"""
    <style>
    .reportview-container {{
        background: url(data:image/png;base64,{background_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)

## title
st.markdown("""
        <div style='background-color: #2c2c69; padding: 10px; border-radius: 15px;'>
            <h2 style='color: white; text-align: center;'>Website Hỗ Trợ Phân Loại Hình Ảnh Ung Thư Cổ Tử Cung</h2>
        </div>
    """, unsafe_allow_html=True)

upload_file = st.file_uploader("Chọn ảnh tải lên", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption='Ảnh đã tải lên.', width=150)
    st.write("Kích thước ảnh: ", image.size)

    # Predict image
    st.write("Đang phân loại ảnh...")
    progress = st.progress(0)
    try:
        for percent_complete in range(100):
            progress.progress(percent_complete + 1)    # Thanh trạng thái 
        predicted_class = predict_image(image)
        label = label_map.get(predicted_class, 'Unknown')
        st.write(f"Ảnh thuộc lớp: {label}")
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán. {e}")

    # Note
    note = st.text_area("Ghi chú")
    st.write("Ghi chú của bạn:", note)

    # Save to history
    if st.button('Lưu kết quả'):
        save_to_history(upload_file, predicted_class, note)
        st.success('Đã lưu kết quả thành công!')

# Hiển thị ra màn hình
display_history()



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