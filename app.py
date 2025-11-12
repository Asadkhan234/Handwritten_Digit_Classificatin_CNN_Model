import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_digit_model():
    try:
        model = load_model('digit_model.h5')
        return model
    except Exception as e:
        st.error("❌ Model file 'digit_model.h5' not found! Please place it in the same folder as app.py.")
        st.stop()

model = load_digit_model()

# -------------------------------
# Image preprocessing function
# -------------------------------
def preprocess_image(image, model_type='cnn'):
    # Convert to grayscale and resize
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0

    # Invert colors if needed
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    # Reshape according to model type
    if model_type == 'cnn':
        img_array = img_array.reshape(1, 28, 28, 1)
    else:  # dense model
        img_array = img_array.reshape(1, 784)
    
    return img, img_array

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="🖊️ Handwritten Digit Classifier", page_icon="✏️", layout="wide")
st.title("🧠 Handwritten Digit Classifier")
st.write("Upload a handwritten digit image (0–9) and see the prediction with probabilities!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file)

    # Detect model type automatically
    model_type = 'cnn' if len(model.input_shape) == 4 else 'dense'

    # Preprocess
    processed_img, processed_array = preprocess_image(image, model_type)

    # Prediction
    prediction = model.predict(processed_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # -------------------------------
    # Horizontal layout with columns
    # -------------------------------
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    # Original uploaded image
    with col1:
        st.subheader("📷 Uploaded")
        st.image(image, use_container_width=True)

    # Processed image
    with col2:
        st.subheader("🖼️ Processed")
        st.image(processed_img, width=120)


    # Prediction info
    with col3:
        st.subheader("🔍 Prediction")
        st.write(f"**Digit:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    # Probability graph
    with col4:
        st.subheader("📊 Probabilities")
        fig2, ax2 = plt.subplots(figsize=(4,1.5))  # short and wide
        ax2.bar(range(10), prediction[0], color='skyblue')
        ax2.set_xticks(range(10))
        ax2.set_xlabel("Digits")
        ax2.set_ylabel("Prob")
        ax2.set_title("Prediction Probabilities")
        st.pyplot(fig2, use_container_width=True)

else:
    st.info("👆 Upload an image above to get started.")
