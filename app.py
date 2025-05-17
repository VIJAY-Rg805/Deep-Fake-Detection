import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('deepfake_detection_model_efficientnet.h5')

# Preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    return "Fake" if prediction < 0.5 else "Real"


# App Title & Banner
st.markdown("<h1 style='text-align: center; color: grey;'>DEEP FAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>", unsafe_allow_html=True)
st.image("coverpage.png", width=700)

# Description
st.header("Understanding Deepfakes")
st.write("Deepfakes are synthetic media... [your full description here]")

# Upload and Prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR")

    try:
        result = predict_image(image)
        if result == "Fake":
            color = "red"
            description = "Deepfake detected based on visual inconsistencies."
        else:
            color = "green"
            description = "Image appears to be Real."

        st.markdown(f"<h1 style='color:{color};'>The image is {result}</h1>", unsafe_allow_html=True)
        st.write(description)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Graphs
st.title("Model Training Graph")
st.markdown("### Model Training Accuracy")
st.image("Figure_2(E).png")
st.markdown("### Model Training Loss")
st.image("Figure_1(E).png")

# Footer
st.markdown("""
---
**Contact Us:**
For more information and queries, please contact us at [contact@example.com](mailto:contact@example.com).

**Follow us on:**
[Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com) | [Facebook](https://facebook.com)
""")
