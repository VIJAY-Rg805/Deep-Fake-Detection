# 🧠 DeepFake Detection in Social Media Content

This project uses deep learning to detect whether a face image is **real or AI-generated (deepfake)**. It combines the power of EfficientNetB3 with a custom classification head to identify fake images from real ones.

## 📂 Project Structure

```
├── app.py                        # Streamlit web app for uploading and detecting images
├── train.py                      # Script to train the model on real/fake face data
├── predict.py                    # Optional helper script for standalone predictions
├── deepfake_detection_model.h5   # Trained model (MObilenetV3)( but in  the code model is in EfficientNet-based)
├── coverpage.png                 # Banner image for the app
├── Figure_1(E).png               # Training Loss Graph
├── Figure_2(E).png               # Training Accuracy Graph
```

---

## 🚀 Features

* 📷 Upload any face image (JPG, JPEG, PNG)
* 🧠 AI tells you if the image is **Fake** or **Real**
* 📊 View training graphs for model accuracy and loss
* ✅ Built using TensorFlow, Keras, Streamlit, OpenCV

---

## 🧑‍💻 How to Run

### 🔹 1. Clone the Repo

```bash
git clone https://github.com/your-username/deepfake-detection
cd deepfake-detection
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> You can manually install important packages:

```bash
pip install streamlit tensorflow opencv-python numpy
```

### 🔹 3. Launch the App

```bash
streamlit run app.py
```

---

## 🏠 Model Architecture

* **Base**: EfficientNetB3 (pre-trained on ImageNet)
* **Custom Layers**:

  * GlobalAveragePooling2D
  * Dense(512) → ReLU → BatchNormalization → Dropout(0.4)
  * Dense(128) → ReLU → Dropout(0.3)
  * Dense(1) → Sigmoid (binary output)

---

## 📈 Training Performance

* **Dataset**: Real and Fake face images
* **Accuracy**: \~92.3% on validation
* **Epochs**: 20
* **Optimizer**: Adam
* **Loss Function**: Binary Crossentropy
* **Learning Rate Scheduler**: Used for better convergence

---

## 🖼️ Sample Results

![Accuracy](Figure_2\(E\).png)
*Train vs Validation Accuracy*

![Loss](Figure_1\(E\).png)
*Train vs Validation Loss*

---

## 💡 Use Case

This project can be used in:

* Digital media verification
* Face dataset authenticity validation
* Research in misinformation detection
* AI content flagging tools








