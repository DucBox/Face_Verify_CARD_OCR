# 🚀 Face Verification & ID Extraction

## 📌 Introduction
This project is a face verification and ID card information extraction system. It utilizes **Streamlit** for the user interface, **Firestore (Firebase)** for storing embeddings, and **deep learning models** for image processing.

---

## 🛠 Installation

### 🔹 1. Clone the Project
```bash
git clone https://github.com/DucBox/OCR.git
cd OCR
```

### 🔹 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔥 Configuration

### 🔹 1. Database Configuration
All configurations are stored in `config.py`. Update the **database path** before running the application.

- Open `config.py` and set `DATABASE_CONFIG_PATH`.
- **How to get Firestore credentials from Firebase**:
  1. Go to [Firebase Console](https://console.firebase.google.com/).
  2. Navigate to **Project Settings** > **Service Accounts**.
  3. Click **Generate new private key**, and download the JSON file.
  4. Place the JSON file in the `src/` directory and update `DATABASE_CONFIG_PATH`.

```python
DATABASE_CONFIG_PATH = "src/firebase_config.json"
```

### 🔹 2. Model Paths Configuration
Set the model paths in `config.py`:
```python
MODEL_FACE_EMBEDDING = "models/facenet.pth"
MODEL_TEXT_RECOGNITION = "models/vietocr.pth"
...
```

---

## 🚀 Running the Application

### 🔹 1. Run Locally
```bash
streamlit run frontend/app.py
```
Access the application at `http://localhost:8501`

---

## 📜 License & Author
- 📌 **Author:** Ngo Quang Duc
- 📌 **Contact:** quangducngo0811@gmail.com

