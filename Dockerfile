# ===== Base Python Image =====
FROM python:3.9-slim

# ===== Install OS dependencies =====
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ===== Install Python dependencies early =====
RUN pip install --upgrade pip
RUN pip install gdown

# ===== Set working directory =====
WORKDIR /app

# ===== Copy code =====
COPY . /app

# ===== Create models folder =====
RUN mkdir -p /app/models

# ===== Copy script download model vào container =====
COPY download_models.py /app/download_models.py

# ===== Download models =====
RUN python3 download_models.py

# ===== Debug xem models tải ok chưa =====
RUN ls -lh /app/models/

# ===== Install project Python packages =====
RUN pip install -r requirements.txt

# ===== Expose port =====
EXPOSE 8501

# ===== Run app =====
CMD ["python3", "-m", "streamlit", "run", "frontend/app.py"]
