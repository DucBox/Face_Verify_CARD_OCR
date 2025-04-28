# ===== Base Python Image =====
FROM python:3.9-slim

# ===== Install OS dependencies (early để cache tốt hơn) =====
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# ===== Set working directory =====
WORKDIR /app

# ===== Copy code =====
COPY . /app

# ===== Install Git LFS and pull real model files =====
RUN git lfs install && git lfs pull

# ===== Debug xem bên trong container đã có những file nào =====
RUN ls -R /app

# ===== Install Python packages (cached!) =====
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ===== Expose port =====
EXPOSE 8501

# ===== Set entrypoint =====
CMD ["python3", "-m", "streamlit", "run", "frontend/app.py"]
