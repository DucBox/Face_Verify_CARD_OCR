# ===== Base Python Image =====
FROM python:3.9-slim

# ===== Install OS dependencies (early để cache tốt hơn) =====
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*


# ===== Set working directory =====
WORKDIR /app

# ===== Copy code =====
COPY . /app

# ===== Download model files =====
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1APfWKSS-lHpI5yERsxD2_FZSntGwUqWT' -O /app/models/card_detect.pt
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cMNwpR9m4QAwv2lK2QXZvGaqgTg904lh' -O /app/models/face_card_detect.pt
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UUXsI_Y1BAiPQ3wuk2gyFfMWhzeLbyJy' -O /app/models/head_detect.pt
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xWOiSHxe_QBzdzYmQ0IM9wWTQU7A5c4n' -O /app/models/text_recog.pt


# ===== Debug xem bên trong container đã có những file nào (tùy chọn, có thể giữ để debug) =====
RUN ls -R /app

# ===== Install Python packages (cached!) =====
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ===== Expose port =====
EXPOSE 8501

# ===== Set entrypoint =====
CMD ["python3", "-m", "streamlit", "run", "frontend/app.py"]
