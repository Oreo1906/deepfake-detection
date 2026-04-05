FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System libs required by MediaPipe/OpenCV at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code and model assets used at startup/inference.
COPY backend /app/backend
COPY eye_model.pth /app/eye_model.pth
COPY lip_model.pth /app/lip_model.pth
COPY nose_model.pth /app/nose_model.pth
COPY skin_model.pth /app/skin_model.pth
COPY geometry_model.pth /app/geometry_model.pth
COPY geometry_scaler.npy /app/geometry_scaler.npy

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
