# Dockerfile for PC-24 Takeoff & Landing Streamlit app
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install system deps (Tesseract if you want OCR inside container) - optional
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy app and resources
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir streamlit pandas numpy pytesseract pillow

# Expose Streamlit port
EXPOSE 8501

# Streamlit configuration to allow running as root and binding to all interfaces
ENV STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the app
CMD ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "pc24_takeoff_landing_app.py"]
