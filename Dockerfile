FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ app/
COPY services/ services/
COPY data/ data/

# Expose port
EXPOSE 8420

# Run the web app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8420"]
