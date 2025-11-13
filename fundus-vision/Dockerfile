FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_PATH=models/checkpoints/best.pt
ENV CLASSES_PATH=models/checkpoints/classes.json
ENV BACKBONE=convnext_base
ENV HEAD=rpattn
ENV IMG_SIZE=224
EXPOSE 8000
CMD ["gunicorn","-w","2","-b","0.0.0.0:8000","src.api.app:app"]
