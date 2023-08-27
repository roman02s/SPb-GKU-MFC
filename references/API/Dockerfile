# Используем базовый образ Python
FROM python:3.9

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости в контейнер
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения в контейнер
COPY . .

# Указываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Запускаем FastAPI при старте контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# uvicorn main:app --host 0.0.0.0 --port 8000
# sudo docker build -t api:v0 .
# sudo docker run -d -v /home/rustem/SPb-GKU-MFC/data:/data --restart unless-stopped -p 8000:8000 -t api:v0
# sudo docker build -t front:v1 .
# sudo docker run -d --restart unless-stopped -p 8501:8501 -t front:v0
