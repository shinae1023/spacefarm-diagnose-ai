# 사용할 기본 이미지 지정
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 및 모델 파일 복사
COPY . .

# 포트 설정
EXPOSE 8080

# Gunicorn을 사용하여 Flask 앱 실행
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flask_app:app"]