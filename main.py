"""
(1) Fast API 설치
pip install "fastapi[all]"

(2) Fast API 실행
uvicorn test:app --reload

(3) API 테스트
http://localhost:8000/api/hello


디렉토리 구조
main.py : FastAPI 애플리케이션 엔트리포인트(진입점)
api : API(Endpoint)
core : 설정, 보안, 환경변수, 인증, JWT 로직 등
schemas : 요청 및 응답 Pydantic 모델
services : 비즈니스 로직 서비스
utils : 유틸리티 함수 및 모듈
"""
#.env File Variable Read
from dotenv import load_dotenv
import os
load_dotenv()

#Fast API framework
from fastapi import FastAPI

#Service Module
from app.api import estimate

app = FastAPI(title="Demo FastAPI for Spring")

# app/api/uploadImage.py 라우터들을 FastAPI 애플리케이션에 등록. endpoint prefix는 /fastapi/image, 태그는 hello로 설정
app.include_router(estimate.router, prefix="/mycar")

