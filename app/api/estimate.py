# MVC에서 Controller 역할을 하는 모듈
from fastapi import APIRouter, File, UploadFile
from app.service import estimate_cv_service
from app.service import estimate_ml_service
from fastapi import File, UploadFile, Form
from typing import List
import json
import os
import io
import zipfile
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.responses import StreamingResponse

router = APIRouter()
current_file = Path(__file__).resolve()
APP_PATH = current_file.parent.parent  # api 상위 폴더 = app

# Spring Boot => FastAPI MultipartFile(이미지 파일) 전달
@router.post("/estimate")
async def upload_file(files: List[UploadFile] = File(...), filePathList: List[str] = Form(...)) :
    # 문자열로 전달되어 리스트로 변환
    filePathList = json.loads(filePathList[0])
    
    idx = 0
    
    for filePath in filePathList : 
        save_path = "app/uploads/" + filePath[0:8]
        
        # 날짜 폴더 생성
        os.makedirs(save_path, exist_ok=True)  
        
        # 파일 읽기
        content = await files[idx].read()
        
        # 저장 경로 설정
        save_path = os.path.join(save_path, filePath)
        
        with open(save_path, "wb") as f:
            f.write(content)
            
        # CV 모델 호출하여 json 반환
        cv_response_json = estimate_cv_service.estimate_custom_vision(save_path)
        
        # torch vision 모델 호출 및 cost json 읽기
        estimate_ml_service.estimate_torch_vision(cv_response_json)
        
        idx+=1
        
    # ZIP 파일을 메모리에서 생성
    zip_buffer = io.BytesIO()
    all_json_results = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filePath in filePathList:
            base_name = os.path.splitext(filePath)[0]  # 20251116153434640_05595

             # JSON 파일 읽기
            json_filename = f"{base_name}_cost.json"
            json_path = os.path.join(APP_PATH / "ml_outputs", json_filename)
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_json_results.append(data)
            else:
                print('cost json file not found !!')

            # JPG 파일 추가
            jpg_filename = f"{base_name}_image.jpg"
            jpg_path = os.path.join(APP_PATH / "ml_outputs", jpg_filename)
            if os.path.exists(jpg_path):
                zipf.write(jpg_path, arcname=jpg_filename)
            else:
                print('result JPG file not found !!')
        
        # 모든 JSON 합쳐서 result.json 생성
        result_json_bytes = json.dumps(all_json_results, ensure_ascii=False, indent=2).encode("utf-8")
        zipf.writestr("result.json", result_json_bytes)

    zip_buffer.seek(0)

    # StreamingResponse로 ZIP 반환
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=result.zip"}
    )
