from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from typing import TypedDict, Optional
import os
import json
import random
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="치매예방 퀴즈 생성 API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 구조화된 출력을 위한 타입 정의
class DementiaQuiz(TypedDict):
    quiz: str
    answer: str
    commentary: str

# 응답 모델
class QuizResponse(BaseModel):
    quiz: str
    answer: str
    commentary: str
    success: bool
    message: Optional[str] = None
    image_path: Optional[str] = None

# 에러 응답 모델
class ErrorResponse(BaseModel):
    success: bool = False
    message: str

def clean_json_response(response_text: str) -> str:
    """
    Gemini API 응답에서 JSON 부분을 추출하고 정리합니다.
    """
    # 코드 블록 제거
    if "```json" in response_text:
        # ```json과 ``` 사이의 내용 추출
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    
    # 일반 JSON 형식인 경우
    if response_text.strip().startswith("{") and response_text.strip().endswith("}"):
        return response_text.strip()
    
    # JSON 부분 찾기
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start != -1 and end != 0:
        return response_text[start:end]
    
    return response_text

def get_random_image_from_folder(folder_path: str = "example_img") -> tuple[str, Image.Image]:
    """
    example_img 폴더에서 임의의 이미지를 선택하여 반환합니다.
    
    Args:
        folder_path: 이미지가 저장된 폴더 경로
        
    Returns:
        tuple: (이미지 파일명, PIL Image 객체)
    """
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 폴더 내 이미지 파일들 찾기
    image_files = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        raise FileNotFoundError(f"{folder_path} 폴더에 이미지 파일이 없습니다.")
    
    # 임의의 이미지 선택
    selected_image = random.choice(image_files)
    image_path = os.path.join(folder_path, selected_image)
    
    # 이미지 로드
    image = Image.open(image_path)
    
    return selected_image, image

@app.get("/")
async def root():
    return {"message": "치매예방 퀴즈 생성 API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API 서버가 정상 작동 중입니다."}

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(file: UploadFile = File(...)):
    """
    이미지를 업로드하여 치매예방 퀴즈를 생성합니다.
    
    Args:
        file: 업로드할 이미지 파일
        
    Returns:
        QuizResponse: 생성된 퀴즈 정보
    """
    try:
        # 파일 타입 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 이미지 읽기
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Gemini 클라이언트 초기화
        client = genai.Client()
        
        # 치매예방 퀴즈 생성 요청 - 수정된 프롬프트
        text_input = ('이 이미지를 참고해서 유사한 주제의 치매예방 퀴즈를 만들어 주세요. '
                    '이미지와 완전히 똑같은 문제가 아니라, 이미지의 주제나 내용을 참고하여 '
                    '비슷한 유형의 새로운 문제를 생성해 주세요. '
                    '다음 JSON 형식으로 응답해 주세요: '
                    '{"quiz": "질문", "answer": "답변", "commentary": "설명"}')
        
        # 이미지 분석용 모델 사용
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[text_input, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
        
        # 응답 파싱
        response_text = response.candidates[0].content.parts[0].text
        
        # JSON 파싱 및 검증
        try:
            # 응답 텍스트 정리
            cleaned_response = clean_json_response(response_text)
            quiz_data = json.loads(cleaned_response)
            
            return QuizResponse(
                quiz=quiz_data['quiz'],
                answer=quiz_data['answer'],
                commentary=quiz_data['commentary'],
                success=True,
                message="퀴즈가 성공적으로 생성되었습니다."
            )
            
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 원본 텍스트 반환
            return QuizResponse(
                quiz="",
                answer="",
                commentary=f"JSON 파싱 오류: {str(e)}\n원본 응답: {response_text}",
                success=False,
                message="JSON 파싱에 실패했습니다. 원본 응답을 확인하세요."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/generate-quiz-random", response_model=QuizResponse)
async def generate_quiz_random():
    """
    example_img 폴더에서 임의의 이미지를 선택하여 치매예방 퀴즈를 생성합니다.
    
    Returns:
        QuizResponse: 생성된 퀴즈 정보와 사용된 이미지 경로
    """
    try:
        # 임의의 이미지 선택
        selected_image_name, image = get_random_image_from_folder()
        
        # Gemini 클라이언트 초기화
        client = genai.Client()
        
        # 치매예방 퀴즈 생성 요청 - 수정된 프롬프트
        text_input = ('이 이미지를 참고해서 유사한 주제의 치매예방 퀴즈를 만들어 주세요. '
                    '이미지와 완전히 똑같은 문제가 아니라, 이미지의 주제나 내용을 참고하여 '
                    '비슷한 유형의 새로운 문제를 생성해 주세요. '
                    '다음 JSON 형식으로 응답해 주세요: '
                    '{"quiz": "질문", "answer": "답변", "commentary": "설명"}')
        
        # 이미지 분석용 모델 사용
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[text_input, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
        
        # 응답 파싱
        response_text = response.candidates[0].content.parts[0].text
        
        # JSON 파싱 및 검증
        try:
            # 응답 텍스트 정리
            cleaned_response = clean_json_response(response_text)
            quiz_data = json.loads(cleaned_response)
            
            return QuizResponse(
                quiz=quiz_data['quiz'],
                answer=quiz_data['answer'],
                commentary=quiz_data['commentary'],
                success=True,
                message="퀴즈가 성공적으로 생성되었습니다.",
                image_path=f"example_img/{selected_image_name}"
            )
            
        except json.JSONDecodeError as e:
            return QuizResponse(
                quiz="",
                answer="",
                commentary=f"JSON 파싱 오류: {str(e)}\n원본 응답: {response_text}",
                success=False,
                message="JSON 파싱에 실패했습니다. 원본 응답을 확인하세요.",
                image_path=f"example_img/{selected_image_name}"
            )
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")

@app.post("/generate-quiz-from-url")
async def generate_quiz_from_url(image_url: str):
    """
    이미지 URL을 제공하여 치매예방 퀴즈를 생성합니다.
    
    Args:
        image_url: 이미지 URL
        
    Returns:
        QuizResponse: 생성된 퀴즈 정보
    """
    try:
        import requests
        
        # 이미지 다운로드
        response = requests.get(image_url)
        response.raise_for_status()
        
        # 이미지 로드
        image = Image.open(BytesIO(response.content))
        
        # Gemini 클라이언트 초기화
        client = genai.Client()
        
        # 치매예방 퀴즈 생성 요청 - 수정된 프롬프트
        text_input = ('이 이미지를 참고해서 유사한 주제의 치매예방 퀴즈를 만들어 주세요. '
                    '이미지와 완전히 똑같은 문제가 아니라, 이미지의 주제나 내용을 참고하여 '
                    '비슷한 유형의 새로운 문제를 생성해 주세요. '
                    '다음 JSON 형식으로 응답해 주세요: '
                    '{"quiz": "질문", "answer": "답변", "commentary": "설명"}')
        
        # 이미지 분석용 모델 사용
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[text_input, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
        
        # 응답 파싱
        response_text = response.candidates[0].content.parts[0].text
        
        # JSON 파싱 및 검증
        try:
            # 응답 텍스트 정리
            cleaned_response = clean_json_response(response_text)
            quiz_data = json.loads(cleaned_response)
            
            return QuizResponse(
                quiz=quiz_data['quiz'],
                answer=quiz_data['answer'],
                commentary=quiz_data['commentary'],
                success=True,
                message="퀴즈가 성공적으로 생성되었습니다."
            )
            
        except json.JSONDecodeError as e:
            return QuizResponse(
                quiz="",
                answer="",
                commentary=f"JSON 파싱 오류: {str(e)}\n원본 응답: {response_text}",
                success=False,
                message="JSON 파싱에 실패했습니다. 원본 응답을 확인하세요."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)