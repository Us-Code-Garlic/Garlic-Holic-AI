from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, TypedDict
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import google.genai as genai_client
from google.genai import types
from PIL import Image
import json
import random
import uvicorn

load_dotenv()

app = FastAPI(title="통합 API 서버", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 텍스트 요약 API 모델 ====================

# 요청 모델
class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 800  # 기본값을 800자로 변경
    chunk_size: Optional[int] = 2000
    chunk_overlap: Optional[int] = 200

# 응답 모델
class SummarizeResponse(BaseModel):
    summary: str
    dementia_suspicion: bool
    original_length: int
    summary_length: int
    compression_ratio: float
    success: bool
    message: Optional[str] = None

# ==================== 퀴즈 생성 API 모델 ====================

# 응답 모델
class QuizResponse(BaseModel):
    quiz: str
    answer: str
    commentary: str
    success: bool
    message: Optional[str] = None
    image_path: Optional[str] = None

# ==================== 텍스트 요약 함수들 ====================

def split_text_into_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """
    긴 텍스트를 청크로 분할합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

def analyze_dementia_suspicion(text: str, model) -> bool:
    """
    텍스트를 분석하여 치매 의심 여부를 판단합니다.
    """
    prompt = f"""
    당신은 치매 전문의입니다. 주어진 텍스트를 분석하여 치매 의심 증상이 있는지 판단해주세요.
    
    치매 의심 증상:
    - 기억력 저하 (최근 일 기억 못함, 반복적인 질문)
    - 언어 능력 저하 (단어 찾기 어려움, 문장 구성 어려움)
    - 시공간 능력 저하 (길 찾기 어려움, 시간 개념 혼란)
    - 판단력 저하 (부적절한 행동, 의사결정 어려움)
    - 성격 변화 (과민, 우울, 무관심)
    - 일상생활 수행 능력 저하
    
    텍스트:
    {text}
    
    위 증상들이 명확하게 나타나는지 분석하고, JSON 형식으로 응답해주세요:
    {{"dementia_suspicion": true/false, "reason": "판단 근거"}}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # JSON 파싱
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text.strip()
        else:
            json_text = response_text.strip()
        
        # JSON 부분 추출
        start = json_text.find("{")
        end = json_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_text = json_text[start:end]
        
        data = json.loads(json_text)
        return data.get("dementia_suspicion", False)
        
    except Exception as e:
        print(f"치매 의심 분석 중 오류: {e}")
        return False

def map_reduce_summarize(text: str, model, max_length: Optional[int] = None) -> str:
    """
    Map-Reduce 방식을 사용하여 텍스트를 요약합니다.
    부모님의 관점에서 자식의 상태를 종합적으로 분석합니다.
    """
    # 텍스트를 청크로 분할
    chunks = split_text_into_chunks(text)
    
    if len(chunks) == 1:
        # 단일 청크인 경우 직접 요약
        prompt = f"""
        당신은 치매 전문의이자 노인 의료 전문가입니다. 
        주어진 텍스트를 분석하여 부모님의 관점에서 자식의 상태를 종합적으로 요약해주세요.
        
        다음 항목들을 포함하여 800자 이내로 작성해주세요:
        1. 치매 의심 여부 및 증상 분석
        2. 복약 상태 및 약물 복용 여부
        3. 우울증 증상 및 정신 건강 상태
        4. 대화 기록을 통한 하루의 일상 생활
        5. 전반적인 건강 상태 및 주의사항
        
        텍스트:
        {text}
        
        요약 (부모님 관점에서):
        """
        response = model.generate_content(prompt)
        summary = response.text
    else:
        # Map 단계: 각 청크를 개별적으로 분석
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            map_prompt = f"""
            당신은 치매 전문의입니다. 
            주어진 텍스트 부분을 분석하여 다음 항목들을 추출해주세요:
            - 치매 관련 증상
            - 복약 상태
            - 우울증 증상
            - 일상 생활 패턴
            - 건강 상태
            
            텍스트 부분 {i+1}:
            {chunk}
            
            분석 결과:
            """
            response = model.generate_content(map_prompt)
            chunk_analyses.append(response.text)
        
        # Reduce 단계: 모든 분석을 하나로 통합
        combined_text = "\n\n".join(chunk_analyses)
        reduce_prompt = f"""
        당신은 치매 전문의이자 노인 의료 전문가입니다. 
        여러 개의 분석 결과를 종합하여 부모님의 관점에서 자식의 상태를 요약해주세요.
        
        다음 형식으로 800자 이내로 작성해주세요:
        
        [치매 의심 여부]
        - 증상 분석 및 위험도
        
        [복약 상태]
        - 약물 복용 여부 및 복용 상태
        
        [정신 건강]
        - 우울증 증상 및 정신 상태
        
        [일상 생활]
        - 대화 기록을 통한 하루의 일상 분석
        
        [전반적 건강 상태]
        - 종합적인 건강 상태 및 주의사항
        
        분석 결과들:
        {combined_text}
        
        종합 요약 (부모님 관점에서):
        """
        response = model.generate_content(reduce_prompt)
        summary = response.text
    
    # 요약 길이 제한 (800자)
    max_summary_length = max_length if max_length else 800
    if len(summary) > max_summary_length:
        short_prompt = f"""
        당신은 치매 전문의입니다. 
        주어진 요약을 {max_summary_length}자 이내로 축약하되, 
        다음 항목들이 모두 포함되도록 해주세요:
        - 치매 의심 여부
        - 복약 상태
        - 우울증 증상
        - 일상 생활
        - 전반적 건강 상태
        
        원본 요약:
        {summary}
        
        축약된 요약 ({max_summary_length}자 이내):
        """
        response = model.generate_content(short_prompt)
        summary = response.text
    
    return summary

# ==================== 퀴즈 생성 함수들 ====================

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

# ==================== 공통 엔드포인트 ====================

@app.get("/")
async def root():
    return {
        "message": "통합 API 서버", 
        "version": "1.0.0",
        "description": "텍스트 요약 및 치매예방 퀴즈 생성 API",
        "endpoints": {
            "POST /summarize": "텍스트 요약",
            "GET /generate-quiz-random": "랜덤 이미지로 퀴즈 생성"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API 서버가 정상 작동 중입니다."}

# ==================== 텍스트 요약 엔드포인트 ====================

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Map-Reduce 방식을 사용하여 텍스트를 요약하고 치매 의심 여부를 분석합니다.
    """
    try:
        # Google API 키 확인
        if not os.getenv("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="GOOGLE_API_KEY가 설정되지 않았습니다."
            )
        
        # 텍스트 길이 확인
        if len(request.text.strip()) == 0:
            raise HTTPException(
                status_code=400, 
                detail="텍스트가 비어있습니다."
            )
        
        original_length = len(request.text)
        
        # Gemini 모델 초기화
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Map-Reduce 요약 실행
        summary = map_reduce_summarize(request.text, model, request.max_length)
        
        # 치매 의심 여부 분석
        dementia_suspicion = analyze_dementia_suspicion(request.text, model)
        
        summary_length = len(summary)
        compression_ratio = (1 - summary_length / original_length) * 100 if original_length > 0 else 0
        
        return SummarizeResponse(
            summary=summary,
            dementia_suspicion=dementia_suspicion,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=round(compression_ratio, 2),
            success=True,
            message=f"텍스트가 성공적으로 요약되었습니다. (압축률: {round(compression_ratio, 2)}%)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"요약 중 오류가 발생했습니다: {str(e)}"
        )

# ==================== 퀴즈 생성 엔드포인트 ====================

@app.get("/generate-quiz-random", response_model=QuizResponse)
async def generate_quiz_random():
    """
    example_img 폴더에서 임의의 이미지를 선택하여 치매예방 퀴즈를 생성합니다.
    """
    try:
        # 임의의 이미지 선택
        selected_image_name, image = get_random_image_from_folder()
        
        # Gemini 클라이언트 초기화
        client = genai_client.Client()
        
        # 치매예방 퀴즈 생성 요청
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

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
