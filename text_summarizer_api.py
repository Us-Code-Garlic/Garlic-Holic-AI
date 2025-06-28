from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import tempfile

load_dotenv()

app = FastAPI(title="텍스트 요약 API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델
class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 1000
    chunk_size: Optional[int] = 2000
    chunk_overlap: Optional[int] = 200

# 응답 모델
class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    success: bool
    message: Optional[str] = None

# 에러 응답 모델
class ErrorResponse(BaseModel):
    success: bool = False
    message: str

def split_text_into_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """
    긴 텍스트를 청크로 분할합니다.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
        
    Returns:
        List[str]: 분할된 텍스트 리스트
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

def map_reduce_summarize(text: str, model, max_length: Optional[int] = None) -> str:
    """
    Map-Reduce 방식을 사용하여 텍스트를 요약합니다.
    
    Args:
        text: 요약할 텍스트
        model: Gemini 모델
        max_length: 최대 요약 길이
        
    Returns:
        str: 요약된 텍스트
    """
    # 텍스트를 청크로 분할
    chunks = split_text_into_chunks(text)
    
    if len(chunks) == 1:
        # 단일 청크인 경우 직접 요약
        prompt = f"""
        당신은 전문적인 텍스트 요약 전문가입니다. 
        주어진 텍스트를 한국어로 간결하고 정확하게 요약해주세요.
        
        텍스트:
        {text}
        
        요약:
        """
        response = model.generate_content(prompt)
        summary = response.text
    else:
        # Map 단계: 각 청크를 개별적으로 요약
        chunk_summaries = []
        for chunk in chunks:
            map_prompt = f"""
            당신은 텍스트 요약 전문가입니다. 
            주어진 텍스트 부분을 한국어로 간결하게 요약해주세요.
            
            텍스트 부분:
            {chunk}
            
            요약:
            """
            response = model.generate_content(map_prompt)
            chunk_summaries.append(response.text)
        
        # Reduce 단계: 모든 요약을 하나로 결합
        combined_text = "\n\n".join(chunk_summaries)
        reduce_prompt = f"""
        당신은 텍스트 요약 전문가입니다. 
        여러 개의 요약을 하나의 통합된 요약으로 만들어주세요. 
        중복된 내용은 제거하고 핵심 내용만 포함해주세요.
        
        요약들:
        {combined_text}
        
        통합된 요약:
        """
        response = model.generate_content(reduce_prompt)
        summary = response.text
    
    # 요약 길이 제한
    if max_length and len(summary) > max_length:
        short_prompt = f"""
        당신은 텍스트 요약 전문가입니다. 
        주어진 텍스트를 {max_length}자 이내로 한국어로 간결하게 요약해주세요.
        
        텍스트:
        {summary}
        
        요약:
        """
        response = model.generate_content(short_prompt)
        summary = response.text
    
    return summary

@app.get("/")
async def root():
    return {
        "message": "텍스트 요약 API", 
        "version": "1.0.0",
        "description": "Map-Reduce 방식을 사용한 텍스트 요약 API"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API 서버가 정상 작동 중입니다."}

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Map-Reduce 방식을 사용하여 텍스트를 요약합니다.
    
    Args:
        request: 요약 요청 데이터
        
    Returns:
        SummarizeResponse: 요약 결과
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
        
        # Gemini 모델 초기화 (LangChain 없이 직접 사용)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Map-Reduce 요약 실행
        summary = map_reduce_summarize(request.text, model, request.max_length)
        
        summary_length = len(summary)
        compression_ratio = (1 - summary_length / original_length) * 100 if original_length > 0 else 0
        
        return SummarizeResponse(
            summary=summary,
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

@app.post("/summarize-file")
async def summarize_file(file_content: str, max_length: Optional[int] = 1000):
    """
    파일 내용을 요약합니다 (텍스트 파일 업로드 대신 내용을 직접 전달).
    
    Args:
        file_content: 파일 내용
        max_length: 최대 요약 길이
        
    Returns:
        SummarizeResponse: 요약 결과
    """
    request = SummarizeRequest(
        text=file_content,
        max_length=max_length
    )
    return await summarize_text(request)

@app.get("/info")
async def get_api_info():
    """
    API 정보와 사용법을 반환합니다.
    """
    return {
        "title": "텍스트 요약 API",
        "version": "1.0.0",
        "description": "Map-Reduce 방식을 사용한 텍스트 요약 API",
        "endpoints": {
            "POST /summarize": "텍스트 요약",
            "POST /summarize-file": "파일 내용 요약",
            "GET /health": "서버 상태 확인",
            "GET /info": "API 정보"
        },
        "features": {
            "method": "Map-Reduce",
            "description": "긴 텍스트를 청크로 분할하여 각각 요약한 후 최종 요약을 생성",
            "advantages": [
                "긴 텍스트 처리 가능",
                "메모리 효율적",
                "병렬 처리 가능"
            ]
        },
        "usage": {
            "text": "요약할 텍스트 (필수)",
            "max_length": "최대 요약 길이 (선택, 기본값: 1000)",
            "chunk_size": "청크 크기 (선택, 기본값: 2000)",
            "chunk_overlap": "청크 겹침 (선택, 기본값: 200)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 