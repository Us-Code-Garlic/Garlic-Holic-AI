import os
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import speech
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import json

load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(
    title="MP4 STT + Gemini API",
    description="MP4 파일에서 음성을 추출하여 STT를 수행하고 Gemini API로 응답을 생성하는 서비스",
    version="1.0.0"
)

# 1. GCP Speech-to-Text 클라이언트 초기화
try:
    speech_client = speech.SpeechClient()
except Exception as e:
    print("GCP Speech-to-Text 클라이언트 초기화에 실패했습니다.")
    print("Google Cloud 인증이 설정되어 있는지 확인해주세요.")
    print("다음 중 하나의 방법으로 인증을 설정하세요:")
    print("1. GOOGLE_APPLICATION_CREDENTIALS 환경 변수 설정")
    print("2. gcloud auth application-default login 실행")
    print(f"오류: {e}")

# 2. Gemini 클라이언트 초기화
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception as e:
    print("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("`.env` 파일에 키를 추가하거나 환경 변수를 설정해주세요.")
    print(f"오류: {e}")

def extract_audio_from_mp4(mp4_file_path: str, output_audio_path: str) -> bool:
    """
    MP4 파일에서 오디오를 추출하여 WAV 파일로 저장합니다.
    
    Args:
        mp4_file_path: 입력 MP4 파일 경로
        output_audio_path: 출력 WAV 파일 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        # ffmpeg를 사용하여 MP4에서 오디오 추출
        cmd = [
            'ffmpeg',
            '-i', mp4_file_path,
            '-vn',  # 비디오 스트림 제외
            '-acodec', 'pcm_s16le',  # 16-bit PCM 코덱
            '-ar', '16000',  # 샘플링 레이트 16kHz
            '-ac', '1',  # 모노 채널
            '-y',  # 기존 파일 덮어쓰기
            output_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg 오류: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {e}")
        return False

def transcribe_audio_with_gcp(audio_file_path: str) -> Optional[str]:
    """
    GCP Speech-to-Text API를 사용하여 오디오를 텍스트로 변환합니다.
    
    Args:
        audio_file_path: 오디오 파일 경로
    
    Returns:
        Optional[str]: 변환된 텍스트 또는 None
    """
    try:
        # 오디오 파일 읽기
        with open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()

        # 오디오 설정
        audio = speech.RecognitionAudio(content=content)
        
        # 인식 설정
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",  # 한국어 설정
            enable_automatic_punctuation=True,  # 자동 문장부호 추가
            model="latest_long"  # 긴 오디오에 최적화된 모델
        )

        # 음성 인식 요청
        response = speech_client.recognize(config=config, audio=audio)

        # 결과 처리
        if response.results:
            # 가장 신뢰도가 높은 결과 반환
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            return transcript.strip()
        else:
            return None
            
    except FileNotFoundError:
        print(f"오류: 오디오 파일을 찾을 수 없습니다. ({audio_file_path})")
        return None
    except Exception as e:
        print(f"GCP Speech-to-Text API 호출 중 오류가 발생했습니다: {e}")
        return None

def get_gemini_response(user_input: str) -> str:
    """
    Gemini를 사용하여 사용자 입력에 대한 응답을 생성합니다.
    
    Args:
        user_input: 사용자 입력 텍스트
    
    Returns:
        str: Gemini 응답
    """
    try:
        # 사용자 입력에 대한 응답 생성
        response = llm.invoke(f"사용자가 말한 내용: '{user_input}'\n\n이에 대해 친근하고 도움이 되는 답변을 해주세요.")
        return response.content
    except Exception as e:
        print(f"Gemini 응답 생성 중 오류가 발생했습니다: {e}")
        return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MP4 STT + Gemini API 서비스",
        "version": "1.0.0",
        "endpoints": {
            "process_mp4": "/process-mp4",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "mp4-stt-gemini"}

@app.post("/process-mp4")
async def process_mp4(file: UploadFile = File(...)):
    """
    MP4 파일을 업로드받아 STT를 수행하고 Gemini 응답을 생성합니다.
    
    Args:
        file: 업로드된 MP4 파일
    
    Returns:
        JSONResponse: 처리 결과
    """
    try:
        # 파일 확장자 검증
        if not file.filename.lower().endswith('.mp4'):
            raise HTTPException(status_code=400, detail="MP4 파일만 업로드 가능합니다.")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # MP4 파일 저장
            mp4_path = temp_dir_path / "input.mp4"
            with open(mp4_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # 오디오 파일 경로
            audio_path = temp_dir_path / "extracted_audio.wav"
            
            # 1. MP4에서 오디오 추출
            print("MP4에서 오디오 추출 중...")
            if not extract_audio_from_mp4(str(mp4_path), str(audio_path)):
                raise HTTPException(status_code=500, detail="오디오 추출에 실패했습니다.")
            
            # 2. GCP STT로 텍스트 변환
            print("음성을 텍스트로 변환 중...")
            transcribed_text = transcribe_audio_with_gcp(str(audio_path))
            
            if not transcribed_text:
                raise HTTPException(status_code=400, detail="음성을 인식할 수 없습니다.")
            
            # 3. Gemini로 응답 생성
            print("Gemini가 응답을 생성 중...")
            gemini_response = get_gemini_response(transcribed_text)
            
            # 결과 반환
            return JSONResponse(content={
                "success": True,
                "transcribed_text": transcribed_text,
                "gemini_response": gemini_response,
                "file_info": {
                    "filename": file.filename,
                    "size": len(content)
                }
            })
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.post("/process-mp4-batch")
async def process_mp4_batch(files: list[UploadFile] = File(...)):
    """
    여러 MP4 파일을 일괄 처리합니다.
    
    Args:
        files: 업로드된 MP4 파일들
    
    Returns:
        JSONResponse: 일괄 처리 결과
    """
    results = []
    
    for file in files:
        try:
            # 파일 확장자 검증
            if not file.filename.lower().endswith('.mp4'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "MP4 파일만 업로드 가능합니다."
                })
                continue
            
            # 임시 디렉토리 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # MP4 파일 저장
                mp4_path = temp_dir_path / "input.mp4"
                with open(mp4_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # 오디오 파일 경로
                audio_path = temp_dir_path / "extracted_audio.wav"
                
                # 1. MP4에서 오디오 추출
                if not extract_audio_from_mp4(str(mp4_path), str(audio_path)):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "오디오 추출에 실패했습니다."
                    })
                    continue
                
                # 2. GCP STT로 텍스트 변환
                transcribed_text = transcribe_audio_with_gcp(str(audio_path))
                
                if not transcribed_text:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "음성을 인식할 수 없습니다."
                    })
                    continue
                
                # 3. Gemini로 응답 생성
                gemini_response = get_gemini_response(transcribed_text)
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "transcribed_text": transcribed_text,
                    "gemini_response": gemini_response,
                    "file_size": len(content)
                })
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "results": results,
        "total_files": len(files),
        "successful_files": len([r for r in results if r["success"]])
    })

if __name__ == "__main__":
    import uvicorn
    
    print("="*50)
    print("MP4 STT + Gemini API 서버 시작")
    print("="*50)
    
    # 서버 실행
    uvicorn.run(
        "mp4main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
