import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, List as TypeList
import operator
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from dotenv import load_dotenv
import sqlite3
import json
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# FastAPI 앱 생성
app = FastAPI(
    title="치매 검사 API",
    description="음성 파일과 대화 텍스트를 기반으로 치매 의심 여부를 검사하는 API",
    version="1.0.0"
)

# Pydantic 모델 정의
class DementiaCheckRequest(BaseModel):
    patient_id: str
    conversation_text: str

class DementiaCheckResponse(BaseModel):
    patient_id: str
    final_diagnosis: str
    diagnosis_summary: str
    check_results: List[Dict[str, Any]]
    confidence_score: float
    timestamp: str
    contents: str  # LLM 응답 내용 추가

# 1. 상태 정의
class DementiaCheckState(TypedDict):
    patient_id: str
    current_audio_file: str  # 현재 음성 파일 경로
    current_text: str  # 현재 대화 텍스트
    messages: Annotated[TypeList[BaseMessage], operator.add]
    check_results: Annotated[TypeList[dict], operator.add]  # 각 노드의 검사 결과
    final_diagnosis: str  # 최종 진단 결과 (yes/no)
    contents: str  # LLM의 응답 내용

# 2. 데이터베이스 초기화
def init_database():
    """치매 검사용 데이터베이스 초기화"""
    conn = sqlite3.connect('dementia_check.db')
    cursor = conn.cursor()
    
    # 대화 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            conversation_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 음성 임베딩 테이블 (정상 발음 기준)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file_path TEXT,
            embedding_data BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# 3. 대화 처리 노드 추가
def process_conversation(state: DementiaCheckState):
    """사용자 대화를 처리하고 LLM 응답을 생성하는 노드"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    # 대화 기반 치매 검사 프롬프트
    conversation_prompt = f"""
    당신은 치매 검사를 위한 대화형 AI 어시스턴트입니다.
    환자와의 대화를 통해 치매 의심 증상을 관찰하고 적절한 응답을 제공해야 합니다.
    
    환자 ID: {patient_id}
    환자 대화: "{current_text}"
    
    다음 지침을 따라 응답해주세요:
    1. 먼저 환자에게 자연스럽고 친근하게 응답하세요
    2. 대화 내용에서 치매 의심 증상이 있는지 관찰하세요:
       - 기억력 문제 (최근 일 기억 못함)
       - 반복적인 발화
       - 혼란스러운 표현
       - 시간/장소 인지 문제
    
    응답 형식:
    - 친근한 응답: [환자에게 할 말]
    - 치매 의심 여부: [yes/no]
    - 관찰 내용: [관찰한 증상이나 정상적인 대화인지 설명]
    """
    
    try:
        response = llm.invoke(conversation_prompt)
        
        # 응답 파싱
        response_text = response.content
        
        # 간단한 파싱 (실제로는 더 정교한 파싱이 필요)
        lines = response_text.split('\n')
        friendly_response = ""
        dementia_suspicion = "no"
        observation = ""
        
        for line in lines:
            if "친근한 응답:" in line:
                friendly_response = line.split("친근한 응답:")[1].strip()
            elif "치매 의심 여부:" in line:
                suspicion = line.split("치매 의심 여부:")[1].strip().lower()
                if "yes" in suspicion:
                    dementia_suspicion = "yes"
            elif "관찰 내용:" in line:
                observation = line.split("관찰 내용:")[1].strip()
        
        return {
            "contents": friendly_response,
            "final_diagnosis": dementia_suspicion,
            "check_results": [{
                "type": "conversation_analysis",
                "result": dementia_suspicion,
                "confidence": 0.8,
                "details": observation
            }]
        }
        
    except Exception as e:
        return {
            "contents": "죄송합니다. 잠시 오류가 발생했습니다.",
            "final_diagnosis": "no",
            "check_results": [{
                "type": "conversation_analysis",
                "result": "error",
                "confidence": 0.0,
                "details": f"대화 처리 오류: {str(e)}"
            }]
        }

# 4. 발음 어눌함 검사 노드 (대화 기반)
def check_pronunciation_clarity(state: DementiaCheckState):
    """발음이 어눌한지 검사하는 노드 (대화 기반)"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    try:
        # 대화 내용에서 발음 관련 문제 분석
        pronunciation_prompt = f"""
        다음 환자의 대화 내용에서 발음이나 언어 표현의 어눌함을 분석해주세요:
        
        대화: "{current_text}"
        
        분석 기준:
        - 문장이 중간에 끊어지는 경우
        - 단어를 찾지 못하는 경우
        - 발음이 어려운 단어를 피하는 경우
        - 문법적으로 어색한 표현
        
        응답 형식:
        - 발음 상태: [clear/unclear]
        - 신뢰도: [0.0-1.0]
        - 분석 내용: [구체적인 관찰 내용]
        """
        
        response = llm.invoke(pronunciation_prompt)
        response_text = response.content
        
        # 파싱
        lines = response_text.split('\n')
        pronunciation_status = "clear"
        confidence = 0.5
        details = ""
        
        for line in lines:
            if "발음 상태:" in line:
                status = line.split("발음 상태:")[1].strip().lower()
                if "unclear" in status:
                    pronunciation_status = "unclear"
            elif "신뢰도:" in line:
                try:
                    conf = line.split("신뢰도:")[1].strip()
                    confidence = float(conf)
                except:
                    pass
            elif "분석 내용:" in line:
                details = line.split("분석 내용:")[1].strip()
        
        return {
            "check_results": [{
                "type": "pronunciation_clarity",
                "result": pronunciation_status,
                "confidence": confidence,
                "details": details
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "pronunciation_clarity",
                "result": "error",
                "confidence": 0.0,
                "details": f"발음 분석 오류: {str(e)}"
            }]
        }

# 5. 기억력 검사 노드 (대화 기반)
def check_memory_recall(state: DementiaCheckState):
    """기억력 검사 노드 (대화 기반)"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    try:
        # 대화 내용에서 기억력 문제 분석
        memory_prompt = f"""
        다음 환자의 대화 내용에서 기억력 문제를 분석해주세요:
        
        대화: "{current_text}"
        
        분석 기준:
        - 최근 일에 대한 기억 부족
        - 시간 개념의 혼란
        - 반복적인 질문
        - 과거 일에 대한 막연한 표현
        
        응답 형식:
        - 기억력 상태: [normal/impaired]
        - 신뢰도: [0.0-1.0]
        - 분석 내용: [구체적인 관찰 내용]
        """
        
        response = llm.invoke(memory_prompt)
        response_text = response.content
        
        # 파싱
        lines = response_text.split('\n')
        memory_status = "normal"
        confidence = 0.5
        details = ""
        
        for line in lines:
            if "기억력 상태:" in line:
                status = line.split("기억력 상태:")[1].strip().lower()
                if "impaired" in status:
                    memory_status = "impaired"
            elif "신뢰도:" in line:
                try:
                    conf = line.split("신뢰도:")[1].strip()
                    confidence = float(conf)
                except:
                    pass
            elif "분석 내용:" in line:
                details = line.split("분석 내용:")[1].strip()
        
        return {
            "check_results": [{
                "type": "memory_recall",
                "result": memory_status,
                "confidence": confidence,
                "details": details
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "memory_recall",
                "result": "error",
                "confidence": 0.0,
                "details": f"기억력 검사 오류: {str(e)}"
            }]
        }

# 6. 반복 발화 검사 노드 (대화 기반)
def check_repetitive_speech(state: DementiaCheckState):
    """반복 발화 검사 노드 (대화 기반)"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    try:
        # 대화 내용에서 반복 패턴 분석
        repetition_prompt = f"""
        다음 환자의 대화 내용에서 반복적인 발화 패턴을 분석해주세요:
        
        대화: "{current_text}"
        
        분석 기준:
        - 같은 내용의 반복
        - 유사한 표현의 반복
        - 특정 주제에 대한 과도한 집착
        - 대화의 자연스러운 흐름 부족
        
        응답 형식:
        - 반복 상태: [normal/repetitive]
        - 신뢰도: [0.0-1.0]
        - 분석 내용: [구체적인 관찰 내용]
        """
        
        response = llm.invoke(repetition_prompt)
        response_text = response.content
        
        # 파싱
        lines = response_text.split('\n')
        repetition_status = "normal"
        confidence = 0.5
        details = ""
        
        for line in lines:
            if "반복 상태:" in line:
                status = line.split("반복 상태:")[1].strip().lower()
                if "repetitive" in status:
                    repetition_status = "repetitive"
            elif "신뢰도:" in line:
                try:
                    conf = line.split("신뢰도:")[1].strip()
                    confidence = float(conf)
                except:
                    pass
            elif "분석 내용:" in line:
                details = line.split("분석 내용:")[1].strip()
        
        return {
            "check_results": [{
                "type": "repetitive_speech",
                "result": repetition_status,
                "confidence": confidence,
                "details": details
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "repetitive_speech",
                "result": "error",
                "confidence": 0.0,
                "details": f"반복 발화 검사 오류: {str(e)}"
            }]
        }

# 7. Map 함수 수정
def map_to_checkers(state: DementiaCheckState):
    """각 검사 노드에 작업을 할당하는 함수"""
    return [
        Send("conversation_processor", {"patient_id": state["patient_id"], "current_text": state["current_text"]}),
        Send("pronunciation_checker", {"patient_id": state["patient_id"], "current_text": state["current_text"]}),
        Send("memory_checker", {"patient_id": state["patient_id"], "current_text": state["current_text"]}),
        Send("repetition_checker", {"patient_id": state["patient_id"], "current_text": state["current_text"]})
    ]

# 8. Combine 함수 수정
def combine_diagnosis(state: DementiaCheckState):
    """모든 검사 결과를 통합하여 최종 치매 진단"""
    check_results = state["check_results"]
    
    # 대화 처리 결과에서 contents와 초기 진단 가져오기
    conversation_result = None
    pronunciation_result = None
    memory_result = None
    repetition_result = None
    
    for result in check_results:
        if result["type"] == "conversation_analysis":
            conversation_result = result
        elif result["type"] == "pronunciation_clarity":
            pronunciation_result = result
        elif result["type"] == "memory_recall":
            memory_result = result
        elif result["type"] == "repetitive_speech":
            repetition_result = result
    
    # 대화 기반 진단이 우선
    final_diagnosis = "no"
    contents = "안녕하세요, 무엇을 도와드릴까요?"
    
    if conversation_result:
        # 대화 처리에서 나온 진단 결과 사용
        final_diagnosis = conversation_result.get("result", "no")
        # contents는 state에서 가져와야 함
        contents = state.get("contents", "안녕하세요, 무엇을 도와드릴까요?")
    
    # 추가 검사 결과로 진단 보정
    dementia_indicators = 0
    total_confidence = 0.0
    details = []
    
    if pronunciation_result and pronunciation_result["result"] == "unclear":
        dementia_indicators += 1
        details.append(f"발음 어눌함 (신뢰도: {pronunciation_result['confidence']:.3f})")
    total_confidence += float(pronunciation_result["confidence"]) if pronunciation_result else 0.0
    
    if memory_result and memory_result["result"] == "impaired":
        dementia_indicators += 1
        details.append(f"기억력 저하 (신뢰도: {memory_result['confidence']:.3f})")
    total_confidence += float(memory_result["confidence"]) if memory_result else 0.0
    
    if repetition_result and repetition_result["result"] == "repetitive":
        dementia_indicators += 1
        details.append(f"반복 발화 (신뢰도: {repetition_result['confidence']:.3f})")
    total_confidence += float(repetition_result["confidence"]) if repetition_result else 0.0
    
    # 대화 기반 진단과 검사 결과를 종합
    if dementia_indicators >= 2:
        final_diagnosis = "yes"
    
    avg_confidence = total_confidence / 3 if total_confidence > 0 else 0.0
    
    diagnosis_summary = f"""
    치매 검사 결과:
    - 대화 분석: {conversation_result['result'] if conversation_result else 'N/A'}
    - 발음 어눌함: {pronunciation_result['result'] if pronunciation_result else 'N/A'}
    - 기억력 저하: {memory_result['result'] if memory_result else 'N/A'}
    - 반복 발화: {repetition_result['result'] if repetition_result else 'N/A'}
    
    치매 의심 지표 수: {dementia_indicators}/3
    평균 신뢰도: {avg_confidence:.3f}
    
    최종 진단: {'치매 의심' if final_diagnosis == 'yes' else '정상'}
    """
    
    return {
        "final_diagnosis": final_diagnosis,
        "contents": contents,  # 대화 처리에서 나온 응답
        "messages": [("diagnosis", diagnosis_summary)]
    }

# 9. 그래프 구성 수정
def create_dementia_check_graph():
    """치매 검사 그래프 생성"""
    workflow = StateGraph(DementiaCheckState)
    
    # 대화 처리 노드 추가
    workflow.add_node("conversation_processor", process_conversation)
    
    # 각 검사 노드 추가
    workflow.add_node("pronunciation_checker", check_pronunciation_clarity)
    workflow.add_node("memory_checker", check_memory_recall)
    workflow.add_node("repetition_checker", check_repetitive_speech)
    
    # Combine 노드 추가
    workflow.add_node("combine", combine_diagnosis)
    
    # START에서 map_to_checkers로 가는 조건부 엣지 추가
    workflow.add_conditional_edges(
        START,
        map_to_checkers,
        {
            "conversation_processor": "conversation_processor",
            "pronunciation_checker": "pronunciation_checker",
            "memory_checker": "memory_checker",
            "repetition_checker": "repetition_checker"
        }
    )
    
    # 각 검사 노드의 결과를 combine으로 연결
    for checker in ["conversation_processor", "pronunciation_checker", "memory_checker", "repetition_checker"]:
        workflow.add_edge(checker, "combine")
    
    workflow.add_edge("combine", END)
    
    return workflow.compile()

# 10. 대화 기록 저장 함수
def save_conversation(patient_id: str, conversation_text: str):
    """대화 기록을 데이터베이스에 저장"""
    conn = sqlite3.connect('dementia_check.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversation_history (patient_id, conversation_text)
        VALUES (?, ?)
    ''', (patient_id, conversation_text))
    conn.commit()
    conn.close()

# 11. 메인 실행 함수 수정
def run_dementia_check(patient_id: str, audio_file_path: str, conversation_text: str):
    """치매 검사 실행"""
    # 데이터베이스 초기화
    init_database()
    
    # 대화 기록 저장
    save_conversation(patient_id, conversation_text)
    
    # 그래프 생성
    app = create_dementia_check_graph()
    
    # 입력 데이터 준비
    inputs = {
        "patient_id": patient_id,
        "current_audio_file": audio_file_path,
        "current_text": conversation_text,
        "messages": [],
        "check_results": [],
        "final_diagnosis": "no",  # default 값
        "contents": ""  # 초기 빈 값
    }
    
    # 그래프 실행
    config = {"recursion_limit": 50}
    result = app.invoke(inputs, config)
    
    return result

# API 엔드포인트 수정
@app.post("/check-dementia", response_model=DementiaCheckResponse)
async def check_dementia(
    patient_id: str = Form(...),
    conversation_text: str = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    치매 검사 API (대화형)
    
    - **patient_id**: 환자 ID
    - **conversation_text**: 대화 텍스트
    - **audio_file**: 음성 파일 (선택사항)
    """
    try:
        # 임시 파일 생성 (음성 파일이 없는 경우 더미 파일 생성)
        temp_audio_file = None
        
        if audio_file:
            # 실제 음성 파일 처리
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_audio_file.name, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            audio_file_path = temp_audio_file.name
        else:
            # 더미 음성 파일 생성 (테스트용)
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            # 간단한 더미 오디오 데이터 생성
            sample_rate = 22050
            duration = 1.0  # 1초
            t = np.linspace(0, duration, int(sample_rate * duration))
            dummy_audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz 사인파
            sf.write(temp_audio_file.name, dummy_audio, sample_rate)
            audio_file_path = temp_audio_file.name
        
        # 치매 검사 실행
        result = run_dementia_check(patient_id, audio_file_path, conversation_text)
        
        # 결과 정리
        check_results = result.get("check_results", [])
        final_diagnosis = result.get("final_diagnosis", "no")
        contents = result.get("contents", "안녕하세요, 무엇을 도와드릴까요?")
        
        # 신뢰도 계산
        total_confidence = sum(float(result.get("confidence", 0)) for result in check_results)
        avg_confidence = total_confidence / len(check_results) if check_results else 0.0
        
        # 진단 요약 생성
        diagnosis_summary = ""
        if result.get("messages"):
            for msg_type, msg_content in result["messages"]:
                if msg_type == "diagnosis":
                    diagnosis_summary = msg_content
                    break
        
        # 임시 파일 정리
        if temp_audio_file:
            os.unlink(temp_audio_file.name)
        
        return DementiaCheckResponse(
            patient_id=patient_id,
            final_diagnosis=final_diagnosis,
            diagnosis_summary=diagnosis_summary,
            check_results=check_results,
            confidence_score=avg_confidence,
            timestamp=datetime.now().isoformat(),
            contents=contents  # LLM 응답 내용 추가
        )
        
    except Exception as e:
        # 임시 파일 정리
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.unlink(temp_audio_file.name)
        
        raise HTTPException(status_code=500, detail=f"치매 검사 중 오류가 발생했습니다: {str(e)}")

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """API 상태 확인"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 루트 엔드포인트
@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "치매 검사 API",
        "version": "1.0.0",
        "endpoints": {
            "check_dementia": "/check-dementia",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)