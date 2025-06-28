import os
import json
import uuid
from typing import Literal, TypedDict, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import tempfile
import shutil
import numpy as np
import soundfile as sf
from datetime import datetime
import librosa
import pymysql
from contextlib import contextmanager

# FAISS 및 음성 분석 관련 import 추가
from transformers import ClapModel, ClapProcessor
import torchaudio
import torch
import torch.nn.functional as F
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings
from tqdm import tqdm

# 환경 변수 로드
load_dotenv()

# ==================== DB 연결 설정 ====================

DB_CONFIG = {
    'host': '14.6.152.212',
    'user': 'root',
    'password': 'qlqjstongue@74',
    'database': 'igarlicyou-dev',
    'port': 3306,
    'charset': 'utf8mb4',
    'autocommit': True
}

@contextmanager
def get_db_connection():
    """DB 연결을 관리하는 컨텍스트 매니저"""
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        yield connection
    except Exception as e:
        print(f"DB 연결 오류: {str(e)}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection:
            connection.close()

def create_chat_table():
    """채팅 테이블이 없으면 생성하는 함수"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_history_tb (
        id INT AUTO_INCREMENT PRIMARY KEY,
        chats TEXT NOT NULL,
        role ENUM('user', 'agent') NOT NULL,
        user_id VARCHAR(10) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                print("채팅 테이블 생성 완료 또는 이미 존재함")
    except Exception as e:
        print(f"테이블 생성 오류: {str(e)}")

def generate_user_id():
    """10자리 숫자 user_id 생성"""
    return str(uuid.uuid4().int)[:10]

def save_chat_to_db(user_id: str, message: str, role: str):
    """대화 내역을 DB에 저장하는 함수"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = "INSERT INTO chat_history_tb (chats, role, user_id) VALUES (%s, %s, %s)"
                cursor.execute(sql, (message, role, user_id))
                print(f"DB 저장 완료: user_id={user_id}, role={role}, message={message[:50]}...")
    except Exception as e:
        print(f"DB 저장 오류: {str(e)}")

# FastAPI 앱 초기화
app = FastAPI(
    title="통합 의료 서비스 API",
    description="치매 검사, 복약 알림, 기분 및 건강체크를 통합 관리하는 API 서버",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# CLAP 모델 및 FAISS 벡터스토어 전역 변수
clap_model = None
clap_processor = None
faiss_vectorstore = None

# 앱 시작 시 테이블 생성 및 모델 로드
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    create_chat_table()
    await load_audio_models()

async def load_audio_models():
    """CLAP 모델과 FAISS 벡터스토어를 로드하는 함수"""
    global clap_model, clap_processor, faiss_vectorstore
    
    try:
        print("Loading CLAP model and processor...")
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        
        # FAISS 벡터스토어 로드
        faiss_path = "./db/faiss"
        if os.path.exists(faiss_path):
            print("Loading FAISS vectorstore...")
            faiss_vectorstore = load_faiss_vectorstore(faiss_path, clap_model, clap_processor)
            if faiss_vectorstore:
                print(f"FAISS vectorstore loaded with {faiss_vectorstore.index.ntotal} vectors")
            else:
                print("Failed to load FAISS vectorstore")
        else:
            print("FAISS vectorstore not found at", faiss_path)
            
    except Exception as e:
        print(f"Error loading audio models: {e}")

# ==================== Pydantic 모델 정의 ====================

# Supervisor 관련 모델
class SupervisorRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    patient_id: Optional[str] = Field(None, description="환자 ID (치매 검사용)")

class SupervisorResponse(BaseModel):
    service_type: str = Field(..., description="서비스 타입 (dementia/medicine/mood_health)")
    selected_agent: str = Field(..., description="선택된 Agent")
    state: dict = Field(..., description="전체 State 정보")
    response: dict = Field(..., description="주요 응답 내용")
    next_action: str = Field(..., description="다음 액션")
    user_id: str = Field(..., description="사용자 ID")

# 복약 관련 모델
class MedicineInfo(BaseModel):
    """복약 정보를 추출하는 모델"""
    time: str = Field(description="복약 시간을 24시간 형식으로 변환 (예: 08:00, 20:30)")
    medicine_name: str = Field(description="약물 이름")
    dosage: str = Field(description="복용량 (예: 2개, 1정, 10ml)")
    needs_reminder: bool = Field(description="복약알림이 필요한지 여부")

# 기분 및 건강 관련 모델
class MoodHealthInfo(BaseModel):
    """기분 및 건강 정보를 추출하는 모델"""
    mood: Literal["좋음", "나쁨", "평범"] = Field(description="기분 상태 (반드시 3개 중 하나)")
    health_status: str = Field(description="건강 상태에 대한 자유로운 텍스트 설명")
    confidence: float = Field(description="분석 신뢰도 (0.0-1.0)")

# ==================== State 및 Router 정의 ====================

class State(MessagesState):
    next: str = "supervisor"
    service_type: Optional[str] = None
    patient_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    selected_agent: Optional[str] = None
    executed_agents: list = []  # 추가: 실행된 Agent 목록
    dementia_result: Optional[dict] = None
    medicine_result: Optional[dict] = None
    mood_result: Optional[dict] = None
    health_result: Optional[dict] = None
    stroke_result: Optional[dict] = None  # 추가: 뇌졸중 검사 결과
    user_id: Optional[str] = None  # 추가: 사용자 ID

members = ["dementia_agent", "medicine_agent", "mood_health_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

# ==================== FAISS 음성 분석 함수들 ====================

class AudioEmbeddings(Embeddings):
    """오디오 임베딩을 위한 커스텀 클래스"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def embed_documents(self, texts):
        """문서 임베딩 (FAISS 호환성을 위해 구현)"""
        embeddings = []
        for text in tqdm(texts, desc="Embedding audio files", unit="file"):
            try:
                embed = load_and_embed_audio(text, self.model, self.processor)
                if embed is not None:
                    embed_np = embed.squeeze().cpu().numpy()
                    embeddings.append(embed_np)
                else:
                    embeddings.append(np.zeros(512))
            except Exception as e:
                print(f"Error embedding {text}: {e}")
                embeddings.append(np.zeros(512))
        return embeddings
    
    def embed_query(self, text):
        """쿼리 임베딩"""
        try:
            embed = load_and_embed_audio(text, self.model, self.processor)
            if embed is not None:
                embed_np = embed.squeeze().cpu().numpy()
                return embed_np
            else:
                return np.zeros(512)
        except Exception as e:
            print(f"Error embedding query {text}: {e}")
            return np.zeros(512)

def load_and_embed_audio(file_path, model, processor):
    """WAV 파일을 로드하고 임베딩을 추출하는 함수"""
    try:
        # WAV 파일 로드
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        
        # 모노로 변환 (스테레오인 경우)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # CLAP 모델이 요구하는 48kHz로 리샘플링
        target_sample_rate = 48000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # 오디오 배열을 numpy로 변환
        audio_array = waveform.squeeze().numpy()
        
        # 프로세서를 사용하여 입력 준비
        inputs = processor(audios=audio_array, sampling_rate=sample_rate, return_tensors="pt")
        
        # 오디오 임베딩 추출
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
            # 임베딩 벡터 정규화 (L2 norm)
            audio_embed = F.normalize(audio_embed, p=2, dim=1)
        
        return audio_embed
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_faiss_vectorstore(save_path="./db/faiss", model=None, processor=None):
    """로컬에서 FAISS 벡터스토어 로드"""
    try:
        if not os.path.exists(save_path):
            print(f"Vectorstore path {save_path} does not exist")
            return None
        
        if model is None or processor is None:
            print("Model and processor are required to load vectorstore")
            return None
        
        audio_embeddings = AudioEmbeddings(model, processor)
        vectorstore = FAISS.load_local(save_path, audio_embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {save_path}")
        return vectorstore
        
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def check_stroke_with_faiss(audio_file_path: str):
    """FAISS 벡터DB를 사용한 뇌졸중 검사 함수"""
    global faiss_vectorstore, clap_model, clap_processor
    
    if faiss_vectorstore is None or clap_model is None or clap_processor is None:
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": "FAISS 벡터스토어 또는 CLAP 모델이 로드되지 않음",
            "similar_count": 0
        }
    
    if not os.path.exists(audio_file_path):
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": "음성 파일이 존재하지 않음",
            "similar_count": 0
        }
    
    try:
        print(f"Checking stroke with FAISS for {audio_file_path}")
        
        # 쿼리 임베딩 생성
        query_embedding = faiss_vectorstore.embedding_function.embed_query(audio_file_path)
        
        # FAISS 인덱스에서 모든 결과 검색
        query_embedding_reshaped = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_vectorstore.index.search(query_embedding_reshaped, faiss_vectorstore.index.ntotal)
        
        # 유사도 점수 계산 (코사인 유사도)
        similarities = 1 - distances[0]
        
        # 0.5 이상 유사한 음성 개수 계산
        similar_count = sum(1 for sim in similarities if sim >= 0.5)
        
        # 뇌졸중 판단: 100개 이상이면 뇌졸중 의심
        stroke_suspicion = similar_count >= 100
        
        # 신뢰도 계산 (유사한 음성 비율 기반)
        confidence = min(similar_count / 100.0, 1.0) if stroke_suspicion else 0.0
        
        result = {
            "type": "stroke_check",
            "result": "stroke_suspicion" if stroke_suspicion else "normal",
            "confidence": float(confidence),
            "details": f"유사한 음성 {similar_count}개 발견 (임계값: 100개)",
            "similar_count": similar_count,
            "total_vectors": len(similarities),
            "similarity_scores": similarities.tolist()[:10]  # 상위 10개 점수만
        }
        
        print(f"Stroke check result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in stroke check: {e}")
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": f"뇌졸중 검사 오류: {str(e)}",
            "similar_count": 0
        }

# ==================== 치매 검사 함수들 ====================

def process_conversation(patient_id: str, conversation_text: str):
    """사용자 대화를 처리하고 LLM 응답을 생성하는 함수"""
    conversation_prompt = f"""
    당신은 치매 검사를 위한 대화형 AI 어시스턴트입니다.
    환자와의 대화를 통해 치매 의심 증상을 관찰하고 적절한 응답을 제공해야 합니다.
    
    환자 ID: {patient_id}
    환자 대화: "{conversation_text}"
    
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
        response_text = response.content
        
        # 간단한 파싱
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

def check_repetitive_speech(conversation_text: str):
    """반복 발화 검사 함수 - 치매 판단용"""
    repetition_prompt = f"""
    다음 환자의 대화 내용에서 반복적인 발화 패턴을 분석해주세요:
    
    대화: "{conversation_text}"
    
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
    
    try:
        response = llm.invoke(repetition_prompt)
        response_text = response.content
        
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
            "type": "repetitive_speech",
            "result": repetition_status,
            "confidence": confidence,
            "details": details
        }
        
    except Exception as e:
        return {
            "type": "repetitive_speech",
            "result": "error",
            "confidence": 0.0,
            "details": f"반복 발화 검사 오류: {str(e)}"
        }

def check_memory_recall(conversation_text: str):
    """기억력 검사 함수 - 치매 판단용"""
    memory_prompt = f"""
    다음 환자의 대화 내용에서 기억력 문제를 분석해주세요:
    
    대화: "{conversation_text}"
    
    분석 기준:
    - 최근 일에 대한 기억 부족
    - 시간 개념의 혼란
    - 반복적인 질문
    - 과거 일에 대한 막연한 표현
    - 이전 대화 내용 참조 능력
    
    응답 형식:
    - 기억력 상태: [normal/impaired]
    - 신뢰도: [0.0-1.0]
    - 분석 내용: [구체적인 관찰 내용]
    """
    
    try:
        response = llm.invoke(memory_prompt)
        response_text = response.content
        
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
            "type": "memory_recall",
            "result": memory_status,
            "confidence": confidence,
            "details": details
        }
        
    except Exception as e:
        return {
            "type": "memory_recall",
            "result": "error",
            "confidence": 0.0,
            "details": f"기억력 검사 오류: {str(e)}"
        }

# ==================== 복약 분석 함수들 ====================

def analyze_medicine(message: str):
    """복약 정보를 분석하는 함수"""
    analysis_msgs = [
        ("system", """
        당신은 복약 관련 메시지를 분석하여 복약 정보를 추출하는 전문가입니다.
        사용자의 메시지에서 다음 정보를 정확히 추출해주세요:

        1. 복약 시간: 시간을 24시간 형식으로 변환 (예: 08:00, 20:30)
        2. 약물 이름: 복용해야 할 약의 이름
        3. 복용량: 몇 개, 몇 정, 몇 ml 등
        4. 복약알림 필요 여부: 복용 스케줄이 있으면 True, 단순 정보 제공이면 False

        예시:
        - "아침 8시마다 관절염 약 2개씩을 먹어야 해" → time: "08:00", medicine_name: "관절염 약", dosage: "2개", needs_reminder: True
        - "저녁 8시에 혈압약 1정" → time: "20:00", medicine_name: "혈압약", dosage: "1정", needs_reminder: True
        - "아스피린에 대해 알고 싶어" → time: "", medicine_name: "아스피린", dosage: "", needs_reminder: False
        """),
        ("user", f"사용자 메시지: {message}\n복약 정보를 추출해주세요:"),
    ]
    analysis_prompt = ChatPromptTemplate.from_messages(analysis_msgs)
    
    model_with_structured_output = llm.with_structured_output(MedicineInfo)
    
    response = model_with_structured_output.invoke(
        analysis_prompt.format_messages(
            messages=message
        )
    )
    
    return response

def generate_medicine_response(medicine_info: MedicineInfo):
    """복약 응답을 생성하는 함수"""
    time = medicine_info.time
    medicine_name = medicine_info.medicine_name
    dosage = medicine_info.dosage
    needs_reminder = medicine_info.needs_reminder
    
    # 24시간 형식을 12시간 형식으로 변환
    formatted_time = ""
    if time:
        try:
            hour, minute = map(int, time.split(':'))
            if hour == 0:
                formatted_time = f"12:{minute:02d} am"
            elif hour < 12:
                formatted_time = f"{hour}:{minute:02d} am"
            elif hour == 12:
                formatted_time = f"12:{minute:02d} pm"
            else:
                formatted_time = f"{hour-12}:{minute:02d} pm"
        except:
            formatted_time = time
    
    return {
        "time": formatted_time,
        "medicine_name": medicine_name,
        "dosage": dosage,
        "needs_reminder": needs_reminder
    }

# ==================== 기분 및 건강체크 함수들 ====================

def analyze_mood_health(message: str):
    """기분 및 건강 상태를 분석하는 함수"""
    analysis_msgs = [
        ("system", """
        당신은 환자의 기분과 건강 상태를 분석하는 전문가입니다.
        사용자의 메시지에서 다음 정보를 정확히 추출해주세요:

        1. 기분 상태: 반드시 "좋음", "나쁨", "평범" 중 하나로 판별
           - "좋음": 긍정적이고 기분이 좋은 상태
           - "나쁨": 부정적이고 기분이 나쁜 상태  
           - "평범": 특별히 좋지도 나쁘지도 않은 보통 상태

        2. 건강 상태: 전반적인 건강 상태를 자유로운 텍스트로 설명
           - 신체적 증상, 컨디션, 불편함 등을 포함
           - 구체적이고 이해하기 쉽게 작성

        3. 신뢰도: 분석의 확신 정도 (0.0-1.0)

        예시:
        - "오늘 기분이 정말 좋아요" → mood: "좋음", health_status: "전반적으로 건강하고 기분이 좋은 상태"
        - "몸이 좀 아파요" → mood: "나쁨", health_status: "신체적 불편함이 있어 건강 상태가 좋지 않음"
        - "컨디션이 평범해요" → mood: "평범", health_status: "특별한 문제없이 평상시와 같은 건강 상태"
        """),
        ("user", f"사용자 메시지: {message}\n기분과 건강 상태를 분석해주세요:"),
    ]
    analysis_prompt = ChatPromptTemplate.from_messages(analysis_msgs)
    
    model_with_structured_output = llm.with_structured_output(MoodHealthInfo)
    
    response = model_with_structured_output.invoke(
        analysis_prompt.format_messages(
            messages=message
        )
    )
    
    return response

def generate_mood_health_response(mood_health_info: MoodHealthInfo):
    """기분 및 건강 응답을 생성하는 함수"""
    mood = mood_health_info.mood
    health_status = mood_health_info.health_status
    confidence = mood_health_info.confidence
    
    # 기분에 따른 응답 메시지 생성
    mood_responses = {
        "좋음": "기분이 좋으시다니 다행이에요! 좋은 기분을 유지하세요.",
        "나쁨": "기분이 좋지 않으시군요. 무리하지 마시고 충분히 휴식을 취하세요.",
        "평범": "평범한 기분이시군요. 괜찮으시면 좋겠어요."
    }
    
    response_message = mood_responses.get(mood, "기분 상태를 확인했습니다.")
    
    return {
        "mood": mood,
        "health_status": health_status,
        "confidence": confidence,
        "response_message": response_message
    }

# ==================== LangGraph 노드들 ====================

# Supervisor 노드 - 단순한 키워드 기반 라우팅 - 1개 Agent만 실행 후 종료
def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    """단순한 키워드 기반 라우팅 - 1개 Agent만 실행 후 종료"""
    message = state["messages"][-1].content.lower()
    
    print(f"\n=== SUPERVISOR NODE ===")
    print(f"입력 메시지: {message}")
    
    # 사용자 입력을 DB에 저장
    user_id = state.get("user_id")
    if user_id:
        save_chat_to_db(user_id, state["messages"][-1].content, "user")
    
    # 복약 관련 키워드 체크 (최우선)
    medicine_keywords = [
        "약", "복용", "복약", "정", "개", "ml", "시간", 
        "아침", "저녁", "점심", "8시", "9시", "10시", "11시", "12시",
        "1시", "2시", "3시", "4시", "5시", "6시", "7시"
    ]
    if any(keyword in message for keyword in medicine_keywords):
        matched_keywords = [kw for kw in medicine_keywords if kw in message]
        print(f"복약 키워드 매칭: {matched_keywords}")
        print(f"라우팅: medicine_agent")
        return Command(
            goto="medicine_agent", 
            update={
                "selected_agent": "medicine_agent",
                "service_type": "medicine"
            }
        )
    
    # 기분/건강 관련 키워드 체크
    mood_keywords = ["기분", "좋음", "나쁨", "평범", "건강", "컨디션", "몸"]
    if any(keyword in message for keyword in mood_keywords):
        matched_keywords = [kw for kw in mood_keywords if kw in message]
        print(f"기분/건강 키워드 매칭: {matched_keywords}")
        print(f"라우팅: mood_health_agent")
        return Command(
            goto="mood_health_agent", 
            update={
                "selected_agent": "mood_health_agent",
                "service_type": "mood_health"
            }
        )
    
    # 치매 관련 키워드 체크 (기본값)
    dementia_keywords = ["기억", "잊어", "날짜", "시간", "혼란", "반복"]
    if any(keyword in message for keyword in dementia_keywords):
        matched_keywords = [kw for kw in dementia_keywords if kw in message]
        print(f"치매 키워드 매칭: {matched_keywords}")
        print(f"라우팅: dementia_agent")
        return Command(
            goto="dementia_agent", 
            update={
                "selected_agent": "dementia_agent",
                "service_type": "dementia"
            }
        )
    
    # 기본값: 치매 검사
    print(f"기본 라우팅: dementia_agent")
    return Command(
        goto="dementia_agent", 
        update={
            "selected_agent": "dementia_agent",
            "service_type": "dementia"
        }
    )

# 치매 검사 Agent 노드 - 실행 후 바로 종료
def dementia_node(state: State) -> Command[Literal["__end__"]]:
    """치매 검사 서비스 - 실행 후 바로 종료"""
    print(f"\n=== DEMENTIA NODE 실행 ===")
    print(f"State 정보: selected_agent={state.get('selected_agent')}, service_type={state.get('service_type')}")
    
    try:
        patient_id = state.get("patient_id", "patient_001")
        conversation_text = state["messages"][-1].content
        audio_file_path = state.get("audio_file_path")
        user_id = state.get("user_id")
        
        print(f"환자 ID: {patient_id}")
        print(f"대화 내용: {conversation_text}")
        print(f"음성 파일: {audio_file_path}")
        
        # 대화 처리
        conversation_result = process_conversation(patient_id, conversation_text)
        
        # 뇌졸중 검사 (음성 파일이 있는 경우)
        stroke_result = None
        if audio_file_path and os.path.exists(audio_file_path):
            stroke_result = check_stroke_with_faiss(audio_file_path)
            print(f"뇌졸중 검사 결과: {stroke_result}")
        
        # 치매 검사: 반복 발화 + 기억력 검사
        repetition_result = check_repetitive_speech(conversation_text)
        memory_result = check_memory_recall(conversation_text)
        
        # 결과 통합
        check_results = [
            conversation_result["check_results"][0],
            repetition_result,
            memory_result
        ]
        
        # 뇌졸중 검사 결과 추가
        if stroke_result:
            check_results.append(stroke_result)
        
        # 치매 판단: 반복 발화 + 기억력 검사 둘 다 0.5점 이상이면 치매 의심
        repetition_score = float(repetition_result.get("confidence", 0)) if repetition_result.get("result") == "repetitive" else 0.0
        memory_score = float(memory_result.get("confidence", 0)) if memory_result.get("result") == "impaired" else 0.0
        
        dementia_indicators = 0
        if repetition_score >= 0.5:
            dementia_indicators += 1
        if memory_score >= 0.5:
            dementia_indicators += 1
        
        final_diagnosis = "yes" if dementia_indicators >= 2 else "no"
        avg_confidence = float((repetition_score + memory_score) / 2) if (repetition_score + memory_score) > 0 else 0.0
        
        print(f"치매 의심 지표: {dementia_indicators}/2")
        print(f"반복 발화 점수: {repetition_score:.3f}")
        print(f"기억력 점수: {memory_score:.3f}")
        print(f"평균 신뢰도: {avg_confidence:.3f}")
        print(f"최종 진단: {final_diagnosis}")
        
        dementia_result = {
            "diagnosis": final_diagnosis,
            "confidence": avg_confidence,
            "contents": conversation_result["contents"],
            "check_results": check_results,
            "summary": f"치매 의심 지표 수: {dementia_indicators}/2, 평균 신뢰도: {avg_confidence:.3f}",
            "repetition_score": repetition_score,
            "memory_score": memory_score,
            "audio_used": audio_file_path is not None
        }
        
        # Agent 응답을 DB에 저장
        if user_id:
            agent_response = f"치매 검사 완료: {final_diagnosis} (신뢰도: {avg_confidence:.3f})"
            save_chat_to_db(user_id, agent_response, "agent")
        
        print(f"치매 검사 완료 - END로 종료")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"치매 검사 완료: {final_diagnosis} (신뢰도: {avg_confidence:.3f})",
                        name="dementia_agent"
                    )
                ],
                "dementia_result": dementia_result,
                "stroke_result": stroke_result  # 뇌졸중 검사 결과 추가
            },
            goto="__end__"  # 실행 후 바로 종료
        )
        
    except Exception as e:
        print(f"치매 검사 오류: {str(e)}")
        error_result = {
            "error": f"치매 검사 중 오류: {str(e)}",
            "diagnosis": "no",
            "confidence": 0.0
        }
        
        # 오류 응답도 DB에 저장
        user_id = state.get("user_id")
        if user_id:
            error_response = f"치매 검사 오류: {str(e)}"
            save_chat_to_db(user_id, error_response, "agent")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"치매 검사 오류: {str(e)}",
                        name="dementia_agent"
                    )
                ],
                "dementia_result": error_result
            },
            goto="__end__"  # 오류가 있어도 실행 후 바로 종료
        )

# 복약 알림 Agent 노드 - 실행 후 바로 종료
def medicine_node(state: State) -> Command[Literal["__end__"]]:
    """복약 알림 서비스 - 실행 후 바로 종료"""
    print(f"\n=== MEDICINE NODE 실행 ===")
    print(f"State 정보: selected_agent={state.get('selected_agent')}, service_type={state.get('service_type')}")
    
    try:
        message = state["messages"][-1].content
        user_id = state.get("user_id")
        print(f"입력 메시지: {message}")
        
        # 복약 정보 분석
        medicine_info = analyze_medicine(message)
        print(f"복약 정보 분석 결과: {medicine_info}")
        
        # 응답 생성
        medicine_result = generate_medicine_response(medicine_info)
        print(f"복약 응답 생성: {medicine_result}")
        
        # Agent 응답을 DB에 저장
        if user_id:
            agent_response = f"복약 정보 추출 완료: {medicine_result['medicine_name']} {medicine_result['dosage']} - {medicine_result['time']}"
            save_chat_to_db(user_id, agent_response, "agent")
        
        print(f"복약 분석 완료 - END로 종료")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"복약 정보 추출 완료: {medicine_result['medicine_name']} {medicine_result['dosage']} - {medicine_result['time']}",
                        name="medicine_agent"
                    )
                ],
                "medicine_result": medicine_result
            },
            goto="__end__"  # 실행 후 바로 종료
        )
        
    except Exception as e:
        print(f"복약 분석 오류: {str(e)}")
        error_result = {
            "error": f"복약 분석 중 오류: {str(e)}",
            "time": "",
            "medicine_name": "",
            "dosage": "",
            "needs_reminder": False
        }
        
        # 오류 응답도 DB에 저장
        user_id = state.get("user_id")
        if user_id:
            error_response = f"복약 분석 오류: {str(e)}"
            save_chat_to_db(user_id, error_response, "agent")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"복약 분석 오류: {str(e)}",
                        name="medicine_agent"
                    )
                ],
                "medicine_result": error_result
            },
            goto="__end__"  # 오류가 있어도 실행 후 바로 종료
        )

# 기분 및 건강체크 Agent 노드 - 실행 후 바로 종료
def mood_health_node(state: State) -> Command[Literal["__end__"]]:
    """기분 및 건강체크 서비스 - 실행 후 바로 종료"""
    print(f"\n=== MOOD_HEALTH NODE 실행 ===")
    print(f"State 정보: selected_agent={state.get('selected_agent')}, service_type={state.get('service_type')}")
    
    try:
        message = state["messages"][-1].content
        user_id = state.get("user_id")
        print(f"입력 메시지: {message}")
        
        # 기분 및 건강 상태 분석
        mood_health_info = analyze_mood_health(message)
        print(f"기분/건강 분석 결과: {mood_health_info}")
        
        # 응답 생성
        mood_health_result = generate_mood_health_response(mood_health_info)
        print(f"기분/건강 응답 생성: {mood_health_result}")
        
        # Agent 응답을 DB에 저장
        if user_id:
            agent_response = f"기분 및 건강체크 완료: {mood_health_result['mood']} - {mood_health_result['health_status']}"
            save_chat_to_db(user_id, agent_response, "agent")
        
        print(f"기분/건강체크 완료 - END로 종료")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"기분 및 건강체크 완료: {mood_health_result['mood']} - {mood_health_result['health_status']}",
                        name="mood_health_agent"
                    )
                ],
                "mood_result": {
                    "mood": mood_health_result["mood"],
                    "confidence": mood_health_result["confidence"]
                },
                "health_result": {
                    "status": mood_health_result["health_status"],
                    "confidence": mood_health_result["confidence"]
                }
            },
            goto="__end__"  # 실행 후 바로 종료
        )
        
    except Exception as e:
        print(f"기분/건강체크 오류: {str(e)}")
        error_result = {
            "error": f"기분 및 건강체크 분석 중 오류: {str(e)}",
            "mood": "평범",
            "health_status": "분석 오류로 인해 확인할 수 없음"
        }
        
        # 오류 응답도 DB에 저장
        user_id = state.get("user_id")
        if user_id:
            error_response = f"기분 및 건강체크 오류: {str(e)}"
            save_chat_to_db(user_id, error_response, "agent")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"기분 및 건강체크 오류: {str(e)}",
                        name="mood_health_agent"
                    )
                ],
                "mood_result": {"mood": "평범", "confidence": 0.0},
                "health_result": {"status": "분석 오류", "confidence": 0.0}
            },
            goto="__end__"  # 오류가 있어도 실행 후 바로 종료
        )

# ==================== LangGraph 그래프 구성 ====================

def create_supervisor_graph():
    """Supervisor 그래프 생성 - 단순한 구조"""
    graph_builder = StateGraph(State)
    
    # 노드 추가
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("dementia_agent", dementia_node)
    graph_builder.add_node("medicine_agent", medicine_node)
    graph_builder.add_node("mood_health_agent", mood_health_node)
    
    # 엣지 추가 - 단순한 구조
    graph_builder.add_edge("supervisor", "dementia_agent")
    graph_builder.add_edge("supervisor", "medicine_agent")
    graph_builder.add_edge("supervisor", "mood_health_agent")
    
    # 각 Agent에서 END로
    graph_builder.add_edge("dementia_agent", "__end__")
    graph_builder.add_edge("medicine_agent", "__end__")
    graph_builder.add_edge("mood_health_agent", "__end__")
    
    # 시작점 설정
    graph_builder.set_entry_point("supervisor")
    
    return graph_builder.compile()

# 그래프 인스턴스 생성
supervisor_graph = create_supervisor_graph()

# ==================== FastAPI 엔드포인트들 ====================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "통합 의료 서비스 API",
        "version": "2.0.0",
        "description": "치매 검사, 복약 알림, 기분 및 건강체크를 통합 관리하는 API 서버",
        "endpoints": {
            "supervisor": "/supervisor (multipart/form-data with audio)",
            "supervisor_json": "/supervisor-json (JSON format)",
            "health": "/health"
        },
        "services": {
            "dementia_check": "치매 검사 서비스 (음성 파일 지원)",
            "medicine_reminder": "복약 알림 서비스",
            "mood_health_check": "기분 및 건강체크 서비스"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy", 
        "service": "통합 의료 서비스 API",
        "version": "2.0.0",
        "available_services": [
            "dementia_check", 
            "medicine_reminder", 
            "mood_health_check"
        ]
    }

# ==================== Supervisor Agent 엔드포인트 ====================

@app.post("/supervisor", response_model=SupervisorResponse)
async def supervisor_endpoint(
    message: str = Form(...),
    patient_id: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Supervisor Agent 엔드포인트 (음성 파일 지원)
    
    사용자 메시지를 분석하여 적절한 의료 서비스로 라우팅합니다.
    """
    print(f"\n=== SUPERVISOR ENDPOINT 호출 ===")
    print(f"메시지: {message}")
    print(f"환자 ID: {patient_id}")
    print(f"음성 파일: {audio_file.filename if audio_file else 'None'}")
    
    try:
        # 사용자 ID 생성
        user_id = generate_user_id()
        print(f"생성된 사용자 ID: {user_id}")
        
        config = {"configurable": {"thread_id": f"supervisor_{patient_id or 'default'}"}}
        
        # 음성 파일 처리
        audio_file_path = None
        temp_file = None
        
        if audio_file:
            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_file.name, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            audio_file_path = temp_file.name
            print(f"음성 파일 저장됨: {audio_file_path}")
        
        # 입력 메시지 준비
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "patient_id": patient_id,
            "audio_file_path": audio_file_path,
            "user_id": user_id  # 사용자 ID 추가
        }
        
        print(f"그래프 입력 데이터: {input_data}")
        
        # 그래프 실행
        response = await supervisor_graph.ainvoke(input_data, config=config)
        
        print(f"그래프 실행 결과: {response}")
        
        # 결과 분석 - State 전체를 응답으로 사용
        selected_agent = response.get("selected_agent", "unknown")
        service_type = response.get("service_type", "unknown")
        
        print(f"선택된 Agent: {selected_agent}")
        print(f"서비스 타입: {service_type}")
        
        # State를 dict로 변환 (JSON 직렬화 가능하도록)
        state_dict = {
            "selected_agent": response.get("selected_agent"),
            "service_type": response.get("service_type"),
            "patient_id": response.get("patient_id"),
            "audio_file_path": response.get("audio_file_path"),
            "dementia_result": response.get("dementia_result"),
            "medicine_result": response.get("medicine_result"),
            "mood_result": response.get("mood_result"),
            "health_result": response.get("health_result"),
            "stroke_result": response.get("stroke_result"),  # 뇌졸중 검사 결과 추가
            "messages": [msg.dict() if hasattr(msg, 'dict') else msg for msg in response.get("messages", [])]
        }
        
        # 주요 응답 데이터 추출
        if selected_agent == "dementia_agent":
            result_data = response.get("dementia_result", {})
            next_action = "치매 검사 완료 - 추가 상담이 필요할 수 있습니다."
        elif selected_agent == "medicine_agent":
            result_data = response.get("medicine_result", {})
            next_action = "복약 정보 추출 완료 - 알림 설정이 필요할 수 있습니다."
        elif selected_agent == "mood_health_agent":
            result_data = {
                "mood": response.get("mood_result", {}),
                "health": response.get("health_result", {})
            }
            next_action = "기분 및 건강체크 완료 - 필요시 추가 상담을 권장합니다."
        else:
            result_data = {"message": "서비스 타입을 결정할 수 없습니다."}
            next_action = "수동 확인이 필요합니다."
        
        print(f"최종 응답 데이터: {result_data}")
        
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
            print(f"임시 파일 삭제됨: {temp_file.name}")
        
        return SupervisorResponse(
            service_type=service_type,
            selected_agent=selected_agent,
            state=state_dict,
            response=result_data,
            next_action=next_action,
            user_id=user_id  # 사용자 ID 추가
        )
        
    except Exception as e:
        print(f"Supervisor 엔드포인트 오류: {str(e)}")
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        raise HTTPException(status_code=500, detail=f"Supervisor 처리 중 오류가 발생했습니다: {str(e)}")

# JSON 형식도 지원하는 엔드포인트 추가
@app.post("/supervisor-json", response_model=SupervisorResponse)
async def supervisor_json_endpoint(request: SupervisorRequest):
    """
    Supervisor Agent 엔드포인트 (JSON 형식)
    
    기존 JSON 형식 호환성을 위한 엔드포인트
    """
    try:
        # 사용자 ID 생성
        user_id = generate_user_id()
        print(f"생성된 사용자 ID: {user_id}")
        
        config = {"configurable": {"thread_id": f"supervisor_{request.patient_id or 'default'}"}}
        
        # 입력 메시지 준비
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
            "patient_id": request.patient_id,
            "audio_file_path": None,  # JSON 형식에서는 음성 파일 없음
            "user_id": user_id  # 사용자 ID 추가
        }
        
        # 그래프 실행
        response = await supervisor_graph.ainvoke(input_data, config=config)
        
        # 결과 분석
        selected_agent = response.get("selected_agent", "unknown")
        service_type = response.get("service_type", "unknown")
        
        # State를 dict로 변환
        state_dict = {
            "selected_agent": response.get("selected_agent"),
            "service_type": response.get("service_type"),
            "patient_id": response.get("patient_id"),
            "audio_file_path": response.get("audio_file_path"),
            "dementia_result": response.get("dementia_result"),
            "medicine_result": response.get("medicine_result"),
            "mood_result": response.get("mood_result"),
            "health_result": response.get("health_result"),
            "stroke_result": response.get("stroke_result"),  # 뇌졸중 검사 결과 추가
            "messages": [msg.dict() if hasattr(msg, 'dict') else msg for msg in response.get("messages", [])]
        }
        
        # 주요 응답 데이터 추출
        if selected_agent == "dementia_agent":
            result_data = response.get("dementia_result", {})
            next_action = "치매 검사 완료 - 추가 상담이 필요할 수 있습니다."
        elif selected_agent == "medicine_agent":
            result_data = response.get("medicine_result", {})
            next_action = "복약 정보 추출 완료 - 알림 설정이 필요할 수 있습니다."
        elif selected_agent == "mood_health_agent":
            result_data = {
                "mood": response.get("mood_result", {}),
                "health": response.get("health_result", {})
            }
            next_action = "기분 및 건강체크 완료 - 필요시 추가 상담을 권장합니다."
        else:
            result_data = {"message": "서비스 타입을 결정할 수 없습니다."}
            next_action = "수동 확인이 필요합니다."
        
        return SupervisorResponse(
            service_type=service_type,
            selected_agent=selected_agent,
            state=state_dict,
            response=result_data,
            next_action=next_action,
            user_id=user_id  # 사용자 ID 추가
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supervisor 처리 중 오류가 발생했습니다: {str(e)}")

# ==================== 테스트용 동기 실행 함수 ====================

def run_supervisor_sync(message: str, patient_id: str = "patient_001"):
    """동기적으로 Supervisor Agent 실행 (테스트용)"""
    async def async_run():
        config = {"configurable": {"thread_id": f"test_{patient_id}"}}
        
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "patient_id": patient_id
        }
        
        response = await supervisor_graph.ainvoke(input_data, config=config)
        return response
    
    return asyncio.run(async_run())

# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=== 통합 의료 서비스 API 시작 ===")
    print("API 서버: http://localhost:8000")
    print()
    
    print("=== 사용 가능한 서비스 ===")
    print("1. 치매 검사 서비스 (음성 파일 지원)")
    print("2. 복약 알림 서비스")
    print("3. 기분 및 건강체크 서비스")
    print()
    
    print("=== API 엔드포인트 ===")
    print("- POST /supervisor: Supervisor Agent (음성 파일 지원)")
    print("- POST /supervisor-json: Supervisor Agent (JSON 형식)")
    print("- GET /health: 헬스 체크")
    print()
    
    print("=== 새로운 기능 ===")
    print("- FAISS 벡터DB를 사용한 뇌졸중 검사")
    print("- 개선된 치매 검사 (반복 발화 + 기억력 검사)")
    print("- 음성 파일 임베딩 기반 유사도 분석")
    print()
    
    print("=== 서버 시작 ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)