import os
import json
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

# 환경 변수 로드
load_dotenv()

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

# ==================== Pydantic 모델 정의 ====================

# Supervisor 관련 모델
class SupervisorRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    patient_id: Optional[str] = Field(None, description="환자 ID (치매 검사용)")

class SupervisorResponse(BaseModel):
    service_type: str = Field(..., description="서비스 타입 (dementia/medicine/mood_health)")
    response: dict = Field(..., description="응답 내용")
    next_action: str = Field(..., description="다음 액션")

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
    dementia_result: Optional[dict] = None
    medicine_result: Optional[dict] = None
    mood_result: Optional[dict] = None
    health_result: Optional[dict] = None

members = ["dementia_agent", "medicine_agent", "mood_health_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

# ==================== 음성 분석 함수들 ====================

def analyze_audio_pronunciation(audio_file_path: str):
    """음성 파일을 분석하여 발음 어눌함을 검사하는 함수"""
    try:
        # 음성 파일 로드
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # 기본적인 음성 특성 분석
        duration = float(len(y) / sr)
        energy = float(np.mean(librosa.feature.rms(y=y)))
        
        # 음성 스펙트럼 특성
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 음성 명확도 (스펙트럼 중심)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = float(np.mean(spectral_centroids))
        
        # 음성 대비 (스펙트럼 대비)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = float(np.mean(spectral_contrast))
        
        # 발음 어눌함 판단 기준
        pronunciation_score = 0.0
        details = []
        
        # 음성 길이가 너무 짧으면 의심
        if duration < 0.5:
            pronunciation_score += 0.3
            details.append("음성 길이가 너무 짧음")
        
        # 에너지가 너무 낮으면 의심
        if energy < 0.01:
            pronunciation_score += 0.2
            details.append("음성 에너지가 낮음")
        
        # 스펙트럼 중심이 너무 낮으면 의심
        if spectral_centroid_mean < 1000:
            pronunciation_score += 0.2
            details.append("스펙트럼 중심 주파수가 낮음")
        
        # 스펙트럼 대비가 낮으면 의심
        if spectral_contrast_mean < 10:
            pronunciation_score += 0.3
            details.append("스펙트럼 대비가 낮음")
        
        # 결과 결정
        if pronunciation_score >= 0.5:
            result = "unclear"
            confidence = float(min(pronunciation_score, 0.9))
        else:
            result = "clear"
            confidence = float(1.0 - pronunciation_score)
        
        return {
            "type": "audio_pronunciation_analysis",
            "result": result,
            "confidence": confidence,
            "details": details,
            "audio_features": {
                "duration": duration,
                "energy": energy,
                "spectral_centroid": spectral_centroid_mean,
                "spectral_contrast": spectral_contrast_mean
            }
        }
        
    except Exception as e:
        return {
            "type": "audio_pronunciation_analysis",
            "result": "error",
            "confidence": 0.0,
            "details": f"음성 분석 오류: {str(e)}",
            "audio_features": {}
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

def check_pronunciation_clarity(conversation_text: str, audio_file_path: Optional[str] = None):
    """발음이 어눌한지 검사하는 함수 (텍스트 + 음성 파일)"""
    results = []
    
    # 1. 텍스트 기반 발음 분석
    pronunciation_prompt = f"""
    다음 환자의 대화 내용에서 발음이나 언어 표현의 어눌함을 분석해주세요:
    
    대화: "{conversation_text}"
    
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
    
    try:
        response = llm.invoke(pronunciation_prompt)
        response_text = response.content
        
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
        
        results.append({
            "type": "text_pronunciation_analysis",
            "result": pronunciation_status,
            "confidence": float(confidence),
            "details": details
        })
        
    except Exception as e:
        results.append({
            "type": "text_pronunciation_analysis",
            "result": "error",
            "confidence": 0.0,
            "details": f"텍스트 발음 분석 오류: {str(e)}"
        })
    
    # 2. 음성 파일 기반 발음 분석 (있는 경우)
    if audio_file_path and os.path.exists(audio_file_path):
        audio_result = analyze_audio_pronunciation(audio_file_path)
        results.append(audio_result)
    
    # 3. 결과 통합
    if len(results) == 2:  # 텍스트 + 음성 모두 있는 경우
        text_result = results[0]
        audio_result = results[1]
        
        # 두 결과를 가중 평균으로 통합
        text_weight = 0.4
        audio_weight = 0.6
        
        combined_confidence = float(text_result["confidence"] * text_weight + 
                                   audio_result["confidence"] * audio_weight)
        
        # 둘 중 하나라도 unclear이면 unclear로 판정
        if text_result["result"] == "unclear" or audio_result["result"] == "unclear":
            final_result = "unclear"
        else:
            final_result = "clear"
        
        return {
            "type": "pronunciation_clarity",
            "result": final_result,
            "confidence": combined_confidence,
            "details": f"텍스트 분석: {text_result['result']}, 음성 분석: {audio_result['result']}",
            "sub_analyses": results
        }
    
    else:  # 텍스트만 있는 경우
        return results[0]

def check_memory_recall(conversation_text: str):
    """기억력 검사 함수"""
    memory_prompt = f"""
    다음 환자의 대화 내용에서 기억력 문제를 분석해주세요:
    
    대화: "{conversation_text}"
    
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

def check_repetitive_speech(conversation_text: str):
    """반복 발화 검사 함수"""
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

# Supervisor 노드
system_prompt = f"""
당신은 의료 서비스를 관리하는 Supervisor Agent입니다.
다음 세 가지 서비스를 관리합니다:

1. dementia_agent: 치매 검사 서비스
   - 환자의 대화나 음성을 분석하여 치매 의심 여부를 검사
   - 발음 어눌함, 기억력 저하, 반복 발화 등을 분석
   - 사용 예시: "아침에 약을 먹었는데 또 먹어야 하나요?", "오늘 날짜가 뭐였지?"

2. medicine_agent: 복약 알림 서비스
   - 복약 관련 메시지를 분석하여 복용 시간, 약물명, 복용량 추출
   - 복약 알림 설정 및 관리
   - 사용 예시: "아침 8시에 혈압약 1정 먹어야 해", "관절염 약 2개씩 복용"

3. mood_health_agent: 기분 및 건강체크 서비스
   - 환자의 기분 상태를 "좋음", "나쁨", "평범" 중 하나로 판별
   - 전반적인 건강 상태를 텍스트로 분석
   - 사용 예시: "오늘 기분이 좋아요", "몸이 좀 아파요", "컨디션이 평범해요"

사용자의 메시지를 분석하여 적절한 서비스로 라우팅하세요:
- 치매 관련 증상이나 인지 기능 문제 → dementia_agent
- 복약 관련 정보나 알림 설정 → medicine_agent
- 기분이나 건강 상태 관련 → mood_health_agent
- 두 서비스 이상 필요하거나 명확하지 않으면 → mood_health_agent (일반적인 대화 우선)

작업이 완료되면 FINISH로 응답하세요.
"""

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    
    if goto == "FINISH":
        goto = END
    
    return Command(goto=goto, update={"next": goto})

# 치매 검사 Agent 노드
def dementia_node(state: State) -> Command[Literal["__end__"]]:
    """치매 검사 서비스를 처리하는 노드"""
    try:
        patient_id = state.get("patient_id", "patient_001")
        conversation_text = state["messages"][-1].content
        audio_file_path = state.get("audio_file_path")
        
        # 대화 처리
        conversation_result = process_conversation(patient_id, conversation_text)
        
        # 추가 검사들 (음성 파일 경로 전달)
        pronunciation_result = check_pronunciation_clarity(conversation_text, audio_file_path)
        memory_result = check_memory_recall(conversation_text)
        repetition_result = check_repetitive_speech(conversation_text)
        
        # 결과 통합
        check_results = [
            conversation_result["check_results"][0],
            pronunciation_result,
            memory_result,
            repetition_result
        ]
        
        # 최종 진단 결정
        dementia_indicators = 0
        total_confidence = 0.0
        
        for result in check_results:
            if result["type"] == "conversation_analysis" and result["result"] == "yes":
                dementia_indicators += 1
            elif result["type"] == "pronunciation_clarity" and result["result"] == "unclear":
                dementia_indicators += 1
            elif result["type"] == "memory_recall" and result["result"] == "impaired":
                dementia_indicators += 1
            elif result["type"] == "repetitive_speech" and result["result"] == "repetitive":
                dementia_indicators += 1
            
            total_confidence += float(result.get("confidence", 0))
        
        final_diagnosis = "yes" if dementia_indicators >= 2 else "no"
        avg_confidence = float(total_confidence / len(check_results)) if check_results else 0.0
        
        dementia_result = {
            "diagnosis": final_diagnosis,
            "confidence": avg_confidence,
            "contents": conversation_result["contents"],
            "check_results": check_results,
            "summary": f"치매 의심 지표 수: {dementia_indicators}/4, 평균 신뢰도: {avg_confidence:.3f}",
            "audio_used": audio_file_path is not None
        }
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"치매 검사 완료: {final_diagnosis} (신뢰도: {avg_confidence:.3f})",
                        name="dementia_agent"
                    )
                ],
                "dementia_result": dementia_result
            },
            goto=END
        )
        
    except Exception as e:
        error_result = {
            "error": f"치매 검사 중 오류: {str(e)}",
            "diagnosis": "no",
            "confidence": 0.0
        }
        
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
            goto=END
        )

# 복약 알림 Agent 노드
def medicine_node(state: State) -> Command[Literal["__end__"]]:
    """복약 알림 서비스를 처리하는 노드"""
    try:
        message = state["messages"][-1].content
        
        # 복약 정보 분석
        medicine_info = analyze_medicine(message)
        
        # 응답 생성
        medicine_result = generate_medicine_response(medicine_info)
        
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
            goto=END
        )
        
    except Exception as e:
        error_result = {
            "error": f"복약 분석 중 오류: {str(e)}",
            "time": "",
            "medicine_name": "",
            "dosage": "",
            "needs_reminder": False
        }
        
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
            goto=END
        )

# 기분 및 건강체크 Agent 노드
def mood_health_node(state: State) -> Command[Literal["__end__"]]:
    """기분 및 건강체크 서비스를 처리하는 노드"""
    try:
        message = state["messages"][-1].content
        
        # 기분 및 건강 상태 분석
        mood_health_info = analyze_mood_health(message)
        
        # 응답 생성
        mood_health_result = generate_mood_health_response(mood_health_info)
        
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
            goto=END
        )
        
    except Exception as e:
        error_result = {
            "error": f"기분 및 건강체크 분석 중 오류: {str(e)}",
            "mood": "평범",
            "health_status": "분석 오류로 인해 확인할 수 없음"
        }
        
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
            goto=END
        )

# ==================== LangGraph 그래프 구성 ====================

def create_supervisor_graph():
    """Supervisor 그래프 생성"""
    graph_builder = StateGraph(State)
    
    # 노드 추가
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("dementia_agent", dementia_node)
    graph_builder.add_node("medicine_agent", medicine_node)
    graph_builder.add_node("mood_health_agent", mood_health_node)
    
    # 엣지 추가
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("supervisor", "dementia_agent")
    graph_builder.add_edge("supervisor", "medicine_agent")
    graph_builder.add_edge("supervisor", "mood_health_agent")
    graph_builder.add_edge("supervisor", END)
    
    # 모든 agent는 END로 직접 이동
    graph_builder.add_edge("dementia_agent", END)
    graph_builder.add_edge("medicine_agent", END)
    graph_builder.add_edge("mood_health_agent", END)
    
    # 메모리 설정
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

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
    try:
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
        
        # 입력 메시지 준비
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "patient_id": patient_id,
            "audio_file_path": audio_file_path
        }
        
        # 그래프 실행
        response = await supervisor_graph.ainvoke(input_data, config=config)
        
        # 결과 분석 - service_type을 결과에서 추론
        service_type = "unknown"
        result_data = {}
        
        if response.get("dementia_result"):
            service_type = "dementia"
            result_data = response.get("dementia_result", {})
            next_action = "치매 검사 완료 - 추가 상담이 필요할 수 있습니다."
        elif response.get("medicine_result"):
            service_type = "medicine"
            result_data = response.get("medicine_result", {})
            next_action = "복약 정보 추출 완료 - 알림 설정이 필요할 수 있습니다."
        elif response.get("mood_result") or response.get("health_result"):
            service_type = "mood_health"
            result_data = {
                "mood": response.get("mood_result", {}),
                "health": response.get("health_result", {})
            }
            next_action = "기분 및 건강체크 완료 - 필요시 추가 상담을 권장합니다."
        else:
            result_data = {"message": "서비스 타입을 결정할 수 없습니다."}
            next_action = "수동 확인이 필요합니다."
        
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        return SupervisorResponse(
            service_type=service_type,
            response=result_data,
            next_action=next_action
        )
        
    except Exception as e:
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
            "audio_file_path": None  # JSON 형식에서는 음성 파일 없음
        }
        
        # 그래프 실행
        response = await supervisor_graph.ainvoke(input_data, config=config)
        
        # 결과 분석
        service_type = "unknown"
        result_data = {}
        
        if response.get("dementia_result"):
            service_type = "dementia"
            result_data = response.get("dementia_result", {})
            next_action = "치매 검사 완료 - 추가 상담이 필요할 수 있습니다."
        elif response.get("medicine_result"):
            service_type = "medicine"
            result_data = response.get("medicine_result", {})
            next_action = "복약 정보 추출 완료 - 알림 설정이 필요할 수 있습니다."
        elif response.get("mood_result") or response.get("health_result"):
            service_type = "mood_health"
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
            response=result_data,
            next_action=next_action
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
    
    print("=== 서버 시작 ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)