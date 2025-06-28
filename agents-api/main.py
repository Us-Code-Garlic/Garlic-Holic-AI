import os
import json
import uuid
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import tempfile
import shutil
import numpy as np
import pymysql
from contextlib import contextmanager, asynccontextmanager

# CLAP 모델 import 추가
try:
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except ImportError:
    print("Warning: CLAP 모델을 사용할 수 없습니다. transformers 라이브러리가 설치되지 않았습니다.")
    CLAP_AVAILABLE = False

from StrokeCheck import (
    AudioEmbeddings, 
    load_and_embed_audio, 
    load_faiss_vectorstore, 
    check_stroke_with_faiss,
    set_global_models,
    get_global_models
)
from RepretitiveCheck import check_repetitive_speech, check_memory_recall

# 모듈화된 파일들 import
from states import State, Router, members, options
from model import SupervisorRequest, SupervisorResponse, MedicineInfo, MoodHealthInfo
from MoodHealthCheck import mood_health_node, set_llm_model
from MedicineCheck import medicine_node, set_llm_model as set_medicine_llm_model

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

# FastAPI 앱 초기화 (lifespan 이벤트로 변경)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # 시작 시
    create_chat_table()
    await load_audio_models()
    yield
    # 종료 시 (필요한 정리 작업)

app = FastAPI(
    title="통합 의료 서비스 API",
    description="치매 검사, 복약 알림, 기분 및 건강체크를 통합 관리하는 API 서버",
    version="3.0.0",
    lifespan=lifespan
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

# MoodHealthCheck 모듈에 LLM 설정
set_llm_model(llm)

# MedicineCheck 모듈에 LLM 설정
set_medicine_llm_model(llm)

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
    
    if not CLAP_AVAILABLE:
        print("CLAP 모델을 사용할 수 없어 음성 분석 기능이 비활성화됩니다.")
        return
    
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
                # StrokeCheck 모듈에 전역 변수 설정
                set_global_models(faiss_vectorstore, clap_model, clap_processor)
            else:
                print("Failed to load FAISS vectorstore")
        else:
            print("FAISS vectorstore not found at", faiss_path)
            
    except Exception as e:
        print(f"Error loading audio models: {e}")
        print("음성 분석 기능이 비활성화됩니다.")

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

# ==================== LangGraph 노드들 ====================

# Supervisor 노드 - 반복 제어 및 라우팅
def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    """Supervisor 노드 - 반복 제어 및 라우팅"""
    message = state["messages"][-1].content.lower()
    iteration_count = state.get("iteration_count", 0)
    
    print(f"\n=== SUPERVISOR NODE (반복 {iteration_count}회) ===")
    print(f"입력 메시지: {message}")
    print(f"현재 반복 횟수: {iteration_count}")
    
    # 사용자 입력을 DB에 저장
    user_id = state.get("user_id")
    if user_id:
        save_chat_to_db(user_id, state["messages"][-1].content, "user")
    
    # 5회 반복 후 종료 조건
    if iteration_count >= 5:
        print(f"최대 반복 횟수(5회) 도달 - __end__로 종료")
        return Command(
            update={
                "iteration_count": iteration_count + 1,
                "next": "__end__"
            },
            goto="__end__"
        )
    
    # Agent 응답 메시지인지 확인 (Agent가 보낸 메시지는 무시하고 종료)
    if any(agent_name in message for agent_name in ["치매 검사 완료", "복약 정보 추출 완료", "기분 및 건강체크 완료"]):
        print(f"Agent 응답 메시지 감지 - __end__로 종료")
        return Command(
            update={
                "iteration_count": iteration_count + 1,
                "next": "__end__"
            },
            goto="__end__"
        )
    
    # 모든 메시지를 치매 검사로 라우팅 (키워드 기반 라우팅 제거)
    print(f"기본 라우팅: dementia_agent")
    return Command(
        update={
            "selected_agent": "dementia_agent",
            "service_type": "dementia",
            "iteration_count": iteration_count + 1,
            "next": "dementia_agent"
        },
        goto="dementia_agent"
    )

# 치매 검사 Agent 노드 - 실행 후 supervisor로 돌아감
def dementia_node(state: State) -> Command[Literal["supervisor"]]:
    """치매 검사 서비스 - 실행 후 supervisor로 돌아감"""
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
        
        # 뇌졸중 검사 (음성 파일이 있는 경우, CLAP 모델이 사용 가능한 경우에만)
        stroke_result = None
        if audio_file_path and os.path.exists(audio_file_path) and CLAP_AVAILABLE:
            try:
                stroke_result = check_stroke_with_faiss(audio_file_path)
                print(f"뇌졸중 검사 결과: {stroke_result}")
            except Exception as e:
                print(f"뇌졸중 검사 중 오류: {e}")
                stroke_result = None
        elif audio_file_path and not CLAP_AVAILABLE:
            print("CLAP 모델이 사용 불가능하여 뇌졸중 검사를 건너뜁니다.")
        
        # 치매 검사: 반복 발화 + 기억력 검사 (LLM 전달)
        repetition_result = check_repetitive_speech(conversation_text, llm)
        memory_result = check_memory_recall(conversation_text, llm)
        
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
            "audio_used": audio_file_path is not None and CLAP_AVAILABLE
        }
        
        # Agent 응답을 DB에 저장
        if user_id:
            agent_response = f"치매 검사 완료: {final_diagnosis} (신뢰도: {avg_confidence:.3f})"
            save_chat_to_db(user_id, agent_response, "agent")
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["dementia_agent"]
        
        # 누적 결과에 추가 (새로운 리스트 생성)
        current_all_dementia_results = state.get("all_dementia_results", [])
        new_all_dementia_results = current_all_dementia_results + [dementia_result]
        
        print(f"치매 검사 완료 - supervisor로 돌아감")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"치매 검사 완료: {final_diagnosis} (신뢰도: {avg_confidence:.3f})",
                        name="dementia_agent"
                    )
                ],
                "dementia_result": dementia_result,
                "stroke_result": stroke_result,
                "executed_agents": ["dementia_agent"],  # 단일 값으로 전달
                "all_dementia_results": new_all_dementia_results  # 새로운 리스트로 전달
            },
            goto="supervisor"  # supervisor로 돌아감
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
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["dementia_agent"]
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"치매 검사 오류: {str(e)}",
                        name="dementia_agent"
                    )
                ],
                "dementia_result": error_result,
                "executed_agents": ["dementia_agent"]  # 단일 값으로 전달
            },
            goto="supervisor"  # 오류가 있어도 supervisor로 돌아감
        )

# ==================== LangGraph 그래프 구성 ====================

def create_supervisor_graph():
    """Supervisor 그래프 생성 - 반복 구조"""
    graph_builder = StateGraph(State)
    
    # 노드 추가
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("dementia_agent", dementia_node)
    graph_builder.add_node("medicine_agent", medicine_node)  # 모듈에서 import한 함수 사용
    graph_builder.add_node("mood_health_agent", mood_health_node)  # 모듈에서 import한 함수 사용
    
    # 각 Agent에서 supervisor로 돌아감
    graph_builder.add_edge("dementia_agent", "supervisor")
    graph_builder.add_edge("medicine_agent", "supervisor")
    graph_builder.add_edge("mood_health_agent", "supervisor")
    
    # 조건부 엣지 추가 - supervisor에서 각 Agent로 또는 __end__로
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "dementia_agent"),
        {
            "dementia_agent": "dementia_agent",
            "medicine_agent": "medicine_agent", 
            "mood_health_agent": "mood_health_agent",
            "__end__": "__end__"
        }
    )
    
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
        "version": "3.0.0",
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
        },
        "features": {
            "iteration_limit": "최대 5회 반복 후 자동 종료",
            "cumulative_results": "모든 Agent 실행 결과 누적 저장",
            "supervisor_control": "Supervisor가 모든 Agent 실행을 제어"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy", 
        "service": "통합 의료 서비스 API",
        "version": "3.0.0",
        "available_services": [
            "dementia_check", 
            "medicine_reminder", 
            "mood_health_check"
        ],
        "iteration_limit": 5
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
    최대 5회 반복 후 자동 종료됩니다.
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
            "user_id": user_id,
            "iteration_count": 0,  # 초기 반복 횟수
            "executed_agents": [],  # 실행된 Agent 목록
            "all_dementia_results": [],  # 누적 치매 결과
            "all_medicine_results": [],  # 누적 복약 결과
            "all_mood_results": [],      # 누적 기분 결과
            "all_health_results": []     # 누적 건강 결과
        }
        
        print(f"그래프 입력 데이터: {input_data}")
        
        # 그래프 실행
        response = await supervisor_graph.ainvoke(input_data, config=config)
        
        print(f"그래프 실행 결과: {response}")
        
        # 결과 분석 - State 전체를 응답으로 사용
        selected_agent = response.get("selected_agent", "unknown")
        service_type = response.get("service_type", "unknown")
        iteration_count = response.get("iteration_count", 0)
        executed_agents = response.get("executed_agents", [])
        
        print(f"선택된 Agent: {selected_agent}")
        print(f"서비스 타입: {service_type}")
        print(f"최종 반복 횟수: {iteration_count}")
        print(f"실행된 Agent들: {executed_agents}")
        
        # State를 dict로 변환 (JSON 직렬화 가능하도록)
        state_dict = {
            "selected_agent": response.get("selected_agent"),
            "service_type": response.get("service_type"),
            "patient_id": response.get("patient_id"),
            "audio_file_path": response.get("audio_file_path"),
            "iteration_count": iteration_count,
            "executed_agents": executed_agents,
            "dementia_result": response.get("dementia_result"),
            "medicine_result": response.get("medicine_result"),
            "mood_result": response.get("mood_result"),
            "health_result": response.get("health_result"),
            "stroke_result": response.get("stroke_result"),
            "all_dementia_results": response.get("all_dementia_results", []),
            "all_medicine_results": response.get("all_medicine_results", []),
            "all_mood_results": response.get("all_mood_results", []),
            "all_health_results": response.get("all_health_results", []),
            "messages": [msg.dict() if hasattr(msg, 'dict') else msg for msg in response.get("messages", [])]
        }
        
        # 주요 응답 데이터 추출 (누적 결과 포함)
        if selected_agent == "dementia_agent":
            result_data = {
                "current_result": response.get("dementia_result", {}),
                "all_results": response.get("all_dementia_results", [])
            }
            next_action = f"치매 검사 완료 (총 {len(response.get('all_dementia_results', []))}회) - 추가 상담이 필요할 수 있습니다."
        elif selected_agent == "medicine_agent":
            result_data = {
                "current_result": response.get("medicine_result", {}),
                "all_results": response.get("all_medicine_results", [])
            }
            next_action = f"복약 정보 추출 완료 (총 {len(response.get('all_medicine_results', []))}회) - 알림 설정이 필요할 수 있습니다."
        elif selected_agent == "mood_health_agent":
            result_data = {
                "current_result": {
                    "mood": response.get("mood_result", {}),
                    "health": response.get("health_result", {})
                },
                "all_mood_results": response.get("all_mood_results", []),
                "all_health_results": response.get("all_health_results", [])
            }
            next_action = f"기분 및 건강체크 완료 (총 {len(response.get('all_mood_results', []))}회) - 필요시 추가 상담을 권장합니다."
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
            user_id=user_id,
            iteration_count=iteration_count
        )
        
    except Exception as e:
        print(f"Supervisor 엔드포인트 오류: {str(e)}")
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        raise HTTPException(status_code=500, detail=f"Supervisor 처리 중 오류가 발생했습니다: {str(e)}")


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
            "patient_id": patient_id,
            "iteration_count": 0,
            "executed_agents": [],
            "all_dementia_results": [],
            "all_medicine_results": [],
            "all_mood_results": [],
            "all_health_results": []
        }
        
        response = await supervisor_graph.ainvoke(input_data, config=config)
        return response
    
    return asyncio.run(async_run())

# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=== 통합 의료 서비스 API v3.0 시작 ===")
    print("API 서버: http://localhost:8000")
    print()
    
    print("=== 새로운 기능 (v3.0) ===")
    print("1. 반복 구조: 각 Agent가 END로 가는 대신 Supervisor로 돌아감")
    print("2. 반복 제어: 최대 5회 반복 후 자동 종료")
    print("3. 누적 결과: 모든 Agent 실행 결과를 누적하여 저장")
    print("4. Supervisor 중심: Supervisor가 모든 Agent 실행을 제어")
    print("5. 모듈화: states.py, model.py, MoodHealthCheck.py로 코드 분리")
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
    
    print("=== 그래프 구조 ===")
    print("Supervisor → Agent → Supervisor → Agent → ... → END (5회 후)")
    print("각 Agent 실행 후 결과를 누적하고 Supervisor로 돌아감")
    print()
    
    print("=== 모듈 구조 ===")
    print("- main.py: 메인 애플리케이션 및 API 엔드포인트")
    print("- states.py: State 클래스 및 Router 정의")
    print("- model.py: Pydantic 모델 정의")
    print("- MoodHealthCheck.py: 기분 및 건강체크 모듈")
    print()
    
    print("=== 서버 시작 ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)