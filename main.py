import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel as LangChainBaseModel
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from dotenv import load_dotenv
import uuid
import asyncio
from datetime import datetime

load_dotenv()

# Google API 키 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FastAPI 앱 초기화
app = FastAPI(
    title="Gemini AI Chat API",
    description="LangChain과 Google Gemini를 활용한 AI 채팅 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 관리 (실제 프로덕션에서는 Redis나 데이터베이스 사용 권장)
chat_sessions = {}

# Pydantic 모델들
class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (없으면 새로 생성)")
    model_name: Optional[str] = Field("gemini-2.0-flash", description="사용할 모델명")
    temperature: Optional[float] = Field(0.7, description="창의성 조절 (0.0 ~ 1.0)")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 응답")
    session_id: str = Field(..., description="세션 ID")
    timestamp: datetime = Field(..., description="응답 시간")

class StreamingChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")

class ToolCallRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    use_tools: bool = Field(True, description="도구 사용 여부")

class StructuredOutputRequest(BaseModel):
    text: str = Field(..., description="구조화할 텍스트")

class Person(LangChainBaseModel):
    """사람 정보를 구조화된 형태로 저장"""
    name: str = Field(..., description="사람의 이름")
    age: int = Field(..., description="나이")
    occupation: str = Field(..., description="직업")

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    model_name: str

# 도구(Tool) 정의
@tool(description="사용자의 이름을 기억하고 인사하는 도구")
def remember_name(name: str) -> str:
    return f"안녕하세요 {name}님! 이름을 기억했습니다."

@tool(description="간단한 계산을 수행하는 도구")
def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"계산 결과: {expression} = {result}"
    except:
        return "계산할 수 없는 표현식입니다."

class GeminiChatService:
    def __init__(self):
        self.llm = None
    
    def get_llm(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.7):
        """LLM 인스턴스 생성 또는 반환"""
        if self.llm is None or self.llm.model_name != model_name:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY
            )
        return self.llm
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """세션 생성 또는 기존 세션 반환"""
        if session_id and session_id in chat_sessions:
            return session_id
        
        new_session_id = str(uuid.uuid4())
        chat_sessions[new_session_id] = {
            "conversation_history": [],
            "created_at": datetime.now(),
            "message_count": 0
        }
        return new_session_id
    
    def add_to_history(self, session_id: str, user_message: str, ai_response: str):
        """대화 기록에 메시지 추가"""
        if session_id in chat_sessions:
            chat_sessions[session_id]["conversation_history"].extend([
                HumanMessage(content=user_message),
                AIMessage(content=ai_response)
            ])
            chat_sessions[session_id]["message_count"] += 1

# 서비스 인스턴스
chat_service = GeminiChatService()

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Gemini AI Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """기본 채팅 엔드포인트"""
    try:
        # 세션 관리
        session_id = chat_service.get_or_create_session(request.session_id)
        
        # LLM 초기화
        llm = chat_service.get_llm(request.model_name, request.temperature)
        
        # 대화 기록 가져오기
        history = chat_sessions[session_id]["conversation_history"]
        
        # 메시지 생성
        messages = history + [HumanMessage(content=request.message)]
        
        # AI 응답 생성
        response = llm.invoke(messages)
        
        # 대화 기록 업데이트
        chat_service.add_to_history(session_id, request.message, response.content)
        
        return ChatResponse(
            response=response.content,
            session_id=session_id,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류: {str(e)}")

@app.post("/chat/stream")
async def streaming_chat(request: StreamingChatRequest):
    """스트리밍 채팅 엔드포인트"""
    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        llm = chat_service.get_llm()
        
        async def generate_stream():
            try:
                async for chunk in llm.astream(request.message):
                    yield f"data: {chunk.content}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: 오류: {str(e)}\n\n"
        
        return generate_stream()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스트리밍 채팅 오류: {str(e)}")

@app.post("/chat/tools")
async def tool_calling_chat(request: ToolCallRequest):
    """도구 호출 채팅 엔드포인트"""
    try:
        llm = chat_service.get_llm()
        
        if request.use_tools:
            # 도구를 모델에 바인딩
            llm_with_tools = llm.bind_tools([remember_name, calculate])
            
            # 도구 호출
            ai_msg = llm_with_tools.invoke(request.message)
            
            # 도구 실행 결과 생성
            tool_messages = []
            for tool_call in ai_msg.tool_calls:
                if tool_call["name"] == "remember_name":
                    result = remember_name(tool_call["args"]["name"])
                elif tool_call["name"] == "calculate":
                    result = calculate(tool_call["args"]["expression"])
                
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
            
            # 최종 응답 생성
            final_response = llm_with_tools.invoke([ai_msg] + tool_messages)
            
            return {
                "response": final_response.content,
                "tool_calls": ai_msg.tool_calls,
                "timestamp": datetime.now()
            }
        else:
            # 일반 채팅
            response = llm.invoke(request.message)
            return {
                "response": response.content,
                "tool_calls": [],
                "timestamp": datetime.now()
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"도구 호출 채팅 오류: {str(e)}")

@app.post("/structured-output")
async def structured_output(request: StructuredOutputRequest):
    """구조화된 출력 엔드포인트"""
    try:
        llm = chat_service.get_llm(temperature=0)
        structured_llm = llm.with_structured_output(Person)
        
        result = structured_llm.invoke(request.text)
        
        return {
            "structured_data": {
                "name": result.name,
                "age": result.age,
                "occupation": result.occupation
            },
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"구조화된 출력 오류: {str(e)}")

@app.post("/google-search")
async def google_search_chat(request: ChatRequest):
    """Google 검색 통합 채팅"""
    try:
        llm = chat_service.get_llm()
        
        response = llm.invoke(
            request.message,
            tools=[GenAITool(google_search={})]
        )
        
        return {
            "response": response.content,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google 검색 오류: {str(e)}")

@app.post("/code-execution")
async def code_execution_chat(request: ChatRequest):
    """코드 실행 채팅"""
    try:
        llm = chat_service.get_llm()
        
        response = llm.invoke(
            request.message,
            tools=[GenAITool(code_execution={})]
        )
        
        code_results = []
        for content in response.content:
            if isinstance(content, dict):
                if content["type"] == "code_execution_result":
                    code_results.append({
                        "type": "execution_result",
                        "result": content["code_execution_result"]
                    })
                elif content["type"] == "executable_code":
                    code_results.append({
                        "type": "executable_code",
                        "code": content["executable_code"]
                    })
            else:
                code_results.append({
                    "type": "text",
                    "content": str(content)
                })
        
        return {
            "response": response.content,
            "code_results": code_results,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"코드 실행 오류: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session = chat_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=session["message_count"],
        model_name="gemini-2.0-flash"
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "세션이 삭제되었습니다"}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

@app.get("/sessions")
async def list_sessions():
    """모든 세션 목록 조회"""
    sessions = []
    for session_id, session_data in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "message_count": session_data["message_count"]
        })
    return {"sessions": sessions}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "api_key_configured": bool(GOOGLE_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    
    if not GOOGLE_API_KEY:
        print("오류: GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        print("다음 명령어로 API 키를 설정해주세요:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        exit(1)
    
    print("=== Gemini AI Chat API 서버 시작 ===")
    print("API 문서: http://localhost:8000/docs")
    print("ReDoc 문서: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
