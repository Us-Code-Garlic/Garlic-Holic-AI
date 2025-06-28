import os
import getpass
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# FastAPI 앱 초기화
app = FastAPI(title="복약알림 API", description="LangGraph 기반 복약알림 분석 API")

# LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# State 정의
class State(MessagesState):
    time: Optional[str] = None
    medicine_name: Optional[str] = None
    dosage: Optional[str] = None
    needs_reminder: Optional[bool] = None

# Pydantic 모델 정의
class MedicineReminderRequest(BaseModel):
    message: str = Field(..., description="복약 관련 메시지")

class MedicineReminderResponse(BaseModel):
    time: str = Field(..., description="복약 시간 (예: 8:00 am)")
    medicine_name: str = Field(..., description="약물 이름")
    dosage: str = Field(..., description="복용량")
    needs_reminder: bool = Field(..., description="복약알림 필요 여부")

# 복약 정보 추출을 위한 구조화된 출력 모델
class MedicineInfo(BaseModel):
    """복약 정보를 추출하는 모델"""
    time: str = Field(description="복약 시간을 24시간 형식으로 변환 (예: 08:00, 20:30)")
    medicine_name: str = Field(description="약물 이름")
    dosage: str = Field(description="복용량 (예: 2개, 1정, 10ml)")
    needs_reminder: bool = Field(description="복약알림이 필요한지 여부")

# 복약 정보 분석 노드
MEDICINE_ANALYSIS_SYSTEM_TEMPLATE = """
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
"""

MEDICINE_ANALYSIS_USER_TEMPLATE = """
사용자 메시지: {messages}
복약 정보를 추출해주세요:
"""

def analyze_medicine(state: State):
    """복약 정보를 분석하는 노드"""
    analysis_msgs = [
        ("system", MEDICINE_ANALYSIS_SYSTEM_TEMPLATE),
        ("user", MEDICINE_ANALYSIS_USER_TEMPLATE),
    ]
    analysis_prompt = ChatPromptTemplate.from_messages(analysis_msgs)
    
    model_with_structured_output = llm.with_structured_output(MedicineInfo)
    
    response = model_with_structured_output.invoke(
        analysis_prompt.format_messages(
            messages=state["messages"][-1].content
        )
    )
    
    print(f"\n[analyze_medicine node]")
    print(f"입력: {state['messages'][-1].content}")
    print(f"시간: {response.time}")
    print(f"약물: {response.medicine_name}")
    print(f"복용량: {response.dosage}")
    print(f"알림 필요: {response.needs_reminder}")
    
    return {
        "time": response.time,
        "medicine_name": response.medicine_name,
        "dosage": response.dosage,
        "needs_reminder": response.needs_reminder
    }

# 응답 생성 노드
RESPONSE_SYSTEM_TEMPLATE = """
당신은 복약 정보를 정리하여 사용자에게 명확하게 전달하는 의료 어시스턴트입니다.
추출된 복약 정보를 바탕으로 구조화된 응답을 생성해주세요.

응답 형식:
- 시간: 12시간 형식으로 변환 (예: 8:00 am, 8:00 pm)
- 약물 이름: 그대로 유지
- 복용량: 그대로 유지
- 복약알림 필요 여부: True/False
"""

RESPONSE_USER_TEMPLATE = """
추출된 복약 정보:
- 시간: {time}
- 약물 이름: {medicine_name}
- 복용량: {dosage}
- 복약알림 필요: {needs_reminder}

구조화된 응답을 생성해주세요.
"""

def generate_response(state: State):
    """응답을 생성하는 노드"""
    time = state.get("time", "")
    medicine_name = state.get("medicine_name", "")
    dosage = state.get("dosage", "")
    needs_reminder = state.get("needs_reminder", False)
    
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
    
    response_content = f"복약 정보가 성공적으로 추출되었습니다:\n- 시간: {formatted_time}\n- 약물: {medicine_name}\n- 복용량: {dosage}\n- 복약알림 필요: {needs_reminder}"
    
    print(f"\n[generate_response node]")
    print(f"응답: {response_content}")
    
    return {
        "messages": [AIMessage(content=response_content)],
        "time": formatted_time,
        "medicine_name": medicine_name,
        "dosage": dosage,
        "needs_reminder": needs_reminder
    }

# LangGraph 그래프 구성
def create_medicine_graph():
    """복약 분석 그래프 생성"""
    graph_builder = StateGraph(State)
    
    # 노드 추가
    graph_builder.add_node("analyze_medicine", analyze_medicine)
    graph_builder.add_node("generate_response", generate_response)
    
    # 엣지 추가
    graph_builder.add_edge(START, "analyze_medicine")
    graph_builder.add_edge("analyze_medicine", "generate_response")
    graph_builder.add_edge("generate_response", END)
    
    # 메모리 설정
    memory = MemorySaver()
    
    # 그래프 컴파일
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph

# 그래프 인스턴스 생성
medicine_graph = create_medicine_graph()

# FastAPI 엔드포인트
@app.post("/analyze-medicine", response_model=MedicineReminderResponse)
async def analyze_medicine_reminder(request: MedicineReminderRequest):
    """
    복약 관련 메시지를 분석하여 복약 정보를 추출합니다.
    
    예시 입력:
    - "아침 8시마다 관절염 약 2개씩을 먹어야 해."
    - "저녁 8시에 혈압약 1정 복용"
    """
    try:
        config = {"configurable": {"thread_id": "medicine_analysis"}}
        
        response = medicine_graph.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                }
            ]
        }, config=config)
        
        return MedicineReminderResponse(
            time=response.get("time", ""),
            medicine_name=response.get("medicine_name", ""),
            dosage=response.get("dosage", ""),
            needs_reminder=response.get("needs_reminder", False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "Medicine Reminder API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
