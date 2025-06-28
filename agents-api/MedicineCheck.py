"""
복약 체크 모듈
복약 정보 분석 및 응답 생성을 담당하는 모듈
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from model import MedicineInfo

# 전역 LLM 모델 변수
llm_model = None

def set_llm_model(model):
    """LLM 모델을 설정하는 함수"""
    global llm_model
    llm_model = model

def analyze_medicine(message: str):
    """복약 정보를 분석하는 함수"""
    if not llm_model:
        raise ValueError("LLM 모델이 설정되지 않았습니다. set_llm_model()을 먼저 호출하세요.")
    
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
    
    model_with_structured_output = llm_model.with_structured_output(MedicineInfo)
    
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

def medicine_node(state) -> Command[Literal["supervisor"]]:
    """복약 알림 서비스 - 실행 후 supervisor로 돌아감"""
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
        
        # Agent 응답을 DB에 저장 (main.py의 save_chat_to_db 함수 사용)
        if user_id:
            agent_response = f"복약 정보 추출 완료: {medicine_result['medicine_name']} {medicine_result['dosage']} - {medicine_result['time']}"
            # save_chat_to_db 함수는 main.py에서 import하여 사용
            from main import save_chat_to_db
            save_chat_to_db(user_id, agent_response, "agent")
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["medicine_agent"]
        
        # 누적 결과에 추가 (새로운 리스트 생성)
        current_all_medicine_results = state.get("all_medicine_results", [])
        new_all_medicine_results = current_all_medicine_results + [medicine_result]
        
        print(f"복약 분석 완료 - supervisor로 돌아감")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"복약 정보 추출 완료: {medicine_result['medicine_name']} {medicine_result['dosage']} - {medicine_result['time']}",
                        name="medicine_agent"
                    )
                ],
                "medicine_result": medicine_result,
                "executed_agents": new_executed_agents,
                "all_medicine_results": new_all_medicine_results
            },
            goto="supervisor"  # supervisor로 돌아감
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
            from main import save_chat_to_db
            save_chat_to_db(user_id, error_response, "agent")
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["medicine_agent"]
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"복약 분석 오류: {str(e)}",
                        name="medicine_agent"
                    )
                ],
                "medicine_result": error_result,
                "executed_agents": new_executed_agents
            },
            goto="supervisor"  # 오류가 있어도 supervisor로 돌아감
        )