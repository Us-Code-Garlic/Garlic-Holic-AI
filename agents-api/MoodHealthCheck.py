"""
기분 및 건강체크 모듈

이 모듈은 사용자의 기분과 건강 상태를 분석하고 응답을 생성하는 기능을 제공합니다.
"""

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from model import MoodHealthInfo
from states import State

# LLM 초기화 (main.py에서 import)
llm = None

def set_llm_model(model):
    """LLM 모델을 설정하는 함수"""
    global llm
    llm = model

def analyze_mood_health(message: str):
    """기분 및 건강 상태를 분석하는 함수"""
    if not llm:
        raise ValueError("LLM 모델이 설정되지 않았습니다. set_llm_model()을 먼저 호출하세요.")
    
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

def mood_health_node(state: State) -> Command[Literal["supervisor"]]:
    """기분 및 건강체크 서비스 - 실행 후 supervisor로 돌아감"""
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
        
        # Agent 응답을 DB에 저장 (main.py의 함수 사용)
        if user_id:
            from main import save_chat_to_db
            agent_response = f"기분 및 건강체크 완료: {mood_health_result['mood']} - {mood_health_result['health_status']}"
            save_chat_to_db(user_id, agent_response, "agent")
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["mood_health_agent"]
        
        # 누적 결과에 추가 (새로운 리스트 생성)
        current_all_mood_results = state.get("all_mood_results", [])
        current_all_health_results = state.get("all_health_results", [])
        new_all_mood_results = current_all_mood_results + [{
            "mood": mood_health_result["mood"],
            "confidence": mood_health_result["confidence"]
        }]
        new_all_health_results = current_all_health_results + [{
            "status": mood_health_result["health_status"],
            "confidence": mood_health_result["confidence"]
        }]
        
        print(f"기분/건강체크 완료 - supervisor로 돌아감")
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
                },
                "executed_agents": new_executed_agents,
                "all_mood_results": new_all_mood_results,
                "all_health_results": new_all_health_results
            },
            goto="supervisor"  # supervisor로 돌아감
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
            from main import save_chat_to_db
            error_response = f"기분 및 건강체크 오류: {str(e)}"
            save_chat_to_db(user_id, error_response, "agent")
        
        # 현재 실행된 Agent를 executed_agents에 추가 (새로운 리스트 생성)
        current_executed_agents = state.get("executed_agents", [])
        new_executed_agents = current_executed_agents + ["mood_health_agent"]
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"기분 및 건강체크 오류: {str(e)}",
                        name="mood_health_agent"
                    )
                ],
                "mood_result": {"mood": "평범", "confidence": 0.0},
                "health_result": {"status": "분석 오류", "confidence": 0.0},
                "executed_agents": new_executed_agents
            },
            goto="supervisor"  # 오류가 있어도 supervisor로 돌아감
        )