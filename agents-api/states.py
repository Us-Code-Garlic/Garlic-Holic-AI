from typing import Literal, TypedDict, Optional, Annotated
from langgraph.graph import MessagesState, add_messages

# ==================== State 및 Router 정의 ====================

class State(MessagesState):
    next: str = "supervisor"
    service_type: Optional[str] = None
    patient_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    selected_agent: Optional[str] = None
    executed_agents: Annotated[list, add_messages] = []  # 실행된 Agent 목록
    iteration_count: int = 0  # 반복 횟수 추가
    dementia_result: Optional[dict] = None
    medicine_result: Optional[dict] = None
    mood_result: Optional[dict] = None
    health_result: Optional[dict] = None
    stroke_result: Optional[dict] = None
    user_id: Optional[str] = None
    # 누적 결과 저장을 위한 리스트 (add_messages 제거)
    all_dementia_results: list = []
    all_medicine_results: list = []
    all_mood_results: list = []
    all_health_results: list = []

members = ["dementia_agent", "medicine_agent", "mood_health_agent"]
options = members + ["__end__"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to END."""
    next: Literal[*options]