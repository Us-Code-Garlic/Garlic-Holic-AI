from typing import Literal, Optional
from pydantic import BaseModel, Field

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
    iteration_count: int = Field(..., description="반복 횟수")

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