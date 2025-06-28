from langchain_google_genai import ChatGoogleGenerativeAI

def check_repetitive_speech(conversation_text: str, llm=None):
    """반복 발화 검사 함수 - 치매 판단용"""
    if llm is None:
        # LLM이 전달되지 않은 경우 기본값 반환
        return {
            "type": "repetitive_speech",
            "result": "normal",
            "confidence": 0.5,
            "details": "LLM이 설정되지 않아 기본값을 반환합니다."
        }
    
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

def check_memory_recall(conversation_text: str, llm=None):
    """기억력 검사 함수 - 치매 판단용"""
    if llm is None:
        # LLM이 전달되지 않은 경우 기본값 반환
        return {
            "type": "memory_recall",
            "result": "normal",
            "confidence": 0.5,
            "details": "LLM이 설정되지 않아 기본값을 반환합니다."
        }
    
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
