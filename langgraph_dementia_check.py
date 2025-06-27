import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from dotenv import load_dotenv
import sqlite3
import json
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# 1. 상태 정의
class DementiaCheckState(TypedDict):
    patient_id: str
    current_audio_file: str  # 현재 음성 파일 경로
    current_text: str  # 현재 대화 텍스트
    messages: Annotated[List[BaseMessage], operator.add]
    check_results: Annotated[List[dict], operator.add]  # 각 노드의 검사 결과
    final_diagnosis: str  # 최종 진단 결과 (yes/no)

# 2. 데이터베이스 초기화
def init_database():
    """치매 검사용 데이터베이스 초기화"""
    conn = sqlite3.connect('dementia_check.db')
    cursor = conn.cursor()
    
    # 대화 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            conversation_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 음성 임베딩 테이블 (정상 발음 기준)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file_path TEXT,
            embedding_data BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# 3. 음성 분석 노드 (발음 어눌함 검사)
def check_pronunciation_clarity(state: DementiaCheckState):
    """발음이 어눌한지 검사하는 노드"""
    current_audio_file = state["current_audio_file"]
    patient_id = state["patient_id"]
    
    try:
        # 1. 현재 음성 파일에서 MFCC 특징 추출
        y, sr = librosa.load(current_audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        current_embedding = np.mean(mfcc, axis=1)
        
        # 2. 데이터베이스에서 정상 발음 기준 임베딩들 가져오기
        conn = sqlite3.connect('dementia_check.db')
        cursor = conn.cursor()
        cursor.execute('SELECT embedding_data FROM voice_embeddings')
        baseline_embeddings = cursor.fetchall()
        conn.close()
        
        if not baseline_embeddings:
            # 기준 데이터가 없으면 정상으로 판단
            similarity_score = 0.8
        else:
            # 유사도 계산
            similarities = []
            for baseline_embedding_blob in baseline_embeddings:
                baseline_embedding = np.frombuffer(baseline_embedding_blob[0])
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1), 
                    baseline_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            
            similarity_score = np.mean(similarities)
        
        # 3. 임계값 기반 판단 (0.7 미만이면 어눌함)
        is_unclear = similarity_score < 0.7
        
        return {
            "check_results": [{
                "type": "pronunciation_clarity",
                "result": "unclear" if is_unclear else "clear",
                "confidence": similarity_score,
                "details": f"발음 유사도 점수: {similarity_score:.3f}"
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "pronunciation_clarity",
                "result": "error",
                "confidence": 0.0,
                "details": f"음성 분석 오류: {str(e)}"
            }]
        }

# 4. 기억력 검사 노드 (며칠 전 일 기억 못함)
def check_memory_recall(state: DementiaCheckState):
    """며칠 전의 일을 기억하지 못하는지 검사하는 노드"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    try:
        # 1. 데이터베이스에서 이전 대화 내역 가져오기
        conn = sqlite3.connect('dementia_check.db')
        cursor = conn.cursor()
        
        # 3일 전부터의 대화 내역 조회
        three_days_ago = datetime.now() - timedelta(days=3)
        cursor.execute('''
            SELECT conversation_text, timestamp 
            FROM conversation_history 
            WHERE patient_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (patient_id, three_days_ago))
        
        previous_conversations = cursor.fetchall()
        conn.close()
        
        if not previous_conversations:
            # 이전 대화 기록이 없으면 정상으로 판단
            memory_score = 0.8
        else:
            # 2. 현재 대화에서 과거 일에 대한 언급 분석
            memory_prompt = f"""
            다음은 환자의 현재 대화 내용입니다:
            "{current_text}"
            
            다음은 이전 대화 기록들입니다:
            {previous_conversations}
            
            현재 대화에서 3일 이내의 과거 일에 대한 언급이 있는지 분석하고,
            기억력 상태를 평가해주세요. 
            
            응답 형식:
            - 기억력 점수 (0.0-1.0): 
            - 분석 결과:
            """
            
            response = llm.invoke(memory_prompt)
            
            # 응답에서 점수 추출 (간단한 파싱)
            try:
                lines = response.content.split('\n')
                score_line = [line for line in lines if '기억력 점수' in line][0]
                memory_score = float(score_line.split(':')[1].strip())
            except:
                memory_score = 0.5
        
        # 3. 임계값 기반 판단 (0.6 미만이면 기억력 문제)
        has_memory_issue = memory_score < 0.6
        
        return {
            "check_results": [{
                "type": "memory_recall",
                "result": "impaired" if has_memory_issue else "normal",
                "confidence": memory_score,
                "details": f"기억력 점수: {memory_score:.3f}"
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "memory_recall",
                "result": "error",
                "confidence": 0.0,
                "details": f"기억력 검사 오류: {str(e)}"
            }]
        }

# 5. 반복 발화 검사 노드 (같은 문장 반복)
def check_repetitive_speech(state: DementiaCheckState):
    """같은 문장을 반복해서 이야기하는지 검사하는 노드"""
    current_text = state["current_text"]
    patient_id = state["patient_id"]
    
    try:
        # 1. 데이터베이스에서 이전 대화 내역 가져오기
        conn = sqlite3.connect('dementia_check.db')
        cursor = conn.cursor()
        
        # 최근 10개 대화 내역 조회
        cursor.execute('''
            SELECT conversation_text 
            FROM conversation_history 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', (patient_id,))
        
        previous_conversations = cursor.fetchall()
        conn.close()
        
        if not previous_conversations:
            # 이전 대화 기록이 없으면 정상으로 판단
            repetition_score = 0.2
        else:
            # 2. 문장 유사도 분석
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # 현재 문장 임베딩
            current_embedding = model.encode([current_text])
            
            # 이전 문장들과 유사도 계산
            similarities = []
            for prev_conv in previous_conversations:
                prev_text = prev_conv[0]
                prev_embedding = model.encode([prev_text])
                similarity = cosine_similarity(current_embedding, prev_embedding)[0][0]
                similarities.append(similarity)
            
            # 높은 유사도 문장이 있으면 반복으로 판단
            max_similarity = max(similarities) if similarities else 0
            repetition_score = max_similarity
        
        # 3. 임계값 기반 판단 (0.8 이상이면 반복)
        is_repetitive = repetition_score > 0.8
        
        return {
            "check_results": [{
                "type": "repetitive_speech",
                "result": "repetitive" if is_repetitive else "normal",
                "confidence": repetition_score,
                "details": f"반복 유사도 점수: {repetition_score:.3f}"
            }]
        }
        
    except Exception as e:
        return {
            "check_results": [{
                "type": "repetitive_speech",
                "result": "error",
                "confidence": 0.0,
                "details": f"반복 발화 검사 오류: {str(e)}"
            }]
        }

# 6. Map 함수: 각 검사 노드에 작업 할당
def map_to_checkers(state: DementiaCheckState):
    """각 검사 노드에 작업을 할당하는 함수"""
    return [
        Send("pronunciation_checker", {"patient_id": state["patient_id"], "current_audio_file": state["current_audio_file"]}),
        Send("memory_checker", {"patient_id": state["patient_id"], "current_text": state["current_text"]}),
        Send("repetition_checker", {"patient_id": state["patient_id"], "current_text": state["current_text"]})
    ]

# 7. Combine 함수: 검사 결과 통합 및 최종 진단
def combine_diagnosis(state: DementiaCheckState):
    """모든 검사 결과를 통합하여 최종 치매 진단"""
    check_results = state["check_results"]
    
    # 각 검사 결과 분석
    pronunciation_result = None
    memory_result = None
    repetition_result = None
    
    for result in check_results:
        if result["type"] == "pronunciation_clarity":
            pronunciation_result = result
        elif result["type"] == "memory_recall":
            memory_result = result
        elif result["type"] == "repetitive_speech":
            repetition_result = result
    
    # 진단 로직
    dementia_indicators = 0
    total_confidence = 0
    details = []
    
    # 발음 어눌함 검사
    if pronunciation_result and pronunciation_result["result"] == "unclear":
        dementia_indicators += 1
        details.append(f"발음 어눌함 (신뢰도: {pronunciation_result['confidence']:.3f})")
    total_confidence += pronunciation_result["confidence"] if pronunciation_result else 0
    
    # 기억력 문제 검사
    if memory_result and memory_result["result"] == "impaired":
        dementia_indicators += 1
        details.append(f"기억력 저하 (신뢰도: {memory_result['confidence']:.3f})")
    total_confidence += memory_result["confidence"] if memory_result else 0
    
    # 반복 발화 검사
    if repetition_result and repetition_result["result"] == "repetitive":
        dementia_indicators += 1
        details.append(f"반복 발화 (신뢰도: {repetition_result['confidence']:.3f})")
    total_confidence += repetition_result["confidence"] if repetition_result else 0
    
    # 최종 진단 (2개 이상의 지표가 있으면 치매 의심)
    final_diagnosis = "yes" if dementia_indicators >= 2 else "no"
    avg_confidence = total_confidence / 3 if total_confidence > 0 else 0
    
    diagnosis_summary = f"""
    치매 검사 결과:
    - 발음 어눌함: {pronunciation_result['result'] if pronunciation_result else 'N/A'}
    - 기억력 저하: {memory_result['result'] if memory_result else 'N/A'}
    - 반복 발화: {repetition_result['result'] if repetition_result else 'N/A'}
    
    치매 의심 지표 수: {dementia_indicators}/3
    평균 신뢰도: {avg_confidence:.3f}
    
    최종 진단: {'치매 의심' if final_diagnosis == 'yes' else '정상'}
    """
    
    return {
        "final_diagnosis": final_diagnosis,
        "messages": [("diagnosis", diagnosis_summary)]
    }

# 8. 그래프 구성
def create_dementia_check_graph():
    """치매 검사 그래프 생성"""
    workflow = StateGraph(DementiaCheckState)
    
    # 각 검사 노드 추가
    workflow.add_node("pronunciation_checker", check_pronunciation_clarity)
    workflow.add_node("memory_checker", check_memory_recall)
    workflow.add_node("repetition_checker", check_repetitive_speech)
    
    # Combine 노드 추가
    workflow.add_node("combine", combine_diagnosis)
    
    # START에서 map_to_checkers로 가는 조건부 엣지 추가
    workflow.add_conditional_edges(
        START,
        map_to_checkers,
        {
            "pronunciation_checker": "pronunciation_checker",
            "memory_checker": "memory_checker",
            "repetition_checker": "repetition_checker"
        }
    )
    
    # 각 검사 노드의 결과를 combine으로 연결
    for checker in ["pronunciation_checker", "memory_checker", "repetition_checker"]:
        workflow.add_edge(checker, "combine")
    
    workflow.add_edge("combine", END)
    
    return workflow.compile()

# 9. 대화 기록 저장 함수
def save_conversation(patient_id: str, conversation_text: str):
    """대화 기록을 데이터베이스에 저장"""
    conn = sqlite3.connect('dementia_check.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversation_history (patient_id, conversation_text)
        VALUES (?, ?)
    ''', (patient_id, conversation_text))
    conn.commit()
    conn.close()

# 10. 메인 실행 함수
def run_dementia_check(patient_id: str, audio_file_path: str, conversation_text: str):
    """치매 검사 실행"""
    # 데이터베이스 초기화
    init_database()
    
    # 대화 기록 저장
    save_conversation(patient_id, conversation_text)
    
    # 그래프 생성
    app = create_dementia_check_graph()
    
    # 입력 데이터 준비
    inputs = {
        "patient_id": patient_id,
        "current_audio_file": audio_file_path,
        "current_text": conversation_text,
        "messages": [],
        "check_results": [],
        "final_diagnosis": ""
    }
    
    # 그래프 실행
    config = {"recursion_limit": 50}
    result = app.invoke(inputs, config)
    
    return result

# 11. 테스트 실행 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_patient_id = "patient_001"
    test_audio_file = "sample_audio.mp3"  # 실제 음성 파일 경로
    test_conversation = "어제 병원에 갔는데 의사 선생님이 뭐라고 하셨는지 기억이 안 나요. 어제 병원에 갔는데 의사 선생님이 뭐라고 하셨는지 기억이 안 나요."
    
    # 치매 검사 실행
    result = run_dementia_check(test_patient_id, test_audio_file, test_conversation)
    
    print("=== 치매 검사 결과 ===")
    print(f"최종 진단: {result['final_diagnosis']}")
    print(f"메시지: {result['messages']}")
    print(f"검사 결과: {result['check_results']}")
