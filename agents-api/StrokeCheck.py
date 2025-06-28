import os
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

# 전역 변수 (main.py에서 설정됨)
faiss_vectorstore = None
clap_model = None
clap_processor = None

class AudioEmbeddings(Embeddings):
    """오디오 임베딩을 위한 커스텀 클래스"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def embed_documents(self, texts):
        """문서 임베딩 (FAISS 호환성을 위해 구현)"""
        embeddings = []
        for text in tqdm(texts, desc="Embedding audio files", unit="file"):
            try:
                embed = load_and_embed_audio(text, self.model, self.processor)
                if embed is not None:
                    embed_np = embed.squeeze().cpu().numpy()
                    embeddings.append(embed_np)
                else:
                    embeddings.append(np.zeros(512))
            except Exception as e:
                print(f"Error embedding {text}: {e}")
                embeddings.append(np.zeros(512))
        return embeddings
    
    def embed_query(self, text):
        """쿼리 임베딩"""
        try:
            embed = load_and_embed_audio(text, self.model, self.processor)
            if embed is not None:
                embed_np = embed.squeeze().cpu().numpy()
                return embed_np
            else:
                return np.zeros(512)
        except Exception as e:
            print(f"Error embedding query {text}: {e}")
            return np.zeros(512)

def load_and_embed_audio(file_path, model, processor):
    """WAV 파일을 로드하고 임베딩을 추출하는 함수"""
    try:
        # WAV 파일 로드
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        
        # 모노로 변환 (스테레오인 경우)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # CLAP 모델이 요구하는 48kHz로 리샘플링
        target_sample_rate = 48000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # 오디오 배열을 numpy로 변환
        audio_array = waveform.squeeze().numpy()
        
        # 프로세서를 사용하여 입력 준비
        inputs = processor(audios=audio_array, sampling_rate=sample_rate, return_tensors="pt")
        
        # 오디오 임베딩 추출
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
            # 임베딩 벡터 정규화 (L2 norm)
            audio_embed = F.normalize(audio_embed, p=2, dim=1)
        
        return audio_embed
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_faiss_vectorstore(save_path="./db/faiss", model=None, processor=None):
    """로컬에서 FAISS 벡터스토어 로드"""
    try:
        if not os.path.exists(save_path):
            print(f"Vectorstore path {save_path} does not exist")
            return None
        
        if model is None or processor is None:
            print("Model and processor are required to load vectorstore")
            return None
        
        audio_embeddings = AudioEmbeddings(model, processor)
        vectorstore = FAISS.load_local(save_path, audio_embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {save_path}")
        return vectorstore
        
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def check_stroke_with_faiss(audio_file_path: str):
    """FAISS 벡터DB를 사용한 뇌졸중 검사 함수"""
    global faiss_vectorstore, clap_model, clap_processor
    
    if faiss_vectorstore is None or clap_model is None or clap_processor is None:
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": "FAISS 벡터스토어 또는 CLAP 모델이 로드되지 않음",
            "similar_count": 0
        }
    
    if not os.path.exists(audio_file_path):
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": "음성 파일이 존재하지 않음",
            "similar_count": 0
        }
    
    try:
        print(f"Checking stroke with FAISS for {audio_file_path}")
        
        # 쿼리 임베딩 생성
        query_embedding = faiss_vectorstore.embedding_function.embed_query(audio_file_path)
        
        # FAISS 인덱스에서 모든 결과 검색
        query_embedding_reshaped = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_vectorstore.index.search(query_embedding_reshaped, faiss_vectorstore.index.ntotal)
        
        # 유사도 점수 계산 (코사인 유사도)
        similarities = 1 - distances[0]
        
        # 0.5 이상 유사한 음성 개수 계산
        similar_count = sum(1 for sim in similarities if sim >= 0.5)
        
        # 뇌졸중 판단: 100개 이상이면 뇌졸중 의심
        stroke_suspicion = similar_count >= 100
        
        # 신뢰도 계산 (유사한 음성 비율 기반)
        confidence = min(similar_count / 100.0, 1.0) if stroke_suspicion else 0.0
        
        result = {
            "type": "stroke_check",
            "result": "stroke_suspicion" if stroke_suspicion else "normal",
            "confidence": float(confidence),
            "details": f"유사한 음성 {similar_count}개 발견 (임계값: 100개)",
            "similar_count": similar_count,
            "total_vectors": len(similarities),
            "similarity_scores": similarities.tolist()[:10]  # 상위 10개 점수만
        }
        
        print(f"Stroke check result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in stroke check: {e}")
        return {
            "type": "stroke_check",
            "result": "error",
            "confidence": 0.0,
            "details": f"뇌졸중 검사 오류: {str(e)}",
            "similar_count": 0
        }

def set_global_models(vectorstore, model, processor):
    """전역 변수 설정 함수"""
    global faiss_vectorstore, clap_model, clap_processor
    faiss_vectorstore = vectorstore
    clap_model = model
    clap_processor = processor
    print("StrokeCheck 모듈의 전역 변수가 설정되었습니다.")

def get_global_models():
    """전역 변수 반환 함수"""
    global faiss_vectorstore, clap_model, clap_processor
    return faiss_vectorstore, clap_model, clap_processor
