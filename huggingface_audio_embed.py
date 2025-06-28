import os
import glob
from transformers import ClapModel, ClapProcessor
import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings
from tqdm import tqdm

def load_and_embed_audio(file_path, model, processor):
    """WAV 파일을 로드하고 임베딩을 추출하는 함수"""
    try:
        # WAV 파일 로드 (sampling_rate 명시)
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
        
        # 프로세서를 사용하여 입력 준비 (sampling_rate 명시)
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

class AudioEmbeddings(Embeddings):
    """오디오 임베딩을 위한 커스텀 클래스"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def embed_documents(self, texts):
        """문서 임베딩 (FAISS 호환성을 위해 구현)"""
        embeddings = []
        # tqdm으로 진행상황 표시
        for text in tqdm(texts, desc="Embedding audio files", unit="file"):
            # text는 실제로는 파일 경로
            try:
                embed = load_and_embed_audio(text, self.model, self.processor)
                if embed is not None:
                    embed_np = embed.squeeze().cpu().numpy()
                    embeddings.append(embed_np)
                else:
                    # 에러 시 0으로 채워진 임베딩 반환
                    embeddings.append(np.zeros(512))  # CLAP 임베딩 크기
            except Exception as e:
                print(f"Error embedding {text}: {e}")
                # 에러 시 0으로 채워진 임베딩 반환
                embeddings.append(np.zeros(512))  # CLAP 임베딩 크기
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

def create_audio_documents(audio_folder):
    """오디오 파일들을 Document 객체로 변환"""
    wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {audio_folder}")
        return []
    
    documents = []
    # tqdm으로 진행상황 표시
    for i, file_path in enumerate(tqdm(wav_files, desc="Creating documents", unit="file")):
        doc = Document(
            page_content=file_path,  # 실제 내용은 파일 경로
            metadata={
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "file_index": i,
                "type": "audio"
            }
        )
        documents.append(doc)
    
    return documents

def setup_faiss_vectorstore(audio_folder, model, processor):
    """FAISS 벡터스토어 설정 및 오디오 파일 임베딩"""
    # 오디오 파일들을 Document 객체로 변환
    documents = create_audio_documents(audio_folder)
    
    if not documents:
        print("No documents to process")
        return None
    
    print(f"Processing {len(documents)} audio files...")
    
    # 커스텀 오디오 임베딩 모델 생성
    audio_embeddings = AudioEmbeddings(model, processor)
    
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(
        documents,
        embedding=audio_embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    print(f"FAISS vectorstore created with {len(documents)} documents")
    print(f"Distance strategy: {vectorstore.distance_strategy}")
    
    return vectorstore

def save_faiss_vectorstore(vectorstore, save_path="./db/faiss"):
    """FAISS 벡터스토어를 로컬에 저장"""
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)
        
        # 벡터스토어 저장
        vectorstore.save_local(save_path)
        print(f"Vectorstore saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error saving vectorstore: {e}")
        return False

def load_faiss_vectorstore(save_path="./db/faiss", model=None, processor=None):
    """로컬에서 FAISS 벡터스토어 로드"""
    try:
        if not os.path.exists(save_path):
            print(f"Vectorstore path {save_path} does not exist")
            return None
        
        # AudioEmbeddings 인스턴스 생성 (모델과 프로세서 필요)
        if model is None or processor is None:
            print("Model and processor are required to load vectorstore")
            return None
        
        audio_embeddings = AudioEmbeddings(model, processor)
        
        # 벡터스토어 로드 (allow_dangerous_deserialization=True 추가)
        vectorstore = FAISS.load_local(save_path, audio_embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {save_path}")
        return vectorstore
        
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def search_similar_audio_faiss(query_audio_path, vectorstore, top_k=10, min_score=0.0):
    """FAISS를 사용한 유사도 검색 (점수 포함, 최소 점수 필터링)"""
    if not os.path.exists(query_audio_path):
        print(f"Query audio file {query_audio_path} not found!")
        return []
    
    print(f"Searching with k={top_k}...")
    
    # 직접 FAISS 인덱스에 접근하여 모든 결과 가져오기
    try:
        # 쿼리 임베딩 생성 (embedding_function 사용)
        query_embedding = vectorstore.embedding_function.embed_query(query_audio_path)
        
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"FAISS index size: {vectorstore.index.ntotal}")
        
        # FAISS 인덱스에서 직접 검색
        query_embedding_reshaped = np.array([query_embedding], dtype=np.float32)
        distances, indices = vectorstore.index.search(query_embedding_reshaped, vectorstore.index.ntotal)
        
        print(f"FAISS search returned {len(indices[0])} results")
        
        # 결과를 더 자세한 형태로 변환
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # FAISS 코사인 거리를 코사인 유사도로 변환
            similarity_score = 1 - distance
            
            # 최소 점수 필터링 (이제 유사도 기준)
            if similarity_score >= min_score:
                # docstore에서 문서 정보 가져오기
                doc_id = list(vectorstore.docstore._dict.keys())[idx]
                doc = vectorstore.docstore._dict[doc_id]
                
                # 메타데이터에서 정보 추출
                metadata = doc.metadata
                results.append({
                    "rank": i + 1,
                    "filename": metadata.get("filename", "Unknown"),
                    "file_path": metadata.get("file_path", "Unknown"),
                    "file_index": metadata.get("file_index", -1),
                    "similarity_score": similarity_score,  # 코사인 유사도 (0~1)
                    "distance_score": distance,  # FAISS 거리 점수
                    "content": doc.page_content,
                    "doc_id": doc_id,
                    "faiss_index": idx
                })
        
        print(f"After filtering (min_score={min_score}): {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Error in direct FAISS search: {e}")
        # 기존 방법으로 fallback
        docs_and_scores = vectorstore.similarity_search_with_score(query_audio_path, k=top_k)
        
        results = []
        for i, (doc, distance) in enumerate(docs_and_scores):
            similarity_score = 1 - distance
            
            if similarity_score >= min_score:
                metadata = doc.metadata
                results.append({
                    "rank": i + 1,
                    "filename": metadata.get("filename", "Unknown"),
                    "file_path": metadata.get("file_path", "Unknown"),
                    "file_index": metadata.get("file_index", -1),
                    "similarity_score": similarity_score,
                    "distance_score": distance,
                    "content": doc.page_content
                })
        
        return results

def main():
    # 모델과 프로세서 로드
    print("Loading CLAP model and processor...")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    # FAISS 벡터스토어 설정
    audio_folder = "extracted_audio_wav"
    save_path = "./db/faiss"
    
    # 기존 벡터스토어 확인 및 로드
    print("Checking for existing vectorstore...")
    vectorstore = load_faiss_vectorstore(save_path, model, processor)
    
    if vectorstore is None:
        print("Creating new FAISS vectorstore...")
        vectorstore = setup_faiss_vectorstore(audio_folder, model, processor)
        
        if vectorstore is not None:
            # 벡터스토어 저장
            print("Saving vectorstore...")
            save_success = save_faiss_vectorstore(vectorstore, save_path)
            if save_success:
                print("Vectorstore successfully created and saved")
            else:
                print("Failed to save vectorstore")
        else:
            print("Failed to create vectorstore")
            return
    else:
        print("Using existing vectorstore from local storage")
    
    # example.wav 파일로 유사도 검색
    query_audio = "example.wav"
    if os.path.exists(query_audio):
        print(f"\nSearching for audio similar to {query_audio}...")
        
        # 벡터스토어 정보 확인
        print(f"Vectorstore contains {len(vectorstore.docstore._dict)} documents")
        print(f"FAISS index contains {vectorstore.index.ntotal} vectors")
        
        # 전체 결과 검색 (FAISS 인덱스의 모든 벡터 수만큼)
        total_vectors = vectorstore.index.ntotal
        all_results = search_similar_audio_faiss(query_audio, vectorstore, top_k=total_vectors)
        
        # 더 낮은 임계값으로 필터링
        filtered_results_05 = [r for r in all_results if r['similarity_score'] >= 0.5]
        filtered_results_03 = [r for r in all_results if r['similarity_score'] >= 0.3]
        filtered_results_01 = [r for r in all_results if r['similarity_score'] >= 0.1]
        
        print(f"\n=== FAISS Similarity Results for {query_audio} ===")
        print(f"Total results found: {len(all_results)}")
        print(f"Results with similarity score >= 0.5: {len(filtered_results_05)}")
        print(f"Results with similarity score >= 0.3: {len(filtered_results_03)}")
        print(f"Results with similarity score >= 0.1: {len(filtered_results_01)}")
        
        # 최고 점수와 최저 점수 표시
        max_similarity = max(r['similarity_score'] for r in all_results)
        min_similarity = min(r['similarity_score'] for r in all_results)
        max_distance = max(r['distance_score'] for r in all_results)
        min_distance = min(r['distance_score'] for r in all_results)
        
        print(f"Similarity score range: {min_similarity:.4f} - {max_similarity:.4f}")
        print(f"Distance score range: {min_distance:.4f} - {max_distance:.4f}")
        
        # 넓은 범위의 점수 분포 출력
        score_ranges = {
            "0.9-1.0": len([r for r in all_results if 0.9 <= r['similarity_score'] < 1.0]),
            "0.8-0.9": len([r for r in all_results if 0.8 <= r['similarity_score'] < 0.9]),
            "0.7-0.8": len([r for r in all_results if 0.7 <= r['similarity_score'] < 0.8]),
            "0.6-0.7": len([r for r in all_results if 0.6 <= r['similarity_score'] < 0.7]),
            "0.5-0.6": len([r for r in all_results if 0.5 <= r['similarity_score'] < 0.6]),
            "0.4-0.5": len([r for r in all_results if 0.4 <= r['similarity_score'] < 0.5]),
            "0.3-0.4": len([r for r in all_results if 0.3 <= r['similarity_score'] < 0.4]),
            "0.2-0.3": len([r for r in all_results if 0.2 <= r['similarity_score'] < 0.3]),
            "0.1-0.2": len([r for r in all_results if 0.1 <= r['similarity_score'] < 0.2]),
            "0.0-0.1": len([r for r in all_results if 0.0 <= r['similarity_score'] < 0.1])
        }
        
        print("\nSimilarity score distribution:")
        for range_name, count in score_ranges.items():
            print(f"  {range_name}: {count} files")
        
        print(f"\nTop 10 results with highest similarity:")
        
        # 상위 10개만 표시 (유사도 높은 순으로 정렬)
        top_results = sorted(all_results, key=lambda x: x['similarity_score'], reverse=True)[:10]
        
        for i, result in enumerate(top_results):
            print(f"{i+1}. {result['filename']}")
            print(f"   File path: {result['file_path']}")
            print(f"   File index: {result['file_index']}")
            print(f"   Similarity score: {result['similarity_score']:.4f}")
            print(f"   Distance score: {result['distance_score']:.4f}")
            if 'faiss_index' in result:
                print(f"   FAISS index: {result['faiss_index']}")
            print()
        
        # 상위 5개 결과의 상세 정보
        print(f"\n=== Top 5 Most Similar Audio Files ===")
        top_5_results = sorted(all_results, key=lambda x: x['similarity_score'], reverse=True)[:5]
        
        for i, result in enumerate(top_5_results):
            print(f"{i+1}. {result['filename']}")
            print(f"   Similarity: {result['similarity_score']:.4f}")
            print(f"   Distance: {result['distance_score']:.4f}")
            print(f"   Path: {result['file_path']}")
            print(f"   Index: {result['file_index']}")
            if 'faiss_index' in result:
                print(f"   FAISS index: {result['faiss_index']}")
            print()
    
    else:
        print(f"Query audio file {query_audio} not found!")
    
    print("Done!")

if __name__ == "__main__":
    main()