from transformers import ClapModel, ClapProcessor
import torchaudio
import torch
import torch.nn.functional as F

def load_and_embed_audio(file_path, model, processor):
    """WAV 파일을 로드하고 임베딩을 추출하는 함수"""
    # WAV 파일 로드
    waveform, sample_rate = torchaudio.load(file_path)
    
    # 모노로 변환 (스테레오인 경우)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 오디오 배열을 numpy로 변환
    audio_array = waveform.squeeze().numpy()
    
    # 프로세서를 사용하여 입력 준비
    inputs = processor(audios=audio_array, return_tensors="pt")
    
    # 오디오 임베딩 추출
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
    
    return audio_embed

# WAV 파일 경로들
audio_file_1 = "extracted_audio_wav/train-00000-of-00018_0.wav"
audio_file_2 = "extracted_audio_wav/train-00000-of-00018_1.wav"

# 모델과 프로세서 로드
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# 두 오디오 파일의 임베딩 추출
embed_1 = load_and_embed_audio(audio_file_1, model, processor)
embed_2 = load_and_embed_audio(audio_file_2, model, processor)

print(f"Embedding 1 shape: {embed_1.shape}")
print(f"Embedding 2 shape: {embed_2.shape}")

# 코사인 유사도 계산
cosine_sim = F.cosine_similarity(embed_1, embed_2, dim=-1)

print(f"\nCosine similarity between the two audio files: {cosine_sim.item():.4f}")

# 추가 정보 출력
print(f"\nFile 1: {audio_file_1}")
print(f"File 2: {audio_file_2}")
print(f"Similarity score: {cosine_sim.item():.4f} (0=completely different, 1=identical)")