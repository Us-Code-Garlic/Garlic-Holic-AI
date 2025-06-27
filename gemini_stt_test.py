import os
import pyaudio
import wave
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import speech

load_dotenv()

# 1. GCP Speech-to-Text 클라이언트 초기화
try:
    speech_client = speech.SpeechClient()
except Exception as e:
    print("GCP Speech-to-Text 클라이언트 초기화에 실패했습니다.")
    print("Google Cloud 인증이 설정되어 있는지 확인해주세요.")
    print("다음 중 하나의 방법으로 인증을 설정하세요:")
    print("1. GOOGLE_APPLICATION_CREDENTIALS 환경 변수 설정")
    print("2. gcloud auth application-default login 실행")
    print(f"오류: {e}")
    exit()

# 2. Gemini 클라이언트 초기화
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception as e:
    print("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("`.env` 파일에 키를 추가하거나 환경 변수를 설정해주세요.")
    print(f"오류: {e}")
    exit()

# 3. 오디오 녹음 설정
FORMAT = pyaudio.paInt16  # 16-bit 오디오 포맷
CHANNELS = 1              # 모노 채널 (마이크는 보통 모노)
RATE = 16000              # 샘플링 레이트 (Hz). GCP STT가 선호하는 값입니다.
CHUNK = 1024              # 한 번에 읽을 프레임 크기
RECORD_SECONDS = 5        # 녹음 시간 (초)
WAVE_OUTPUT_FILENAME = Path(__file__).parent / "recorded_audio.wav"

def record_audio():
    """헤드셋으로 오디오를 녹음합니다."""
    audio = pyaudio.PyAudio()

    # PyAudio 스트림 시작 (마이크 입력)
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"마이크를 열 수 없습니다. 마이크가 연결되어 있는지 확인해주세요. 오류: {e}")
        audio.terminate()
        return None

    print(f"녹음을 시작합니다. {RECORD_SECONDS}초 동안 말씀해주세요...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음이 완료되었습니다.")

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 녹음된 오디오를 .wav 파일로 저장
    with wave.open(str(WAVE_OUTPUT_FILENAME), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"오디오를 '{WAVE_OUTPUT_FILENAME}' 파일로 저장했습니다.")
    return WAVE_OUTPUT_FILENAME

def transcribe_audio_with_gcp(audio_file_path):
    """GCP Speech-to-Text API를 사용하여 오디오를 텍스트로 변환합니다."""
    try:
        # 오디오 파일 읽기
        with open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()

        # 오디오 설정
        audio = speech.RecognitionAudio(content=content)
        
        # 인식 설정
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="ko-KR",  # 한국어 설정
            enable_automatic_punctuation=True,  # 자동 문장부호 추가
            model="latest_long"  # 긴 오디오에 최적화된 모델
        )

        # 음성 인식 요청
        response = speech_client.recognize(config=config, audio=audio)

        # 결과 처리
        if response.results:
            # 가장 신뢰도가 높은 결과 반환
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            return transcript.strip()
        else:
            print("음성을 인식할 수 없습니다.")
            return None
            
    except FileNotFoundError:
        print(f"오류: 녹음된 파일을 찾을 수 없습니다. ({audio_file_path})")
        return None
    except Exception as e:
        print(f"GCP Speech-to-Text API 호출 중 오류가 발생했습니다: {e}")
        return None

def get_gemini_response(user_input):
    """Gemini를 사용하여 사용자 입력에 대한 응답을 생성합니다."""
    try:
        # 사용자 입력에 대한 응답 생성
        response = llm.invoke(f"사용자가 말한 내용: '{user_input}'\n\n이에 대해 친근하고 도움이 되는 답변을 해주세요.")
        return response.content
    except Exception as e:
        print(f"Gemini 응답 생성 중 오류가 발생했습니다: {e}")
        return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

def main():
    """메인 함수: 음성 녹음 -> GCP STT -> Gemini 응답"""
    print("="*50)
    print("음성 인식 + Gemini AI 챗봇 (GCP STT 사용)")
    print("5초간 음성을 입력하면 Gemini가 응답합니다.")
    print("="*50)
    
    while True:
        # 사용자 입력을 받아 녹음 시작
        user_input = input("\n녹음을 시작하려면 Enter를 누르세요 (종료하려면 'q' 입력 후 Enter): ")
        if user_input.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        
        # 1. 오디오 녹음
        audio_file = record_audio()
        if not audio_file:
            continue
        
        # 2. GCP STT로 텍스트 변환
        print("음성을 텍스트로 변환 중...")
        transcribed_text = transcribe_audio_with_gcp(audio_file)
        
        if not transcribed_text:
            print("음성 인식에 실패했습니다. 다시 시도해주세요.")
            continue
        
        print(f"\n--- 음성 인식 결과 ---")
        print(f"인식된 텍스트: {transcribed_text}")
        print("----------------------")
        
        # 3. Gemini로 응답 생성
        print("Gemini가 응답을 생성 중...")
        gemini_response = get_gemini_response(transcribed_text)
        
        print(f"\n--- Gemini 응답 ---")
        print(f"AI: {gemini_response}")
        print("-------------------")

if __name__ == "__main__":
    main()
