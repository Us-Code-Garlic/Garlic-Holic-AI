import os
import pyaudio
import wave
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. OpenAI 클라이언트 초기화
try:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
except TypeError:
    print("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("`.env` 파일에 키를 추가하거나 환경 변수를 설정해주세요.")
    exit()

# 2. 오디오 녹음 설정
FORMAT = pyaudio.paInt16  # 16-bit 오디오 포맷
CHANNELS = 1              # 모노 채널 (마이크는 보통 모노)
RATE = 16000              # 샘플링 레이트 (Hz). Whisper가 선호하는 값입니다.
CHUNK = 1024              # 한 번에 읽을 프레임 크기
RECORD_SECONDS = 5        # 녹음 시간 (초)
WAVE_OUTPUT_FILENAME = Path(__file__).parent / "recorded_audio.wav"

print("="*20)
print("오디오 장치 진단을 시작합니다.")
print(f"PyAudio 버전: {pyaudio.__version__}")
print("="*20)

p = pyaudio.PyAudio()

try:
    print("\n--- 기본 장치 정보 ---")
    default_input = p.get_default_input_device_info()
    print(f"기본 입력 장치: {default_input['name']} (Device ID: {default_input['index']})")
except IOError:
    print("기본 입력 장치를 찾을 수 없습니다. 사용 가능한 장치가 없는 것 같습니다.")


print("\n--- 전체 장치 목록 ---")
try:
    device_count = p.get_device_count()
    if device_count == 0:
        print("사용 가능한 오디오 장치가 없습니다.")
    else:
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            device_type = "입력" if device_info.get('maxInputChannels') > 0 else "출력"
            print(f"  [{device_type}] Device ID {i}: {device_info.get('name')}")
except Exception as e:
    print(f"장치 목록을 가져오는 중 오류 발생: {e}")


p.terminate()

print("="*20)
print("진단이 완료되었습니다.")
print("위 목록에 '[입력]' 장치가 보이지 않는다면, WSL에서 마이크를 사용할 수 있도록 추가 설정이 필요합니다.")
print("이 스크립트를 실행한 후 전체 출력 결과를 복사해서 알려주세요.")

def record_and_transcribe():
    """헤드셋으로 오디오를 녹음하고 텍스트로 변환합니다."""
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
        return

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

    print(f"오디오를 '{WAVE_OUTPUT_FILENAME}' 파일로 저장했습니다. 이제 텍스트로 변환합니다.")

    # 3. STT API 호출
    try:
        with open(WAVE_OUTPUT_FILENAME, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
            print("\n--- 음성 인식 결과 ---")
            print(transcription.text)
            print("----------------------\n")

    except FileNotFoundError:
        print(f"오류: 녹음된 파일을 찾을 수 없습니다. ({WAVE_OUTPUT_FILENAME})")
    except Exception as e:
        print(f"API 호출 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    while True:
        # 사용자 입력을 받아 녹음 시작
        user_input = input("녹음을 시작하려면 Enter를 누르세요 (종료하려면 'q' 입력 후 Enter): ")
        if user_input.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        record_and_transcribe()
