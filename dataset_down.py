from datasets import load_dataset
import os

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yoona-J/ASR_Stroke_Dataset")

# 데이터셋 정보 출력
print("Dataset info:")
print(ds)

# 캐시 디렉토리 확인
from datasets import config
print(f"\nCache directory: {config.HF_DATASETS_CACHE}")

# 데이터셋을 로컬 폴더로 저장
output_dir = "./ASR_Stroke_Dataset"
ds.save_to_disk(output_dir)

print(f"\nDataset saved to: {os.path.abspath(output_dir)}")

# 저장된 데이터셋 확인
print(f"\nSaved files:")
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")