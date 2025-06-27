from datasets import load_dataset
from transformers import ClapModel, ClapProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[0]

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt")
audio_embed = model.get_audio_features(**inputs)

print(audio_embed)