from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = [
    "en: Take the predic caption of trained model."
    ]

outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
from transformers import VitsModel, AutoTokenizer
import torch
import torchaudio
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

text = "bản tiếng Việt của output caption"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
torchaudio.save("output.wav", output.squeeze(0), 22050)