# pip install bitsandbytes accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xxl", device_map="auto", load_in_8bit=True
)

input_text = "Question: 世界の人口はおよそ何人ですか？\n Answer: Let's think step by step."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids,max_length=1000,min_length=50, num_beams=4)

for out in outputs:
    print(tokenizer.decode(out, skip_special_tokens=True))