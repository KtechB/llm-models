from transformers import AutoModelForCausalLM, AutoTokenizer


checkpoint = "bigscience/bloomz-7b1"  # "bigscience/bloom"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype="auto", device_map="auto", load_in_8bit=True
)

inputs = tokenizer.encode(
    "Question: What is Prompt engineering?\n Answer: Let's think step by step.",
    return_tensors="pt",
).to("cuda")
outputs = model.generate(inputs, max_new_tokens=100, min_new_tokens=30)

for out in outputs:
    print(tokenizer.decode(out))
