
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

print("---------------------XPU Check:--------------------------")
print(torch.xpu.get_device_properties(0))
print("-----------------------Done------------------------------")
 
model = AutoModelForCausalLM.from_pretrained("gpt2").to("xpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "GPT2 is a model developed by OpenAI."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('xpu')

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("---------------------------Done-----------------------------------")
print(" Text generated:", gen_text)

