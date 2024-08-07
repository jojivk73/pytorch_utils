

from transformers import AutoTokenizer, BertForMaskedLM
import torch

print("---------------------XPU Check:--------------------------")
print(torch.xpu.get_device_properties(0))
print("-----------------------Done------------------------------")
 
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = BertForMaskedLM.from_pretrained("google-bert/bert-large-uncased").to("xpu")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt").to("xpu")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

print("\n Answer :", tokenizer.decode(predicted_token_id))

