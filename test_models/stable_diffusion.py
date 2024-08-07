import torch
from diffusers import StableDiffusionPipeline

print("---------------------XPU Check:--------------------------")
print(torch.xpu.get_device_properties(0))
print("-----------------------Done------------------------------")
 
model_id = "CompVis/stable-diffusion-v1-4"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("xpu")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
