
import torch
import torchvision.models as models

print("---------------------XPU Check:--------------------------")
print(torch.xpu.get_device_properties(0))
print("-----------------------Done------------------------------")
 
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
######## code changes #######

with torch.no_grad():
  for i in range(1000):
    print(i)
    model(data)

print("Execution finished")
