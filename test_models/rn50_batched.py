
import torch
import torchvision.models as models

print("---------------------XPU Check:--------------------------")
print(torch.xpu.get_device_properties(0))
print("-----------------------Done------------------------------")
 
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()

all_data=[]
data_count = 100

print(" Generating Data....!")
for i in range(data_count):
  data = torch.rand(32, 3, 224, 224).to("xpu")
  all_data.append(data)

######## code changes #######
model = model.to("xpu")
######## code changes #######


print(" Starting inference Check....!")
with torch.no_grad():
  for i in range(data_count):
    print(" Batch :", i)
    data = all_data[i]
    model(data)

print("Execution finished")
