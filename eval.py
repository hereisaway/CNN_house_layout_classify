import torch
import torchvision
from PIL import Image
from model import *

image_path = 'test_data/img.png'
image=Image.open(image_path)
image=image.convert('L')

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                          torchvision.transforms.ToTensor()])
image=transform(image)

net=SimpleCNN()
net.eval()
net.load_state_dict(torch.load('model/model9.pth')) 

output=net(image)
print(output.argmax(1))





