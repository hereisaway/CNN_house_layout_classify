import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from dataset import *
from model import *

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
dataset = (MyDataset('data','0',transform=transform)
            +MyDataset('data','1',transform=transform)
            +MyDataset('data','2',transform=transform))

torch.manual_seed(42)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% 作为训练集
test_size = dataset_size - train_size  # 剩余 20% 作为测试集

# 使用 random_split 分割数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)  # 将模型转移到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for images, labels in train_loader:
        # print(images,labels)
        images, labels = images.to(device), labels.to(device)  # 将数据转移到 GPU
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

    # 测试阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据转移到 GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), 'model/model{}.pth'.format(epoch))