import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第一层卷积 + 池化
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 第二层卷积 + 池化
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 第三层卷积 + 池化
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 224/2/2/2 = 28
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # 第一层
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层
        x = self.pool(F.relu(self.conv2(x)))
        # 第三层
        x = self.pool(F.relu(self.conv3(x)))

        # 展平
        x = x.view(-1, 64 * 28 * 28)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 创建模型
if __name__ == '__main__':
    model = SimpleCNN()
    print(model)