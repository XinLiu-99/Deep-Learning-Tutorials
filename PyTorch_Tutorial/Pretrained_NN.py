# %%

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.randint(0, 1000, (1,)) 

# %%
# 定义损失函数（交叉熵损失）
criterion = torch.nn.CrossEntropyLoss()  
# 加载一个优化器，它是一个SGD，学习率0.01，动量0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# %%
for i in range(50):
    optim.zero_grad()  # 梯度归零

    prediction = model(data)  # 重新计算前向传播
    loss = criterion(prediction, labels)  # 使用交叉熵损失计算损失
    loss.backward()  # 反向传播计算梯度
    optim.step()  # 优化器更新参数

    if i % 10 == 0:
        print(f'Step {i}, Loss: {loss.item()}')

# %%
    
