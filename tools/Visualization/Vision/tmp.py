'''
测试 PyTorch 中的 forward hook
'''

import torch
import torch.nn as nn

# 定义一个处理中间层输出的函数
def forward_hook(module, input, output):
    # input: 层的输入（元组）
    # output: 层的输出
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}\n")

# 示例网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 26 * 26, 10)  # 假设输入尺寸为 28x28

        # 注册前向 Hook
        self.conv.register_forward_hook(forward_hook)
        self.fc.register_forward_hook(forward_hook)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 测试
model = Net()
dummy_input = torch.randn(1, 3, 28, 28)
output = model(dummy_input)