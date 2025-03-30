import torch
import torch.nn as nn

"""
论文链接：https://arxiv.org/abs/2404.13537
论文标题：利用高低频分解进行图像复原与增强（CVPR 2024）
"""

class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        """
            小卷积核策略：采用尺寸较小的卷积核（如3x3）可以更精细地捕获图像中的纹理与细微结构，
            这些细节构成了高频信息的核心部分。
        """
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        # ai缝合大王

        self.gelu = nn.GELU()

    def forward(self, x):

        """
            密集与残差连接机制：通过在各层之间构建密集连接，网络能够融合初级和高级特征，
            从而增强对图像细节的识别；同时，残差连接使网络直接学习输入与输出间的差分，
            专注于那些需要强化的细微部分。
        """

        # 输入张量形状示例：[1, 32, 64, 64]
        x1 = self.conv1(x)      # 输出尺寸仍为 [1, 32, 64, 64]
        x1 = self.gelu(x1 + x)  # 结合输入形成残差，保持尺寸不变

        x2 = self.conv2(x1)     # 经过第二个卷积层后尺寸：[1, 32, 64, 64]
        x2 = self.gelu(x2 + x1 + x)  # 密集连接，将前几层信息叠加

        x3 = self.conv3(x2)             # 第三层卷积输出：[1, 32, 64, 64]
        x3 = self.gelu(x3 + x2 + x1 + x) # 叠加前面所有层的特征

        x4 = self.conv4(x3)             # 第四层卷积处理后尺寸不变
        x4 = self.gelu(x4 + x3 + x2 + x1 + x)

        x5 = self.conv5(x4)                 # 第五层卷积，输出尺寸依然为 [1, 32, 64, 64]
        x5 = self.gelu(x5 + x4 + x3 + x2 + x1 + x)

        x6 = self.conv6(x5)                 # 最后一层卷积
        x6 = self.gelu(x6 + x5 + x4 + x3 + x2 + x1 + x)  # 最终融合所有层的信息
        # ai缝合大王

        return x6

# if __name__ == '__main__':

#     # 创建Dense模块实例，输入通道数设为32
#     model = Dense(32)

#     # 构造随机输入，形状为 (1, 32, 64, 64)
#     input = torch.randn(1, 32, 64, 64)

#     # 执行前向传播过程
#     output = model(input)
#     print('input_size:', input.size())
#     print('output_size:', output.size())
#     # ai缝合大王
