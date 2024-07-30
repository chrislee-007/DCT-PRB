import torch
import torch.nn as nn


class SimAMWithSlicing(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAMWithSlicing, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        #if height % 2 != 0 or width % 2 != 0:
            #print("Input height or width is odd, returning original input tensor.")
            #return x

        block_size_h = height // 2
        block_size_w = width // 2

        # 分割输入张量成四个局部区域
        block1 = x[:, :, :block_size_h, :block_size_w]
        block2 = x[:, :, :block_size_h, block_size_w:]
        block3 = x[:, :, block_size_h:, :block_size_w]
        block4 = x[:, :, block_size_h:, block_size_w:]

        #print("Block 1 shape:", block1.shape)
        #print("Block 2 shape:", block2.shape)
        #print("Block 3 shape:", block3.shape)
        #print("Block 4 shape:", block4.shape)
#
        # 计算每个局部区域的增强值
        enhanced_blocks = []
        for block in [block1, block2, block3, block4]:
            n = block_size_h * block_size_w - 1
            block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = block_minus_mu_square / (
                        4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
            enhanced_blocks.append(block * self.activaton(y))

        # 合并增强的块以重构输出图像
        enhanced_image = torch.cat([torch.cat([enhanced_blocks[0], enhanced_blocks[1]], dim=3),
                                    torch.cat([enhanced_blocks[2], enhanced_blocks[3]], dim=3)], dim=2)

        #print("Output shape:", enhanced_image.shape)

        return enhanced_image


if __name__ == '__main__':
    input = torch.randn(1, 1024, 7, 7)  # 示例输入大小（根据需要调整）
    model = SimAMWithSlicing()
    outputs = model(input)
