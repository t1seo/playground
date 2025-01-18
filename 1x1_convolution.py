import torch
import torch.nn as nn


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        # 1x1 Convolution을 정의한다
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


# 사용 예시
if __name__ == "__main__":
    # 입력 텐서 (배치, 채널, 높이, 너비)
    x = torch.randn(1, 64, 32, 32)

    # 1x1 Convolution 레이어 생성
    conv1x1 = PointwiseConv(64, 32)

    # 출력 계산
    out = conv1x1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
