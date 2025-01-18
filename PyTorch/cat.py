import torch

# 2x3 크기의 랜덤 텐서 생성
x = torch.randn(2, 3)

# dim=0 (세로 방향)으로 연결
result1 = torch.cat((x, x, x), 0)  # 6x3 텐서가 됨

# dim=1 (가로 방향)으로 연결
result2 = torch.cat((x, x, x), 1)  # 2x9 텐서가 됨

print(x)
print(result1)
print(result2)
