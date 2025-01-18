import torch

# 11개의 원소를 가진 1차원 텐서를 6개의 덩어리로 나누기
x = torch.arange(10)
chunks = torch.chunk(x, 6)

# 출력: 각각 2개의 원소를 가진 5개의 덩어리와
# 1개의 원소를 가진 1개의 덩어리가 생성됨
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
