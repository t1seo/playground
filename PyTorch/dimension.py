import torch

# 3채널 RGB 이미지 생성 (3x4x4)
tensor_3d = torch.tensor(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],  # R 채널
        [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],  # G 채널
        [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]],  # B 채널
    ],
    dtype=torch.float32,
)

print("3차원 텐서 (RGB 이미지):")
print(f"Shape: {tensor_3d.shape}")  # torch.Size([3, 4, 4])

# dim=0 (채널 방향) 평균 - 그레이스케일 변환과 유사
channel_mean = torch.mean(tensor_3d, dim=0)
print("\n채널 평균 (dim=0):")
print(channel_mean)

# dim=1 (높이 방향) 평균
height_mean = torch.mean(tensor_3d, dim=1)
print("\n높이 평균 (dim=1):")
print(height_mean)

# dim=2 (너비 방향) 평균
width_mean = torch.mean(tensor_3d, dim=2)
print("\n너비 평균 (dim=2):")
print(width_mean)

print("=" * 10)
# 배치 크기 2, 채널 3, 높이 4, 너비 4인 텐서 생성
tensor_4d = torch.rand(2, 3, 4, 4)
print("4차원 텐서 (이미지 배치):")
print(f"Shape: {tensor_4d.shape}")  # torch.Size([2, 3, 4, 4])

# dim=0 (배치 방향) 평균
batch_mean = torch.mean(tensor_4d, dim=0)
print(f"\n배치 평균 shape: {batch_mean.shape}")  # torch.Size([3, 4, 4])

# dim=1 (채널 방향) 평균
channel_mean = torch.mean(tensor_4d, dim=1)
print(f"채널 평균 shape: {channel_mean.shape}")  # torch.Size([2, 4, 4])

# 여러 차원에 대한 평균
multi_dim_mean = torch.mean(tensor_4d, dim=(2, 3))
print(f"높이와 너비 평균 shape: {multi_dim_mean.shape}")  # torch.Size([2, 3])

print(batch_mean)
print(channel_mean)
print(multi_dim_mean)
