import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams["axes.unicode_minus"] = False
try:
    font_path = font_manager.findfont(font_manager.FontProperties(family="AppleGothic"))
    rc("font", family="AppleGothic")
except:
    try:
        font_path = font_manager.findfont(font_manager.FontProperties(family="NanumGothic"))
        rc("font", family="NanumGothic")
    except:
        print("Warning: 한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")


# 1. NumPy로 구현한 BCE Loss
def numpy_bce_loss(y_true, y_pred):
    """
    Parameters:
    y_true: 실제 레이블 (0 또는 1)
    y_pred: 예측값 (0~1 사이의 확률)
    """
    # 수치적 안정성을 위해 클리핑
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # BCE Loss 계산
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


# 2. PyTorch로 구현한 BCE Loss
def torch_bce_loss(y_true, y_pred):
    criterion = nn.BCELoss()
    return criterion(y_pred, y_true)


# 3. 손실 함수 시각화
def visualize_bce_loss():
    # 예측값 범위 생성
    predictions = np.linspace(0.001, 0.999, 1000)

    # 실제값이 1일 때와 0일 때의 손실 계산
    loss_true_1 = [-np.log(pred) for pred in predictions]
    loss_true_0 = [-np.log(1 - pred) for pred in predictions]

    plt.figure(figsize=(10, 6))
    plt.plot(predictions, loss_true_1, label="실제값 = 1")
    plt.plot(predictions, loss_true_0, label="실제값 = 0")
    plt.xlabel("예측값")
    plt.ylabel("손실값")
    plt.title("Binary Cross Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 4. 실제 예제
def example_usage():
    # 예시 데이터
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

    # NumPy로 계산
    numpy_loss = numpy_bce_loss(y_true, y_pred)
    print(f"NumPy로 계산한 BCE Loss: {numpy_loss:.4f}")

    # PyTorch로 계산
    torch_true = torch.FloatTensor(y_true)
    torch_pred = torch.FloatTensor(y_pred)
    torch_loss = torch_bce_loss(torch_true, torch_pred)
    print(f"PyTorch로 계산한 BCE Loss: {torch_loss.item():.4f}")


if __name__ == "__main__":
    visualize_bce_loss()
    example_usage()
