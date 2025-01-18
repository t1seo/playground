import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


def gelu(x):
    """
    GELU 활성화 함수를 구현합니다.
    입력값 x에 대해 가우시안 오차 선형 단위를 계산합니다.
    """
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def relu(x):
    """
    비교를 위한 ReLU 함수를 구현합니다.
    """
    return np.maximum(0, x)


# 입력값 생성
x = np.linspace(-5, 5, 1000)

# GELU와 ReLU 값 계산
y_gelu = gelu(x)
y_relu = relu(x)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(x, y_gelu, label="GELU", color="blue")
plt.plot(x, y_relu, label="ReLU", color="red", linestyle="--")
plt.grid(True)
plt.legend()
plt.title("GELU vs ReLU Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.show()
