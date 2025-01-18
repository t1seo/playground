import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 더미 데이터 생성
np.random.seed(42)  # 재현성을 위한 시드 설정
days = 100  # 100일간의 데이터
price = 100 + np.random.randn(days).cumsum()  # 랜덤 워크로 주가 데이터 생성


def calculate_ema(data, alpha):
    """
    지수 이동 평균을 계산하는 함수

    매개변수:
        data: 시계열 데이터
        alpha: 평활화 계수 (0 < alpha <= 1)
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]  # 첫 번째 값은 원본 데이터 값으로 초기화

    # EMA 계산
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


# 여러 알파 값으로 EMA 계산
alphas = [0.1, 0.3, 0.5]
ema_results = {f"EMA (α={alpha})": calculate_ema(price, alpha) for alpha in alphas}

# 데이터프레임 생성
df = pd.DataFrame({"원본 데이터": price, **ema_results})

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["원본 데이터"], label="원본 데이터", alpha=0.5)
for col in df.columns[1:]:
    plt.plot(df[col], label=col)
plt.title("지수 이동 평균 (EMA) 비교")
plt.xlabel("시간")
plt.ylabel("가격")
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print("처음 10개의 데이터 포인트:")
print(df.head(10))
