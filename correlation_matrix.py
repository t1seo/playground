import numpy as np
import pandas as pd

# 예시 데이터 생성
data = {"키": [170, 175, 160, 180, 165], "몸무게": [65, 70, 55, 75, 60], "나이": [25, 30, 28, 35, 27]}

df = pd.DataFrame(data)

# 상관관계 행렬 계산
correlation_matrix = df.corr()
print(correlation_matrix)
