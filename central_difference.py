import numpy as np


def central_difference(f, x, h=1e-5):
    """
    중앙 차분법을 이용하여 함수 f의 x점에서의 도함수 값을 계산한다.

    매개변수:
        f: 미분하고자 하는 함수
        x: 도함수를 구하고자 하는 지점
        h: 스텝 크기 (기본값: 0.00001)

    반환값:
        x점에서의 도함수 근사값
    """
    return (f(x + h) - f(x - h)) / (2 * h)


# 사용 예시
def f(x):
    """테스트를 위한 예시 함수: f(x) = x^2"""
    return x**2


# 테스트
x = 2.0  # x = 2에서의 도함수 값을 계산
derivative = central_difference(f, x)
print(f"f'({x}) ≈ {derivative}")  # 실제 도함수 값은 4

# 여러 지점에서의 도함수 값 계산
x_points = np.linspace(-2, 2, 5)  # -2부터 2까지 5개의 균등한 간격의 점
for x in x_points:
    derivative = central_difference(f, x)
    print(f"f'({x:.1f}) ≈ {derivative:.2f}")
