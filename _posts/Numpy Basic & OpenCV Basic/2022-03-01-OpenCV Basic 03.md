# OpenCV Basic 03



## OpenCV 이미지 변형

```python
import cv2
import numpy as np

moon_gray = cv2.imread('Moon_grayscale.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('1', moon_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 이미지 크기 조절
# 보간법(interpolation) : 이미지 사이즈가 변할 때 픽셀 사이의 값을 조절하는 방법
# INTER_CUBIC : 주로 사이즈 크게 할 때 사용
# INTER_AREA : 주로 사이즈 작게 할 때 사용
shr = cv2.resize(moon_gray, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
ex = cv2.resize(moon_gray, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
cv2.imshow('2', shr)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('3', ex)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 위치 변경
# 변환행렬 # 앞에 2X2행렬과는 곱, 뒤 M13, M23으로 이루어진 행렬은 합
M1 = np.array([
    [1, 0, 50],
    [0, 1, 10]
], dtype = np.float32)

height, width = moon_gray.shape[:2] # 좌표 주의 (y, x, 3(color))에서 y좌표, x좌표로 받는다. 
# 위치 이동
im2 = cv2.warpAffine(moon_gray, M1, (width, height)) # 좌표 주의
cv2.imshow('4', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 회전
M2 = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5) # 회전 행렬생성기를 이용(회전의 중심점, 각도, 스케일)
im3 = cv2.warpAffine(moon_gray, M2, (width, height))
cv2.imshow('5', im3)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

