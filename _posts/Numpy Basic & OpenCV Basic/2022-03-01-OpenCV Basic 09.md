# OpenCV Basic 09



## Contours 처리

```python
import cv2
import numpy as np

im = cv2.imread('mnist.jpg', cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, im1 = cv2.threshold(im1, 127, 255, 0)
im1 = cv2.bitwise_not(im1)
cv2.imshow('1', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 사각형 외각 찾기
contours = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # index 0 은 컨투어 전체를 반환한다.
image = cv2.drawContours(im, contours, -1, (0, 0, 255), 4)

cv2.imshow('contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


contour = contours[1]
x, y, w, h = cv2.boundingRect(contour) # 각 컨투어 경계부분(사각형 형태)의 좌표(왼쪽 아래), 길이를 반환시켜주는 함수
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('contour_rect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```