# OpenCV Basic 06



## OpenCV Tracker

```python
import numpy as np
import cv2

#Traker : 사용자가 슬라이드 바를 이용해서 값을 편하게 바꿀 수 있게 해주는 기능
def change_color(x):
    r = cv2.getTrackbarPos("R", "im")
    g = cv2.getTrackbarPos("G", 'im')
    b = cv2.getTrackbarPos("B", 'im')
    im[:] = [b, g, r]
    cv2.imshow('Im', im)

im = np.zeros((600, 400, 3), np.uint8)
cv2.namedWindow("im")
cv2.createTrackbar("R", "im", 0, 255, change_color)
cv2.createTrackbar("G", "im", 0, 255, change_color)
cv2.createTrackbar("B", "im", 0, 255, change_color)

cv2.imshow('im', im)
cv2.waitKey(0)

```