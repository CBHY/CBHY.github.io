---
categories: OpenCV_Basic
tag: [OpenCV, cv2, numpy, np, python, lib]
toc: true
---
# OpenCV Basic 08



## OpenCV Contour(외곽, 태두리)

```python
import cv2
import numpy as np

im = cv2.imread('mnist.jpg', cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im1 = cv2.adaptiveThreshold(im1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
cv2.imshow('ath_grayscale', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

contour= cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # 모든 값을 추출
im = cv2.drawContours(im, contour, -1, (0, 255, 0), 4)
cv2.imshow('countour', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

