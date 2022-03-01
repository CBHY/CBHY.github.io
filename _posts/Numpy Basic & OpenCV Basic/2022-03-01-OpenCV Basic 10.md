---
categories: OpenCV_Basic
tag: [OpenCV, cv2, numpy, np, python, lib]
toc: true
---
# OpenCV Basic 10



## OpenCV  Filtering

```python
import cv2
import numpy as np
im = cv2.imread('mnist.jpg', cv2.IMREAD_COLOR)
cv2.imshow('1', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 커널(필터) 생성
size = 4
kernel_Basic = np.ones((size, size), dtype=np.float32) / (size ** 2) # Basic Kernel
print(kernel_Basic)

# 필터링
im_BK = cv2.filter2D(im, -1, kernel_Basic)
cv2.imshow('2', im_BK)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 간단하게 블러 필터링 도와주는 함수 (blur)
im_BK_bf = cv2.blur(im, (4, 4))
cv2.imshow('3', im_BK_bf)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 가우시안 블러 필터링
im_gauss = cv2.GaussianBlur(im, (5, 5), 0) # 가우시안 블러의 커널 사이즈는 홀수 사이즈.
cv2.imshow('4', im_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

