# OpenCV Basic 04



## 이미지 합치기

```python
import cv2
import numpy as np

im1 = cv2.imread('Winter.jpg', cv2.IMREAD_COLOR)
im2 = cv2.imread('Sunset.jpg', cv2.IMREAD_COLOR)
im1 = cv2.resize(im1, (128, 128))
im2 = cv2.resize(im2, (128, 128))

# 각 픽셀의 합 (saturation 연산)
im_plus_sat = cv2.add(im1, im2)
cv2.imshow('im sat', im_plus_sat)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 각 픽셀의 합 (Modulo 연산) like overflow.
im_plus_Mod = im1 + im2
cv2.imshow('im mod', im_plus_Mod)
cv2.waitKey(0)
cv2.destroyAllWindows()	
```

