---
categories: OpenCV_Basic
tag: [OpenCV, cv2, numpy, np, python, lib]
toc: true
---
# OpenCV Basic 07



## OpenCV 도형 그리기

```python
import cv2
import numpy as np

# 직선 그리기
im = np.full((512, 512, 3), 255, dtype=np.uint8)
line1 = cv2.line(im, (0, 0), (255, 255), (255,0, 0), 5)
cv2.imshow('line', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 사각형 그리기
rect = cv2.rectangle(im, (23, 23), (500, 500), (125, 45, 66), 10)
cv2.imshow('rectangle', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 원 그리기
circ = cv2.circle(im, (300, 300), 10, (222, 22, 21), 5)
cv2.imshow('circle', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 다각형 그리기
point = np.array([
    [5, 5],
    [24, 24],
    [300, 244],
    [444, 2]
])
poli = cv2.polylines(im, [point], True, (222, 221, 22), 5)
cv2.imshow('polyline', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 텍스트 그리기
text = cv2.putText(im, 'KHY', (45, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (21, 0, 22))
cv2.imshow('Text', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

