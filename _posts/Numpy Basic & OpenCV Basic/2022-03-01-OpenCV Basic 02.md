# OpenCV Basic 02



## OpenCV 이미지 연산

```python
# OpenCV 이미지 연산
import cv2
im1 = cv2.imread('Sunset.jpg', cv2.IMREAD_COLOR)

# 픽셀 수 및 이미지 확인
print(im1.shape)
print(im1.size)

# 각 픽셀 하나씩 읽기
px = im1[1300, 2000]
print(px)

# Row 읽기
pxr = im1[500]
print(pxr)

# 특정 범위의 픽셀 변경
for i in range(100):
    for j in range(100):
        im1[i, j] = [255, 255, 255]

cv2.imshow('1', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()


im1[0:100, 300:1000] = [0, 0, 0]
cv2.imshow('2', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ROI(region of interest)추출 및 복사

#픽셀의 색상별 변경
im1[:,:,2] = 0

cv2.imshow('3', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()


```

