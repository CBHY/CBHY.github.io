# OpenCV Basic 05



## 임계점 처리

```python
import cv2

im = cv2.imread('mnist.jpg', cv2.IMREAD_GRAYSCALE)

# 임계점thres hold
ret, im1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
ret, im2 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
ret, im3 = cv2.threshold(im, 127, 255, cv2.THRESH_TRUNC)
ret, im4 = cv2.threshold(im, 127, 255, cv2.THRESH_TOZERO)
ret, im5 = cv2.threshold(im, 127, 255, cv2.THRESH_TOZERO_INV)

image_list = []

image_list.append(im1)
image_list.append(im2)
image_list.append(im3)
image_list.append(im4)
image_list.append(im5)


for i in image_list:
    cv2.imshow('i', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 적응 임계점 처리(임계점 자동 지정)
im6 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
im7 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)

cv2.imshow('6', im6)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('7', im7)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

