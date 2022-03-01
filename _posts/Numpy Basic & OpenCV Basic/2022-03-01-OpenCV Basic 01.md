# OpenCV Basic 01



## OpenCV 소개

```python
import cv2

# 이미지 읽기 cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
im1 = cv2.imread('Moon.jpg', cv2.IMREAD_COLOR)

# 이미지 보여주기
cv2.imshow('mo', im1)
cv2.waitKey(0)

# 이미지 창 닫기
cv2.destroyAllWindows()

# 이미지 저장
cv2.imwrite('Moon.jpg', im1)

# 이미지 형식 변경
im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
cv2.imshow('mog', im2)
cv2.waitKey(0)

cv2.imwrite('Moon_grayscale.jpg', im2)
```

