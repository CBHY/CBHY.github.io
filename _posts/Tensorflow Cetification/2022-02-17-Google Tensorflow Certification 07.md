# Google Tensorflow Certification 07



## ★ 전이 학습(Transfer Learning)



### VGG-Net

#### 	가장 많이 활용되는 모델 

###### 		특히, VGG-16 > 아래 사진에 모델 
![tensorflow certification 07-1](D:\CodingData\github\cbhy.github.io\GitHub\_posts\images\2022-02-17-Google Tensorflow Certification 07\tensorflow certification 07-1.png)


## 실습(VGG16 Transfer Learning)

```python
# 모델 불러오기
from tensorflow.keras.applications import VGG16
```

```python
# 모델 값 설정(★위 이미지 참조)

# VGG16모델을 불러옴(가중치(Weight)= '이미지에 사용하는 가중치', Top포함 = ㄴㄴ, input_shape = 사이즈, 3체널)
transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 전이 모델 학습 = False
transfer_model.trainable=False # 모델이 학습하면 기껏 불러온 가중치가 변경되므로 Trainable = False 
```

```python
# Modeling
model = Sequential([
    transfer_model, # 앞에서 불러와서 설정한 모델을 가져옴
    Flatten(), # 2D -> 1D with Flatten
    Dropout(0.5),
    Dense(512, activation='relu'), # include_top = False 이기 때문에 이 아래 부분은 용도에 맞춰 Custom
    Dense(128, activation='relu'),
    Dense(2, activation='softmax'),
])
```

##### 	

