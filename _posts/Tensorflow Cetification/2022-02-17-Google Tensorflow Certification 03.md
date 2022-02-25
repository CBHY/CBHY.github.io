# Google Tensorflow Certification 03



## Category 2 - 심층신경망 모델(이미지)



#### 	학습(Train)에 대한 Loss와 Acc는 Epoch이 증가할 수록 무한히 좋아지지만,

#### 	과연 실제 데이터에 적용했을 경우 어떻게 될까?



### 과대적합(Overfitting)/과소적합(Underfitting)

#### 	과대적합 :학습 데이터(Train Data)에 너무 많이 적합되는 경우

#### 	과소적합 : 학습 데이터(Train Date)에 너무 조금 적합되는 경우

###### 		쉽게 보면, 학습데이터를 계속 학습시켰을 경우, 학습데이터'만' 구분할 정도로 학습되는 경우를 과대적합,

###### 		학습데이터의 학습이 덜 된 경우를 과소적합이라고 생각하면 쉽다.



### Train/Validation

#### 적절한 학습(Approrpriate Train)을 위해, 학습 데이터(Train data)로 학습시킨 모델을 

#### 검증 데이터(Validation data)로 검증하는 방식을 이용한다.



###  ModelCheckpoint

#### 	학습 후 검증된 모델에서, 가중치를 저장하기 위해 필요한 과정이다.



### 세로운 전처리(preprocessing)과정

#### 	이미지 정규화(Image Normalization) : 이미지의 픽셀 값들을 0 ~ 1 사이의 값으로 맞춘다.

###### 			이미지 정규화를 수행한 데이터는 학습 성능이 올라간다.

#### 	원핫인코딩(One-Hot Incoding) : label 들의 독립적인 관계를 나타내기 위한 인코딩

###### 			범주형 데이터에서는 필수적인 과정이다.



### 활성함수(Activation Function)

#### 	선형함수X선형함수 = 선형관계

##### 		딥러닝이 층을 깊게 쌓아 복잡한 문제를 해결하는 과정인데, 선형함수들만으로 모델을 구축하면 선형관계를 이루기 때문에

##### 		복잡한 문제를 해결할 수 없다.

#### 	선형함수X비선형함수(활성함수)X선형함수

##### 		이러한 관계를 이루기 위해 비선형함수로 활성함수(Activation Function)을 사용한다.



#### 	마지막 출력층의 활성함수에 대한 Loss Function 설정

![img](https://learnaday.kr/media/BXnew9OZiEM8Ck5wIXQ17JhGmCa9WmIg)





## 실습(Fashion Mnist)



### Step 1. Import

```python
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Dense, Flatten # Dense Layer, Flatten Layer import
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import
```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드(Tensorflow datasets - FashionMnist)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data() # 데이터 로드 형식은 아래 데이터셋 소개에서 확인

# 정규화(Nomalization)
x_train = x_train / 255.0
x_valid = x_valid / 255.0
```

[Tensorflow datasets 소개](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)

```python
# 형식 확인
print(x_train.shape)
```

###### (60000, 28, 28)

```python
# 2D -> 1D With Flatten
x = Flatten(input_shape=(28, 28))

# 형식 확인
print(x(x_train).shape)
```

###### (60000, 784)



### Step 3. Modeling

```python
model = Sequential([
    # 2D -> 1D With Flatten
    Flatten(input_shape=(28, 28)),
    # Dense Layer
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'), 
    Dense(10, activation='softmax'), # 분류(Classification) label이 10개 -> 이진분류X, activation = 'softmax'
])
```

##### 	

### Step 4. Compile

```python
# optimizer = 'adam' (분류 최적화는 adam이 가장 좋다(?))
# loss = 'sparse(원핫인코딩X)_categorical(모델 마지막 활성함수 Softmax)_crossentropy'
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
```



### Step 4.5. ModelCheckpoint 생성

```python
checkpoint_path = "my_checkpoint.ckpt" # 체크포인트 위치는 로컬, 이름.ckpt or 이름.m5
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             save_weights_only=True, # 가중치만 저장
                             save_best_only=True, # 가장 좋은 결과만 저장
                             monitor='val_loss',  # 기준 = 'validation_loss가 가장 낮은 것'
                             verbose=1) # 출력
```



### Step 5. Fit

```python
# 학습(train data, Validation_data, epochs, callbacks[ckpt])
history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=20,
                    callbacks=[checkpoint],
                   )
```



### Step 5.5. Ckpt Load Weight

```python
# 이 코드가 없다면, Ckpt 만드는 이유가 없음(가중치 저장만 해두고 사용 안하는 것)
model.load_weights(checkpoint_path)
```



### Step 6. Predict

```python
model.evaluate(x_valid, y_valid)
```

