# Google Tensorflow Certification 06



## Category 3 - 이미지 분류 모델 (CNN) - Type B



### With

#### tensorflow_datasets



## 실습(Cat vs Dog)



### Step 1. Import

```python
import tensorflow_datasets as tfds # tfds import
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import
```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드
dataset_name = 'cats_vs_dogs'

# Cat_vs_dog data set도 train data set만 있음
train_dataset = tfds.load(name=dataset_name, split='train[:80%]')
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')
```

#### 전처리(preprocessing) 요구 조건

1. 이미지 정규화 (Normalization)
2. 이미지 사이즈 맞추기: (224 X 224)
3. image(x), label(y)를 분할



```python
# 전처리 함수
def preprocess(data):
    x = data['image'] # 분할
    y = data['label'] # 분할
    x = x / 255 # 정규화(Normalization)
    x = tf.image.resize(x, size=(224, 224)) # 사이즈 변환(resize함수) > (224, 224)
    return x, y
```

```python
# 전처리 함수 적용
batch_size = 32
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)
```



### Step 3. Modeling

```python
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'), # 사이즈 224X224, RGB 3체널(컬러)
    MaxPooling2D(2, 2), # 2X2 픽셀을 1픽셀로 (MaxPooling > 가장 큰 값)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(), # 2D -> 1D With Flatten
    Dropout(0.5), # 학습에 사용하는 Node 수 x0.5 > 과대적합 방지
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax'), # 분류(Classification) label이 2개 -> 이진분류O,  activation = 'softmax'
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
model.fit(train_data,
          validation_data=(valid_data),
          epochs=20,
          callbacks=[checkpoint],
          )
```



### Step 5.5. Ckpt Load Weight

```python
# 이 코드가 없다면, Ckpt 만드는 이유가 없음(가중치 저장만 해두고 사용 안하는 것)
model.load_weights(checkpoint_path)
```



