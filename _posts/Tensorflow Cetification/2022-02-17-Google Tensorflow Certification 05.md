# Google Tensorflow Certification 05



## Category 3 - 이미지 분류 모델 (CNN) - Type A



### IDG(Image Data Gererator)

#### 	전처리(preprocess)과정에서, 

#### 	이미지를 변형(Arguemenration)을 통해 Data set을 구성해주는 역할을 함

###### 			flow_from_driectory()함수를 이용해서 이미지를 불러올 때 폴더명에 맞춰서 자동으로 라벨링, 사이즈조절, 베치조절 등을 한다.



### CNN(Convolution Neural Network) : 합성 곱 신경망

#### 	 효율적인 연산을 위해이미지의 특성을 추출하는 필터를 통한 Feature Map을 생성함



### Pooling Layer

#### 	이미지의 사이즈를 줄여주는 레이어

#### 	- Max Pooling : 주변 픽셀 값 중, 최대값으로만 표시

#### 	- Average Pooling : 주변 픽셀 값들의 평균값으로 표시



#### 보통 CNN + Activation + Pooling 을 이용해서 모델을 쌓는다.



## 실습(RPS)



### Step 1. Import

```python
import urllib.request
import zipfile
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout # 레이어 설명은 아래에서 
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import

from tensorflow.keras.preprocessing.image import ImageDataGenerator # ImageDataGenerator import

```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

# Data Set 경로 지정
TRAINING_DIR = "tmp/rps/" #flow_from_directory 함수에 의해 각 레이블(r,p,s)는 루트 폴더(tmp/rps)아레 폴더에 분류
```

```python
# Image Data Generator 생성
training_datagen = ImageDataGenerator(
    rescale=1. / 255, # 정규화(Normalization)
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest', 
    validation_split=0.2 # rps data set에는 train data만 존제, 8:2 == train:valid 분할
    )
```

#### 대표적인 변형(Argumentation) 설정값

- `rescale`: 이미지의 픽셀 값을 조정
- `rotation_range`: 이미지 회전
- `width_shift_range`: 가로 방향으로 이동
- `height_shift_range`: 세로 방향으로 이동
- `shear_range`: 이미지 굴절
- `zoom_range`: 이미지 확대
- `horizontal_flip`: 횡 방향으로 이미지 반전
- `fill_mode`: 이미지를 이동이나 굴절시켰을 때 빈 픽셀 값에 대하여 값을 채우는 방식
- `validation_split`: validation set의 구성 비율

###### 

```python
# IDG 적용 with flow_from_directory
training_generator = training_datagen.flow_from_directory(TRAINING_DIR, # root 폴더 경로
                                                          batch_size=32, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical', # 이진분류 X 
                                                          subset='training', # IDG설정값 Validation_split존재
                                                         )

validation_generator = training_datagen.flow_from_directory(TRAINING_DIR, 
                                                          batch_size=32, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical',
                                                          subset='validation', 
                                                         )
```





### Step 3. Modeling

```python
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)), # 사이즈 150X150, RGB 3체널(컬러)
    MaxPooling2D(2, 2), # 2X2 픽셀을 1픽셀로 (MaxPooling > 가장 큰 값)  
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Flatten(), # 2D -> 1D With Flatten
    Dropout(0.5), # 학습에 사용하는 Node 수 x0.5 > 과대적합 방지
    Dense(512, activation='relu'), 
    Dense(3, activation='softmax'),# 분류(Classification) label이 3개 -> 이진분류X, activation = 'softmax'
])
```

##### 	

### Step 4. Compile

```python
# optimizer = 'adam' (분류 최적화는 adam이 가장 좋다(?))
# loss = 'sparse(원핫인코딩X)_categorical(모델 마지막 활성함수 Softmax)_crossentropy'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
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
epochs = 25
history = model.fit(training_generator,  # IDG 적용 data set
                    validation_data=(validation_generator), # IDG 적용 data set
                    epochs=epochs,
                    callbacks=[checkpoint],
                    )
```



### Step 5.5. Ckpt Load Weight

```python
# 이 코드가 없다면, Ckpt 만드는 이유가 없음(가중치 저장만 해두고 사용 안하는 것)
model.load_weights(checkpoint_path)
```



