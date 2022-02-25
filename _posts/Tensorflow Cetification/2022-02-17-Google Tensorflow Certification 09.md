# Google Tensorflow Certification 09



## Category 5 - 시퀀스와 시계열 (Sequence, Time Series) - 1



### Windowd Dataset

#### 	시계열 데이터에서 데이터를 예측하기 위해 참고할 데이터를 세팅하는 것

###### 			window_size, shift, drop_remainder # 실습 참조

### Conv1D(Convolution Neural Network 1-Dimention)

#### 	효율적인 연산을 위해 특성을 추출하는 필터

###### 		kernel_size, strides, padding # 실습 참조

### Optimizer 튜닝

#### 	기본적인 최적화(Optimizer) 알고리즘인 SGD, ADAM 등을 튜닝할 수 있다.

###### 		momentum, Learning Rate 등.. # 실습 참조

### Huber Loss

#### 	MSE와 MAE 중에 낮은 값을 따라가는 오차 함수



## 실습(Sunspots)

### Step 1. Import

```python
import csv # csv(comma separated value) import
import urllib
import tensorflow_datasets as tfds # tfds import
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import

from tensorflow.keras.optimizers import SGD # SGD optimizer import
from tensorflow.keras.losses import Huber # Huber loss import
```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
urllib.request.urlretrieve(url, 'sunspots.csv')

with open('sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    i = 0
    for row in reader:
        print(row)
        i+=1
        if i > 10:
            break
```

#### 전처리(preprocessing) 요구 조건

1. sunspots(x), time_step(y)를 분할

```python
# 빈 리스트에 분할시키기
sunspots = []
time_step = []
with open('sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # 첫 줄은 header이므로 skip.
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
        
# List to np.ndarray
series = np.array(sunspots)
time = np.array(time_step)

# Train data set / Validation data set 분할
split_time = 3000

time_train = time[:split_time]
time_valid = time[split_time:]

x_train = series[:split_time]
x_valid = series[split_time:]
```

```python
# Window Dataset Setting
window_size=30
batch_size = 32
shuffle_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
```

```python
# window dataset 적용
train_set = windowed_dataset(x_train, 
                             window_size=window_size, 
                             batch_size=batch_size,
                             shuffle_buffer=shuffle_size)

validation_set = windowed_dataset(x_valid, 
                                  window_size=window_size,
                                  batch_size=batch_size,
                                  shuffle_buffer=shuffle_size)
```



### Step 3. Modeling

```python
model = Sequential([
    # Conv1D(몇 개의 데이터를 가지고 특성을 추출할건지(커널사이즈), padding = causal(사이즈 그대로))
    # input_shape에 None = 어떤 모양이 오든 무시
    Conv1D(60, kernel_size=5, padding="causal", activation="relu",input_shape=[None, 1]),
    LSTM(60, return_sequences=True),
    LSTM(60, return_sequences=True), # many to many 기법 쓰기 때문에 return)sequence = True
    Dense(30, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1),
    Lambda(lambda x: x * 400) #Lambda Layer
])
```

##### 	

### Step 4. Compile

```python
# SGD Custom
optimizer = SGD(lr=0.0001, momentum=0.9) # lr = 학습률, momentum = 관성(가중치)
# Huber
loss= Huber()
# Compile
model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])
```



### Step 4.5. ModelCheckpoint 생성

```python
checkpoint_path = "my_checkpoint.ckpt" # 체크포인트 위치는 로컬, 이름.ckpt or 이름.m5
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             save_weights_only=True, # 가중치만 저장
                             save_best_only=True, # 가장 좋은 결과만 저장
                             monitor='val_mae',  # 기준 = 'validation_MAE가 가장 낮은 것'
                             verbose=1) # 출력
```



### Step 5. Fit

```python
# 학습(train data, Validation_data, epochs, callbacks[ckpt])
epochs=100
history = model.fit(train_set, 
                    validation_data=(validation_set), 
                    epochs=epochs, 
                    callbacks=[checkpoint],
                   )
```



### Step 5.5. Ckpt Load Weight

```python
# 이 코드가 없다면, Ckpt 만드는 이유가 없음(가중치 저장만 해두고 사용 안하는 것)
model.load_weights(checkpoint_path)
```



