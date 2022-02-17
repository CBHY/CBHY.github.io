# Google Tensorflow Certification 04



## Category 2 - 심층신경망 모델(정형데이터)



### Tensorflow Datasets 활용

[Tensorflow datasets 소개](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)



## data load 방법

#### tfds.load문서 발췌

The easiest way of loading a dataset is [`tfds.load`](https://www.tensorflow.org/datasets/api_docs/python/tfds/load). It will:

1. Download the data and save it as [`tfrecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord) files.
2. Load the `tfrecord` and create the [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

```python
ds = tfds.load('mnist', split='train', shuffle_files=True)
                 .
                 .
                 .
```

Some common arguments:

- `split=`: Which split to read (e.g. `'train'`, `['train', 'test']`, `'train[80%:]'`,...). See our [split API guide](https://www.tensorflow.org/datasets/splits).
- `shuffle_files=`: Control whether to shuffle the files between each epoch (TFDS store big datasets in multiple smaller files).
- `data_dir=`: Location where the dataset is saved ( defaults to `~/tensorflow_datasets/`) 
- `with_info=True`: Returns the `tfds.core.DatasetInfo` containing dataset metadata
- `download=False`: Disable download

#### 	

## iris data 분석

#### iris문서 발췌





- **Description**:

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains ***<u>3 classes of 50 instances each</u>***, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 



- **Homepage**: https://archive.ics.uci.edu/ml/datasets/iris
- **Source code**: [`tfds.structured.Iris`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/structured/iris.py)
- **Versions**:
  - **`2.0.0`** (default): New split API (https://tensorflow.org/datasets/splits)
- **Download size**: `4.44 KiB` 
- **Dataset size**: `Unknown size`
- **Auto-cached** ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): Unknown
- **Splits**:

| Split     | Examples |
| :-------- | -------: |
| `'train'` |      150 |

<u>***data set 은 train만 있음을 알 수 있음(나중에 train/valid data로 갈라서 사용)***</u>



- **Features**:

```python
FeaturesDict({
    'features': Tensor(shape=(4,), dtype=tf.float32),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=3),
})
```

<u>***features는 4개, label은 3개임을 알 수 있음***</u>



- **Supervised keys** (See [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)): `('features', 'label')`
- **Figure** ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)): Not supported.
- **Examples** ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):



- **Citation**:

```python
@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences"
}
```





## 실습(Iris)



### Step 1. Import

```python
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Dense, Flatten # Dense Layer, Flatten Layer import
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import

import tensorflow_datasets as tfds # tensorflow_dataset import
```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드(Tensorflow datasets - iris)
# load('데이터셋 이름', split='train데이터의 시작부터 80%까지') > train_dataset으로 이용
train_dataset = tfds.load('iris', split='train[:80%]') 
# load('데이터셋 이름', split='train데이터의 80%부터 끝까지') > valid_dataset으로 이용
valid_dataset = tfds.load('iris', split='train[80%:]')
```

전처리 요구 조건

1. label 값을 one-hot encoding 할 것

2. feature (x), label (y)를 분할할 것

```python
# 전처리 함수 생성
def preprocess(data):
    x = data['features'] # data를 받아 feature를 x에 할당
    y = data['label']	# data를 받아 label을 y에 할당
    y = tf.one_hot(y, 3) # 원핫인코딩, label(y)을 3개로 ex) [1, 0, 0], [0, 1, 0], [0, 0, 1]
    return x, y
```

```python
# 전처리 함수 적용
# train_dataset(에).map(이 함수를 적용).batch(베치사이즈)
batch_size = 10
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)
```



### Step 3. Modeling

```python
model = Sequential([
    # input_shape : feature == 4 이므로, (4, ) or [4]
    Dense(512, activation='relu', input_shape=(4,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'), # 분류(Classification) label이 3개 -> 이진분류X, activation = 'softmax'
])
```

##### 	

### Step 4. Compile

```python
# optimizer = 'adam' (분류 최적화는 adam이 가장 좋다(?))
# loss = '(원핫인코딩O)categorical(모델 마지막 활성함수 Softmax)_crossentropy'
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
history = model.fit(train_data,
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

