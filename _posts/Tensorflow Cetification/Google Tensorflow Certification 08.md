# Google Tensorflow Certification 08



## Category 4 - 자연어처리 (NLP)



### 자언어 전처리(Nature Language Preprocessing)

#### 	Step 1. 토큰화(Tokenizer) : 자연어를 단위로 쪼갠다.(문장 단위, 단어 단위, 알파벳 단위 등...)

#### 	Step 2. 치환 : 토큰을 숫자(인덱스)로 치환한다.

#### 	Step 3. 길이 맞추기 : 기준이 되는 길이로 문장을 자르거나 0을 붙혀서 길이를 맞춘다.



### 모델링 레이어(추가)

#### 	Embedding Layer : 차원을 감소시켜주는 레이어. 단어의 관계도 파악하기 수월

##### 	기존값을 넣어주면 차원의 저주에 빠져서 값이 0에 수렴함

#### RNN(Recurrent Neural Network, 순환 신경망) : 순서(시간)을 반영하는 레이어

##### 	문장이 길어지면 Gradient 소실이 발생

#### LSTM(Long-Short Term Memory) : RNN의 단점을 개선한 레이어

##### 	장기-단기 기억을 모두 활용해서 Gradient  소실을 최소화.

##### 	- many to one 기법

##### 	- many to many 기법

#### Bidirectional Layer : 단어를 예측할 때 양방향에서 예측할 수 있게 해주는 레이어





## 실습(Sarcasm)

### Step 1. Import

```python
import json
import urllib
import tensorflow_datasets as tfds # tfds import
import numpy as np #numpy import
import tensorflow as tf # tensorflow import
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential # Sequential Model import
from tensorflow.keras.callbacks import ModelCheckpoint # ModelCheckpoint import

from tensorflow.keras.preprocessing.text import Tokenizer # Tokenizer import
from tensorflow.keras.preprocessing.sequence import pad_sequences # pad_sequences import
```



### Step 2. Preprocessing

```python
# 전처리할 데이터 로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')
with open('sarcasm.json') as f:
    datas = json.load(f)
```

#### 전처리(preprocessing) 요구 조건

1. image(x), label(y)를 분할

```python
# 빈 리스트에 분할시키기
sentences = []
labels = []
for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])

# Train data set / Validation data set 분할
training_size = 20000

train_sentences = sentences[:training_size]
train_labels = labels[:training_size]

validation_sentences = sentences[training_size:]
validation_labels = labels[training_size:]
```

```python
# Tokenizer Setting
vocab_size = 1000 # Vocab(던어) 개수
oov_tok = "<OOV>" #Out Of Vocab(빈도수 기준 단어 개수를 넘는 것들)은 <OOV>토큰으로 표시

# 토큰화
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences) # 딕셔너리로 반환(key = 단어, vlaue = 숫자)
```

<OOV>  	======>	 1  				# 빈도수 1위 

to  	======>	 2 						#빈도수 2위	

of  	======>	 3 					 . . .	# 1000위까지 표시

```python
# texts_to_sequences(문장을 숫자로 치환)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
```

```python
# 문장 길이 맞추기
max_length = 120
trunc_type='post' # post는 뒤를 자름, pre는 앞을 자름
padding_type='post' # post는 뒤를 채움, pre는 앞을 채움

#적용
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# list > np.ndarray 변환
train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)
```



### Step 3. Modeling

```python
model = Sequential([
    # Embedding Layer(단어 개수(차원 개수)1000, 줄일 차원 수 16, 최대 길이(한 문장 단어 수) 120)
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),# 앞 64, 뒤 64 = 128필터
    Bidirectional(LSTM(64)), # LSTM 이진분류 마지막에는 return_sequences X
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

##### 	

### Step 4. Compile

```python
# optimizer = 'adam' (분류 최적화는 adam이 가장 좋다(?))
# loss = '(원핫인코딩O) binary(모델 마지막 활성함수 sigmoid)_crossentropy'
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
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
epochs=10
history = model.fit(train_padded, train_labels, 
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)
```



### Step 5.5. Ckpt Load Weight

```python
# 이 코드가 없다면, Ckpt 만드는 이유가 없음(가중치 저장만 해두고 사용 안하는 것)
model.load_weights(checkpoint_path)
```



