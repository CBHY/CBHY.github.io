---
categories: tensorflow
tag: [tensorflow, google, google_developer,tensorflow_certification, python]
toc: true
author_profile: false
---
# Google Tensorflow Certification 02



## Category 1 - Basic 모델



### Dense Layer(Fully Connected Layer, FC)

각 노드(Node, 혹은 neuron)가 완전하게 연결되어 있는 Layer

input_shape를 지정해주어야 한다.

![img](https://blog.kakaocdn.net/dn/bvENxB/btqFA6fIM29/lQFQDsq2fF1ovns5PxH2l1/img.png)



## 실습(Simple Regression)



### Step 1. Import

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential
```



### Step 2. Preprocessing

```python
# Category 1 - Basic 모델에서, 데이터의 전처리 과정은 필요없다. 
# ndarray형식으로 데이터가 표현되어 있다.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
```



### Step 3. Modeling

```python
# Sequential modeling(4층)
# 모델명은 model
model = Sequential([
    Dense(3, input_shape=[1]),
    Dense(4),
    Dense(4),
    Dense(1),
])

# Dense layers Image
from IPython.display import Image

Image('https://cs231n.github.io/assets/nn1/neural_net2.jpeg')
```

##### 	주어진 데이터가 간단한데, 이렇게 복잡하게 4층으로 모델링 할 이유가 없다.

```python
# 1층 Sequential model
model = Sequential([
    Dense(1, input_shape=[1]),
])
```



### Step 4. Compile

```python
# optimizer(최적화는) = sgd(확률적 경사하강법 알고리즘을 사용하겠다.)
# loss(lossfunction은) = mse(Mean Squared Error함수를 사용하겠다.) 
# >> 단순 회귀(regression)에서는 mse 사용

model.compile(optimizer='sgd', loss='mse')
```



### Step 5. Fit

```python
# model을 학습(fit)시키겠다. 
#(features = xs, labels = ys, epochs(학습 횟수) = 1200, verbose(학습로그) = 0(출력안함))

model.fit(xs, ys, epochs=1200, verbose=0)
```



### Step 6. Predict

```python
# 모델에 10.0을 입력하면 어떤 값이 나올지 예측
model.predict([10.0])
```
