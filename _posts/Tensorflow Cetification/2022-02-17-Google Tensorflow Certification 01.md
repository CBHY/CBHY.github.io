# Google Tensorflow Certification 01



## 개요(기초)



### 용어 정리

#### 	Epoch : 전체 데이터가 모두 학습한 단위

#### Batch : 1개의 Epoch에서 **여러 개의 샘플을 나누어 학습하는 단위**(OOM(Out Of Memory) 방지)

##### 	ex) **batch_size=10** => (1000개 이미지) / (batch_size=10) = 총 **100개의 batch**

##### 		  batch_size=20 => (1000개 이미지) / (batch_size=20) = 총 **50개의 batch**

##### 		  **batch_size=50** => (1000개 이미지) / (batch_size=50) = 총 **20개의 batch**

#### 	Loss : 정답값 과의 오차

#### 	Accuracy : 정확도

#### 	Supervised Learning(지도학습) : 입력 데이터와 출력 데이터 모두 존제하는 학습

##### 		- 분류(Classification)

##### 		- 회귀(Regression)

##### 		(Google Tensorflow Certification에서는 지도학습만 다룬다.)

#### 	Unsupervised Learning(비지도학습) : 입력 데이터만 존제하는 학습

##### 		- 군집(Clustering)





### 딥러닝의 학습 순서 ★

#### Step 1. Import : 필요한 모듈(라이브러리)를 Import 합니다.

#### Step 2. 전처리(preprocessing) : 학습에 필요한 데이터 전처리를 수행합니다,

#### Step 3. 모델링(model) : 모델을 정의합니다.

##### 	(Google Tensorflow Certification에서는 Sequential 모델만 사용)

#### Step 4. 컴파일(compile) : 모델을 생성합니다.

#### Step 5. 학습(fit) : 모델을 학습시킵니다.

###### 		Step 6. 예측(predict) 



### 선형함수와 오차

#### 선형함수 모델 : Y(예측 데이터) = W(가중치) * X(입력 데이터) + b(bias)

##### 	단순하게, W(가중치)의 변화는 기울기의 변화, b(bias)의 변화는 절편의 변화, 학습하면서 수시로 변하는 값.

#### 오차함수(Loss Function) : 오차를 나타내는 함수

###### 		ex) 만약, 5개의 데이터에 오차가 2, -2, 0, -2, 2 라면,

###### 				오차의 총합은 0이므로, 오차가 존제하지 않는 것 처럼 처리된다. 이것을 방지하기위해, MAE, MSE 함수를 사용한다.

##### 	- MAE(Mean Absolute Error) : 오차의 절댓값 평균

###### 			(abs(2) + abs(-2) + abs(0) + abs(-2) + abs(2)) / 5 = 1.6

##### 	- MSE(Mean Squared Error) : 오차의 제곱 평균

###### 			(2^2 + -(2)^2 + 0^2 + (-2)^2 + 2^2) / 5 = 3.2

