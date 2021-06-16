# CatBoost

- Gradient Boosting 기반한 알고리즘 (ex. LightGBM, XGBM, ...)
- 다른 gbm 보다 사용하기 쉽고, 좋은 결과도 나타나는 경우가 있다.
- Symmetric Tree를 사용하기 때문에, 모델 학습시에는 오랜시간이 소요되어도, 예측시에는 몇 십배 빠른 속도를 가진다.


- Gradient Boosting 방법에 기반한 알고리즘이지만, 약간 다른 방식을 취하고 있다.
    - General algorithm of Gradient Boosting
        1. 모든 데이터 셋에 대하여 잔차를 구하며 학습하여 편의가 큰 모델을 만든다.
        2. 이전에 학습한 모델로 부터 계산된 잔차를 통해 새로운 모델을 만든다.
        3. 일정 시점까지 위를 반복
        
    - CatBoosting Algoritm
        1. 모든 시점에 대하여 잔차를 구하는데 적용하는 모델은 그 시점의 데이터에 대한 정보를 포함하지 않는다. 
        2. 나온 잔차를 통해 모델을 학습한다.
        3. 일정 시점까지 위를 반복
        
- 데이터의 크기가 클 수록 엄청난 계산 수를 요하는 작업이다.
- 이런 방식의 알고리즘을 Ordered boosting이라고 부른다.
- 이런 순차적인 알고리즘 방법 때문에 변수의 순서가 중요하다. 따라서 모델 내에서 여러가지 조합의 무작위 순열로 바꾸어주는 기능이 내장되어있다.(Ramdom Permutation) 기본적으로 아무런 세팅을 하지 않았다면 4가지 무작위 순열이 생성될것이다. 이는 오버피팅을 방지해줄 것이며, ```bagging_temperature``` 라는 파라미터를 통해서 조정가능하다. 

- categorical feature handling
    - 몇 가지 다른 GBM과는 달리 범주형 변수를 따로 원핫인코딩을 하지 않아도 된다. 내장되어있는 인코딩 함수가 있기때문. 다음과 같은 방식의 새로운 접근법으로 인코딩한다. Ordered Target encoding.
    - Ordered boosting과 유사한 방법으로, 현 시점의 데이터를 인코딩하기 위해 이전에 사용한 데이터들의 평균 값으로 인코딩한다. 이 방법은 현 시점의 데이터에 대한 Target value의 정보가 포함되지 않아 Data leakage 문제를 일으키지 않는다는 장점과 다양한 값을 나타 낼 수 있다는 장점이 있다.
    
    - 기본적으로 2개의 범주를 가지는 범주형 변수는 OneHotEncoding을 해준다. 이는 ```one_hot_max_size``` 로 최대 개수를 정해서 원핫인코딩할 범주형 변수의 기준을 만들어 줄 수 있다.
    
    
- 단점
    - sparse matrix를 학습하기 위한 기능을 제공하지 않는다.
    - LightGBM보다 학습에 많은 시간이 소요된다.
    
    
- CatBoost를 사용할만한 여러가지 상황  
    하이퍼 파라미터 튜닝은 Catboost에서 크게 중요한 측면은 아니지만, 풀어야할 문제에 대해서 올바른 파라미터를 설정하는 것은 매우 중요하다. 다음과 같은 상황이 있다.
    1. 시간의 흐름이 있는 데이터  
        시계열 데이터를 학습시키기 위해서는 ```has_time = True```으로 설정해준다.
    2. 예측 시간의 지연이 적은 곳  
        XGBM의 예측보다 무려 8배나 빠르다.
    3. 중요한 관측치가 있을때 그 관측치에 가중치를 설정할 수 있다.  
        가중치를 설정한 데이터는 무작위 순열을 생성할 때, 선택될 기회를 더 많이 가져가게 된다.  
        ex. 모든 데이터에 가중치를 부여하는 경우는 다음과 같다. ```sample_weight = [x for x in range(train.shape[0])]```
    4. 데이터 셋의 크기가 작을 때  
        ```fold_len_multiplier``` as close as 1 (must be >1)  
        ```approx_on_full_history```=True  
        이러한 파라미터를 설정하게 되면, CatBoost는 각각의 데이터 관측치의 잔차를 계산할 때 각각 다른 모델을 사용하게 된다.
    5. 데이터 셋의 크기가 클 때  
        ```task_type = 'GPU'``` 으로 속도를 올릴 수 있다. Colab에서 지원하는 오래된 GPU도 사용할 수 있다.
    6. 학습되고 있는 모델의 성능을 각각 다른 평가지표로 확인하고 싶을 때  
        ex. AUC, Log-loss 평가지표로 확인하고 싶을 때, ```custom_metric = ['AUC', 'Logloss']  
        또한, jupyter notebook에서 시각적으로 확인하고 싶다면, ```ipywidgets```를 설치해주어야하고, ```plot = True```
    7. Staged prediction & Shrink Models  
        ```staged_predict()```를 통해서 매 스테이지의 모델의 성능을 확인할 수 있다.  
        ```shrink()``` 특정 시점의 stage의 모델을 추출할 수 있다.
        [자세히 -> documentation](https://catboost.ai/)
    8. 서로 다른 상황의 결합  
        축제기간 또는 주말 또는 평일이든지 아니든지 우리는 주어진 모든 상황에서 최고의 예측값을 만들어야한다. cross-validation을 통해 여러가지 모델을 만들 수 있고 ```sum_models()```를 통해 모델을 섞을 수 있다.
        
        
### Many More…
- By default, CatBoost has an overfitting detector that stops training when CV error starts increasing. You can set parameter od_type = Iter to stop training your model after few iterations.
- We can also balance an imbalanced dataset with the class_weight parameter.
- CatBoost not only interprets important features, but it also returns important features for a given data point what are the important features.
- The code for training CatBoost is simply straight forwarded and is similar to the sklearn module. You can go through the documentation of CatBoost [here](https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/) for a better understanding.

### Goodbye to Hyper-parameter tuning?
- CatBoost is implemented by powerful theories like ordered Boosting, Random permutations. It makes sure that we are not overfitting our model. It also implements symmetric trees which eliminate parameters like (min_child_leafs ). We can further tune with parameters like learning_rate, random_strength, L2_regulariser, but the results don’t vary much.

### EndNote:
CatBoost is freaking fast when most of the features in your dataset are categorical. A model that is robust to over-fitting and with very powerful tools, what else you are waiting for? Start working on CatBoost !!!


# 참고문서
[참고 1](https://hanishrohit.medium.com/whats-so-special-about-catboost-335d64d754ae)
[참고 2](https://dailyheumsi.tistory.com/136)