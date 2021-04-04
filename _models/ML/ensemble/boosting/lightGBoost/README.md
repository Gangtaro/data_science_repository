

### 이해를 돕는 자료
- 참고문서
    - [공식문서](https://lightgbm.readthedocs.io/en/latest/index.html)
    - [참고논문, LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NIPS 2017)](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
    - [Gradient Boosting with Scikit-learn, ...](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
    - [What is LightGBM, how to implement it? how to fine tune the parameters? - Pushkar Mandot](https://medium.com/@pushkarmandot/what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
        + [파생 notebook, LightGBM Classifier in Python](https://www.kaggle.com/prashant111/lightgbm-classifier-in-python)
    - [xgBoosting and lightGBM parameter explanation](https://sites.google.com/view/lauraepp/parameters)
    parameter tuning 시에 보고 참고하길 권장

    - cross-entropy
        - [Cross-entropy 의 이해: 정보이론과의 관계, 3months](https://3months.tistory.com/436)
        - [초보를 위한 정보이론 안내서 - Cross Entropy 파헤쳐보기, hyunw.kim](https://hyunw.kim/blog/2017/10/26/Cross_Entropy.html)
    - Objective function
        - [머신러닝 Loss Function 이해하기, keepdev](https://keepdev.tistory.com/48)
    
- LightGBM의 특징
    - Tree기반의 모델이 수직적으로 확장되는 반면 **LightGBM은 수평적으로 확장되는 특성을 지니고 있다.** 
    - **빠른 속도 & 적은 메모리 소모 + 좋은 성능(높은 정확도)!**
    - 데이터의 개수가 적은 데이터에는 적합하지 않다. (Leaf-Wise 특성이 강해서, 과적합이 될 우려가 크다.)
    - 
    
## Parameter tuning
LightGBM은 [**leaf-wise tree**](https://lightgbm.readthedocs.io/en/latest/Features.html#leaf-wise-best-first-tree-growth) 기반의 알고리즘을 사용한다. **depth-wise** growth와 비교하여 **leaf-wise** growth는 더 빨리 값에 수렴한다. 하지만 이는 즉 하이퍼 파라미터를 제대로 설정해주지 않으면 쉽게 과적합(Over-fitting)될 수 있다는 것을 의미한다. 그래서 우리는 다음과 같은 사실을 먼저 인지해야한다.

### Leaf-Wise Tree를 위한 파라미터 튜닝 올바른 방향으로 하는 방법
1. ```num_leaves```  
    - 사실상 모델의 복잡도 구성에 있어서 가장 중요한 파라미터. 
    - 이론적으로, ```num_leaves = 2^(max_depth)``` 이렇게 설정해두면 **depth-wise tree**와 같은 수의 잎을 만들어 내게 된다. **근데 이것은 실제 수행에 있어서 그리 좋지 못하다.** 왜냐하면 **leaf-wise tree**는 같은 잎의 개수를 가정할 때, **depth-wise tree**보다 더 깊은 tree를 만들어 내게 된다. 제한이 없는 depth에 대하여 과적합을 유도할 수 있다는 말이다. 
    - 그래서, ```num_leaves < 2^(max_depth)```와 같이 ```num_leaves```를 설정해주면 더 좋은 성능을 기대 할 수 있다.

2. ```min_data_in_leaf```
    - **leaf-wise**에서 과적합을 방지하는데에 중요한 파라미터
    - 이 값의 최적의 값은 training samples의 개수와 ```num_leaves```의 값에 의존적이다.
    - 이 값을 크게 설정해두면 tree가 너무 깊어지는 것을 방지할 수 있지만, 과소적합이 우려된다. 실제 수행에서, 큰 데이터에 대해, 천에서 백정도가 적당하다.
    
3. ```max_depth```
    - tree의 최대 깊이에 제한을 걸어둠으로서 leaf-wise가 빠져들기 쉬운 과적합에서 벗어날 수 있다.
    
### For Better Accuracy
[This comes from ...](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#decrease-num-leaves)
- Use large ```max_bin``` (may be slower)
- Use small l```earning_rate``` with large num_iterations
- Use large ```num_leaves``` (may cause over-fitting)
- Use bigger training data
- Try ```dart```

### Deal with Over-fitting
[This comes from ...](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#decrease-num-leaves)
- Use small ```max_bin```
- Use small ```num_leaves```
- Use ```min_data_in_leaf``` and ```min_sum_hessian_in_leaf```
- Use bagging by set ```bagging_fraction``` and ```bagging_freq```
- Use feature sub-sampling by set ```feature_fraction```
- Use bigger training data
- Try ```lambda_l1```, ```lambda_l2``` and ```min_gain_to_split``` for regularization
- Try ```max_depth``` to avoid growing deep tree
- Try ```extra_trees```
- Try increasing ```path_smooth```



    
### [basic] Parameter introduction
- **```max_depth``` :** (defaults to -1, meaning infinite depth, **typically 6, usually [3, 12]**)
    - Range : [1, ~], int
    - Decription : 각각의 훈련된 Tree의 최대 깊이, **크기가 커질수록 좋은 결과가 나오나, 과적합의 속도도 빨라진다.** --> 과적합이 의심되면 줄여야한다. **과적합과 비례**
    - **tips :** 필요에 따라 무한정으로 이 값을 늘려 가지수를 늘릴 수 있다. **가장 예민한 파라미터로, 가장 먼저 최적의 값을 찾아내야 한다.**
    - have to figure out :
        - Beliefs
            Unlimited depth is essential for training models whose branching is one-sided (instead of balanced branching).
            Such as for long chain of features, like 50 to get to the expected real rule.
    - [How to tune this](#Leaf-Wise-Tree를-위한-파라미터-튜닝-올바른-방향으로-하는-방법)

    
- **```min_data_in_leaf``` :** (defaults to 20, **typically 100**)
    - Range : [0, ~], int
    - Discription : <u>Leaf가 가지고 있어야 하는 최소한의 관측치 수</u>,  **과적합과 비례**
    - **tips :** 어떤 방향으로 흘러가고 있는지 모를때는 조정하지 않기를 권장. 관측치의 개수가 100개 정도로 매우 적다면 이 수치를 줄이고 반대는 필요시에 늘려야한다. 진짜 아무것도 모르겠는 상태에는 xgboost와 같이 1로 놔둔다.
    - [How to tune this](#Leaf-Wise-Tree를-위한-파라미터-튜닝-올바른-방향으로-하는-방법)
    
- **```feature_fraction``` :** (defaults to 1, 사용시에 보통 0.7 사용, **Boosting : rf 일때 사용**)
    - Range : [0, 1], float 
    - Description : iteration마다 랜덤으로 선택하는 feature의 비율. 
    - **tips :** 미세하게 조정할 필요는 없다. 이 방법이 항상 더 좋은 것은 아니다. (SGD is not always better than GD)
    
- **```bagging_fraction``` :** (defaults to 1, 사용시에 보통 0.7 사용)
    - Range : [0, 1], float
    - Description : iteration마다 랜덤으로 선택하는 관측치의 비율 (Row sampling). 
    - **tips :** 미세하게 조정할 필요는 없다. 이 방법이 항상 더 좋은 것은 아니다. (SGD is not always better than GD)
    
- **```early_stopping_round``` :** (defaults to NULL, typically 50)
    - Range : [0, ~]
    - **Needs :** 평가로 사용할 데이터를 지정해줘야한다. + [more information about ```metrics```](#[Metric-parameters])
    - Description : iteration 진행중, 여기서 지정해준 ```early_stopping_round```에서 Validation data의 지표가 이전보다 증가하지 않았다면 학습을 중단한다.
    - **tips :** Validation data set을 추가하지 않았다면 쓸모없는 파라미터 -> lgb.dataset 참고, **```learning_rate```**와 같이 적절히 조정해주면 좋다.
    
    - details
        이 값을 너무 높게 설정해두는것은 운 덕분에 멈추는 것을 허용하지 않으면서도 과적합의 위험이 있다. 
    - examples
        ```python
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        lgb_model = lgb.LGBMClassifier(early_stopping_round = 50, metric = 'multi_logloss')
        lgb_model.fit(X_train, y_train,
                     eval_set = [(X_valid, y_valid)]) # NEEDS
        ```
        
- **```reg_alpha``` :** (Defaults to 0, typically 0, aliases : ```lambda_l1```)
    - Range : [0, ~]
    - Description : [**L1-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.
        

- **```reg_lambda``` :** (Defaults to 0, typically 0, aliases : ```lambda_l2```)
    - Range : [0, ~]
    - Description : [**L2-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.


- **```min_gain_to_split``` :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Desription : 하나의 노드에서 가지를 치기위해 있어야하는 최소한의 관측치 수 설정.
    - **tips :** 뭐하고 있는지 모를땐 놔도라, **DEEP tree model**을 설계할때 유용하게 사용된다.
    
- **```max_cat_threshold``` :** (Defaults to 32, typically 32)
    - Range : [1, ~]
    - Description : 범주형 변수 가운데, 과적합을 막기위해 LightGBM 알고리즘이 새로운 변수 max_cat_group으로 만들어서 그룹 경계를 다시 설정했다.
   
### [Core Parameters]
- **```task``` :** (Defaults to 'train')
    - Description : 데이터에 대하여 모델이 어떤 임무를 목적으로 학습을 수행할 것인지 구체화 하는 역할을 한다. ~~각각 어떤 차이가 있는지 인지하지 못했다.~~
        - **```train``` :** 모델학습이 목적일 때, aliases: training 
        - **```predict``` :** 새로운 값 예측이 목적일 때, aliases: prediction, test
        - **```convert_model``` :** (아직 무슨 역할인지 잘 모르겠으나 학습한 모델을 다른 프로그램 형식의 파일로 변환하는 기능을 지원하는 것 같음. C++의 cpp 형식의 파일로의 변환과 같은.)for converting model file into if-else format, see more information in [Convert Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#convert-parameters)
        - **```refit``` :** 미리 학습한 모델에 대해 **새로운 데이터**를 통해 누적시켜 학습시키고 싶을 때 사용, aliases: refit_tree
        - **```save_binary``` :** (어떤 역할을 수행하고 어떻게 사용하는지 올바르게 알지는 못했지만, 학습을 하기전에 Train data( and validation data)를 이진형태의 파일로 저장하고 동시에 이를 학습하게 한다.)load train (and validation) data then save dataset to binary file. Typical usage: save_binary first, then run multiple train tasks in parallel using the saved binary file
        - **Note:** can be used only in CLI version; for language-specific packages you can use the correspondent functions

- **```application``` :** (Defaults to 'regression') aliases : **objective**
    - Description : **Regressor, Classifier**(Multi, binary)**, Ranker, ...** 어떤 모델을 사용할지 결정했다면, 모델에 쓰일 가중치(또는 회귀계수)를 학습할때 사용되는 목적함수를 결정해야하는데, 여기서 목적함수(objective)로 쓰일 손실함수(loss function)들을 지정할 수 있다. 즉, **분류기+목적함수**의 조합을 지정할 수 있다.
      
        **다음은 분류기에 따른 목적함수로 쓰일 수 있는 모든 모델들이다**
        - regression application
            - **```regression``` :** L2 loss, MSE, 즉, Ridge regression의 방법을 채택한다.
            - **```regression_l1``` :** L1 loss, MAE, LASSO regression의 방법을 채택한다.
            - **```huber``` :** [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) ?
            - **```fair``` :** [Fair loss](https://www.kaggle.com/c/allstate-claims-severity/discussion/24520) ?
            - **```poisson``` :** 회귀모델 자체를 [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression)을 채택한다.
            - **```quantile``` :** [Quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) ?
            - **```mape``` :** [MAPE loss](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)를 목적함수로 사용.
            - **```gamma``` :** 회귀모델 자체를 gamma regression with log-link으로 채택한다. 이는 insurance claims severity과 같은 데이터를 모델링 할 때나 타겟 변수가 [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Occurrence_and_applications)인 데이터를 모델링 할 때, 유용할 수 있다.
            - **```tweedie``` :** 회귀모델 자체를 tweedie regression with log-link으로 채택한다. 이는 total loss in insurance와 같은 데이터를 모델링 할 때나 타겟 변수가 [tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Occurrence_and_applications)인 데이터를 모델링 할 때, 유용할 수 있다.        
            
        - binary classification application
            - **```binary``` :** binary log loss classification or logistic regression
             
        - multi-class classification application
            - **```multiclass``` :** [softmax](https://en.wikipedia.org/wiki/Softmax_function) objective function
            - **```multivlassova``` :** [One-vs-all](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) binary objective function
            - **```num_class``` :** should be as well
            
        - cross-entropy application
            - **```cross_entropy``` :** objective function for cross-entropy (with optional linear weights)
            - **```cross_entropy_lambda``` :** alternative parameterization of cross-entropy
             
        - ranking application
            - **```lambdarank``` :** [lambdarank](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf) objective. [```label_gain```](https://lightgbm.readthedocs.io/en/latest/Parameters.html#label_gain) can be used to set the gain (weight) of int label and all values in label must be smaller than number of elements in ```label_gain```
            - **```rank_xendcg``` :** [XE_NDCG_MART](https://arxiv.org/abs/1911.09798) ranking objective function
            - ```rank_xendcg``` is faster than and achieves the similar performance as lambdarank
            - label should be int type, and larger number represents the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

- **```boosting``` :** (Defaults to 'gdbt')
    - Description : Boosting method. In belief, the boster method has a huge impact on traning performance.
    - Options
        - **```rf```   :**  (Random Forest) which is builds a Random Forest model **(not boosting)**  
            useful when avoiding cross-validation is possible step.
        - **```gbdt``` :**  (Gradient Boosted Decision Trees) which is the default boosting method using Decision Trees and Stochastic Gradient Descent
        - **```dart``` :**  (Dropout Additive Regression Trees) which is a method employing the Dropout method from Neural Networks
            ```dart``` is similar to DropOut in neural networks, except you are applying this idea to trees(droping trees ramdomly)
        - **```goss``` :**  (Gradient-based One-Side Sampling) which is a method using subsampling to converge faster/better using Stochastic Gradient Descent.
            ```goss``` is an adaptive novel method for reconstructing gradients based on other gradients, to converge faster while trying to provide better performance 

- **```num_boost_round``` :** (Defaults to 100) aliases : **n_estimators**, num_iterations, num_iteration, n_iter, num_tree, num_round, ...
    - Range : [1, ~], int
    - Description : Number of boosting iterations. (모델을 학습하는데 boosting을 통해 가중치를 학습하는 round 수)
    - **tips :** 보통, 이 값을 높게 설정할 수록 더 좋은 결과가 나오게 된다(<u>과적합되기 전까지!</u>). 따라서 이 파라미터를 설정할때는 언제 과적합이 발생하는지 주시하는 것이 중요하다. 자동적으로 라운드를 종료시키기 위해 ```early_stopping_round```와 같이 사용하면 좋다.  
        **보통 다음 두 방법을 통해 이 값을 설정한다.**
        - 값을 높게 설정해주고 + ```early_stoppping_round```와 같이 사용
        - Cross-validation을 통해 발견된 반복수의 평균의 1.1배를 이 값으로 사용

- **```learning_rate``` :** (Defaults to 0.1, typically 0.1, 0.05, 0.001, 0.003)
    - Range : (0, ~], float
    - Description : 학습속도를 조절해준다. 매번의 iteration마다 training loss가 나오는데 여기에 learning rate가 곱해져서 학습 속도를 조절하게 된다. 이를 작게 설정하면 조금 더 섬세하게 학습이 되고 과적합까지 가는데 천천히 가는 대신에 iteration의 수가 많아져야 하기 때문에 학습속도가 느리게 된다. 반대로 이 값이 커지면 학습속도는 빠를 수 있으나 최적의 값을 찾는대 섬세하지 않을 수 있다.
    - **tips :** 하이퍼 파라미터 튜닝 할 때는, 이 값을 큰 값으로 설정해두는 것이 좋다.
    
    - details
        - ```learning_rate```는 한 번 정하면 더 이상 변화를 주는 값이 아니다, **하이퍼 파라미터 튜닝의 대상으로 여기는 것은 좋은 것이 아니다.**
        - ```learning_rate```는 '학습속도'&'성능'과 **Trade-off** 관계에 있다.
        - 보통 다음과 같은 값으로 사용한다.(typically)
            + when hyper-parameter tuning, **learning_rate = 0.10**
            + when training the model,     **learning_rate = 0.05**

- **```num_leaves``` :** (Defaults to 31)
    - Range : [1, ~], int
    - Description : 학습된 Tree의 최대 잎의 개수
    - **tips :** 
    
    - details
        - 반드시 ```max_depth```와 함께 튜닝해야한다.
        - 이 값이 커질 수록 training data에 더 잘맞게 학습이 된다. (validation data에 잘맞게는 당연히 아니다. -> 과적합의 문제가 있다.)
        - **두 번째로 민감한 변수.**
    
    - [How to tune this](#Leaf-Wise-Tree를-위한-파라미터-튜닝-올바른-방향으로-하는-방법)

### [Metric parameters]
**모델 학습 중, 분류기가 매 iteration마다 만들어지는데**, 그 때마다 분류기의 성능을 설정해둔 ```eval_set``` (= [(X_valid, y_valid)] ) evaluation 데이터셋을 상대로 성능을 평가한다. 이때 ```metric```을 지정해주지 않으면 자동적으로 분류기가 알아서 평가지표를 선택한다. 여기서는 ```metric```을 지정해주고 싶을 때, 어떤 평가지표를 사용할 수 있는지 알 수 있다.

- ```l1```, absolute loss, aliases: ```mean_absolute_error```, ```mae```, ```regression_l1```
- ```l2```, square loss, aliases: ```mean_squared_error```, ```mse```, ```regression_l2```, ```regression```
- ```rmse```, root square loss, aliases: ```root_mean_squared_error```, ```l2_root```
- ```quantile```, [Quantile regression](https://en.wikipedia.org/wiki/Quantile_regression)
- ```mape```, [MAPE loss](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error), aliases: ```mean_absolute_percentage_error```
- ```huber```, [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
- ```fair```, [Fair loss](https://www.kaggle.com/c/allstate-claims-severity/discussion/24520)
- ```poisson```, negative log-likelihood for [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression)
- ```gamma```, negative log-likelihood for **Gamma** regression
- ```gamma_deviance```, residual deviance for **Gamma** regression
- ```tweedie```, negative log-likelihood for **Tweedie** regression
- ```ndcg```, [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG), aliases: ```lambdarank```, ```rank_xendcg```, ```xendcg```, ```xe_ndcg```, ```xe_ndcg_mart```, ```xendcg_mart```
- ```map```, [MAP](https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/), aliases: ```mean_average_precision```
- ```auc```, [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
- ```average_precision```, [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- ```binary_logloss```, [log loss](https://en.wikipedia.org/wiki/Cross_entropy), aliases: ```binary```
- ```binary_error```, for one sample: ```0``` for correct classification, ```1``` for error classification
- ```auc_mu```, [AUC-mu](http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf)
- ```multi_logloss```, log loss for multi-class classification, aliases: ```multiclass```, ```softmax```, ```multiclassova```, ```multiclass_ova```, ```ova```, ```ovr```
- ```multi_error```, error rate for multi-class classification
- ```cross_entropy```, cross-entropy (with optional linear weights), aliases: ```xentropy```
- ```cross_entropy_lambda```, “intensity-weighted” cross-entropy, aliases: ```xentlambda```
- ```kullback_leibler```, [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence), aliases: ```kldiv```
- support multiple metrics, separated by ```,```




num class ?








