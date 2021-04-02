

### 이해를 돕는 자료
- [공식문서](https://lightgbm.readthedocs.io/en/latest/index.html)
    - [Gradient Boosting with Scikit-learn, ...](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
    - [What is LightGBM, how to implement it? how to fine tune the parameters? - Pushkar Mandot](https://medium.com/@pushkarmandot/what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
    - [xgBoosting and lightGBM parameter explanation](https://sites.google.com/view/lauraepp/parameters)  
    parameter tuning 시에 보고 참고하길 권장
    
- 참고문서
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
    
### [basic] Parameter introduction
- **max_depth :** (defaults to -1, meaning infinite depth, **typically 6, usually [3, 12]**)
    - Range : [1, ~], int
    - Decription : 각각의 훈련된 Tree의 최대 깊이, **크기가 커질수록 좋은 결과가 나오나, 과적합의 속도도 빨라진다.** --> 과적합이 의심되면 줄여야한다. **과적합과 비례**
    - **tips :** 필요에 따라 무한정으로 이 값을 늘려 가지수를 늘릴 수 있다. **가장 예민한 파라미터로, 가장 먼저 최적의 값을 찾아내야 한다.**
    - have to figure out :
        - Beliefs
            Unlimited depth is essential for training models whose branching is one-sided (instead of balanced branching).
            Such as for long chain of features, like 50 to get to the expected real rule.

    
- **min_data_in_leaf :** (defaults to 20, **typically 100**)
    - Range : [0, ~], int
    - Discription : Leaf가 가지고 있는 최소한의 관측치 수,  **과적합과 비례**
    - **tips :** 어떤 방향으로 흘러가고 있는지 모를때는 조정하지 않기를 권장. 관측치의 개수가 100개 정도로 매우 적다면 이 수치를 줄이고 반대는 필요시에 늘려야한다. 진짜 아무것도 모르겠는 상태에는 xgboost와 같이 1로 놔둔다.
    
- **feature_fraction :** (defaults to 1, 사용시에 보통 0.7 사용, **Boosting : rf 일때 사용**)
    - Range : [0, 1], float 
    - Description : iteration마다 랜덤으로 선택하는 feature의 비율. 
    - **tips :** 미세하게 조정할 필요는 없다. 이 방법이 항상 더 좋은 것은 아니다. (SGD is not always better than GD)
    
- **bagging_fraction :** (defaults to 1, 사용시에 보통 0.7 사용)
    - Range : [0, 1], float
    - Description : iteration마다 랜덤으로 선택하는 관측치의 비율 (Row sampling). 
    - **tips :** 미세하게 조정할 필요는 없다. 이 방법이 항상 더 좋은 것은 아니다. (SGD is not always better than GD)
    
- **early_stopping_round :** (defakults to NULL, typically 50)
    - Range : [0, ~]
    - Description : iteration 진행중, 여기서 지정해준 early_stopping_round에서 Validation data의 지표가 이전보다 증가하지 않았다면 학습을 중단한다.
    - **tips :** Validation data set을 추가하지 않았다면 쓸모없는 파라미터 -> lgb.dataset 참고, **Learning rate**와 같이 적절히 조정해주면 좋다.
    
    - details
        이 값을 너무 높게 설정해두는것은 운 덕분에 멈추는 것을 허용하지 않으면서도 과적합의 위험이 있다. 
        
- **reg_alpha :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Description : [**L1-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.
        

- **reg_lambda :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Description : [**L2-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.


- **min_gain_to_split :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Desription : 하나의 노드에서 가지를 치기위해 있어야하는 최소한의 관측치 수 설정.
    - **tips :** 뭐하고 있는지 모를땐 놔도라, **DEEP tree model**을 설계할때 유용하게 사용된다.
    
- **max_cat_threshold :** (Defaults to 32, typically 32)
    - Range : [1, ~]
    - Description : 범주형 변수 가운데, 과적합을 막기위해 LightGBM 알고리즘이 새로운 변수 max_cat_group으로 만들어서 그룹 경계를 다시 설정했다.
   
### [Core Parameters]
- **task :** (Defaults to 'train')
    - Description : 데이터에 대하여 모델이 어떤 임무를 목적으로 학습을 수행할 것인지 구체화 하는 역할을 한다. ~~각각 어떤 차이가 있는지 인지하지 못했다.~~
        - **```train``` :** 모델학습이 목적일 때, aliases: training 
        - **```predict``` :** 새로운 값 예측이 목적일 때, aliases: prediction, test
        - **```convert_model``` :** (아직 무슨 역할인지 잘 모르겠으나 학습한 모델을 다른 프로그램 형식의 파일로 변환하는 기능을 지원하는 것 같음. C++의 cpp 형식의 파일로의 변환과 같은.)for converting model file into if-else format, see more information in [Convert Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#convert-parameters)
        - **```refit``` :** 미리 학습한 모델에 대해 **새로운 데이터**를 통해 누적시켜 학습시키고 싶을 때 사용, aliases: refit_tree
        - **```save_binary``` :** (어떤 역할을 수행하고 어떻게 사용하는지 올바르게 알지는 못했지만, 학습을 하기전에 Train data( and validation data)를 이진형태의 파일로 저장하고 동시에 이를 학습하게 한다.)load train (and validation) data then save dataset to binary file. Typical usage: save_binary first, then run multiple train tasks in parallel using the saved binary file
        - **Note:** can be used only in CLI version; for language-specific packages you can use the correspondent functions

- **application :** (Defaults to 'regression') aliases : **objective**
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

- **boosting :** (Defaults to 'gdbt')
    - Description : Boosting method. In belief, the boster method has a huge impact on traning performance.
    - Options
        - **```rf```   :**  (Random Forest) which is builds a Random Forest model **(not boosting)**  
            useful when avoiding cross-validation is possible step.
        - **```gbdt``` :**  (Gradient Boosted Decision Trees) which is the default boosting method using Decision Trees and Stochastic Gradient Descent
        - **```dart``` :**  (Dropout Additive Regression Trees) which is a method employing the Dropout method from Neural Networks
            ```dart``` is similar to DropOut in neural networks, except you are applying this idea to trees(droping trees ramdomly)
        - **```goss``` :**  (Gradient-based One-Side Sampling) which is a method using subsampling to converge faster/better using Stochastic Gradient Descent.
            ```goss``` is an adaptive novel method for reconstructing gradients based on other gradients, to converge faster while trying to provide better performance 

- **num_boost_round :** 














