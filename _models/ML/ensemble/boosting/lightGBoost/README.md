

### 이해를 돕는 자료
- [공식문서](https://lightgbm.readthedocs.io/en/latest/index.html)
    - [Gradient Boosting with Scikit-learn, ...](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
    - [What is LightGBM, how to implement it? how to fine tune the parameters? - Pushkar Mandot](https://medium.com/@pushkarmandot/what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
    - [xgBoosting and lightGBM parameter explanation](https://sites.google.com/view/lauraepp/parameters)  
    parameter tuning 시에 보고 참고하길 권장
    
    
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
        
- **lambda_l1 :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Description : [**L1-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.
        

- **lambda_l2 :** (Defaults to 0, typically 0)
    - Range : [0, ~]
    - Description : [**L2-정규화**](https://www.notion.so/L1-L2-Regularization-2a53907271bb4bbabe44fd4cf0c140c8)
    - **tips :** 뭐하고 있는지 모를땐 놔둬라
    
    - details
        Regularization은 항상 좋은 방법은 아니다. Data set과 Weight에 의존적이다.























