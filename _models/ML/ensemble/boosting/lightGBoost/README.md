

- 참고 문서
    - [공식문서](https://lightgbm.readthedocs.io/en/latest/index.html)
    - [Gradient Boosting with Scikit-learn, ...](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
    - [What is LightGBM, how to implement it? how to fine tune the parameters? - Pushkar Mandot](https://medium.com/@pushkarmandot/what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
    
    
    
- LightGBM의 특징
    - Tree기반의 모델이 수직적으로 확장되는 반면 **LightGBM은 수평적으로 확장되는 특성을 지니고 있다.** 
    - **빠른 속도 & 적은 메모리 소모 + 좋은 성능(높은 정확도)!**
    - 데이터의 개수가 적은 데이터에는 적합하지 않다. (Leaf-Wise 특성이 강해서, 과적합이 될 우려가 크다.)
    - 
    
### [basic] Parameter introduction
- max_depth : Tree의 최대 깊이. --> 과적합이 의심되면 줄여야한다. 
- min_data_in_leaf : Leaf가 가지고 있는 최소한의 record 수.