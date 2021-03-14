# xgBoost

# Tips

요약하자면

1. feature dimension의 수가 클 때,
2. Sample의 수가 작을 때
3. Computer vision 문제
4. NLP문제

에서는 사용을 지양한다.

ref.

[When to NOT use XGBoost? | Data Science and Machine Learning](https://www.kaggle.com/discussion/196542?fbclid=IwAR132c6fD_Rh3ZR1io_FInrLmSHVdPI53pw0q51kaUplR1LiDOsQ0rnVtIg)

→ 여기에서 확인해 볼 수 있다. 

- can work fine with tabular data even with billions of data points
- can work fine with lot of columns (such as >millions)
- can work fine for anything computer vision / image recognition / NLP etc. you can fit with xgboost (obviously you need to turn it into tabular data, but for that, tou need to do feature engineering before), but it happens usually specialized solutions do better (ex for line pathing in a pic: image geometric consensus sharpening + contrast combo)
- if we can say "deep learning is the bazooka of computer vision", then "XGboost is the bazooka of tabular data" (along with lightGBM)
- there's no single "best" ML model exist in the world, ML models are complementary, and in practice a "not too strong" ML model is better than the "best" model
- can simulate random forest (beware of result averaging xgboost in RF mode is strictly not RF although adhering to its principles, so you may end with results you don't expect)
- great for testing feature engineering (at least for detecting large signals in a new feature - not for small signal)
- hyperparameter tuning is not needed unless you are wanting to spend most of your time getting a slight model performance increase (just use: depth 6, learning rate depending on how long you are ready to wait, hist mode, and you are usually good to go out of the box)
- can choose between having fast results with average model performance (high learning rate) or slow result with good model performance (low learning rate)
- fast enough for most needs
- use on tabular data if you have enough RAM if you are not in distributed mode
- use in ensembles (avoid in real world usage, usually no one wants your 10+ model ensemble)
- better use LightGBM if tabular data with categoricals
- penalization (L1/L2) is useful for hand tuning if tou have an interpretability workflow afterwards bunch of objective functions but you can roll your own objective functions to optimize what you wish for isotonic(monotonic) mode per variable
- old versions work fine especially in production systems (pre 0.7 xgboost), newer xgboost versions suffered a LOT of issues (in APIs, in performance, and in consistency)
- don't work fine in tree mode if you want y=ax+b (and if you already knew you want y=ax+b, you knew already didn't want a tree-based model without a linear regression on each leaf end)
- can y=ax+b in linear mode but better use other (faster, better, and more stable) stuff for that unless you have a very specific need
- if you got choice between xgboost and SVM, SVM bad as always unless you have a very specific need matching exactly the specialty use case of SVM (that is, computing the maximum margin hyperplane, which it excels in) need gcc-4.9 or higher to compile (if I remember correctly), so in production systems you might not have gcc-4.9 or higher especially in regulated environment (think: aircraft embarked stuff) need to export model and re-use the model elsewhere with other libraries to have super fast single predictions (if aiming for sub-millisecond single prediction)
- happens (as with most boosting optimizers) to usually overfit white noise if you don't hold it back (but sometimes, it's good) 
- ex of good:  signal detection in fraud in accounting 
- ex of bad:  high frequency trading, trading in general (better hold it back)
- when you need parametric ML don't use non-parametric ML, as simple as that (usually for regulatory purposes) interpretability sometimes make literally no sense 
(ex: because combination(A,B,C) -> pred, so if you change one value you can't expect pred to move the same way in respect to the 2 other control parameters, unless you specified it explicitly - and it will still move non-linearly but at least following the control parameters)
- must reclaim RAM (garbage collect) manually the training models and datasets in the APIs else your interactive or not) session crashes over time because OOM (especially important in production systems to reclaim back the RAM from the training and the xgboost datasets)
- if you need an extremely simple model you can decipher and read, don't use xgboost unless you really want to boost one(!) single tree or use the linear mode
- getting maximum training speed on xgboost requires to understand the hyperparameters, the dataset, and the hardware you are using, so you can decide the right number of threads to use on which pinned cores to use GPU crash OOM can be a common problem anything related to "before doing supervised machine learning", such as drifting (ex: swiss franc crash fundamental analysis), manipulated data (fix your data), uncleaned data (fix your data), etc.
- don't use xgboost to replace NMAR (not missing at random) or MAR (missing at random) blindly if you expect to use traditional statistical parametric analysis on prediction outputs, don't use sgboost but use a parametric ML model
- don't use xgboost for tasks another ML model is specialized at (looking specifically at KNN for instance, unless you hit an edge case which do happen often especially in the world of nearest neighbors calculation)