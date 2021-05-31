{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 신용카드 사용자 연체 예측 AI 경진대회\n",
    "## 월간 데이콘 14 | 금융 | 정형 | Logloss\n",
    "- 2021.04.05 ~ 2021.05.24 17:59\n",
    "\n",
    "## 개요\n",
    "\n",
    "1. 주제\n",
    "\n",
    "신용카드 사용자 데이터를 보고 사용자의 대금 연체 정도를 예측하는 알고리즘 개발 \n",
    "\n",
    "\n",
    "2. 배경\n",
    "\n",
    "신용카드사는 신용카드 신청자가 제출한 개인정보와 데이터를 활용해 신용 점수를 산정합니다. 신용카드사는 이 신용 점수를 활용해 신청자의 향후 채무 불이행과 신용카드 대급 연체 가능성을 예측합니다.\n",
    "현재 많은 금융업계는 인공지능(AI)를 활용한 금융 서비스를 구현하고자 합니다. 사용자의 대금 연체 정도를 예측할 수 있는 인공지능 알고리즘을 개발해 금융업계에 제안할 수 있는 인사이트를 발굴해주세요!\n",
    "\n",
    "\n",
    "3. 대회 설명\n",
    "\n",
    "신용카드 사용자들의 개인 신상정보 데이터로 **사용자의 신용카드 대금 연체 정도를 예측**\n",
    "\n",
    "---\n",
    "- 참여목적: \n",
    "    - LGBM의 개념 공부 후, 이를 응용해보는 기회를 가진다.\n",
    "    - 이 프로젝트를 통해 신용카드 관련 데이터분석에서 어떠한 변수가 학습모델에 유의하는지 파악할 기회를 가지고 인사이트를 확보한다.\n",
    "    \n",
    "- 느낀점:\n",
    "    - 분석시 채택한 모델은 ```LGBM```과```XGBM``` 딱 두 가지 모델이였다.(```SVM```또한 학습시켜봤지만 터무니 없는 score로 배제하였다.) 두 학습모델을 통해서 분석한 결과 전반적인 학습시에 성능은 LGBM이 XGBM보다 좋은 것으로 파악되었다. 하지만 두 모델 모두 너무 train data에 과적합 되는 모습을 보이며, 가장 raw한 data에서 학습한 모델이 public score에서 가장 좋은 결과를 나타내었다. **26451**개의 데이터로 **10000**개의 데이터를 예측하기에는 너무 학습 데이터가 적은 것이 문제였다고 생각한다. (실제로 변수를 추가할수록 Cross-validation 평가 지표는 계속 좋아졌지만, public score 값은 눈에 띄게 오르는 현상이 발견되었다.) LGBM과 XGBM 모두 feature importance 지표를 통해 확인해본 결과, 주로 Categorical 변수가 중요도가 낮게 나오는것을 확인하였다. 범주형 변수의 특성상 그런 결과가 나왔을거라 예상했기에 이를 통해서 더 나은 지표로 변환하는 것을 떠올려봤지만 범주형 변수들의 조합에서는 그렇게 특별한 인사이트가 나오지 않았다.(+ 학습 결과에도 별로 좋은 효과는 나타나지 않았다.) 따라서 연속형 변수끼리의 조합 만들어 학습력을 높이길 원했고, 실제로 그런식으로 모델 학습을 진행하였다. 위에서 말했듯이, 제일 처음에 학습시킨 raw data에 가까운 데이터로 학습한 모델의 public score를 넘지 못했고, 이는 모델 학습에 있어서, 올바르고 유의한 변수를 만들어내지 못했다라는 판단을 하게했다. 하지만 상위 private score를 기록한 notebook을 확인 후, 사용했던 학습 모델에서 좋지 못한 결과가 나왔라는게 드러났다. 대부분의 사람들이 CatBoost Model을 사용하여 좋은 결과를 기록했고, 좀 더 다양한 모델을 통해서 학습시켜보지 않았던 잘못이라고 결론지었다.\n",
    "    \n",
    "- 고쳐나갈점 및 배운점:\n",
    "    - LightGBM과 XGBM이 꼭 모든 데이터에서 최고의 성능을 발휘하는 것은 아니다. 데이터의 특성에 따라 심한 과적합을 발생시키기도 한다.\n",
    "    - CatBoost도 아직 충분히 좋은 결과를 만들어내고 있다.\n",
    "    - 데이터 분석 및 예측 매뉴얼 :\n",
    "        1. 간단한 EDA\n",
    "        2. 최고 간결한 Data Preprocessing (handling Missing Values, Outliers, ...)\n",
    "        3. 모델 선택\n",
    "            1. 분석에 사용가능할만한 모든 학습 모델 파악\n",
    "            2. 타 모델보다 확연하게 좋은 결과를 나타내는 학습 모델 2-3개 간추리기\n",
    "        4. 선택한 모델에 맞고, 분석의 취지에 알맞는 Feature를 재 생성 및 편집하여 예측의 정확도를 올린다.\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
