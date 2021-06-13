# import modules
import warnings
warnings.filterwarnings('ignore')

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, log_loss, f1_score

import random
from math import floor
from scipy.stats import mode, scoreatpercentile

print("Seaborn version : ", sns.__version__)
sns.set()
#sns.set_style('whitegrid')
sns.set_color_codes()
sns.set_theme(style="ticks", palette="pastel")



# upload data
train = pd.read_csv('/Users/gangtaro/competition_data/DACON/14thMonthlyDacon/open/train.csv',
                   index_col=0)
test = pd.read_csv('/Users/gangtaro/competition_data/DACON/14thMonthlyDacon/open/test.csv',
                  index_col=0)
submit = pd.read_csv('/Users/gangtaro/competition_data/DACON/14thMonthlyDacon/open/sample_submission.csv')


for df in [train, test]:
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['gender', 'car', 'reality', 'work_phone', 'phone', 'email']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df.drop('FLAG_MOBIL', axis = 1, inplace = True)
    df['adj_DAYS_EMPLOYED_replace_0'] = -df.DAYS_EMPLOYED.replace({365243 : 0})
    df['DAYS_EMPLOYED_missing'] = (df.DAYS_EMPLOYED == 365243).astype('int')
    df['adj_begin_month'] = -df.begin_month
    df['adj_income_type'] = df.income_type
    df.loc[df.income_type == 'Student', 'adj_income_type'] = 'Working'
    df['adj_edu_type'] = df.edu_type 
    df.loc[df.edu_type == 'Academic degree', 'adj_edu_type'] = 'Higher education'
    df['adj_family_type'] = df['family_type']
    df['adj_family_type'].loc[(df.family_type == 'Single / not married')&(df.family_size - df.child_num == 2)] = 'Married'
    df['exp_num'] = 0
    df['exp_num'].loc[df.family_type == 'Married'] = 2
    df['exp_num'].loc[df.family_type == 'Civil marriage'] = 2
    df['exp_num'].loc[df.family_type == 'Separated'] = 1
    df['exp_num'].loc[df.family_type == 'Single / not married'] = 1
    df['exp_num'].loc[df.family_type == 'Widow'] = 1
    df['odd_family_size'] = 0
    df['odd_family_size'].loc[(df.family_size - df.child_num) != df.exp_num] = 1
    df['_single_parents'] = ((df.family_type == 'Single / not married')&(df.child_num != 0)).astype('int')
    df['_single_live'] = (df.family_size == 1).astype('int')
    df['adj_occyp_type'] = df.occyp_type.fillna('missing')
    df['_missing_occyp_type'] = df.occyp_type.isna().astype('int')
    df.loc[(df.DAYS_EMPLOYED == 365243)&(df.occyp_type.isna()), 'adj_occyp_type'] = 'inoccyp'
    df.loc[(df.DAYS_EMPLOYED != 365243)&(df.occyp_type.isna()), 'adj_occyp_type'] = 'non_enter'
    df['_age'] = -df.DAYS_BIRTH/365.25

    df['ID'] = \
    df['gender'].astype('str') + \
    df['car'].astype('str') + \
    df['reality'].astype('str') + '_' + \
    df['child_num'].astype('str') + '_' + \
    df['income_total'].astype('str') + '_' + \
    df['income_type'].astype('str') + '_' + \
    df['family_type'].astype('str') + '_' + \
    df['house_type'].astype('str') + '_' + \
    df['phone'].astype('str') + '_' + \
    df['email'].astype('str') + '_' + \
    df['family_size'].astype('str') + '_' + \
    df['DAYS_BIRTH'].astype('str') + '_' + \
    df['DAYS_EMPLOYED'].astype('str') + '_' + \
    df['occyp_type'].astype('str') 
    
    df['_card_num'] = df.groupby('ID').ID.transform(len)
    
    df['adj_begin_month'] = -df.begin_month
    df['_begin_month_max'] = df.groupby('ID').adj_begin_month.transform(max)
    df['_begin_month_mean'] = df.groupby('ID').adj_begin_month.transform(np.mean)
    df['_begin_month_min'] = df.groupby('ID').adj_begin_month.transform(min)
    

personal_info = train.drop(['credit', 'begin_month'], axis = 1).drop_duplicates(subset="ID", keep='first', inplace=False, ignore_index=True)
personal_info_test = pd.concat([train.drop(['credit'], axis = 1), test]).drop(['begin_month'], axis = 1).drop_duplicates(subset="ID", keep='first', inplace=False, ignore_index=True)

for personal_df in [personal_info, personal_info_test]:
    child_fee = (personal_df.income_total/personal_df.family_size)[(personal_df._age > 33) & (personal_df._age < 37) & (personal_df.child_num == 1)].mean()

    def child_fee_age_weights(x) : 
        from scipy.stats import norm
        sd = personal_df._age.std()
        return norm(35, scale = sd).pdf(x) / norm(35, scale = sd).pdf(35)
    personal_df['child_fees'] = (np.log(personal_df.child_num + 1)/np.log(2)) * (child_fee) * personal_df._age.apply(child_fee_age_weights)

    personal_df.income_total.median()*0.1
    def car_weight(x):
        _med = personal_df.income_total.median()
        _max = personal_df.income_total.max()
        if x < _med : 
            return 1
        else:
            return 1+(x-_med)/(_max-_med)*5
    personal_df['car_fees'] = personal_df.income_total.median()*0.1*personal_df.car*personal_df.income_total.apply(car_weight)

    personal_df['_save_income'] = personal_df.income_total - personal_df.child_fees - personal_df.car_fees

    personal_df['_ability_income_per_age'] = 0
    personal_df['_ability_employ_per_age'] = 0
    personal_df['_ability_income_per_emp'] = 0
    for i in range(len(personal_df)) : 
        L_age = personal_df._age[i] - 3
        R_age = personal_df._age[i] + 3
        _gen = personal_df.gender[i]
        _ages_df = personal_df[['income_total', 'adj_DAYS_EMPLOYED_replace_0']][(personal_df._age > L_age) & (personal_df._age < R_age) & (personal_df.gender == _gen)]
        _med_income = _ages_df['income_total'].median()
        _std_income = _ages_df['income_total'].std()
        _med_employ = _ages_df['adj_DAYS_EMPLOYED_replace_0'].median()
        _std_employ = _ages_df['adj_DAYS_EMPLOYED_replace_0'].std()
        _n_df       = _ages_df.shape[0]
        personal_df.loc[i, '_ability_income_per_age'] = (personal_df.income_total.iloc[i] - _med_income) / (_std_income /np.sqrt(_n_df))
        personal_df.loc[i, '_ability_employ_per_age'] = (personal_df.adj_DAYS_EMPLOYED_replace_0.iloc[i] - _med_employ) / (_std_employ/np.sqrt(_n_df))

        if personal_df.adj_DAYS_EMPLOYED_replace_0.iloc[i] != 0:    
            L_emp = personal_df.adj_DAYS_EMPLOYED_replace_0 - 365
            R_emp = personal_df.adj_DAYS_EMPLOYED_replace_0 + 365
            _emps_df = personal_df.income_total[(personal_df.adj_DAYS_EMPLOYED_replace_0 > L_emp)&(personal_df.adj_DAYS_EMPLOYED_replace_0 < R_emp)]
            _med = _emps_df.median()
            _std = _emps_df.std()
            personal_df.loc[i, '_ability_income_per_emp'] = (personal_df.income_total.iloc[i] - _med)/(_std/np.sqrt(_n_df))

train  = pd.merge(train, personal_info[['ID', '_save_income', 'child_fees', 'car_fees', '_ability_income_per_age', '_ability_employ_per_age', '_ability_income_per_emp']], on = 'ID', how = 'left')
test = pd.merge(test, personal_info_test[['ID', '_save_income', 'child_fees', 'car_fees', '_ability_income_per_age', '_ability_employ_per_age', '_ability_income_per_emp']], on = 'ID', how = 'left')


for df in [train, test]:
    df['_income_per_cards']  = df.income_total / np.log(1+df._card_num)
    df['_save_per_cards']    = df._save_income / np.log(1+df._card_num)

    df['_income_per_family'] = df.income_total / df.family_size
    df['_save_per_family']   = df._save_income / df.family_size
    
    df['_age'] = df['_age'].apply(lambda x: floor(x))
    df['adj_DAYS_EMPLOYED_mm'] = df['adj_DAYS_EMPLOYED_replace_0'].apply(lambda x: floor(x/30))
    df['adj_DAYS_EMPLOYED_yy'] = df['adj_DAYS_EMPLOYED_replace_0'].apply(lambda x: floor(x/365.25))
    df['log_income_total'] = df['income_total'].apply(lambda x: np.log(1+x))
    
features = ['gender', 'car', 'reality', 'child_num', 'adj_income_type', 'adj_edu_type', 
            'adj_family_type', 'house_type', '_age', 'adj_DAYS_EMPLOYED_replace_0', 'adj_DAYS_EMPLOYED_mm', 'adj_DAYS_EMPLOYED_yy',
            'adj_begin_month', 'work_phone', 'phone', 'email', 'adj_occyp_type', 
            'ID', '_begin_month_min',
            '_save_income','_income_per_family', 'log_income_total',
            '_ability_income_per_age']

numerical_feats = train[features].dtypes[train[features].dtypes != "object"].index.tolist()
#numerical_feats.remove('credit')
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = train[features].dtypes[train[features].dtypes == "object"].index.tolist()
print("Number of Categorical features: ", len(categorical_feats))


target = 'credit'
X = train.drop(target, axis=1)[features]
y = train[target]
X_test = test[features]

def objective(trial):
    param = {
      "random_state":42,
      'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
      'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
      "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
      "max_depth":trial.suggest_int("max_depth", 4, 16),
      'random_strength' :trial.suggest_int('random_strength', 0, 100),
      "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "max_bin": trial.suggest_int("max_bin", 200, 500),
      'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2)

    cat_features = categorical_feats + ['gender', 'car', 'reality', 'phone', 'email', 'work_phone']
    cat = CatBoostClassifier(**param)
    cat.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid,y_valid)],
          early_stopping_rounds=35,cat_features=cat_features,
          verbose=100)
    cat_pred = cat.predict_proba(X_valid)
    log_score = log_loss(y_valid, cat_pred)

    return log_score

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name = 'cat_parameter_opt',
    direction = 'minimize',
    sampler = sampler,
)
study.optimize(objective, n_trials=10)
print("Best Score:",study.best_value)
print("Best trial",study.best_trial.params)