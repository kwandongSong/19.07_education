import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LogisticRegression
%matplotlib inline

rawdata=pd.read_csv("/notebooks/data/onechu/modeling/kwandong/final_plus_rolling_2.csv") #train 데이터 6개월
rawdata_test=pd.read_csv("/notebooks/data/onechu/modeling/kwandong/churn7_plus_rolling_2.csv")

data=rawdata.copy()
data_test=rawdata_test.copy()

#NULL 값 확인
def check_null(data):
    data=data.fillna(0)
    print('-----------------------------------------------------------------------')
    for feature in data.columns:
        if data[feature].isnull().sum() > 0:
            print('column {} null data count : {}'.format(feature,  data[feature].isnull().sum()))
    return data

data=check_null(data)
data_test=check_null(data_test)

def make_dummies(what_column,data):
    dummies=pd.get_dummies(data[what_column],prefix=what_column)
    data=pd.concat([data,dummies],axis=1)
    data=data.drop([what_column],axis=1)
    return data

columns_for_dummies=['mno_nm','rf_cluster_num','past_churn','pass_yn','second_cluster_num','DP13_new_dan_yn','DP13_nomal_yn','DP29_yn', 'DP26_yn']

for i in columns_for_dummies:
    data= make_dummies(i,data)
    data_test= make_dummies(i,data_test)

data.info()
data_test.info()

droplist=['sex_clsf_cd','age_cat','init_prchs_date','last_prchs_date']
"""
data
""" #'cust_payment_amt_min','clsf_new_cat_n' ,'free_pref' , 'sett_target_cpn_amt_var','days_from_register' ,  'DP29_count',  'DP29_nuniq','DP26_amt', 'w2v_prod_clr_6','w2v_prod_clr_9'
dropped_data = data.drop(droplist, axis=1)
X=dropped_data.drop(['churn'], axis=1)
y=dropped_data['churn']

#val 비율 8:2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0) # 이전 0
X_train.shape, X_val.shape
"""
data_val
"""
dropped_data_test = data_test.drop(droplist, axis=1)
X_test=dropped_data_test.drop(['churn'], axis=1)
y_test=dropped_data_test['churn']


from sklearn.metrics import classification_report
"""

6개월 train 데이터 8:2 train: val 

"""
# 랜덤 포레스트
RF= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=15, max_features=11, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

RF.fit(X_train, y_train)

# Make predictions
predictions = RF.predict(X_val)
probs = RF.predict_proba(X_val)
display(predictions)

score = RF.score(X_val, y_val)
print("Accuracy: ", score)
print(classification_report(y_val, predictions))
data['churn'].value_counts()

# get_ipython().magic('matplotlib inline')
# confusion_matrix = pd.DataFrame(
#     confusion_matrix(y_val, predictions),
#     columns=["Predicted False", "Predicted True"],
#     index=["Actual False", "Actual True"]
# )
# display(confusion_matrix)

#ROC 커브 그리기
fpr, tpr, threshold = roc_curve(y_val, probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

"""

3개월 test 데이터 

"""

predictions = RF.predict(X_test)
probs = RF.predict_proba(X_test)
display(predictions)
score = RF.score(X_test, y_test)
print("Accuracy: ", score)
print(classification_report(y_test, predictions))
data_test['churn'].value_counts()

# get_ipython().magic('matplotlib inline')
# confusion_matrix = pd.DataFrame(
#     confusion_matrix(y_test, predictions),
#     columns=["Predicted False", "Predicted True"],
#     index=["Actual False", "Actual True"]
# )
# display(confusion_matrix)

#confusion_matrix(y_test, predictions)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

from sklearn.metrics import classification_report
from xgboost import plot_importance
"""

6개월 train 데이터 8:2 train: val 

"""
#feature importance 뽑기 위해 xg
#XGB=XGBClassifier(n_jobs=-1,Eta=0.1,min_child_weight=3, n_estimators=10, max_depth=3, random_state=0 )
# XGB=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.8, gamma=0.1,
#               learning_rate=0.05, max_delta_step=0, max_depth=5,
#               min_child_weight=1, missing=None, n_estimators=300, n_jobs=1,
#               nthread=-1, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=10,
#               silent=None, subsample=0.8, verbosity=1)
#XGB= xgb.XGBClassifier( leaning_rate= 0.02, max_bin=10,num_leaves=16, subsample=0.7, max_depth=4, subsample_freq=2, colsample_bytree= 0.3, min_child_samples=500,seed=99,n_estimators=300, objective= 'binary:logistic',n_jobs=-1)
XGB=XGBClassifier(gamma=0.1, subsmaple=1.0, colsample_bytree=1.0, n_estimators=500,max_depth=10, min_child_weight=10, learning_rate=0.01, objective= 'binary:logistic', n_jobs=-1)
XGB.fit(X_train, y_train)

# Make predictions
predictions = XGB.predict(X_val)
probs = XGB.predict_proba(X_val)
display(predictions)

score = XGB.score(X_val, y_val)
print("Accuracy: ", score)
print(classification_report(y_val, predictions))
data['churn'].value_counts()


# confusion_matrix(y_val, predictions)
# display(confusion_matrix)

#ROC 커브 그리기
fpr, tpr, threshold = roc_curve(y_val, probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# feature importance - F score XG꺼 씀
fig, ax = plt.subplots(figsize=(20,15))
plot_importance(XGB, ax=ax)


"""

3개월 test 데이터 

"""

predictions = XGB.predict(X_test)
probs = XGB.predict_proba(X_test)
display(predictions)
score = XGB.score(X_test, y_test)
print("Accuracy: ", score)
print(classification_report(y_test, predictions))
data_test['churn'].value_counts()

# get_ipython().magic('matplotlib inline')
# confusion_matrix = pd.DataFrame(
#     confusion_matrix(y_test, predictions),
#     columns=["Predicted False", "Predicted True"],
#     index=["Actual False", "Actual True"]
# )
# display(confusion_matrix)

#confusion_matrix(y_test, predictions)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

models = [
    #     RandomForestClassifier(random_state=0, n_jobs=-1,
    #                            n_estimators=100, max_depth=3),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=15, max_features=11, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=2, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=300,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False),

    #    XGBClassifier(n_jobs=-1,Eta=0.1,min_child_weight=3, n_estimators=10, max_depth=3, random_state=0 ),
    XGBClassifier(gamma=0.1, subsmaple=1.0, colsample_bytree=1.0, n_estimators=500, max_depth=10, min_child_weight=10,
                  learning_rate=0.01, objective='binary:logistic', n_jobs=-1),

    LogisticRegression(C=1.0, max_iter=140, dual=False, n_jobs=-1)

]

S_train, S_test = stacking(models,
                           # X_train, y_train, X_test,
                           X_train, y_train, X_test,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=10,  # 몇번 실행

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)

model = XGBClassifier(gamma=0.1, subsmaple=1.0, colsample_bytree=1.0, n_estimators=500, max_depth=10,
                      min_child_weight=10, learning_rate=0.01, objective='binary:logistic', n_jobs=-1)

model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
print('Final accuracy_score: [%.4f]' % accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

fpr, tpr, threshold = roc_curve(y_test, probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# 그리드로 큰 틀 잡기
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [300], 'max_depth': [10, 15], 'min_samples_leaf': [1, 2], 'max_features': [11],
               'bootstrap': [True, False], 'random_state': [0, 20, 42], 'n_jobs': [-1]}]
RF_G = RandomForestClassifier()
grid_search = GridSearchCV(RF_G, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)
RF_G = grid_search.best_estimator_
RF_G

#랜덤으로 세부 조정
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

RF_R = RandomForestClassifier()
param_dist = {"max_depth": [10, 20],
              "max_features": sp_randint(1, 15),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False]
             }

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(RF_R, param_distributions=param_dist,
                                   n_iter=n_iter_search)

random_search.fit(X_train, y_train)
RF_R = grid_search.best_estimator_
RF_R


import lime
import lime.lime_tabular

predict_fn_rf = lambda X_test: RF.predict_proba(X_test).astype(float)
X_values = X_test.values
explainer = lime.lime_tabular.LimeTabularExplainer(X_values,feature_names = X_test.columns,class_names=['잔존','이탈'],kernel_width=100)

#3개월치 테스트 데이터 new_id 보기
X_test.index.tolist()

#확인해볼 타겟 new_id(한 사람)
target_id = '8256356212841621953-4779977290935141705'
#3개월치 테스트 데이터 new_id 하나 골라서 values보기
X_test.loc[[target_id ]]
#3개월치 테스트 데이터 new_id 하나 골라서 churn보기
y_test.loc[[target_id]]

choosen_instance = X_test.loc[[target_id]].values[0]
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=100)
exp.show_in_notebook(show_all=False)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 함수화 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 6개월

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
%matplotlib inline



data = pd.read_csv('/notebooks/data/onechu/Original_Data(do_not_touch)/ebook_20180101_20180630_2.csv.gz',usecols=['prchs_id','partition_dt','prod_amt','insd_usermbr_no','insd_device_id']) #partition_dt

#data.info()

data['new_id'] = data["insd_usermbr_no"].astype('str') + data["insd_device_id"].astype('str')
del data["insd_usermbr_no"]
del data["insd_device_id"]


pd.to_datetime(data['partition_dt'])
data=data.sort_values(by='partition_dt')
id_data= data['new_id']
id_data=id_data.drop_duplicates()
id_data

data.index=data['new_id']
data=data.drop('new_id',axis=1)
rawdata=data.copy()
# data.index=data['new_id']
# data=data.drop('new_id',axis=1)

data=pd.pivot_table(data,index="new_id",columns='partition_dt',values='prod_amt')
data= data.T
#data_p= data.pivot(index='new_id',columns='partition_dt',values='prod_amt')
#data=data.fillna(0)
data_7mean = data.rolling(window=7,min_periods=1).mean()
data_7mean=data_7mean.T
rol_avg_amt_7days=data_7mean.iloc[:, 167:174 ] #175:181

data_30mean = data.rolling(window=30,min_periods=1).mean()
data_30mean=data_30mean.T
data_30mean=data_30mean.iloc[:, 167:174 ]
# 167, 174 17일 ~ 23일
data=pd.read_csv("/notebooks/data/onechu/churn/churn7_train_0825.csv") #partition_dt
#data.set_index('new_id',inplace=True)
#!!!!!!!!!!!!!

rol_avg_amt_7days.rename(columns = {20180617: "rolling_average_M_7_day17"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180618: "rolling_average_M_7_day18"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180619: "rolling_average_M_7_day19"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180620: "rolling_average_M_7_day20"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180621: "rolling_average_M_7_day21"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180622: "rolling_average_M_7_day22"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180623: "rolling_average_M_7_day23"}, inplace = True)
result = pd.merge(data,rol_avg_amt_7days,right_index=True,left_index=True)



data_30mean.rename(columns = {20180617: "rolling_average_M_30_day17"}, inplace = True)
data_30mean.rename(columns = {20180618: "rolling_average_M_30_day18"}, inplace = True)
data_30mean.rename(columns = {20180619: "rolling_average_M_30_day19"}, inplace = True)
data_30mean.rename(columns = {20180620: "rolling_average_M_30_day20"}, inplace = True)
data_30mean.rename(columns = {20180621: "rolling_average_M_30_day21"}, inplace = True)
data_30mean.rename(columns = {20180622: "rolling_average_M_30_day22"}, inplace = True)
data_30mean.rename(columns = {20180623: "rolling_average_M_30_day23"}, inplace = True)
result = pd.merge(result,data_30mean,right_index=True,left_index=True)

result.to_csv('final_plus_rolling_pre.csv',index_label=False)
data = pd.read_csv('final_plus_rolling_pre.csv') #partition_dt
fre_data=pd.pivot_table(rawdata,index="new_id",columns='partition_dt',values='prchs_id', aggfunc="count")

fre_data=fre_data.T
data_7mean = fre_data.rolling(window=7,min_periods=1).mean()
data_7mean = data_7mean.T

rol_avg_freq_7days=data_7mean.iloc[:, 167:174 ]

rol_avg_freq_7days.rename(columns = {20180617: "rolling_average_F_7_day17"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180618: "rolling_average_F_7_day18"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180619: "rolling_average_F_7_day19"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180620: "rolling_average_F_7_day20"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180621: "rolling_average_F_7_day21"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180622: "rolling_average_F_7_day22"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180623: "rolling_average_F_7_day23"}, inplace = True)
result = pd.merge(data,rol_avg_freq_7days,right_index=True,left_index=True)
data_30mean = fre_data.rolling(window=30,min_periods=1).mean()
data_30mean=data_30mean.T
data_30mean=data_30mean.iloc[:, 167:174 ]


data_30mean.rename(columns = {20180617: "rolling_average_F_30_day17"}, inplace = True)
data_30mean.rename(columns = {20180618: "rolling_average_F_30_day18"}, inplace = True)
data_30mean.rename(columns = {20180619: "rolling_average_F_30_day19"}, inplace = True)
data_30mean.rename(columns = {20180620: "rolling_average_F_30_day20"}, inplace = True)
data_30mean.rename(columns = {20180621: "rolling_average_F_30_day21"}, inplace = True)
data_30mean.rename(columns = {20180622: "rolling_average_F_30_day22"}, inplace = True)
data_30mean.rename(columns = {20180623: "rolling_average_F_30_day23"}, inplace = True)
result = pd.merge(result,data_30mean,right_index=True,left_index=True)


#data.set_index('new_id',inplace=True)


#result
result.to_csv('final_plus_rolling.csv',index_label=False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 함수화 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3개월

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
%matplotlib inline



data = pd.read_csv('/notebooks/data/onechu/Original_Data(do_not_touch)/ebook_20180701_20180931_2.csv.gz',usecols=['prchs_id','partition_dt','prod_amt','insd_usermbr_no','insd_device_id']) #partition_dt

#data.info()

data['new_id'] = data["insd_usermbr_no"].astype('str') + data["insd_device_id"].astype('str')
del data["insd_usermbr_no"]
del data["insd_device_id"]


pd.to_datetime(data['partition_dt'])
data=data.sort_values(by='partition_dt')
id_data= data['new_id']
id_data=id_data.drop_duplicates()
id_data

data.index=data['new_id']
data=data.drop('new_id',axis=1)
rawdata=data.copy()
# data.index=data['new_id']
# data=data.drop('new_id',axis=1)

data=pd.pivot_table(data,index="new_id",columns='partition_dt',values='prod_amt')
data= data.T
#data_p= data.pivot(index='new_id',columns='partition_dt',values='prod_amt')
#data=data.fillna(0)
data_7mean = data.rolling(window=7,min_periods=1).mean()
data_7mean=data_7mean.T
rol_avg_amt_7days=data_7mean.iloc[:, 78:85 ] #175:181

data_30mean = data.rolling(window=30,min_periods=1).mean()
data_30mean=data_30mean.T
data_30mean=data_30mean.iloc[:, 78:85 ]
# 167, 174
data = pd.read_csv('../churn/churn7_test.csv') #partition_dt
#data.set_index('new_id',inplace=True)
#!!!!!!!!!!!!!

rol_avg_amt_7days.rename(columns = {20180917: "rolling_average_M_7_day17"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180918: "rolling_average_M_7_day18"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180919: "rolling_average_M_7_day19"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180920: "rolling_average_M_7_day20"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180921: "rolling_average_M_7_day21"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180922: "rolling_average_M_7_day22"}, inplace = True)
rol_avg_amt_7days.rename(columns = {20180923: "rolling_average_M_7_day23"}, inplace = True)
result = pd.merge(data,rol_avg_amt_7days,right_index=True,left_index=True)



data_30mean.rename(columns = {20180917: "rolling_average_M_30_day17"}, inplace = True)
data_30mean.rename(columns = {20180918: "rolling_average_M_30_day18"}, inplace = True)
data_30mean.rename(columns = {20180919: "rolling_average_M_30_day19"}, inplace = True)
data_30mean.rename(columns = {20180920: "rolling_average_M_30_day20"}, inplace = True)
data_30mean.rename(columns = {20180921: "rolling_average_M_30_day21"}, inplace = True)
data_30mean.rename(columns = {20180922: "rolling_average_M_30_day22"}, inplace = True)
data_30mean.rename(columns = {20180923: "rolling_average_M_30_day23"}, inplace = True)
result = pd.merge(result,data_30mean,right_index=True,left_index=True)

result.to_csv('churn7_plus_rolling_pre.csv',index_label=False)
data = pd.read_csv('churn7_plus_rolling_pre.csv') #partition_dt
fre_data=pd.pivot_table(rawdata,index="new_id",columns='partition_dt',values='prchs_id', aggfunc="count")

fre_data=fre_data.T
data_7mean = fre_data.rolling(window=7,min_periods=1).mean()
data_7mean = data_7mean.T

rol_avg_freq_7days=data_7mean.iloc[:, 78:85 ]

rol_avg_freq_7days.rename(columns = {20180917: "rolling_average_F_7_day17"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180918: "rolling_average_F_7_day18"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180919: "rolling_average_F_7_day19"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180920: "rolling_average_F_7_day20"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180921: "rolling_average_F_7_day21"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180922: "rolling_average_F_7_day22"}, inplace = True)
rol_avg_freq_7days.rename(columns = {20180923: "rolling_average_F_7_day23"}, inplace = True)
result = pd.merge(data,rol_avg_freq_7days,right_index=True,left_index=True)
data_30mean = fre_data.rolling(window=30,min_periods=1).mean()
data_30mean=data_30mean.T
data_30mean=data_30mean.iloc[:, 78:85 ]


data_30mean.rename(columns = {20180917: "rolling_average_F_30_day17"}, inplace = True)
data_30mean.rename(columns = {20180918: "rolling_average_F_30_day18"}, inplace = True)
data_30mean.rename(columns = {20180919: "rolling_average_F_30_day19"}, inplace = True)
data_30mean.rename(columns = {20180920: "rolling_average_F_30_day20"}, inplace = True)
data_30mean.rename(columns = {20180921: "rolling_average_F_30_day21"}, inplace = True)
data_30mean.rename(columns = {20180922: "rolling_average_F_30_day22"}, inplace = True)
data_30mean.rename(columns = {20180923: "rolling_average_F_30_day23"}, inplace = True)

result = pd.merge(result,data_30mean,right_index=True,left_index=True)


#data.set_index('new_id',inplace=True)


#result
result.to_csv('churn7_plus_rolling.csv',index_label=False)
