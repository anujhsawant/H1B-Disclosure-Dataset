#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("cleaned_h1b.csv")

df.head()

# Dropping columns passing redundant information.
df.drop(['WORKSITE_POSTAL_CODE','EMPLOYER_POSTAL_CODE'],axis=1,inplace=True)

numerical = df.select_dtypes(include=np.number)
categorical = df.select_dtypes(exclude=np.number)

df.head()

for col in list(categorical.columns):
    try:
        categorical[col] = LabelEncoder().fit_transform(categorical[col])
    except:
        print('Issue in label encoding.')

df.columns

categorical.dtypes

categorical.CASE_STATUS.value_counts(0)

for col in list(categorical.columns):
    try:
        categorical[col] = categorical[col].astype(int)
    except:
        print('Error!')

categorical.dtypes

categorical.drop('NAICS_CODE',1,inplace=True)

# Merging Dataframes.
encoded_dataframe = pd.concat([numerical,categorical],axis=1)

x = encoded_dataframe.drop('CASE_STATUS',1)
y = encoded_dataframe.CASE_STATUS
random_under_sampler = RandomUnderSampler(random_state=0)
x_us,y_us = random_under_sampler.fit_resample(x,y)
y_us = pd.Series(y_us)
x_us = pd.DataFrame(x_us,columns=x.columns)
x_train,x_test,y_train,y_test = train_test_split(x_us,y_us,test_size=0.3,random_state=0)

y_us.value_counts()

param_rfc = {
    'n_estimators' : [50,100,200],
    'min_samples_split' : [2,3,5],
    'max_depth' : [2,3,8]
}

param_xgb = {
    'n_estimators' : [50,100,200],
    'max_depth' : [2,3,8],
    'learning_rate' : [0.01,0.02,0.03,0.1]
}

param_lgb = {
    'n_estimators' : [50,100,200],
    'max_depth' : [2,3,8],
    'learning_rate' : [0.01,0.02,0.03,0.1],
    'min_split_gain' : [0.0,0.01,0.1,0.3,0.5]
}

# Random Forest
model_rfc = RandomForestClassifier(class_weight='balanced',n_estimators=100,max_depth=20)
y_pred = model_rfc.fit(x_train,y_train).predict(x_test)
print('Train Accuracy Score: {}'.format(model_rfc.score(x_train,y_train)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

model_rfc

# AdaBoost
model_adaboost = AdaBoostClassifier(learning_rate=0.1)
ada_bagging_model = BaggingClassifier(base_estimator=model_adaboost)
y_pred = ada_bagging_model.fit(x_train,y_train).predict(x_test)
y_pred_train = ada_bagging_model.fit(x_train,y_train).predict(x_train)
print('Train Accuracy Score: {}'.format(ada_bagging_model.score(x_train,y_train)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# XGBoost
model_xgboost = XGBClassifier(learning_rate=0.1)
y_pred = model_xgboost.fit(x_train,y_train).predict(x_test)
print('Train Accuracy Score: {}'.format(model_xgboost.score(x_train,y_train)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# LightGBM
model_lgb = LGBMClassifier(class_weight='balanced',learning_rate=0.1)
y_pred = model_lgb.fit(x_train,y_train).predict(x_test)
y_pred_train = model_lgb.fit(x_train,y_train).predict(x_train)
print('Train Accuracy Score: {}'.format(accuracy_score(y_train,y_pred_train)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),)

# XGBoost Feature Importance
feature_importance = pd.Series(model_xgboost.feature_importances_,index=list(x_train.columns))
feature_importance = feature_importance[feature_importance>0.03].sort_values(ascending=True)
feature_importance.plot('barh')

# Random Forest Feature Importance
feature_importance = pd.Series(model_rfc.feature_importances_,index=list(x_train.columns))
feature_importance = feature_importance[feature_importance>0.03].sort_values(ascending=True)
feature_importance.plot('barh')

# LightGBM Feature Importance
feature_importance = pd.Series(model_lgb.feature_importances_,index=list(x_train.columns))
feature_importance = feature_importance[feature_importance>400].sort_values(ascending=True)
feature_importance.plot('barh')

# Stacking Classifier
classifiers = [model_rfc,model_xgboost,model_lgb]
meta_classifier = KNeighborsClassifier()
stacker_model = StackingClassifier(classifiers=classifiers,meta_classifier=meta_classifier)
y_pred = stacker_model.fit(x_train,y_train).predict(x_test)
print('Train Accuracy Score: {}'.format(stacker_model.score(x_train,y_train)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print('Test Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

y.shape

df[df.DECISION_DURATION<20].DECISION_DURATION.hist()

df.DECISION_DURATION.value_counts()

