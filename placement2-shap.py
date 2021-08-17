import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xai
import json

categ_num = 5

#[0.5627521872520447, 0.8307692408561707] [1.249988317489624, 0.7846153974533081] [0.7035016417503357, 0.6769230961799622] [2.6354684829711914, 0.6769230961799622]

df=pd.read_csv("../dataset/placement.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)

df = df.fillna(value={"salary": 0})
df.dropna(inplace=True)

salarylevel = pd.cut(df.salary,bins=[-np.inf,1,200000,300000,400000,np.max(df['salary'])],labels=[0,1,2,3,4],include_lowest =True)
#salarylevel = pd.qcut(df.salary,q=categ_num)
df.insert(len(df.columns),'salarylevel',salarylevel)

#sscp_range = pd.cut(df['ssc_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
sscp_range = pd.qcut(df['ssc_p'],q=20)
df.insert(len(df.columns),'sscp_range',sscp_range)

#hscp_range = pd.cut(df['hsc_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
hscp_range = pd.qcut(df['hsc_p'],q=20)
df.insert(len(df.columns),'hscp_range',hscp_range)

#degreep_range = pd.cut(df['degree_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
degreep_range = pd.qcut(df['degree_p'],q=10)
df.insert(len(df.columns),'degreep_range',hscp_range)

#mbap_range = pd.cut(df['mba_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
mbap_range = pd.qcut(df['mba_p'],q=20)
df.insert(len(df.columns),'mbap_range',mbap_range)

#etestp_range = pd.cut(df['etest_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
etestp_range = pd.qcut(df['etest_p'],q=20)
df.insert(len(df.columns),'etestp_range',etestp_range)


final_df = df.drop(columns=['sl_no', 'salary','ssc_p','hsc_p','degree_p','mba_p','etest_p'])
#y = df[['salarylevel']]

columns_valuesdict = xai.getColumnvaluedict(final_df)

# gather value dict
with open("analysis/placement-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.interval_encoder)+"}\n")


final_df["gender"] = xai.encodeCategory(final_df,"gender")
final_df["ssc_b"] = xai.encodeCategory(final_df,"ssc_b")
final_df["hsc_b"] = xai.encodeCategory(final_df,"hsc_b")
final_df["hsc_s"] = xai.encodeCategory(final_df,"hsc_s")
final_df["degree_t"] = xai.encodeCategory(final_df,"degree_t")
final_df["workex"] = xai.encodeCategory(final_df,"workex")
final_df["specialisation"] = xai.encodeCategory(final_df,"specialisation")
final_df["status"] = xai.encodeCategory(final_df,"status")

final_df["sscp_range"] = xai.encodeCategory(final_df,"sscp_range")
final_df["hscp_range"] = xai.encodeCategory(final_df,"hscp_range")
final_df["degreep_range"] = xai.encodeCategory(final_df,"degreep_range")
final_df["mbap_range"] = xai.encodeCategory(final_df,"mbap_range")
final_df["etestp_range"] = xai.encodeCategory(final_df,"etestp_range")
final_df["salarylevel"] = xai.encodeCategory(final_df,"salarylevel")

y =  final_df[['salarylevel']]


#checking the unique values for categorical columns
#categorical_columns = final_df.columns[final_df.dtypes == object]

#one hot encoding on categorical variables
# for col in categorical_columns:
#     if(len(final_df[col].unique()) == 2):
#         final_df[col] = pd.get_dummies(final_df[col], drop_first=True)
#
# final_df = pd.get_dummies(final_df)



#print(final_df.columns)

# split train and test data set
df.sort_values('salarylevel',inplace=True)

X = final_df.drop(columns=['salarylevel'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
# model =GradientBoostingClassifier(random_state=77)
# model.fit(scaler.fit_transform(X_train), y_train.values.ravel())
# y_pred = model.predict(scaler.transform(X_test))
#
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_pred)
# print('Accuracy Score on test data :',score)

import tensorflow as tf
from tensorflow import keras
#save model to file
model = keras.models.load_model("model-placement.h5")


#processing time: 2:18
import shap
explainer = shap.KernelExplainer(model.predict,X_train)
shap_values = explainer.shap_values(X_test,nsamples=1000)
shap.summary_plot(shap_values,X_test,feature_names=X.columns)
