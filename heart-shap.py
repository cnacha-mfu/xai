import pandas as pd
import numpy as np
import xai
#https://www.kaggle.com/ronitf/heart-disease-uci
#accurary [0.4888615012168884, 0.8351648449897766]
df=pd.read_csv("../dataset/heart.csv")
df.head()
print(df.describe())

#group columns continuous data
df.dropna(inplace=True)


#agerange = pd.cut(df['age'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
agerange = pd.qcut(df['age'],q=10)
df.insert(len(df.columns),'agerange',agerange)


#bpsrange = pd.cut(df.trestbps,bins=[-np.inf,90,115,125,135,145,155,165,175,185,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
bpsrange = pd.qcut(df.trestbps,q=5)
df.insert(len(df.columns),'bpsrange',bpsrange)

#cholrange = pd.cut(df.chol,bins=[-np.inf,126,170,214,258,302,345,389,433,np.inf],labels=[0,1,2,3,4,5,6,7,8],include_lowest =True)
cholrange = pd.qcut(df.chol,q=10)
df.insert(len(df.columns),'cholrange',bpsrange)

#thalachrange = pd.cut(df.thalach,bins=[-np.inf,71,84,97,110,124,137,150,163,176,189,np.inf],labels=[0,1,2,3,4,5,6,7,8,9,10],include_lowest =True)
thalachrange = pd.qcut(df.thalach,q=10)
df.insert(len(df.columns),'thalachrange',thalachrange)

#oldpeakrange = pd.cut(df.oldpeak,bins=[-np.inf,0.62,1.24,1.86,2.48,3.10,3.72,4.34,np.inf],labels=[0,1,2,3,4,5,6,7],include_lowest =True)
oldpeakrange = pd.qcut(df.oldpeak,q=8,duplicates='drop')
df.insert(len(df.columns),'oldpeakrange',oldpeakrange)


final_df = df.drop(columns=['age','trestbps','chol','thalach','oldpeak','target'])
y = df['target']

xai.encodeCategory(final_df,"agerange")
xai.encodeCategory(final_df,"bpsrange")
xai.encodeCategory(final_df,"cholrange")
xai.encodeCategory(final_df,"thalachrange")
ai.encodeCategory(final_df,"oldpeakrange")

X = final_df

# split train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print(X.columns)

#normalise the value to 0-1 range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras
#save model to file
model = keras.models.load_model("model-heart.h5")


#processing time:5:46
import shap
explainer = shap.KernelExplainer(model.predict,X_train)
shap_values = explainer.shap_values(X_test,nsamples=1000)
shap.summary_plot(shap_values,X_test,feature_names=final_df.columns)
