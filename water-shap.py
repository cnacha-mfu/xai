import pandas as pd
import numpy as np
import xai
#https://www.kaggle.com/adityakadiwal/water-potability
#accurary  [0.6195583939552307, 0.667344868183136], max accuracy is : 0.7182729840278625
df=pd.read_csv("../dataset/water_potability.csv")
df.head()

#group columns continuous data
#df.dropna(inplace=True)

df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())

X = df.drop(columns=['Potability'])
y = df[['Potability']]


# convert to categorical
qgroup = 10
for col in X.columns:
    colrange = pd.qcut(df[col],q=qgroup,duplicates='drop')
    X[col] = colrange

# convert interval to number range
for col in X.columns:
    xai.encodeCategory(X,col)

# split train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras
#save model to file
model = keras.models.load_model("model-water.h5")


#processing time:
import shap
explainer = shap.KernelExplainer(model.predict,X_train[:100])
shap_values = explainer.shap_values(X_test,nsamples=1000)
shap.summary_plot(shap_values,X_test,feature_names=X.columns)
