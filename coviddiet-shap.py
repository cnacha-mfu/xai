import pandas as pd
import numpy as np
import xai
#https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset
#accurary [3.41115140914917, 0.6938775777816772] [0.7415797710418701, 0.7755101919174194]

categ_num = 4

df=pd.read_csv("../dataset/covid-diet.csv")
df.head()

df['Obesity'] = df['Obesity'].fillna(df['Obesity'].mean())
df['Confirmed'] = df['Confirmed'].fillna(0)
df['Deaths'] = df['Deaths'].fillna(0)
df['Recovered'] = df['Recovered'].fillna(0)
df['Active'] = df['Active'].fillna(0)
df.isnull().sum()


df.dropna(inplace=True)
df = df.drop(columns=['Undernourished',
      'Country', 'Unit (all except Population)']) #'Confirmed','Recovered','Active'

# convert to categorical
for col in df.columns:
    qgroup = 10
    if col == 'Deaths':
        qgroup = categ_num
    colrange = pd.qcut(df[col],q=qgroup,duplicates='drop')
    df[col] = colrange

# convert to number range

for col in df.columns:
    xai.encodeCategory(df,col)

# split train and test data set
X = df.drop(columns=['Deaths'])
y = df[['Deaths']]
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
model = keras.models.load_model("model-coviddiet.h5")


#processing time:5:46
import shap
explainer = shap.KernelExplainer(model.predict,X_train)
shap_values = explainer.shap_values(X_test,nsamples=1000)
shap.summary_plot(shap_values,X_test,feature_names=X.columns)
