import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xai

#https://www.kaggle.com/camnugent/california-housing-prices
#accuracy [0.6699084043502808, 0.7100290656089783]

df=pd.read_csv("../dataset/housing.csv")
df.head()

df = df.drop(columns=['longitude', 'latitude'])

#housevalue_level = pd.cut(df.median_house_value,bins=[-np.inf,15000,30000,60000,100000,150000,200000,300000,400000,500000,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
housevalue_level = pd.qcut(df.median_house_value,q=3)
df.insert(len(df.columns),'housevalue_level',housevalue_level)


#age_range = pd.cut(df.housing_median_age,bins=[-np.inf,5,10,20,30,40,50,np.inf],labels=[0,1,2,3,4,5,6],include_lowest =True)
age_range = pd.qcut(df.housing_median_age,q=10)
df.insert(len(df.columns),'age_range',age_range)

#totalrooms_range = pd.cut(df.total_rooms,bins=[-np.inf,2000,4000,6000,8000,10000,20000,30000,np.inf],labels=[0,1,2,3,4,5,6],include_lowest =True)
totalrooms_range = pd.qcut(df.total_rooms,q=10)
df.insert(len(df.columns),'totalrooms_range',totalrooms_range)


bedrooms_range = pd.qcut(df.total_bedrooms,q=10)
df.insert(len(df.columns),'bedrooms_range',bedrooms_range)

population_range = pd.qcut(df.population,q=10)
df.insert(len(df.columns),'population_range',population_range)

households_range = pd.qcut(df.households,q=10)
df.insert(len(df.columns),'households_range',households_range)

income_range = pd.qcut(df.median_income,q=10)
df.insert(len(df.columns),'income_range',income_range)

#longitude_range = pd.qcut(df.longitude,q=10)
#df.insert(len(df.columns),'longitude_range',longitude_range)

#latitude_range = pd.qcut(df.latitude,q=10)
#df.insert(len(df.columns),'latitude_range',latitude_range)




final_df = df.drop(columns=['housevalue_level','median_house_value', 'housing_median_age','total_rooms','total_bedrooms','population','households','median_income'])
y = df[['housevalue_level']]


# #gather value dict
# columns_valuesdict = xai.getColumnvaluedict(final_df)
# columns_valuesdict.update(xai.getColumnvaluedict(y))
#
# print(columns_valuesdict)
#
# # gather value dict
# with open("analysis/house-valuedict.txt", "w") as f:
#     for col in columns_valuesdict:
#         f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.interval_encoder)+"}\n")
#


xai.encodeCategory(final_df,"age_range")
xai.encodeCategory(final_df,"totalrooms_range")
xai.encodeCategory(final_df,"bedrooms_range")
xai.encodeCategory(final_df,"population_range")
xai.encodeCategory(final_df,"households_range")
xai.encodeCategory(final_df,"income_range")
#df["longitude_range"] = encodeCategory(df["longitude_range"])
#df["latitude_range"] = encodeCategory(df["latitude_range"])
xai.encodeCategory(final_df,"ocean_proximity")

xai.encodeCategory(y,"housevalue_level")


pd.set_option("display.max_rows", None, "display.max_columns", None)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df,y,test_size=0.2,random_state=42)

print(X_train.head())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(len(X_train.columns),activation='relu'))
model.add(Dense(24,  activation='relu'))
model.add(Dense(24,  activation='relu'))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics="accuracy")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model("model-house.h5")

import shap
#print(shap.datasets.adult(display=True)[0].values)
#background = shap.maskers.Independent(X_train, max_samples=100)
#explainer = shap.Explainer(model,background)
#shap_values = explainer(X_train[:100])
#print(shap_values)
#shap.summary_plot(shap_values, X_train[:100])
# processing time: 8:06

#processing time 6:33
explainer = shap.KernelExplainer(model.predict,X_train[:10])
shap_values = explainer.shap_values(X_test,nsamples=10)
shap.summary_plot(shap_values,X_test,feature_names=final_df.columns)
