import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
import xai
import tensorflow_decision_forests as tfdf

#https://www.kaggle.com/camnugent/california-housing-prices
#accuracy [0.6699084043502808, 0.7100290656089783,0.7093023061752319]

df=pd.read_csv("../dataset/housing.csv")
df.head()

df = df.drop(columns=['longitude', 'latitude'])

housevalue_level = pd.qcut(df.median_house_value,q=3)
df.insert(len(df.columns),'housevalue_level',housevalue_level)

df = df.drop(columns=['median_house_value'])

xai.encodeCategory(df,"housevalue_level")

train_df = df.iloc[:3000,:]
test_df = df.iloc[3000:,:]

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="housevalue_level")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="housevalue_level")


#create model

model = tfdf.keras.RandomForestModel()
model.fit(train_ds)#model.fit(X_train,y_train)

model.compile(metrics=["accuracy"])
print(model.evaluate(test_ds))

with open("plot.html", "w") as f:
    f.write(tfdf.model_plotter.plot_model(model, tree_idx=0))


#model.save("model-house.h5")
