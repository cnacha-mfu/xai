import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
import xai
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

#https://www.kaggle.com/camnugent/california-housing-prices
#accuracy [0.6699084043502808, 0.7100290656089783,0.7093023061752319]

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
column_num = len(X_train.columns)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
def model_builder(hp):
    model = Sequential()
    model.add(Dense(column_num,activation='relu'))
    model.add(Dense(hp.Int(name='dense_units1', min_value=8, max_value=1024, step=4),  activation='relu'))
    model.add(Dropout(hp.Float('dropout1', 0, 0.5, step=0.1, default=0.3)))
    model.add(Dense(hp.Int(name='dense_units2', min_value=8, max_value=1024, step=4),  activation='relu'))
    model.add(Dropout(hp.Float('dropout2', 0, 0.5, step=0.1, default=0.3)))
    model.add(Dense(3, activation="softmax"))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics="accuracy")
    return model


#tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     directory='house_dir',
                     project_name='housepredict')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(X_train, y_train, epochs=1000, validation_data=(X_test,y_test), callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters()[0]


model = tuner.hypermodel.build(best_hps)
history =  model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=128,epochs=1000,  verbose=1)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
hypermodel.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=128,epochs=best_epoch,  verbose=1)

eval_results = hypermodel.evaluate(X_test, y_test)
print(eval_results)


#model.save("model-house.h5")
