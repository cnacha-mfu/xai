import pandas as pd
import numpy as np
import xai
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


#https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset
#accurary [3.41115140914917, 0.6938775777816772] [0.7415797710418701, 0.7755101919174194]

categ_num = 3

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

# create NN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

def model_builder(hp):
    model = Sequential() #len(X.columns)
    model.add(Dense(len(X.columns), activation='relu'))
    hp1_units = hp.Int(name='units1', min_value=8, max_value=1024, step=4)
    model.add(Dense(units=hp1_units, activation='relu'))

    model.add(Dropout(hp.Float('dropout1', 0, 0.5, step=0.1, default=0.3)))
    hp2_units = hp.Int(name='units2', min_value=8, max_value=1024, step=4)
    model.add(Dense(units=hp2_units, activation='relu'))
    model.add(Dropout(hp.Float('dropout2', 0, 0.5, step=0.1, default=0.3)))
    model.add(Dense(categ_num, activation="softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 5e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics="accuracy")

    return model


#tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     directory='covid_dir',
                     project_name='coviddiet')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(X_train, y_train, epochs=1000, validation_data=(X_test,y_test), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters()[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')} {best_hps.get('units2')}  and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
print(best_hps.values)
summary = tuner.results_summary(num_trials=10)
print(summary)

#model = tuner.get_best_models(num_models=1)[0]


model = tuner.hypermodel.build(best_hps)
history =  model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            #validation_split=0.2,
            batch_size=128,epochs=1000,  verbose=1)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
history = hypermodel.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            #validation_split=0.2,
            batch_size=1024,epochs=best_epoch,  verbose=1)
print(history)

eval_results = hypermodel.evaluate(X_test, y_test)
print(eval_results)


model.save("model-coviddiet.h5")
