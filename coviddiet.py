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

# create NN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

model = Sequential() #len(X.columns)
model.add(Dense(len(X.columns),input_shape=(28,), activation='relu'))
model.add(Dense(48,  activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(48,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24,  activation='relu'))
model.add(Dense(categ_num, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")

fitModel =  model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=1024,epochs=1000,  verbose=1)

eval_results = model.evaluate(X_test, y_test)
print(eval_results)


#model.save("model-coviddiet.h5")
