
import pandas as pd
import numpy as np
import xai

#accuracy [0.46062612533569336, 0.8507462739944458]
df = pd.read_csv("./insurance/insurance3r2.csv")

categ_num = 5

df['age'] = pd.qcut(df['age'],q=10)
df['bmi'] = pd.qcut(df['bmi'],q=10)
df['steps'] = pd.qcut(df['steps'],q=10)
df['charges'] = pd.qcut(df['charges'],q=categ_num)

df["age"] = xai.encodeCategory(df,"age")
df["bmi"] = xai.encodeCategory(df,"bmi")
df["charges"] = xai.encodeCategory(df,"charges")
df["steps"] = xai.encodeCategory(df,"steps")

X = df.drop(['charges','region','insuranceclaim'], axis = 1)
y = df['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(6, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(categ_num, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")


fitModel =  model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=1024,epochs=1000,  verbose=1)

eval_results = model.evaluate(X_test, y_test)
print(eval_results)

model.save("model-insurance.h5")
