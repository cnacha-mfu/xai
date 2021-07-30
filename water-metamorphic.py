import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
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

final_df = df.drop(columns=['Potability'])
y = df[['Potability']]

# convert to categorical
qgroup = 10
for col in final_df.columns:
    colrange = pd.qcut(df[col],q=qgroup,duplicates='drop')
    final_df[col+"range"] = colrange
    final_df = final_df.drop(columns=[col])

columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))

print(columns_valuesdict)

# gather value dict
with open("analysis/water-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        print(col)
        valuestr = json.dumps(columns_valuesdict[col], default=xai.jsencoder)
        f.write("{\"column\":\""+col+"\",\"values\":"+valuestr+"}\n")

# convert interval to number range
for col in final_df.columns:
    xai.encodeCategory(final_df,col)

#encoding the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
encoded_df= scaler.fit_transform(final_df)

model = keras.models.load_model("model-water.h5")


result = model.predict(scaler.transform(final_df))
print(np.argmax(result)," : ",np.max(result))
print(np.argmin(result)," : ",np.min(result))


factor_inc = xai.findIncreaseFactor(model,final_df, np.argmax(result), np.argmin(result), np.max(result), np.min(result),0)
factor_dec = xai.findDecreaseFactor(model,final_df, np.argmax(result), np.argmin(result), np.max(result), np.min(result),0)
#print("increase factor:",factor_inc)
#print("decrease factor:",factor_dec)

#find unique factor in increasing factor that no exist in decreasing factor, same factor does not have significant effect on results
factor = xai.findFactor(factor_inc, factor_dec)
print("valuedict: ", columns_valuesdict)
print("factor:",factor)

factorList = [{'output':1,'factor':factor}]

analysis_file = open("analysis/water-analysis.txt", "w")
analysis_file.write(json.dumps(factorList,default=xai.np_encoder))
analysis_file.close()


# refresh column dict
columns_valuesdict = {}
for col in final_df.columns:
    columns_valuesdict[col] = final_df[col].value_counts().index.tolist()

# generate test data
testdata = xai.generateData(factorList, columns_valuesdict,'Potability', 50)
print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['Potability']
gentest_df = gen_df.drop(columns=['Potability'])
print(gentest_df)
gendata_results = model.predict(scaler.transform(gentest_df))
gendata_results = np.round(gendata_results)
for i in range(len(gendata_results)):
    print(gendata_results[i],"<==",gen_df.iloc[[i]]['Potability'].values[0])
    testdata[i]['Potability_predict'] = gendata_results[i][0].astype(int)
# write generated data to files
gendata_file = open("analysis/water-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
