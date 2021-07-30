import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import tensorflow as tf
from tensorflow import keras
import json
import xai

#[0.4640905261039734, 0.8131868243217468]

df=pd.read_csv("../dataset/heart.csv")
df.head()
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
y = df[['target']]

#checking the unique values for categorical columns
categorical_columns = final_df.columns[final_df.dtypes == object]

categorical_columns = final_df.columns
columns_valuesdict = {}

columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))

print(columns_valuesdict)

# gather value dict
with open("analysis/heart-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        print(col)
        valuestr = json.dumps(columns_valuesdict[col], default=xai.jsencoder)
        f.write("{\"column\":\""+col+"\",\"values\":"+valuestr+"}\n")

final_df["agerange"] = xai.encodeCategory(final_df,"agerange")
final_df["bpsrange"] = xai.encodeCategory(final_df,"bpsrange")
final_df["cholrange"] = xai.encodeCategory(final_df,"cholrange")
final_df["thalachrange"] = xai.encodeCategory(final_df,"thalachrange")
final_df["oldpeakrange"] = xai.encodeCategory(final_df,"oldpeakrange")

#encoding the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
encoded_df= scaler.fit_transform(final_df)

model = keras.models.load_model("model-heart.h5")


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

factorList = {'output':1,'factor':factor}

analysis_file = open("analysis/heart-analysis.txt", "w")
analysis_file.write(json.dumps(factorList,default=xai.np_encoder))
analysis_file.close()


# refresh column dict
columns_valuesdict = {}
for col in final_df.columns:
    columns_valuesdict[col] = final_df[col].value_counts().index.tolist()

# generate test data
testdata = xai.generateData(factorList, columns_valuesdict,'target', 50)
print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['target']
gentest_df = gen_df.drop(columns=['target'])
print(gentest_df)
gendata_results = model.predict(scaler.transform(gentest_df))
gendata_results = np.round(gendata_results)
for i in range(len(gendata_results)):
    print(gendata_results[i],"<==",gen_df.iloc[[i]]['target'].values[0])
    testdata[i]['target_predict'] = gendata_results[i][0].astype(int)
# write generated data to files
gendata_file = open("analysis/heart-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
