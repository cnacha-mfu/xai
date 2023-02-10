import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import tensorflow as tf
from tensorflow import keras
import json
import xai

df = pd.read_csv("./insurance/insurance3r2.csv")

categ_num = 5

df['age'] = pd.qcut(df['age'],q=10)
df['bmi'] = pd.qcut(df['bmi'],q=10)
df['steps'] = pd.qcut(df['steps'],q=10)
df['charges'] = pd.qcut(df['charges'],q=categ_num)


final_df = df.drop(['charges','region','insuranceclaim'], axis = 1)
y = df[['charges']]

#gather value dict
columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))

print(columns_valuesdict)

# gather value dict
with open("analysis/insurance-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        print(col)
        f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.jsencoder)+"}\n")

final_df["age"] = xai.encodeCategory(final_df,"age")
final_df["bmi"] = xai.encodeCategory(final_df,"bmi")
final_df["steps"] = xai.encodeCategory(final_df,"steps")
y["charges"] = xai.encodeCategory(y,"charges")

model = keras.models.load_model("model-insurance.h5")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
encoded_df= scaler.fit_transform(final_df)

results = model.predict(scaler.transform(final_df))

import time

start = time.time()

#find max min in each category
result_dict_list = []
for i in range(categ_num): # loop through all 4 categories
    max = -np.inf; # value for maximum prediction
    min = np.inf; # value for minimum prediction
    maxind = 0 #index for maximum prediction
    minind = 0 #index for minimum prediction value
    for j in range(len(results)):
        if (np.argmax(results[j]) == i) :
                if results[j][i] > max :
                    max = results[j][i]
                    maxind = j
                if results[j][i] < min:
                    min = results[j][i]
                    minind = j

    print("max for ",i,"at ",maxind," : ",max)
    print("min for ",i,"at ",minind,": ",min)
    result_dict = {'catg':i,'maxind':maxind,'minind':minind,'maxval':max,'minval':min}
    result_dict_list.append(result_dict)

factorList = []
analysis_file = open("analysis/insurance-analysis.txt", "w")
for resdict in result_dict_list:
    factor_inc = xai.findIncreaseFactor(model,final_df, resdict['maxind'],resdict['minind'], resdict['maxval'], resdict['minval'],resdict['catg'])
    factor_dec = xai.findDecreaseFactor(model,final_df, resdict['maxind'],resdict['minind'], resdict['maxval'], resdict['minval'],resdict['catg'])
    factor = xai.findFactor(factor_inc, factor_dec)
    print(" catg:",resdict['catg'])
    print("     factor_inc:",factor_inc)
    print("     factor_dec:",factor_dec)
    analysis_file.write(json.dumps({'output':resdict['catg'],'factor':factor},default=xai.np_encoder)+"\n")
    factorList.append({'output':resdict['catg'],'factor':factor})

print("====================================================================")
print("factorList:",factorList)
analysis_file.close()

end = time.time()
print("Elapse-time:",end - start)

print("====================================================================")
## gennerate data for testing
# refresh column dict
columns_valuesdict = {}
for col in final_df.columns:
    columns_valuesdict[col] = final_df[col].value_counts().index.tolist()

testdata = xai.generateData(factorList, columns_valuesdict,'charges',100)

print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['charges']
gentest_df = gen_df.drop(columns=['charges'])
#print(gentest_df)
gendata_results = model.predict(scaler.transform(gentest_df))
for i in range(len(gendata_results)):
    print(np.argmax(gendata_results[i]),"<==",gen_df.iloc[[i]]['charges'].values[0])
    testdata[i]['charges'] = np.argmax(gendata_results[i])
# write generated data to files
gendata_file = open("analysis/insurance-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
