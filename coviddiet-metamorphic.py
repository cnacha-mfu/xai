import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import xai
#https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset
#accurary [0.7415797710418701, 0.7755101919174194]

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
      'Country', 'Unit (all except Population)'])

# convert to categorical
for col in df.columns:
    qgroup = 10
    if col == 'Deaths':
        qgroup = categ_num
    colrange = pd.qcut(df[col],q=qgroup,duplicates='drop')
    df[col.replace(' ', '')+"range"] = colrange
    df = df.drop(columns=[col])



# split train and test data set
final_df = df.drop(columns=['Deathsrange'])
y = df[['Deathsrange']]

#gather value dict
columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))

# gather value dict
with open("analysis/coviddiet-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.interval_encoder)+"}\n")

# convert to number range
for col in df.columns:
    xai.encodeCategory(df,col)

# split train and test data set
final_df = df.drop(columns=['Deathsrange'])
y = df[['Deathsrange']]

#encoding the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
encoded_df= scaler.fit_transform(final_df)

model = keras.models.load_model("model-coviddiet.h5")
#
# print(final_df)
results = model.predict(scaler.transform(final_df))
#find max min in each category
result_dict_list = []

import time
start = time.time()

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
analysis_file = open("analysis/coviddiet-analysis.txt", "w")
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

testdata = xai.generateData(factorList, columns_valuesdict,'Deathsrange',100)

print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['Deathsrange']
gentest_df = gen_df.drop(columns=['Deathsrange'])
#print(gentest_df)
true_pos = [0,0,0,0]
gendata_results = model.predict(scaler.transform(gentest_df))
for i in range(len(gendata_results)):
    print(np.argmax(gendata_results[i]),"<==",gen_df.iloc[[i]]['Deathsrange'].values[0])
    testdata[i]['Deathsrange_predict'] = np.argmax(gendata_results[i])
    if testdata[i]['Deathsrange_predict'] == gen_df.iloc[[i]]['Deathsrange'].values[0]:
        true_pos[gen_df.iloc[[i]]['Deathsrange'].values[0]] = true_pos[gen_df.iloc[[i]]['Deathsrange'].values[0]]+1
# write generated data to files
print(true_pos)
gendata_file = open("analysis/coviddiet-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
