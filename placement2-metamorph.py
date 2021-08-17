import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import tensorflow as tf
from tensorflow import keras
import json
import xai

categ_num = 5

df=pd.read_csv("../dataset/placement.csv")

pd.set_option("display.max_rows", None, "display.max_columns", None)

df = df.fillna(value={"salary": 0})
df.dropna(inplace=True)

salarylevel = pd.cut(df.salary,bins=[-np.inf,1,200000,300000,400000,np.max(df['salary'])],labels=[0,1,2,3,4],include_lowest =True)
#salarylevel = pd.qcut(df.salary,q=4,duplicates='drop')
df.insert(len(df.columns),'salarylevel',salarylevel)

#sscp_range = pd.cut(df['ssc_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
sscp_range = pd.qcut(df['ssc_p'],q=10)
df.insert(len(df.columns),'sscp_range',sscp_range)

#hscp_range = pd.cut(df['hsc_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
hscp_range = pd.qcut(df['hsc_p'],q=10)
df.insert(len(df.columns),'hscp_range',hscp_range)

#degreep_range = pd.cut(df['degree_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
degreep_range = pd.qcut(df['degree_p'],q=10)
df.insert(len(df.columns),'degreep_range',hscp_range)

#mbap_range = pd.cut(df['mba_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
mbap_range = pd.qcut(df['mba_p'],q=10)
df.insert(len(df.columns),'mbap_range',mbap_range)

#etestp_range = pd.cut(df['etest_p'],bins=[-np.inf,10,20,30,40,50,60,70,80,90,np.inf],labels=[0,1,2,3,4,5,6,7,8,9],include_lowest =True)
etestp_range = pd.qcut(df['etest_p'],q=10)
df.insert(len(df.columns),'etestp_range',etestp_range)
#print(df.head())

final_df = df.drop(columns=['sl_no', 'salary','salarylevel','ssc_p','hsc_p','degree_p','mba_p','etest_p'])
y = df[['salarylevel']]


#gather value dict
columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))
# gather value dict
with open("analysis/placement-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.interval_encoder)+"}\n")


final_df["gender"] = xai.encodeCategory(final_df,"gender")
final_df["ssc_b"] = xai.encodeCategory(final_df,"ssc_b")
final_df["hsc_b"] = xai.encodeCategory(final_df,"hsc_b")
final_df["hsc_s"] = xai.encodeCategory(final_df,"hsc_s")
final_df["degree_t"] = xai.encodeCategory(final_df,"degree_t")
final_df["workex"] = xai.encodeCategory(final_df,"workex")
final_df["specialisation"] = xai.encodeCategory(final_df,"specialisation")
final_df["status"] = xai.encodeCategory(final_df,"status")

final_df["sscp_range"] = xai.encodeCategory(final_df,"sscp_range")
final_df["hscp_range"] = xai.encodeCategory(final_df,"hscp_range")
final_df["degreep_range"] = xai.encodeCategory(final_df,"degreep_range")
final_df["mbap_range"] = xai.encodeCategory(final_df,"mbap_range")
final_df["etestp_range"] = xai.encodeCategory(final_df,"etestp_range")

y["salarylevel"] = xai.encodeCategory(y,"salarylevel")
#

#
# #checking the unique values for categorical columns
# #categorical_columns = final_df.columns[final_df.dtypes == object]
#
# #one hot encoding on categorical variables
# # for col in categorical_columns:
# #     if(len(final_df[col].unique()) == 2):
# #         final_df[col] = pd.get_dummies(final_df[col], drop_first=True)
# #
# # final_df = pd.get_dummies(final_df)
#
#
# #print(final_df.columns)
#
#
# #print(X_train)
#
#encoding the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
encoded_df= scaler.fit_transform(final_df)

model = keras.models.load_model("model-placement.h5")
#
# print(final_df)
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
analysis_file = open("analysis/placement-analysis.txt", "w")
for resdict in result_dict_list:
    factor_inc = xai.findIncreaseFactor(model,final_df, resdict['maxind'],resdict['minind'], resdict['maxval'], resdict['minval'],resdict['catg'])
    factor_dec = xai.findDecreaseFactor(model,final_df, resdict['maxind'],resdict['minind'], resdict['maxval'], resdict['minval'],resdict['catg'])
    factor = xai.findFactor(factor_inc, factor_dec)
    print(" catg:",resdict['catg'])
    print("     factor_inc:",factor_inc)
    print("     factor_dec:",factor_dec)
    analysis_file.write(json.dumps({'output':resdict['catg'],'factor':factor},default=xai.np_encoder)+"\n")
    factorList.append({'output':resdict['catg'],'factor':factor})

end = time.time()
print("Elapse-time:",end - start)


print("====================================================================")
print("factorList:",factorList)
analysis_file.close()

print("====================================================================")

## gennerate data for testing
# refresh column dict
columns_valuesdict = {}
for col in final_df.columns:
    columns_valuesdict[col] = final_df[col].value_counts().index.tolist()

testdata = xai.generateData(factorList, columns_valuesdict,'salarylevel',100)

print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['salarylevel']
gentest_df = gen_df.drop(columns=['salarylevel'])
#print(gentest_df)
gendata_results = model.predict(scaler.transform(gentest_df))
true_pos = [0,0,0,0,0]
for i in range(len(gendata_results)):
    print(np.argmax(gendata_results[i]),"<==",gen_df.iloc[[i]]['salarylevel'].values[0])
    testdata[i]['salarylevel_predict'] = np.argmax(gendata_results[i])
    if testdata[i]['salarylevel_predict'] == gen_df.iloc[[i]]['salarylevel'].values[0]:
        true_pos[gen_df.iloc[[i]]['salarylevel'].values[0]] = true_pos[gen_df.iloc[[i]]['salarylevel'].values[0]]+1

print(true_pos)
# write generated data to files
gendata_file = open("analysis/placement-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
