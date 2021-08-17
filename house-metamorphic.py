import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import tensorflow as tf
from tensorflow import keras
import json
import xai

#https://www.kaggle.com/camnugent/california-housing-prices
#accuracy [0.6239417791366577, 0.724321722984314]

df=pd.read_csv("../dataset/housing.csv")
df.dropna(inplace=True)
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


#gather value dict
columns_valuesdict = xai.getColumnvaluedict(final_df)
columns_valuesdict.update(xai.getColumnvaluedict(y))

print(columns_valuesdict)

# gather value dict
with open("analysis/house-valuedict.txt", "w") as f:
    for col in columns_valuesdict:
        f.write("{\"column\":\""+col+"\",\"values\":"+json.dumps(columns_valuesdict[col],default=xai.interval_encoder)+"}\n")


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


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_df)

model = keras.models.load_model("model-house.h5")
results = model.predict(scaled_data)

import time
start = time.time()

#find max min in each category
result_dict_list = []
for i in range(3): # loop through all 4 categories
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
analysis_file = open("analysis/house-analysis.txt", "w")
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

testdata = xai.generateData(factorList, columns_valuesdict,'housevalue_level',100)

print("================= predicting generated test data =========================")
gen_df = pd.DataFrame.from_dict(testdata)
gentest_y = gen_df['housevalue_level']
gentest_df = gen_df.drop(columns=['housevalue_level'])
#print(gentest_df)
gendata_results = model.predict(scaler.transform(gentest_df))
for i in range(len(gendata_results)):
    print(np.argmax(gendata_results[i]),"<==",gen_df.iloc[[i]]['housevalue_level'].values[0])
    testdata[i]['housevaluelevel_predict'] = np.argmax(gendata_results[i])
# write generated data to files
gendata_file = open("analysis/house-gendata.txt", "w")
gendata_file.write(json.dumps(testdata,default=xai.np_encoder))
gendata_file.close()
