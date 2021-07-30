import pandas as pd
import numpy as np


# def encodeCategory(df,col):
#     #df.sort_values(col,inplace=True)
#     col = df[col].astype('category')
#     col = col.cat.codes
#     return col

def encodeCategory(df,col):
    tempdf = df.sort_values(col,inplace=False)
    valuelist = tempdf[col].unique()
    valuedict = {}
    for i in range(len(valuelist)):
        valuedict[valuelist[i]] = i
    df[col] = df[col].replace(valuedict)
    return df[col]

# def getColumnvaluedict(df):
#     columns_valuesdict = {}
#     tempdf = df.copy(deep=True)
#     for col in tempdf.columns:
#         #tempdf.sort_values(col,inplace=True)
#         columns_valuesdict[col] = list(tempdf[col].unique())
#     return columns_valuesdict

def getColumnvaluedict(df):
    columns_valuesdict = {}
    tempdf = df.copy(deep=True)
    for col in tempdf.columns:
        tempdf.sort_values(col,inplace=True)
        columns_valuesdict[col] = list(tempdf[col].unique())
    return columns_valuesdict

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def interval_encoder(object):
    if isinstance(object, pd.Interval):
        return str(object.left)+"-"+str(object.right)
    else:
        return object

def jsencoder(object):
    if isinstance(object, pd.Interval):
        return str(object.left)+"-"+str(object.right)
    elif isinstance(object, np.generic):
        return object.item()
    else:
        return object



def findFactor(factor_inc, factor_dec):
    factor = {}
    for col in factor_inc:
        factor[col] = list(set(factor_inc[col] ) - set(factor_dec[col] ))
    return factor

def findIncreaseFactor(model, df,maxindex, minindex, maxpredict, minpredict, catg):
    dfmin = df.iloc[[minindex]] # [[...]] get specific role in dataframe
    dfmax = df.iloc[[maxindex]]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("start finding catg:",catg)
    factor_list = {}

    #encoding the features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    encoded_df= scaler.fit_transform(df)

    for col in dfmin.columns:
        #if  dfmin[col].values[0] != dfmax[col].values[0] : # when the attributes value are different we find whther it has any impact
            #print("  col:",col," ->",dfmin[col].values[0], "  === ",dfmax[col].values[0])
            # sampling to all possible value that increase prediction
            effected_value = []
            for possiblevalue in df[col].unique():
                newdf = dfmin.copy(deep=True)
                newdf[col].values[0] = possiblevalue #change value to sampling value
                results = model.predict(scaler.transform(newdf))


                if (results[0][catg] > minpredict) :
                    print("     effected  val:",possiblevalue, "new:",results," min:",minpredict," max:",maxpredict)
                    effected_value.append(possiblevalue)

            factor_list[str(col)] = effected_value
        #else: # when the attributes value are the same, find factor that decrease prediction
            #factor_list[str(col)] = [dfmax[col].values[0]]

    print("stop finding.....")
    return factor_list

def findDecreaseFactor(model, df,maxindex, minindex, maxpredict, minpredict,catg):
    dfmin = df.iloc[[minindex]] # [[...]] get specific role in dataframe
    dfmax = df.iloc[[maxindex]]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("start finding.....")
    factor_list = {}

    #encoding the features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    encoded_df= scaler.fit_transform(df)

    for col in dfmin.columns:
        #if  dfmin[col].values[0] != dfmax[col].values[0] : # when the attributes value are different we find whther it has any impact
            #print("  col:",col," ->",dfmin[col].values[0], "  === ",dfmax[col].values[0])
            # sampling to all possible value that increase prediction
            effected_value = []
            for possiblevalue in df[col].unique():
                newdf = dfmax.copy(deep=True)
                newdf[col].values[0] = possiblevalue #change value to sampling value
                results = model.predict(scaler.transform(newdf))

                #print("     val:",possiblevalue, "new:",results," min:",minpredict," max:",maxpredict)
                if (results[0][catg] < maxpredict) :
                    #print("         --->this factor effected....")
                    effected_value.append(possiblevalue)

            factor_list[str(col)] = effected_value
        #else: # when the attributes value are the same, find factor that decrease prediction
            #factor_list[str(col)] = [dfmax[col].values[0]]

    print("stop finding.....")
    return factor_list

def generateData(factorList, columns_valuesdict, outputColName,number):
    import copy
    catgValueList = copy.deepcopy(factorList)

    # fill possible values
    for catg in catgValueList:
        for f in catg['factor']:
            if len(catg['factor'][f]) == 0:
                catg['factor'][f] = columns_valuesdict[f]

    # generate data
    import random
    testdata = []
    for catg in catgValueList:
        for i in range(number):
            record = {outputColName:catg['output']}
            for f in catg['factor']:
                record[f] = random.choice(catg['factor'][f])
            testdata.append(record)

    return testdata
