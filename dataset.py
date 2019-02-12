import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from keras.preprocessing.text import Tokenizer
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

df = pd.read_excel('./dataset/raw_data.xlsx')
data = df.fillna(method='ffill')
vocabSize = 410
tokenizer = Tokenizer(num_words=vocabSize)

def processData(data):
    dataList = []
    dataName = data.replace('^','_').split('_')
    n = 1
    for names in dataName:
        if (n%2) == 0:
            dataList.append(names)
        n += 1
    return dataList

def cleanData():
    diseaseList = []
    diseaseSymptomDict = defaultdict(list)
    diseaseSymptomCount = {}
    count = 0

    for idx, row in data.iterrows():
        #get disease name
        if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
            disease = row['Disease']
            diseaseList = processData(data=disease)
            count = row['Count of Disease Occurrence']

        #get symptoms name
        if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
            symptom = row['Symptom']
            symptomList = processData(data=symptom)
            for d in diseaseList:
                for s in symptomList:
                    diseaseSymptomDict[d].append(s)
                diseaseSymptomCount[d] = count

    #print diseaseSymptomDict
    #print diseaseSymptomCount
    dfClean = pd.DataFrame(list(diseaseSymptomDict.items()), columns=['Disease','Symptom'])
    dfClean.to_csv('./dataset/clean_data.csv', encoding='utf-8')
    return dfClean

def loadData():

    data = cleanData()
    disease = data['Disease']
    symptom = data['Symptom']
    
    '''
    #count symptoms
    cc = []
    for i in range (149):
        for j in range(len(symptom[i])):
            aa = symptom[i][j]
            cc.append(aa)
            cc = list(OrderedDict.fromkeys(cc))
    print len(cc)
    '''

    encoder = LabelEncoder()
    encoder.fit(disease)
    encoded_Y = encoder.transform(disease)
    dummyY = np_utils.to_categorical(encoded_Y)

    tokenizer.fit_on_texts(symptom)
    xTrain = tokenizer.texts_to_matrix(symptom, mode='count')
    yTrain = dummyY

    return xTrain, yTrain
