import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler



def read_data(dataset, processing='norm'):
    """ Read in data and normalize """
    if dataset == 'lstm_toy':
        dataDir = "/Users/leonardrychly/Dropbox/Projects/Weather Fusion/"\
                   "LSTM_weather/data/"
        fname = 'international-airline-passengers.csv'
        dataset = pd.read_csv(dataDir + fname, usecols=[1], engine='python', 
                                  skipfooter=3)
        return dataset
    elif dataset == 'sanmateo':
        dataDir = "/Users/leonardrychly/Dropbox/Projects/Weather Fusion/"\
                  "LSTM_weather/data/"
        fname = 'q_rec.csv'
        # read data
        dataset = pd.read_csv(dataDir + fname)
        datetimeData = pd.to_datetime(dataset['Date Time'].values)\
                       .to_pydatetime()
        dtData = np.asarray([i.timestamp() for i in datetimeData])
        tempData = dataset['TempOutside [C]'].values
        humData = dataset['HumInside [%]'].values
        #dataset = np.vstack((dtData,tempData, humData))
        dataset = tempData.reshape(-1, 1)
        # normalize data with data: (n_channels,n_samples)
        if processing == 'norm':
            dataset = normalize(dataset)
            return dataset.T, 0
        elif processing == 'std':
            scaler = StandardScaler().fit(dataset.T) #T
            dataset = scaler.transform(dataset.T) #T
            return dataset, scaler
        elif processing == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(dataset) #T
            dataset = scaler.transform(dataset) #T
            return dataset, scaler
        elif processing == 'none':
            return dataset, 0




def create_dataset(dataset, look_back=1):
    """ Convert an array of values into a data matrix
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)