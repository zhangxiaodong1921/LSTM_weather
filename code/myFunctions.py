import os
import sys
import numpy as np
import pandas



def read_data():
    dataDir = "/Users/leonardrychly/Dropbox/Projects/Weather Fusion/"\
               "LSTM_weather/data/"
    fname = 'international-airline-passengers.csv'
    dataset = pandas.read_csv(dataDir + fname, usecols=[1], engine='python', 
                              skipfooter=3)
    return dataset


def create_dataset(dataset, look_back=1):
    """ Convert an array of values into a data matrix
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)