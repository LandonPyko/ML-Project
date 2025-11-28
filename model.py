from keras.layers import Dense
import pandas as pd
from tcn import TCN
import os
import numpy as np

# Temporal Convolutional Network
# Built for time series


def process_data():
    # just processing one file for now

    mean = np.loadtxt("data_cleaning/time_series_data_64/Indesit/2_1647850500_1647854400_fast_means.txt")
    median = np.loadtxt("data_cleaning/time_series_data_64/Indesit/2_1647850500_1647854400_fast_medians.txt")

    data = np.column_stack((mean,median))  # Array containing mean and median for each window
    windows = create_windows(data)
    print(windows.shape)

    
def create_windows(data):

    windows = []

    num_points = data.shape[0]
    sequence = 64
    stride = 16   # Allows for overlap to view patterns
    for start in range(0, num_points - sequence + 1, stride):
        seq = data[start:start+sequence]
        windows.append(seq)
    
    return np.array(windows)


def main():
    time = 0   # 

    process_data()
    '''
    window_size = 64
    num_features = 2
    model = Sequential([
        TCN(
            nb_filters = 64,
            kernel_size = 3,
            dilations = [1,2,4,8,16,32,64],
            dropout_rate = .05,
            return_sequences = False,
            input_shape=(window_size,num_features)
            ),

        Dense(1,activation=None)  # Dense layer with one value as output (?)
    ])
    

    #model.compile(optimizer='adam',loss='mse')

    #model.fit()

    #model.evaluate()

    #model.predict()
    '''

if __name__ == "__main__":
    main()