from keras.layers import Dense
from keras.models import Input, Model
import pandas as pd
from tcn import TCN
# Temporal Convolutional Network
# Built for time series


def process_data():
    train_frame = pd.read_csv("data/C-MAPSS/train_FD001.txt")
    test_frame = pd.read_csv("data/C-MAPSS/test_FD001.txt")
    rul_frame = pd.read_csv("data/C-MAPSS/RUL_FD001.txt")
    return train_frame, test_frame, rul_frame

time = 0   # 

train,test,rul = process_data()

model = Sequential([
    TCN(input_shape=(time,8)),  # Length and dimension of input
    Dense(1)  # Dense layer with one value as output (?)
])

model.compile(optimizer='adam',loss='mse')

# model.fit()

# model.evaluate()

# model.predict()

