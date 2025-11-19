from keras.layers import Dense
from keras.models import Input, Model

from tcn import TCN
# Temporal Convolutional Network
# Built for time series


time = 0   # 

model = Sequential([
    TCN(input_shape=(time,8)),  # Length and dimension of input
    Dense(1)  # Dense layer with one value as output (?)
])

model.compile(optimizer='adam',loss='mse')

# model.fit()

# model.evaluate()

# model.predict()