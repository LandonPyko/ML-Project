from keras.layers import Dense
import pandas as pd
from tcn import TCN
import os

# Temporal Convolutional Network
# Built for time series


def process_data():
    raw_data = []  # Holds list of pandas dataframes for each raw 

    fast = os.listdir("data_cleaning/data/fast")
    for file in fast:
        print(file)
        fullPath = os.path.join("data_cleaning/data/fast",file)
        df = pd.read_csv(fullPath)
        raw_data.append(df)

    stream_data = pd.read_csv("data_cleaning/data/stream_labels.csv")
    print(stream_data)

    


def main():
    time = 0   # 

    process_data()

    #model = Sequential([
     #   TCN(input_shape=(time,8)),  # Length and dimension of input
      #  Dense(1)  # Dense layer with one value as output (?)
    #])

    #model.compile(optimizer='adam',loss='mse')

    # model.fit()

    # model.evaluate()

    # model.predict()

if __name__ == "__main__":
    main()