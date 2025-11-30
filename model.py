from keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
import pandas as pd
from tcn import TCN
import matplotlib.pyplot as plt
import os
import numpy as np

# Temporal Convolutional Network
# Built for time series

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()

    self.encoder = Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])

    self.decoder = Sequential([
      layers.Dense(64, activation='linear'),
      layers.Reshape((64,1))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded




def process_data():
    # just processing one file for now


    data = np.loadtxt("data_cleaning/data/fast/2_1639574100_1639584900_fast.csv",
                      delimiter=',',
                      skiprows=1,
                      usecols=2)
    
    data2 = np.loadtxt("data_cleaning/data/fast/3_1648707840_1648711920_fast.csv",
                       delimiter=',',
                       skiprows=1,
                       usecols=2)

    #data = np.column_stack((mean,median))  # Array containing mean and median for each window
    windows = create_windows(data)
    data = windows.reshape(-1, 64, 1)

    windows2 = create_windows(data2)
    data2 = windows2.reshape(-1,64,1)


    #X_train, X_test = train_test_split(data,test_size=0.2,shuffle=True)

    X_train = data
    X_test = data2

    print(X_train.shape)
    print(X_test.shape)

    return X_train, X_test

    
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

    X_train, X_test = process_data()

    mean_val = X_train.mean()
    std_val = X_train.std()

    X_train_norm = (X_train - mean_val) / std_val
    X_test_norm = (X_test - mean_val) / std_val
    # instantiate model

    autoencoder = Autoencoder(latent_dim=64)
    autoencoder.compile(optimizer='adam',loss='mse')
    autoencoder.fit(X_train_norm, X_train_norm,
                epochs=30,
                shuffle=True,
                validation_data=(X_test_norm, X_test_norm))
    
    reconstructions = autoencoder.predict(X_test_norm)
    errors = tf.reduce_mean((X_test_norm - reconstructions)**2, axis=(1,2))

    plt.hist(errors.numpy(), bins=50)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of windows")
    plt.title("Histogram of reconstruction error for healthy data")
    plt.savefig("output.png")
    plt.close()

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