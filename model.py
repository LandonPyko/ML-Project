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
        self.latent_dim = 64

        self.encoder = Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        ])

        self.decoder = Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='linear'),
        layers.Reshape((64, 1)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
  
    def get_config(self):
        # Return only serializxable arguments
        return {"latent_dim": self.latent_dim}

    def from_config(cls, config):
        # Keras will pass config={"latent_dim": 64}
        return cls(**config)




def process_data(file):
    # just processing one file for now


    data = np.loadtxt(file,
                      delimiter=',',
                      skiprows=1,
                      usecols=2)
    
    data2 = np.loadtxt("data_cleaning/data/fast/3_1648707840_1648711920_fast.csv",
                       delimiter=',',
                       skiprows=1,
                       usecols=2)
    
# 2_1648023300_1648035300_fast.csv  = Good File
# 2_1639574100_1639584900_fast.csv = Good File
# 3_1640012400_1640019000_fast.csv = Good File
# 3_1648707840_1648711920_fast.csv = Bad File


# 2_1639641000_1639651080_fast.csv

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

    X_train, X_test = process_data("data_cleaning/data/fast/2_1639574100_1639584900_fast.csv")
    X_train2,X_test2 = process_data("data_cleaning/data/fast/2_1639641000_1639651080_fast.csv")

    all_train = np.concatenate([X_train,X_train2],axis=0)

    mean_val = all_train.mean()
    std_val = all_train.std()

    print("Mean val: ", mean_val)
    print("Std val: ", std_val)

    X_train_norm = (all_train - mean_val) / std_val
    X_test_norm = (X_test - mean_val) / std_val
    # instantiate model

    autoencoder = Autoencoder(latent_dim=64)
    autoencoder.compile(optimizer='adam',loss='mse')
    autoencoder.fit(X_train_norm, X_train_norm,
                epochs=10,
                shuffle=True,
                validation_split = 0.1)
    '''
    

    X_train_norm = (X_train - mean_val) / std_val

    autoencoder.fit(X_train_norm, X_train_norm,
                epochs=10,
                shuffle=True,
                validation_split = 0.1)
    '''

    autoencoder.save("model.keras")
    autoencoder.encoder.summary()

    reconstructions = autoencoder.predict(X_test_norm)
    errors = tf.reduce_mean((X_test_norm - reconstructions)**2, axis=1)
    np_errors = errors.numpy()
    filtered_errors = np_errors[np_errors>0]


    # Threshold from training error
    train_recon = autoencoder.predict(X_train_norm)
    train_errors = tf.reduce_mean((X_train_norm - train_recon)**2, axis=1)  # This needs to be looked at


    #threshold = train_errors.numpy().mean() + 3*train_errors.numpy().std()
    threshold = np.percentile(train_errors.numpy(), 99.99)
    # Classify test windows
    labels = ["Working" if e <= threshold else "Failure" for e in errors.numpy()]

    plt.hist(filtered_errors, bins=50, alpha=0.7, label="Bad file errors")
    plt.axvline(threshold, color='red', linestyle='--', label="Threshold")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of windows")
    plt.title("Histogram of reconstruction error for bad file")
    plt.legend()
    plt.savefig("output_bad.png")
    plt.close()


    plt.plot(X_train_norm[0].squeeze(), label="Original")
    plt.plot(autoencoder.predict(X_train_norm[0:1])[0].squeeze(), label="Reconstructed")
    plt.legend()
    plt.savefig("test.png")
    plt.close()


if __name__ == "__main__":
    main()