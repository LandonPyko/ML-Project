import os
import numpy as np
import pandas as pd

# function assumes that the zip file is in the same directory
def create_dirs(window_size):

    commands = [
        "unzip -qq ./archive.zip -d ./data/",
        "mkdir data/slow",
        "mkdir data/fast",
        f"mkdir time_series_data_{window_size}",
        f"mkdir time_series_data_{window_size}/kunft",
        f"mkdir time_series_data_{window_size}/Indesit",
        "cp ./data/stream_labels.csv ./stream_labels.csv",
    ]

    for cmd in commands:
      os.system(cmd)

    dir_fast = "data/fast"
    dir_slow = "data/slow"
    dir = "data"
    
    files = os.listdir(dir)

    for file in files:
        if not file.endswith(".csv"):
            continue
        if "slow" in file:
            os.rename(os.path.join(dir, file), os.path.join(dir_slow, file))
        if "fast" in file:
            os.rename(os.path.join(dir, file), os.path.join(dir_fast, file))
    

def chunk(array, window_size):
    n = len(array) // window_size
    total = n * window_size
    return array[:total].reshape(n, window_size, 2)

def reshape1D(arr2d):
    for i, arr in enumerate(arr2d):
        arr2d[i] = arr.reshape(len(arr))

def computeMeanMedian(x_ts, y):
    mean = (x_ts.mean(axis=1)[:,1:])
    median = (np.median(x_ts, axis=1)[:,1:])

    reshape1D(mean)
    reshape1D(median)

    return mean, median

# returns a dictionary of (device_id, start, end) = (lbl, brand)
def constructLabels(file_name="stream_labels.csv"):
    df_labels = pd.read_csv(file_name)
    labels = {}
    for i, row in df_labels.iterrows():
        label = 0 if row["failure"] == "Working" else 1
        labels[(row["device_id"], row["timestamp_begin"], row["timestamp_end"])] = (label, row["brand"])

    return labels

# pass a file name and this will retrieve the label of the file
def retrieveLabel(file_name, labels):
    split_name = file_name.split("_")
    key = (int(split_name[0]), int(split_name[1]), int(split_name[2]))

    return labels[key]

def convertToTimeSeries(window_size):
    dir_fast = "data/fast"

    ts_dir = f"time_series_data_{window_size}/"
    files = os.listdir(dir_fast)
    
    labels = constructLabels()

    for i, file in enumerate(files):
        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(dir_fast, file))
        sub_np = df[["UnixTimestamp (us)", "Vibration"]].to_numpy()

        x_ts = chunk(sub_np, window_size)     # window size of 64 --> non-overlapping

        if len(x_ts) == 0:
            continue

        # file key stuff, can uncomment later
        file_split_time = file.split("_")
        key = (int(file_split_time[0]), int(file_split_time[1]), int(file_split_time[2]))
        lbl, brand = labels[key]

        # median and mean
        means, medians = computeMeanMedian(x_ts, lbl)

        full_path = os.path.join(ts_dir, brand, file[:file.find(".csv")])

        # write results to file. 
        with open(full_path + "_means.txt", "w") as f:
            np.savetxt(f, means)
        
        with open(full_path + "_medians.txt", "w") as f:
            np.savetxt(f, medians)

def main():
    window_size = int(input("Window Size (32, 64, 128, etc.): "))
    
    create_dirs(window_size)

    # will store the the np.means and np.medians in folders
    convertToTimeSeries(window_size)

if __name__ == "__main__":
    main()