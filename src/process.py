import numpy as np

def process_data(data,mean=None,std=None):
    # normalize (add 1e-15 to std to avoid numerical issue)
    if mean is not None: #if mean and std passed in

        data = (data - mean) / (std + 1e-15)

        return data
    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        data = (data - mean) / (std + 1e-15)

        return data, mean, std

def process_label(label):
    # convert into one-hot vector
    one_hot = np.zeros([len(label),10])
    
    for i in range(len(label)):
        r = label[i]
        one_hot[i][r] = 1


    return one_hot