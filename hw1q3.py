import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(i):
    return np.longfloat(1.0 / (1 + np.exp(-i)))

def GradAscent(data_features, data_labels):
    data_matrix = np.mat(data_features)
    label_matrix = np.mat(data_labels).transpose()
    m, n = np.shape(data_matrix) #3067*57
    alpha = 0.01
    maxCycles = 1000
    weights = np.ones((n,1))
    #for k in range(maxCycles):
    while 1:
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h #3067*1
        weights_pre = weights;
        deta = alpha * data_matrix.transpose() * error #57*1
        weights = weights + deta
        #print abs(sum(weights-weights_pre)/n)
        if abs(sum(weights_pre - weights) / n) < 0.05:
            break
    #print weights
    return weights

def error_rate(data_test_features, data_test_labels, weights):
    m = len(data_test_features)
    error = 0
    for i in range(m):
        prob = sigmoid(sum(data_test_features[i] * weights))
        if prob[0] > 0.5:
            #print "bigger than 0.5"
            if data_test_labels[i] == 0:
                error += 1
        else:
            #print "smaller than 0.5"
            if data_test_labels[i] == 1:
                error += 1
    return float(error) / len(data_test_labels)

def standardize(data):
    m, n = np.shape(data)
    for i in range(n):
        features = data[:, i]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            data[:, i] = (features - meanVal) / std
        else:
            data[:, i] = 0
    return data

if __name__ == '__main__':
    # Setup command line Argumnets.
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='the path of data.')

    # Parse arguments
    args = parser.parse_args()
    file_path = args.file_path
    try:
        file = open(file_path, 'r')
    except IOError:
        print "Error: cannot find actual file"
    else:
        data = np.loadtxt(file, delimiter = ",")
        file.close()

    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]

    data_labels = data[:,len(data[0]) - 1]
    data_features = data[:, 0:len(data[0]) - 1]

    rows = 2 * len(data) / 3
    rest = len(data) - rows

    data_train_labels = data_labels[:rows]
    #print np.count_nonzero(data_train_labels==1)
    #print len(data_train_labels) - np.count_nonzero(data_train_labels==1)
    data_test_labels = data_labels[-rest:]
    data_test_features = data_features[-rest:]
    data_train_features = data_features[:rows]
    data_train_features_normal = standardize(data_train_features)
    data_test_features_normal = standardize(data_test_features)
    #print data_train_features_normal
    update_weights = GradAscent(data_train_features_normal, data_train_labels)
    #print update_weights
    error_rate = error_rate(data_test_features_normal, data_test_labels, update_weights)
    print 1 - error_rate
