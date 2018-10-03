import numpy as np
import matplotlib.pyplot as plt

def train_SGD(data, w, learning_rate):
    for rounds in range(1000):
        for i in range(0, len(data)):
            predict = w[0] + w[1] * data[i][0] + w[2] * data[i][1]
            if data[i][2] * predict <= 0:
                w[1] = w[1] + learning_rate * data[i][0] * data[i][2]
                w[2] = w[2] + learning_rate * data[i][1] * data[i][2]
                w[0] = w[0] + learning_rate * data[i][2]
        if complete(data, w):
            break
    return w

def complete(data, w):
    for i in range(0, len(data)):
        predict = w[0] + w[1] * data[i][0] + w[2] * data[i][1]
        if data[i][2] * predict <= 0:
            return False
    return True

if __name__ == '__main__':
    data = np.array([[1, 2, 1],
            [1, 4, 1],
            [2, 2, 1],
            [4, 2, -1],
            [3, 4, -1],
            [2, 3, -1]])
    x1, x2, y1, y2 =[],[],[],[]
    x1 = data[0:3,0]
    x2 = data[3:6,0]
    y1 = data[0:3,1]
    y2 = data[3:6,1]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x1, y1, c = 'r')
    ax1.scatter(x2, y2, c = 'b')

    w = [0, 0, 0]
    learning_rate = 0.5
    w_update = train_SGD(data, w, learning_rate)
    print w_update

    x = np.linspace(1, 4)
    y = (w[1] * x + w[0]) / -w[2]
    ax1.plot(x, y, c = 'g')
    plt.show()


plt.show()
