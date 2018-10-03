import argparse
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(data):
    learning_rate = 0.1
    inital_b = 0
    inital_m = 1
    num_iter = 100

    [a, b] = least_square_1st_polynomial(data)
    ##[a, b, c] = least_square_2nd_polynomial(data)
    ##[a, b, c, d] = least_square_3rd_polynomial(data)
    ##[a, b, c, d, e] = least_square_4th_polynomial(data)
    return [a, b]

def least_square_1st_polynomial(data):
    n = len(data)
    sumX, sumY, sumXY, sumXX = 0,0,0,0
    for i in range(0,n):
        sumX  += data[i, 0]
        sumY  += data[i, 1]
        sumXX += data[i, 0]*data[i, 0]
        sumXY += data[i, 0]*data[i, 1]
    """
    a = (n*sumXY -sumX*sumY)/(n*sumXX -sumX*sumX)
    b = (sumXX*sumY - sumX*sumXY)/(n*sumXX-sumX*sumX)
    print a, b
    """
    a = np.array([[sumX, n], [sumXX, sumX]])
    b = np.array([sumY, sumXY])
    [m, n] = np.linalg.solve(a, b)
    print m, n
    return [m, n]

def least_square_2nd_polynomial(data):
    n = len(data)
    sumX,sumY,sumXY, sumXX,sumXXX, sumXXY, sumXXXX = 0,0,0,0,0,0,0
    for i in range(0, n):
        sumX  += data[i, 0]
        sumY  += data[i, 1]
        sumXX += data[i, 0]*data[i, 0]
        sumXY += data[i, 0]*data[i, 1]
        sumXXX += data[i, 0]*data[i, 0]*data[i, 0]
        sumXXY += data[i, 0]*data[i, 0]*data[i, 1]
        sumXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
    a = np.array([[sumXX, sumX, n], [sumXXX, sumXX, sumX], [sumXXXX, sumXXX, sumXX]])
    b = np.array([sumY, sumXY, sumXXY])
    [m, n, l] = np.linalg.solve(a, b)
    print m, n, l
    return [m, n, l]

def least_square_3rd_polynomial(data):
    n = len(data)
    sumX,sumY,sumXY, sumXX,sumXXX, sumXXY, sumXXXX, sumXXXXX, sumXXXXXX, sumXXXY = 0,0,0,0,0,0,0,0,0,0
    for i in range(0, n):
        sumX  += data[i, 0]
        sumY  += data[i, 1]
        sumXX += data[i, 0]*data[i, 0]
        sumXY += data[i, 0]*data[i, 1]
        sumXXX += data[i, 0]*data[i, 0]*data[i, 0]
        sumXXY += data[i, 0]*data[i, 0]*data[i, 1]
        sumXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXY += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 1]
    a = np.array([[sumXXX, sumXX, sumX, n],
                [sumXXXX, sumXXX, sumXX, sumX],
                [sumXXXXX, sumXXXX, sumXXX, sumXX],
                [sumXXXXXX, sumXXXXX, sumXXXX, sumXXX]])
    b = np.array([sumY, sumXY, sumXXY, sumXXXY])
    [q, w, e, r] = np.linalg.solve(a, b)
    print q, w, e, r
    return [q,w,e,r]

def least_square_4th_polynomial(data):
    n = len(data)
    sumX,sumY,sumXY, sumXX,sumXXX, sumXXY, sumXXXX, sumXXXXX, sumXXXXXX, sumXXXY, sumXXXXXXX, sumXXXXXXXX, sumXXXXY = 0,0,0,0,0,0,0,0,0,0,0,0,0
    for i in range(0, n):
        sumX  += data[i, 0]
        sumY  += data[i, 1]
        sumXX += data[i, 0]*data[i, 0]
        sumXY += data[i, 0]*data[i, 1]
        sumXXX += data[i, 0]*data[i, 0]*data[i, 0]
        sumXXY += data[i, 0]*data[i, 0]*data[i, 1]
        sumXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXY += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 1]
        sumXXXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXXXXX += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]
        sumXXXXY += data[i, 0]*data[i, 0]*data[i, 0]*data[i, 0]*data[i, 1]
    a = np.array([[sumXXXX, sumXXX, sumXX, sumX, n],
                [sumXXXXX, sumXXXX, sumXXX, sumXX, sumX],
                [sumXXXXXX, sumXXXXX, sumXXXX, sumXXX, sumXX],
                [sumXXXXXXX, sumXXXXXX, sumXXXXX, sumXXXX, sumXXX],
                [sumXXXXXXXX, sumXXXXXXX, sumXXXXXX, sumXXXXX, sumXXXX]])
    b = np.array([sumY, sumXY, sumXXY, sumXXXY, sumXXXXY])
    [q, w, e, r, t] = np.linalg.solve(a, b)
    print q, w, e, r, t
    return [q,w,e,r, t]

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
        file.readline()
        data = np.loadtxt(file, delimiter = ",")
        file.close()
    data = np.delete(data,0, axis=1)
    x_array = data[:,0]
    y_array = data[:,1]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x_array, y_array, c = 'r', marker = 'o')
    plt.legend('x1')

    [a, b] = linear_regression(data)

    x = np.linspace(0, 30)
    y = a * x + b
    ax1.plot(x, y)
    plt.show()
