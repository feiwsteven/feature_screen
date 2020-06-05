import numpy as np
from numpy.random import binomial
from scipy.stats import chi2



def screen_test(y, x):
    p = x.shape[1]
    n = x.shape[0]
    pi_y = np.zeros((2,1))
    pi_y[0, :] = (n - y.sum()) / n
    pi_y[1, :] = y.sum() / n

    pi_x = np.zeros((2, p))
    pi_x[0, :] = (n - x.sum(axis=0)) / n
    pi_x[1, :] = x.sum(axis=0) / n

    pi_x_0 = np.zeros((2, p))
    pi_x_1 = np.zeros((2, p))

    pi_x_0[0, :] = (n - y.sum() - x[y == 0, :].sum(axis=0)) / n
    pi_x_0[1, :] = x[y == 0, :].sum(axis=0) / n

    pi_x_1[0, :] = (y.sum() - x[y == 1, :].sum(axis=0)) / n
    pi_x_1[1, :] = x[y == 1, :].sum(axis=0) / n

    stat = (pi_x[0, :] * pi_y[0] - pi_x_0[0, :]) ** 2 / (pi_x[0, :] * pi_y[0]) + \
           (pi_x[1, :] * pi_y[0] - pi_x_0[1, :]) ** 2 / (pi_x[1, :] * pi_y[0]) + \
           (pi_x[0, :] * pi_y[1] - pi_x_1[0, :]) ** 2 / (pi_x[0, :] * pi_y[1]) + \
           (pi_x[1, :] * pi_y[1] - pi_x_1[1, :]) ** 2 / (pi_x[1, :] * pi_y[1])

    return stat, 1 - chi2.cdf(stat * n, 1), stat.argsort()[::-1]
if __name__ == '__main__':
    y = np.array(binomial(1, 0.5, 100))
    x = np.array(binomial(1, 0.2, 1000000)).reshape((100, 10000))

    stat, pvalue, index = screen_test(y, x)

    print(pvalue)
