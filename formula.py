from math import pi, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def gaussian(x, mean, sigma):
    power = -0.5 * np.square(x - mean)/np.square(sigma)
    coef = 1/(sigma * sqrt(2 * pi))
    return coef * np.exp(power)




if __name__ == "__main__":
    rand_array = np.random.uniform(low=-3, high=3, size=(10000,))
    sns.lineplot(x=rand_array, y=gaussian(rand_array, mean=0, sigma=1))
    plt.show()
