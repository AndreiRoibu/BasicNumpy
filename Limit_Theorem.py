'''Limit Theorem

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code demonstrates the central limit theorem. This states that, if we set a random variable to be the sum of other random variables from any distribution, then as the number of random values tends to infinity, the distribution of the sum approaches the normal distribution. 

In mathematics, this is: Y = X1 + X2 + ... + Xn

As N -> infinity, Y -> Gaussian distribution

This code proves this both numerically and graphically. Drawing N=1000 samples from a uniform distribution, and then for that variable, 1000 Ys are drawn. 

The code then plots an instagram of the samples. The mean and the variance of Y are also calculated.
'''

import numpy as np
import matplotlib.pyplot as plt

def Y_generator(n=1000):
    '''
    This function sums random variables and returns the Y.

    Args:
        n (int): The number of random samples

    Returns:
        Y (float): The value of the sum of random values
    '''

    x = np.random.random(n)
    return np.sum(x)

def Y_samples_generator(N=1000):
    '''
    This function generates a N -seris of random Y_samples

    Args:
        N (int): The number of samples

    Returns:
        Y_samples (list): The list with values of the sum of random values
        Y_standard_deviation (float): Standard deviation of data
        Y_mean (float): Mean of data
    '''

    Y_samples = np.zeros(N)
    for i in range(N):
        Y_samples[i] = Y_generator()

    Y_mean = np.mean(Y_samples)
    Y_standard_deviation = np.std(Y_samples)

    return Y_samples, Y_mean, Y_standard_deviation

if __name__ == '__main__':
    Y_samples, Y_mean, Y_standard_deviation = Y_samples_generator(N=10000)
    plt.hist(Y_samples, bins=25)
    plt.savefig('LimitTheorem.png')
    print("Mean is {} and standard deviation is {}".format(Y_mean, Y_standard_deviation))