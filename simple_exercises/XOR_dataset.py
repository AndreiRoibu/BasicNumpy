''' XOR Dataset

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code generates and plots the XOR dataset. This is characterised by:

0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0

'''

import numpy as np
import matplotlib.pyplot as plt

def generate_data(number_of_points):
    """ Data point generator

    Function generating the points required for XOR

    Args:
        number_of_points (int): Number of points to be generated

    Returns:
        X (np.array): Array of required points
    """

    X = np.random.random((number_of_points, 2))
    X = X * 2 - 1
    return X

def generate_labels(X):
    """ Label Generator

    Function which generates the required labels for the data

    Args:
        X (np.array): Array of required points

    Returns:
        y (np.array): Labels array

    """

    y = np.zeros(X.shape[0])
    y[(X[:,0] < 0) & (X[:,1] > 0)] = 1
    y[(X[:,0] > 0) & (X[:,1] < 0)] = 1

    return y

def plot_XOR(number_of_points):
    """ XOR plotter

    Function plotting the XOR distribution.

    Args:
        number_of_points (int): Number of points to be generated
    """

    X = generate_data(number_of_points)
    y = generate_labels(X)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

if __name__ == "__main__":
    number_of_points = 2000
    plot_XOR(number_of_points)