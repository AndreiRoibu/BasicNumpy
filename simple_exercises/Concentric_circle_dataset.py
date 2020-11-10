'''Concentric Circles dataset

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code generates and plots the donuts, or concentric circles dataset, with some added noise.

'''

import numpy as np
import matplotlib.pyplot as plt

def generate_data(number_of_points, inner_radius, outer_radius):
    """ Data point generator

    Function generating the points required for the concentric circles

    Args:
        number_of_points (int): Number of points to be generated
        inner_radius (int): Inner circle radius
        outer_radius (int): Outer circle radius

    Returns:
        X (np.array): Array of required points
    """

    inner_data = np.random.randn(number_of_points) + inner_radius
    inner_theta = 2 * np.pi * np.random.random(number_of_points)
    X_inner = np.concatenate([[inner_data * np.sin(inner_theta)], [inner_data * np.cos(inner_theta)]]).T

    outer_data = np.random.randn(number_of_points) + outer_radius
    outer_theta = 2 * np.pi * np.random.random(number_of_points)
    X_outer = np.concatenate([[outer_data * np.sin(outer_theta)], [outer_data * np.cos(outer_theta)]]).T

    X = np.concatenate([X_inner, X_outer])

    return X

def generate_labels(X):
    """ Label Generator

    Function which generates the required labels for the data

    Args:
        X (np.array): Array of required points

    Returns:
        y (np.array): Labels array

    """
    N = X.shape[0] // 2
    y = np.array([0] * N + [1] * N)

    return y

def plot_concentric(number_of_points, inner_radius, outer_radius):
    """ Concentric circle plotter

    Function plotting the XOR distribution.

    Args:
        number_of_points (int): Number of points to be generated
    """

    X = generate_data(number_of_points, inner_radius, outer_radius)
    y = generate_labels(X)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

if __name__ == "__main__":
    number_of_points = 2000
    inner_radius = 5
    outer_radius = 25
    plot_concentric(number_of_points, inner_radius, outer_radius)