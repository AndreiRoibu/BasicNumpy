'''Spiral Dataset

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code generates and plots the spiral dataset. It parameterizes the radius and angle, and makes them both proportional to a parametr t.

The ideea here is to increase the radius from low to high, while also increasing the angle proportional to the radius.

X is defined as radius * cos(angle), while y is defined as r * sin(angle)

'''

import numpy as np
import matplotlib.pyplot as plt

def generate_radius_and_angle(arms,number_of_points):
    """ Radius and Angle generator

    This function generates the information required for the radius and angle.

    Args:
        arms (int): Number of spiral arms to be generated.
        number_of_points (int): Number of points to be generated for each arm

    Returns:
        radius (np.array): The radius of each arm
        thetas (np.array): An array containing the angles for each of the arms

    """

    radius = np.linspace(1,10,number_of_points)
    thetas = np.empty((arms,number_of_points))

    for arm_index in range(arms):
        start_angle = np.pi * arm_index / 3.0
        end_angle = start_angle + np.pi / 2
        angles = np.linspace(start_angle, end_angle, number_of_points)
        thetas[arm_index] = angles

    return radius, thetas

def generate_data(arms,number_of_points):
    """ Data point generator

    Function generating the points required for the concentric circles

    Args:
        arms (int): Number of spiral arms to be generated.
        number_of_points (int): Number of points to be generated for each arm

    Returns:
        X (np.array): Array of required points
    """

    radius, thetas = generate_radius_and_angle(arms,number_of_points)
    x1 = np.empty((arms,number_of_points))
    x2 = np.empty((arms,number_of_points))
    for arm_index in range(arms):
        x1[arm_index] = radius * np.cos(thetas[arm_index])
        x2[arm_index] = radius * np.sin(thetas[arm_index])

    X = np.empty((number_of_points * arms, 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()

    X += np.random.randn(X.shape[0], X.shape[1]) * 0.5 # Adding some gaussian noise

    return X

def generate_labels(arms,number_of_points):
    """ Label Generator

    Function which generates the required labels for the data

   Args:
        arms (int): Number of spiral arms to be generated.
        number_of_points (int): Number of points to be generated for each arm

    Returns:
        y (np.array): Labels array

    """

    N = int(number_of_points)
    y = [0]*N + [1]*N
    y = y * int(arms//2)
    y = np.array(y)

    return y

def plot_spiral(arms,number_of_points):
    """ Spiral plotter

    Function plotting the XOR distribution.

   Args:
        arms (int): Number of spiral arms to be generated.
        number_of_points (int): Number of points to be generated for each arm
    """

    X = generate_data(arms,number_of_points)
    y = generate_labels(arms,number_of_points)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

if __name__ == "__main__":
    number_of_points = 2000
    arms = 6
    plot_spiral(arms,number_of_points)