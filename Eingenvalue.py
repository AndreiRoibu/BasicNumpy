''' Eingenvector for an eingenvalue of 1

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

Defines a matrix A and a vector V which is uniform and sums to one.

This code multiplies A and V 25 times, obtaining a new v every time. 

The code then plots the euclidian distance between each new v, and the original v, ie. |v'-v| . This converges to 0. 

This means that if |v'-v| = 0, we have found the eigenvector of A, for which the corresponding eingenvalue is 1.
'''

import numpy as np
import matplotlib.pyplot as plt

def data_generator():
    '''
    This function generates the A and v matrices.

    Args:
        none
    Returns:
        A (array): Starting Matrix
        v (vector): Original value of vector
    '''

    A = np.array(
        [[0.3, 0.6, 0.1],
        [0.5, 0.2, 0.3],
        [0.4, 0.1, 0.5]]
    )

    v = np.ones(3) / 3.0

    return A,v

def eigenvectors(iterations=25):
    '''
    This function calculates the eingenvector

    Args:
        iterations (int): The number of iterations for running the function

    Returns:
        distances (list): A list containing the different euclidian distances
    '''

    A, v = data_generator()

    distances = []

    for _ in range(iterations):
        v_new = v.dot(A)
        distance = np.linalg.norm(v_new - v)
        distances.append(distance)
        v = v_new

    return distances

if __name__ == '__main__':
    distances = eigenvectors()
    plt.plot(distances)
    plt.savefig('Eigenvalue.png')
