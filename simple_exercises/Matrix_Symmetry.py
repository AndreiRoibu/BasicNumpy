'''Matrix Symmetry

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code tests if a matrix is symmetric. This is done either manually, and using numpy functions.

'''

import numpy as np
import matplotlib.pyplot as plt

def symmetric_test(array):
    """ Simple symmetry test

    Function wrapper which calls to check if a matrix is symmetric.

    Args:
        array (np.array): Array to be tested if symmetric
    Returns:
        (bool): Truth value indicating if array is symmetric
    """

    return np.all(array == array.T)

def symmetric_check(array, truth):
    """ Symmetric assertion checker

    This function checks if an array is symmetric or not.

    Args:
        array (np.array): Array to be tested if symmetric
        truth (bool): Truth value indicating if array is symmetric or not
    """
    print("Testing: ", '\n', array)
    assert(symmetric_test(array) == truth)

if __name__ == '__main__':
    A = np.zeros((3, 3))
    symmetric_check(A, True)

    A = np.eye(3)
    symmetric_check(A, True)

    A = np.random.randn(3, 2)
    A = A.dot(A.T)
    symmetric_check(A, True)

    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    symmetric_check(A, True)

    A = np.random.randn(3, 2)
    symmetric_check(A, False)

    A = np.random.randn(3, 3)
    symmetric_check(A, False)

    A = np.arange(9).reshape(3, 3)
    symmetric_check(A, False)