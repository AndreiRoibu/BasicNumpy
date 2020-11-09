''' Numpy Speed Comparison

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python


'''
import numpy as np
from datetime import datetime

def speed_comparison():

    a = np.random.randn(100)
    b = np.random.randn(100)
    T = 100000

    def slow_dot_product(a, b):
        result = 0
        for e, f in zip(a, b):
            result += e*f
        return result

    t0 = datetime.now()
    for _ in range(T):
        slow_dot_product(a, b)
    dt1 = datetime.now() - t0

    t0 = datetime.now()
    for t in range(T):
        a.dot(b)
    dt2 = datetime.now() - t0

    return dt1, dt2

if __name__ == '__main__':

    for i in range(10):
        dt1,dt2 = speed_comparison()
        print("dt1 / dt2:", dt1.total_seconds() / dt2.total_seconds())