'''MNIST flip

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code loads the MNIST data set, found at this link: https://www.kaggle.com/c/digit-recognizer/data 

This code contains a function that flips an image 90 degrees clockwise, using both for loops and numpy functions, comparing their performance.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MNIST_means import data_loader

def rotate90(image):
    """ Rotate the image by 90 degrees

    Args:
        image (np.array): Array containing the image information
    """

    return np.rot90(image, 3)

def rotate_caller(data_path):
    """ Rotation caller

    This function loads the MNIST data and rotates the images.

    Args:
        data_path (str): Path to where the data file is
    """

    labels, images = data_loader(data_path)
    number_of_images = images.shape[0]

    for index in range(number_of_images):
        image = images[index].reshape(28,28)
        image = rotate90(image)

        plt.imshow(image, cmap='gray')
        plt.title("Label: "+ str(labels[index]))
        plt.show()

        ans = input("Continue [y/n]: ")
        if ans and ans[0].lower() == 'n':
            break

if __name__ == "__main__":
    data_path = '../../large_files/train.csv'
    rotate_caller(data_path)