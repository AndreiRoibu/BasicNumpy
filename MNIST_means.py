'''MNIST means

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code loads the MNIST data set, found at this link: https://www.kaggle.com/c/digit-recognizer/data 

The code then plots the mean image for each digit from 0 to 9. 

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_loader(data_path):
    '''
    This function loads the MNIST training dataset and returns arrays of the labels and images.

    Args:
        data_path (string): The path to where the data file is.

    Returns:
        labels (array): An array containing the labels
        images (array): An array containing the individual images
    '''
    df = pd.read_csv(data_path)
    data = df.values
    labels = data[:,0]
    images = data[:,1:]
    return labels, images

def mean_images(data_path):
    '''
    This function calculates the mean for each of each number label in the MNIST dataset.

    Args:
        data_path (string): The path to where the data file is.
    '''

    labels, images = data_loader(data_path)
    for k in range(10):
        images_k = images[labels == k]
        mean_image = images_k.mean(axis=0)
        image = mean_image.reshape(28,28)

        plt.clf()
        plt.imshow(image, cmap='gray')
        plt.title("Label: {}".format(k))
        plt.savefig("MNIST_mean_"+str(k)+".png")

if __name__ == '__main__':
    data_path = './data/train.csv'
    mean_images(data_path)