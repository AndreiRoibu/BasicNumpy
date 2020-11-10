'''CSVs with Pandas

Author: Andrei-Claudiu Roibu, 2019

This code has been created to compare the speed difference between a numpy dot product and a slow dot product. This code was written in support of my learning.

The original code comes from these two sources: 

    # https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
    # https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

Description:

This code uses previously generated datasets, and saves them as a CSV, with headers, using the Pandas library.
'''

import numpy as np
import pandas as pd
from Spiral_dataset import generate_data, generate_labels

number_of_points = 2000
arms = 6

X = generate_data(arms,number_of_points)
y = generate_labels(arms,number_of_points)

data = np.concatenate((X, np.expand_dims(y, 1)), axis=1)

df = pd.DataFrame(data)
df.columns = ['x1', 'x2', 'y']
df.to_csv('mydata.csv', index=False)