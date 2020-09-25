
# Problem Statement: https://www.hackerrank.com/challenges/predicting-office-space-price/problem

# Import dependencies
import numpy as np
from sklearn.linear_model import LinearRegression

dataset = open('reg_office_prices_input.txt', 'r').read() # input()
dataset = dataset.split('\n')

# Get the number of features and number of rows in the dataset.
f, n = tuple(map(lambda x: int(x), dataset[0].split(' ')))

# Initialize variables.
arr = []
j = 1

# Get the data.
for i in range(n):
    x = list(map(lambda x: float(x), dataset[i + 1].split(' ')))

    j = j + 1
    arr.append(x)

# Create the numpy array from the Python list.
narray = np.array(arr, dtype='float32')

# Get features list.
X = narray[:, :-1]

# Get labels list.
y = narray[:, -1:]

# Initialize the LinearRegression() object.
reg = LinearRegression()

# Fit the features to the labels using LinearRegression.
reg.fit(X, y)

# Get the number of test cases.
t = int(dataset[j])

predictions = []

# Predict the output for all the test cases.
for i in range(t):
    j = j + 1

    # Get the test case.
    testcase = list(map(lambda x: float(x), dataset[j].split(' ')))

    # Make predictions on the new data.
    predictions.append(str(round(reg.predict([testcase]).reshape(1)[0], 2)))

# Print the predictions
print('\n'.join(predictions))
