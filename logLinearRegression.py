import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load the diabetes dataset
# sklearn offers the dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]


# Split the data into training/testing sets
# choose the data(except for the last 20 data) for the training set
diabetes_X_train = diabetes_X[:-20]
#print "\ntrain:\n", diabetes_X_train

# choose the last 20 data as for the testing set
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
diabetes_y_original = diabetes.target[-20:]
plt.scatter(diabetes_X_test, diabetes_y_original,  color='black')

#logarithm to the training/testing sets
for i in range(len(diabetes_y_train)):
	diabetes_y_train[i] = math.log(diabetes_y_train[i])

for j in range(len(diabetes_y_test)):
	diabetes_y_test[j] = math.log(diabetes_y_test[j])


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets(log)
# Create the ln y linear regression
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set(log)
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print'Coefficients:', regr.coef_


# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

# Calculate the y (exp(ln y))
for k in range(len(diabetes_y_pred)):
	diabetes_y_pred[k] = math.exp(diabetes_y_pred[k])

plt.scatter(diabetes_X_test, diabetes_y_pred, color='orange')

# Error analysis
diabetes_error_analysis = datasets.load_diabetes()
diabetes_y_test_error_analysis = diabetes_error_analysis.target[-20:]
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test_error_analysis, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test_error_analysis, diabetes_y_pred))

plt.xlabel('x')
plt.ylabel('y')
plt.show()