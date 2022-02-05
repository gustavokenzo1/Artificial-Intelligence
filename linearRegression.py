"""
Implementation of Linear Regression
from professor Diogo Cortiz's AI Crash Course
"""

from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Get data set
dataSet = 'https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/RegressaoLinear/FuelConsumptionCo2.csv'
df = pd.read_csv(dataSet)

# Store only engine and co2 data in two separate dataframes
engines = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]

# Divide the database between train and test data
# test_size is the percentage of data which will be used for training
engines_train, engines_test, co2_train, co2_test = train_test_split(engines, co2, test_size=0.2, random_state=42)

# Plot train data
plt.scatter(engines_train, co2_train, color='blue')
plt.title('Train data')
plt.xlabel('Engine')
plt.ylabel('CO2 Emission')
plt.show()

# Apply the Linear Regression model
model = linear_model.LinearRegression()

# Train the model using engine and co2 data
# Returns a function y = a + bx
model.fit(engines_train, co2_train)

print('Intercepts at a =', model.intercept_)
print('Angular coeficient b =', model.coef_)

# Plot linear regression model
plt.scatter(engines_train, co2_train, color='blue') # Same as before
# engines_train is our f(x), 
# model.coef_[0][0] is the angular coeficiente, which multiplies the 'x' 
# model.intercept_[0] is the height our function crosses when x = 0 
plt.plot(engines_train, model.coef_[0][0]*engines_train + model.intercept_[0], '-r')
plt.ylabel('CO2 Emission')
plt.xlabel('Engine')
plt.title('Train data with linear regression')
plt.show()

# Execute our model in the test dataset to see how accurate it is
predict_co2 = model.predict(engines_test)

# Plot linear regression in test dataset
plt.scatter(engines_test, co2_test, color='red')
plt.plot(engines_test, model.coef_[0][0]*engines_test + model.intercept_[0], '-r')
plt.ylabel('CO2 Emission')
plt.xlabel('Engine')
plt.title('Test data with linear regression')
plt.show()

# Evaluate accuracy
print('Squared Sum Error =', np.sum((predict_co2 - co2_test)**2))
print('Median Squared Error =', mean_squared_error(co2_test, predict_co2))
print('Mean Absolute Error =', mean_absolute_error(co2_test, predict_co2))
print('Root Mean Squared Error =', sqrt(mean_squared_error(co2_test, predict_co2)))
print('R2-score =', r2_score(co2_test, predict_co2))

