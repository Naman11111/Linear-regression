# importing necessary libraries for use

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# downloading dataset

!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

# Reading the data in

df = pd.read_csv("FuelConsumption.csv")
 
# take a look at the dataset

df.head()#displays first five rows of data 
#df.head(10) display specified rows of data, in this case 10

# summarize the data
df.describe()

# select some features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# plot each of these features
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_CITY','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()

# plot each of these features vs the Emission, to see how linear is their relation
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.title("SCATTER PLOT")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS,color='red')
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("EMISSION")
plt.title("SCATTER PLOT")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color=('yellow'))
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="green")
plt.xlabel('Cylinder')
plt.ylabel('Emission')
plt.show()

# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print(train)
print(test)

'''
Coefficient and Intercept in the simple linear regression, are the parameters of the fit line.
Given that it is a simple linear regression, with multiple parameters.
sklearn can estimate them directly from our data. Notice that all of the data must be available to traverse and calculate the parameters.
'''

from sklearn import linear_model

regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
x_test = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train,y_train)

# The coefficients

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# Plot outputs
# CO2EMISSION / Y = Intercept + cofficient1*ENGINESIZE + cofficient2*CYLINDERS + cofficient3*FUELCONSUMPTION_COMB

plt.scatter(train.ENGINESIZE, y_train,  color='blue')
plt.scatter(train.CYLINDERS, y_train, color='green')
plt.scatter(train.FUELCONSUMPTION_COMB, y_train, color='orange')
plt.plot(x_train, regr.coef_[0][0]*x_train+ regr.intercept_[0], '-r')
plt.plot(x_train, regr.coef_[0][1]*x_train+ regr.intercept_[0], 'black')
plt.plot(x_train, regr.coef_[0][2]*x_train+ regr.intercept_[0], 'purple')
plt.xlabel("Engine size, Cylinders, Fuelconsumption")
plt.ylabel("Emission")

#predict actual values

pred = regr.predict(x_test) # predicts the corresponding values of y
print(pred)

'''
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model.

Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

        Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.

        Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.

        Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.

        R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
'''

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , y_test) )
print("Regresion score: %.2f" % (regr.score(x_test,y_test)*100))
=======
# importing necessary libraries

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# downloading dataset

!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

# Reading the data in

df = pd.read_csv("FuelConsumption.csv")
 
# take a look at the dataset

df.head()#displays first five rows of data 
#df.head(10) display specified rows of data, in this case 10

# summarize the data
df.describe()

# select some features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# plot each of these features
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_CITY','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()

# plot each of these features vs the Emission, to see how linear is their relation
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.title("SCATTER PLOT")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS,color='red')
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("EMISSION")
plt.title("SCATTER PLOT")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color=('yellow'))
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="green")
plt.xlabel('Cylinder')
plt.ylabel('Emission')
plt.show()

# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print(train)
print(test)

'''
Coefficient and Intercept in the simple linear regression, are the parameters of the fit line.
Given that it is a simple linear regression, with multiple parameters.
sklearn can estimate them directly from our data. Notice that all of the data must be available to traverse and calculate the parameters.
'''

from sklearn import linear_model

regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
x_test = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train,y_train)

# The coefficients

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# Plot outputs
# CO2EMISSION / Y = Intercept + cofficient1*ENGINESIZE + cofficient2*CYLINDERS + cofficient3*FUELCONSUMPTION_COMB

plt.scatter(train.ENGINESIZE, y_train,  color='blue')
plt.scatter(train.CYLINDERS, y_train, color='green')
plt.scatter(train.FUELCONSUMPTION_COMB, y_train, color='orange')
plt.plot(x_train, regr.coef_[0][0]*x_train+ regr.intercept_[0], '-r')
plt.plot(x_train, regr.coef_[0][1]*x_train+ regr.intercept_[0], 'black')
plt.plot(x_train, regr.coef_[0][2]*x_train+ regr.intercept_[0], 'purple')
plt.xlabel("Engine size, Cylinders, Fuelconsumption")
plt.ylabel("Emission")

#predict actual values

pred = regr.predict(x_test) # predicts the corresponding values of y
print(pred)

'''
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model.

Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

        Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.

        Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.

        Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.

        R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
'''

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , y_test) )
print("Regresion score: %.2f" % (regr.score(x_test,y_test)*100))
>>>>>>> b71309a4c05ec668e86f20443d234349d2c414d3
