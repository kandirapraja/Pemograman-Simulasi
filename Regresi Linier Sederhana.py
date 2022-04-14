import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

datasets = pd.read_csv('Data1.csv')

X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Fitting Simple Linear Regression to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set result ￼

Y_Pred = regressor.predict(X_Test)

# Visualising the Training set results

plt.scatter(X_Train, Y_Train, color='blue')
plt.plot(X_Train, regressor.predict(X_Train), color='red')
plt.title('Berat Mobil vs Konsumsi Bahan Bakar  (Training Set)')
plt.xlabel('Berat Mobil')
plt.ylabel('Konsumsi Bahan Bakar')
plt.show()

# Visualising the Test set results

plt.scatter(X_Test, Y_Test, color='blue')
plt.plot(X_Train, regressor.predict(X_Train), color='red')
plt.title('Berat Mobil vs Konsumsi Bahan Bakar  (Test Set)')
plt.xlabel('Berat Mobil')
plt.ylabel('Konsumsi Bahan Bakar')
plt.show()
