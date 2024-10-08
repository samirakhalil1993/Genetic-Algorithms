import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense





#Generate data points in Python using:
x_data = np.linspace(-0.5 , 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_data, y_data)

# Predict y values based on the x_data
y_pred = model.predict(x_data)

# Plot the original noisy data and the linear regression line
plt.scatter(x_data, y_data, color='blue', label='Noisy Data')
plt.plot(x_data, y_pred, color='red', label='Linear Regression')

# Adding labels and title
plt.title('Linear Regression Model on Noisy Data')
plt.xlabel('x')
plt.ylabel('y')

#Show the plot
plt.legend()
plt.grid=True
plt.show()

# Trsnform the data for a polynomial regression model for degree 2
ploy = PolynomialFeatures(degree=2)
x_ploy=ploy.fit_transform(x_data)

# Create and train the polynomial regression model
ploy_model = LinearRegression()
ploy_model.fit(x_ploy,y_data)

# Predict y values based on the polynomial model
y_poly_pred = ploy_model.predict(x_ploy)

# Plot the original noisy data and the liner regression line
plt.scatter(x_data , y_data,color='blue', label = 'Data eith noise')
plt.plot(x_data,y_poly_pred,color='red',label='liner regression')

# Adding labels and title
plt.title('lenier regression model based on noisy data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid=True
plt.show()

# Split the dataset into training (80%) and testing data (20%)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

# Build the neural network model
model = Sequential()

# Input layer (1 node) and hidden layer (6 nodes) with ReLU activation
model.add(Dense(6, input_dim=1, activation='relu'))

# Output layer (1 node)
model.add(Dense(1))

# Compile the model using Mean Squared Error loss and Adam optimizer
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(x_train, y_train, epochs=100, verbose=0, validation_split=0.2)

# Predict y values for the test set
y_pred = model.predict(x_test)

# Plot the results
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.scatter(x_test, y_pred, color='red', label='Predicted Data')
plt.title('Neural Network Predictions vs Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()