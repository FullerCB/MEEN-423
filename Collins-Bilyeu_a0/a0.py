import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import plotly.graph_objects as go
import numpy as np

#data handeling
data = pd.read_csv("a0_data.csv")

data_train, data_test = sklearn.model_selection.train_test_split(data, train_size = 0.8, random_state=2024)

x_train = data_train[['x1', 'x2']]
y_train = data_train['y']
x_test = data_test[['x1', 'x2']]
y_test = data_test['y']

#model creation
#linear
linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(x_train, y_train)
linear_train_error = sklearn.metrics.mean_squared_error(y_train, linear_model.predict(x_train))
linear_test_error = sklearn.metrics.mean_squared_error(y_test, linear_model.predict(x_test))
linear_errors = (linear_train_error, linear_test_error)

#quadratic
quadratic_model = sklearn.linear_model.LinearRegression()
quadratic_feature = sklearn.preprocessing.PolynomialFeatures(degree=2)

x_train_quadratic = quadratic_feature.fit_transform(x_train)
quadratic_model.fit(x_train_quadratic, y_train)
y_train_pred_quadratic = quadratic_model.predict(x_train_quadratic)

x_test_quadratic = quadratic_feature.transform(x_test)
y_test_pred_quadratic = quadratic_model.predict(x_test_quadratic)

quadratic_train_error = sklearn.metrics.mean_squared_error(y_train, y_train_pred_quadratic)
quadratic_test_error = sklearn.metrics.mean_squared_error(y_test, y_test_pred_quadratic)
quadratic_errors = (quadratic_train_error, quadratic_test_error)

#cubic
cubic_model = sklearn.linear_model.LinearRegression()
cubic_feature = sklearn.preprocessing.PolynomialFeatures(degree=3)

x_train_cubic = cubic_feature.fit_transform(x_train)
cubic_model.fit(x_train_cubic, y_train)
y_train_pred_cubic = cubic_model.predict(x_train_cubic)

x_test_cubic = cubic_feature.transform(x_test)
y_test_pred_cubic = cubic_model.predict(x_test_cubic)

cubic_train_error = sklearn.metrics.mean_squared_error(y_train, y_train_pred_cubic)
cubic_test_error = sklearn.metrics.mean_squared_error(y_test, y_test_pred_cubic)
cubic_errors = (cubic_train_error, cubic_test_error)

# display_results_table
#
# A specialized display utility for MEEN 423 A0. 
# Creates a table for visualizing the training and test loss statistics
# for three different fits.
#
# Takes three arguments, each of which is a two-element tuple. Assumes the 
# first element of the tuple is for training MSE and the second element is for 
# test MSE. 
def display_results_table(linear, quadratic, cubic):
  print('\n\t\t\tTrain MSE\tTest MSE')
  print('Linear    \t%6.4f\t\t%6.4f' % linear)
  print('Quadratic \t%6.4f\t\t%6.4f' % quadratic)
  print('Cubic     \t%6.4f\t\t%6.4f' % cubic)

display_results_table(linear_errors, quadratic_errors, cubic_errors)
