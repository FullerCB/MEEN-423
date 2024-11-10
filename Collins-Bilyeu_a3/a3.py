'''

I modified the gamma parameters because the values >0.01 take reallyyyyy long to compute

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("steel_strength.csv")
output_var = 'tensile strength' #change to yield strength, tensile strength, or elongation
data = data.dropna(subset=[f'{output_var}'])

X = data[['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]
y = data[f'{output_var}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {
    'kernel': ['poly', 'rbf'],  # Chosen kernels
    'C': [0.1, 1, 10, 100, 250, 500, 1000],  # Chosen C values
    'gamma': [0.001, 0.005, 0.01]  # Chosen gamma values
}

estimator = SVR(epsilon=50)

grid_search = GridSearchCV(estimator, param_grid)
grid_search.fit(X_train, y_train)

#Best parameters
best_parameters = grid_search.best_params_
kernel_name = best_parameters['kernel']
C = best_parameters['C']
gamma = best_parameters['gamma']

best_svr = grid_search.best_estimator_
y_train_pred = best_svr.predict(X_train)
y_test_pred = best_svr.predict(X_test)

#Quantifying test and training datas
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


#output
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel(f'True {output_var}')
plt.ylabel(f'Predicted {output_var}')
plt.title(f'True vs Predicted {output_var}')
plt.show()

print('Best hyperparameters:')
print('\t Kernel={}'.format(kernel_name))
print('\t C={}'.format(C))
print('\t gamma={}'.format(gamma))
print('\n\t\t\t MSE\t\t R2')
print('Train \t {:7.4f}\t {:7.4f}'.format(train_mse, train_r2))
print('Test \t {:7.4f}\t {:7.4f}'.format(test_mse, test_r2))