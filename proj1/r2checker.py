import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#read data
X_raw = np.load('xs_train.npy') # shape of (32497, 192, 2)
y_raw = np.load('ys_train.npy') # shape of (32497, 2)

y = y_raw[:, 0] / y_raw[:, 1] # this will convert the output to a list of lift / drag rather than train each individually
scaler = StandardScaler()
X = scaler.fit_transform(X_raw.reshape(-1, 2)).reshape(X_raw.shape)

#split up everything
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

# Load the trained model
model = tf.keras.models.load_model('98.68.h5')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")