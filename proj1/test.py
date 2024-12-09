import numpy as np
import matplotlib.pyplot as plt

#tensor flow stuff
import datetime, os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, MaxPooling1D, Dropout

#sklearn stuff
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read data
X_raw = np.load('xs_train.npy') # shape of (32497, 192, 2)
y_raw = np.load('ys_train.npy') # shape of (32497, 2)

y = y_raw[:, 0] / y_raw[:, 1] # this will convert the output to a list of lift / drag rather than train each individually

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X_raw.reshape(-1, 2)).reshape(X_raw.shape)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

#split up everything
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)


model = models.Sequential([
    Input(shape=(192, 2)),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.0001)),
    
    #output layer
    layers.Dense(1, kernel_regularizer=regularizers.l2(0.0001))
])



model.compile(
    optimizer=optimizers.Adam(0.001),
    loss='mean_squared_error',  # Mean Squared Error for regression
    metrics=[tf.keras.metrics.R2Score()]  # Mean Absolute Error for easier interpretability
)

print(model.summary())

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1)

class DynamicBatchSize(Callback):
    def __init__(self, initial_batch_size=128, final_batch_size=16, decay_rate=0.9):
        super(DynamicBatchSize, self).__init__()
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.decay_rate = decay_rate
    
    def on_epoch_begin(self, epoch, logs=None):
        new_batch_size = max(self.final_batch_size, int(self.initial_batch_size * self.decay_rate ** epoch))
        self.params['batch_size'] = new_batch_size
        print(f"Epoch {epoch+1}: Reducing batch size to {new_batch_size}")

# Example usage in training
dynamic_batch = DynamicBatchSize(initial_batch_size=128, final_batch_size=16, decay_rate=0.95)

history = model.fit(
    X_train,
    y_train,  # single target: lift-to-drag ratio
    epochs=350,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose = 1,
    callbacks=[tensorboard_callback,  early_stopping, lr_scheduler, dynamic_batch]
)

mse_train = mean_squared_error(y_train, model.predict(X_train))

y_pred = np.array(model.predict(X_test))
mse_test = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training error (MSE):", mse_train)
print("Testing error (MSE):", mse_test)
print("Testing R2 score:", r2)

model.save('model.h5')