import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load - Replace with File Path
df = pd.read_csv('updated_file.csv')
data = df.dropna(subset=['FantasyPoints_x', 'FantasyPoints_y'])

# Define your feature columns and target column (replace these with actual column names)
feature_columns = columns = [
    "Age", "G", "GS", "Tgt", "Rec", "PassingYds", "PassingTD", 
    "PassingAtt", "RushingYds", "RushingTD", "RushingAtt", 
    "ReceivingYds", "ReceivingTD", "FantasyPoints_x", "Int", 
    "Fumbles", "FumblesLost"
]
target_column = 'FantasyPoints_y'  # target column name


# Split the dataset into features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.dropna()

# Apply PCA
pca = PCA(n_components=0.975)  # Capture 97.5% of the variance
X_pca = pca.fit_transform(X_scaled)
print(f"Number of components selected: {pca.n_components_}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25)

#Build the neural network
from tensorflow.keras.regularizers import l2

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)  # Output layer
])

model.compile(optimizer='rmsprop', loss='mean_squared_error')

#Training

early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=3, 
    validation_data=(X_test, y_test),
    verbose = 2,
    callbacks=[early_stopping]
    )


#Evaluate
from sklearn.metrics import r2_score, mean_squared_error
mse_train = mean_squared_error(y_train, model.predict(X_train))
y_pred = np.array(model.predict(X_test))
mse_test = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training error (MSE):", mse_train)
print("Testing error (MSE):", mse_test)
print("Testing R2 score:", r2)
r2 = r2_score(y_test,  data.loc[y_test.index, "FantasyPoints_x"])
print("NONTesting R2 score:", r2)
#Saving
model.save('fantasy_points_predictor_model.h5')

differences = y_test - y_pred.flatten()

import matplotlib.pyplot as plt
from scipy.stats import norm

plt.figure(figsize=(18, 6))

# Plot 1: Histogram of Residuals (Differences)
plt.subplot(1, 3, 1)
plt.hist(differences, bins=10, density=True, alpha=0.6, color='g', edgecolor='black', label='Residuals (Differences)')
plt.axhline(0, color='black', linestyle='--')
plt.title('Difference Between Actual and Predicted Fantasy Points')
plt.xlabel('Difference')
plt.ylabel('Relative Frequency')

# Calculate and plot the normal distribution for the residuals
mean_residual = np.mean(differences)
std_residual = np.std(differences)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_residual, std_residual)
plt.plot(x, p, 'k', linewidth=2)  # Normal distribution curve

# Plot 2: Actual vs Predicted Fantasy Points
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred.flatten(), color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Fantasy Points')
plt.xlabel('Actual Fantasy Points')
plt.ylabel('Predicted Fantasy Points')

# Plot 3: Actual vs Last Year Fantasy Points (FantasyPoints_x)
plt.subplot(1, 3, 3)
plt.scatter(y_test, data.loc[y_test.index, "FantasyPoints_x"], color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Last Year Fantasy Points')
plt.xlabel('Actual Fantasy Points')
plt.ylabel('Last Year Fantasy Points')

# Show the plots
plt.tight_layout()
plt.show()


# PCA explained variance (optional, helps to understand the variance captured by the components)
print(f'Explained variance ratio for each principal component: {pca.explained_variance_ratio_}')
print(f'Total variance explained by the selected components: {sum(pca.explained_variance_ratio_)}')

data = data.copy()

# Predict fantasy points for all rows in the dataset
all_predictions = model.predict(X_pca)

# Add the predictions as a new column to the dataset
data['PredictedFantasyPoints'] = all_predictions.flatten()

# Save the updated dataset back to the original CSV file
data.to_csv('updated_file.csv', index=False)

print("Predicted fantasy points have been added to 'updated_file.csv' for all rows.")