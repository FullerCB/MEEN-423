import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the dataset
df = pd.read_csv('updated_file.csv')
data = df.dropna(subset=['FantasyPoints_x', 'FantasyPoints_y'])

# Define features and target
feature_columns = [
    "Age", "G", "GS", "Tgt", "Rec", "PassingYds", "PassingTD",
    "PassingAtt", "RushingYds", "RushingTD", "RushingAtt",
    "ReceivingYds", "ReceivingTD", "FantasyPoints_x", "Int",
    "Fumbles", "FumblesLost"
]
target_column = 'FantasyPoints_y'

X = data[feature_columns]
y = data[target_column]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.975)
X_pca = pca.fit_transform(X_scaled)

# Split the PCA-transformed data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define the objective function for Bayesian Optimization
def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    regularization_strength = 1e-5  # Reduced L2 regularization strength

    # Build the model
    model = Sequential()
    model.add(Dense(
        n_units,
        activation="relu",
        kernel_regularizer=l2(regularization_strength),
        input_shape=(X_train.shape[1],)
    ))
    for _ in range(n_layers - 1):
        model.add(Dense(
            n_units,
            activation="relu",
            kernel_regularizer=l2(regularization_strength)
        ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    # Train the model
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=30,
        verbose=0
    )

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    return r2

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Retrain the model with optimal hyperparameters and L2 regularization
final_model = Sequential()
regularization_strength = 1e-5  # Reduced L2 regularization strength
final_model.add(Dense(
    best_params['n_units'],
    activation="relu",
    kernel_regularizer=l2(regularization_strength),
    input_shape=(X_train.shape[1],)
))
for _ in range(best_params['n_layers'] - 1):
    final_model.add(Dense(
        best_params['n_units'],
        activation="relu",
        kernel_regularizer=l2(regularization_strength)
    ))
    final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(1))

# Compile with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss="mean_squared_error")
history = final_model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=best_params['batch_size'],
    epochs=100,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the final model
y_final_pred = final_model.predict(X_test).flatten()
final_r2 = r2_score(y_test, y_final_pred)
final_mse = mean_squared_error(y_test, y_final_pred)

print("Final R2 Score:", final_r2)
print("Final MSE:", final_mse)

# Residuals for plotting
differences = y_test - y_final_pred

# Generate plots
plt.figure(figsize=(18, 6))

# Plot 1: Histogram of Residuals
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
plt.scatter(y_test, y_final_pred, color='blue')
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

# PCA explained variance
print(f'Explained variance ratio for each principal component: {pca.explained_variance_ratio_}')
print(f'Total variance explained by the selected components: {sum(pca.explained_variance_ratio_)}')
