import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load dataset
try:
    df = pd.read_csv('netflix_minimal_dataset.csv')
except FileNotFoundError:
    print("Error: File 'beauty_minimal_dataset.csv' not found. Please check the file path.")
    exit()

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isna().sum())

# Ensure 'learner_rating' column exists
if 'learner_rating' not in df.columns:
    print("\nError: 'learner_rating' column is missing in the dataset. Check column names!")
    exit()

# Visualize the distribution of the 'learner_rating' column
sns.histplot(df['learner_rating'], kde=True)
plt.title("Learner Rating Distribution")
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Normalize data (Standardization)
df_norm = (df - df.mean()) / df.std()

# Convert features (X) and target (Y)
X = df_norm.drop(columns=['learner_rating'])  # All columns except 'learner_rating'
Y = df_norm['learner_rating']  # Target variable

# Convert to NumPy arrays
X_arr = X.values
Y_arr = Y.values

# Print dataset shapes
print("\nDataset Shapes:")
print(f"X shape: {X_arr.shape}, Y shape: {Y_arr.shape}")

# Split data into training and testing sets (5% test data)
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=0.05, shuffle=True, random_state=0)

# Print train/test split shapes
print("\nTrain/Test Split:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Calculate mean & std of 'learner_rating' to revert normalization later
y_mean = df['learner_rating'].mean()
y_std = df['learner_rating'].std()

# Function to revert normalization for predictions
def convert_label_value(pred):
    return pred * y_std + y_mean  # Keep as float instead of int

# Define neural network model
def get_model():
    model = Sequential([
        Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Initialize model
model = get_model()
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training & validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

plot_loss(history)

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Convert normalized predictions back to actual learner ratings
y_test_actual = [convert_label_value(y) for y in y_test.flatten()]
y_pred_actual = [convert_label_value(y) for y in y_pred]

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print("\nModel Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Function to compare true vs predicted learner ratings
def compare_predictions(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='red')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='blue', linestyle='--')  # Ideal line
    plt.xlabel("True Learner Ratings")
    plt.ylabel("Predicted Learner Ratings")
    plt.title("True vs Predicted Learner Ratings")
    plt.show()

compare_predictions(y_test_actual, y_pred_actual)

