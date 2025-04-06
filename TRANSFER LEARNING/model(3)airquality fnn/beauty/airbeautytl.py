import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
try:
    df = pd.read_csv("beauty_minimal_dataset.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File 'beauty_minimal_dataset.csv' not found!")
    exit()

# Normalize data (Standardization)
df_norm = (df - df.mean()) / df.std()

# Convert features (X) and target (Y)
X = df_norm.drop(columns=['learner_rating'])  
Y = df_norm['learner_rating']  

# Convert to NumPy arrays
X_arr = X.values
Y_arr = Y.values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=0)

# Load pre-trained model
pretrained_model = load_model('learner_rating_model.h5')
print("✅ Pre-trained model loaded successfully!")

# Freeze all layers except the last one
for layer in pretrained_model.layers[:-1]:
    layer.trainable = False

# Recompile the model
pretrained_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Fine-tune the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = pretrained_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[early_stopping], verbose=1)

# Save the fine-tuned model
pretrained_model.save('fine_tuned_learner_rating_model.h5')
print("✅ Fine-tuned model saved successfully as 'fine_tuned_learner_rating_model.h5'!")

# Predict on test set
y_pred = pretrained_model.predict(X_test).flatten()

# Compute RMSE & MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n✅ Fine-Tuned Model Evaluation Metrics:")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE: {mae:.2f}")

# Plot Training vs Validation Loss
def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss (Fine-Tuned Model)')
    plt.show()

plot_loss(history)

# Plot True vs Predicted Learner Ratings
def compare_predictions(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='red', label='Predicted')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='blue', linestyle='--', label='Ideal Fit')
    plt.xlabel("True Learner Ratings")
    plt.ylabel("Predicted Learner Ratings")
    plt.title("True vs Predicted Learner Ratings (Fine-Tuned Model)")
    plt.legend()
    plt.show()

compare_predictions(y_test, y_pred)

