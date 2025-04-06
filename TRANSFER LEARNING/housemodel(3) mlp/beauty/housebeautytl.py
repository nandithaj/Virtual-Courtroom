import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the pre-trained model
try:
    pretrained_model = load_model('learner_rating_model.h5')  # Ensure model exists
    print("✅ Pre-trained model loaded successfully!")
except:
    print("❌ Error: Pre-trained model not found!")
    exit()

# Freeze all layers except the last one
for layer in pretrained_model.layers[:-1]:
    layer.trainable = False

# Replace the last layer with a new trainable one
x = pretrained_model.layers[-2].output  # Take output from the second-last layer
new_output = Dense(1, activation='linear', name='new_output_layer')(x)

# Create a new model
transfer_model = Model(inputs=pretrained_model.input, outputs=new_output)

# Compile the model
transfer_model.compile(loss='mse', optimizer='adam')

# Display model summary
transfer_model.summary()

# Load dataset
try:
    df = pd.read_csv('beauty_minimal_dataset.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found!")
    exit()

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isna().sum())

# Ensure 'learner_rating' column exists
if 'learner_rating' not in df.columns:
    print("\n❌ Error: 'learner_rating' column is missing in the dataset!")
    exit()

# Normalize data (Standardization)
df_norm = (df - df.mean()) / df.std()

# Convert features (X) and target (Y)
X = df_norm.drop(columns=['learner_rating'])  
Y = df_norm['learner_rating']  

# Convert to NumPy arrays
X_arr = X.values
Y_arr = Y.values

# Split data into training and testing sets (5% test data)
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=0.05, shuffle=True, random_state=0)

# Store mean & std of 'learner_rating' for re-scaling
y_mean = df['learner_rating'].mean()
y_std = df['learner_rating'].std()

# Function to revert normalization for predictions
def convert_label_value(pred):
    return pred * y_std + y_mean  

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fine-tune the model
history = transfer_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,  # Fine-tuning for fewer epochs
    batch_size=16,
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
    plt.title('Training vs Validation Loss (Fine-Tuned Model)')
    plt.show()

plot_loss(history)

# Predict on test set
y_pred = transfer_model.predict(X_test).flatten()

# Convert normalized predictions back to actual values
y_test_actual = [convert_label_value(y) for y in y_test.flatten()]
y_pred_actual = [convert_label_value(y) for y in y_pred]

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print("\n✅ Model Evaluation Metrics:")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE: {mae:.2f}")

# Function to compare true vs predicted values
def compare_predictions(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='red')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='blue', linestyle='--')  # Ideal line
    plt.xlabel("True Learner Ratings")
    plt.ylabel("Predicted Learner Ratings")
    plt.title("True vs Predicted Learner Ratings (Fine-Tuned Model)")
    plt.show()

compare_predictions(y_test_actual, y_pred_actual)

# Save the fine-tuned model
transfer_model.save('fine_tuned_learner_rating_model.h5')
print("✅ Fine-tuned model saved successfully!")

