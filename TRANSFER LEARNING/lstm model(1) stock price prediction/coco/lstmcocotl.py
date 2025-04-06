import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the pre-trained model
pretrained_model = load_model('learner_rating_prediction_model.h5')

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

# Summary of the modified model
transfer_model.summary()

# Load the new dataset and preprocess it
file_path = "coco_minimal_dataset.csv"
df = pd.read_csv(file_path)

# Selecting relevant numeric columns
df_numeric = df[["learner_rating", "course_id"]]  

# Convert to numeric and handle errors
df_numeric["learner_rating"] = pd.to_numeric(df_numeric["learner_rating"], errors='coerce')
df_numeric["course_id"] = pd.to_numeric(df_numeric["course_id"], errors='coerce')

# Drop rows with missing values
df_numeric = df_numeric.dropna()

# Load the scaler used in the previous training
scalerfile = 'scaler_learner_rating.pkl'
scaler = pickle.load(open(scalerfile, 'rb'))

# Scale the new data
df_scaled = scaler.transform(df_numeric)

# Function to create time-series sequences
def createXY(dataset, n_past):
    dataX, dataY = [], []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, :])  
        dataY.append(dataset[i, 0])  
    return np.array(dataX), np.array(dataY)

# Define past time steps
n_past = 30  

# Create new training data
X_new, Y_new = createXY(df_scaled, n_past)

# Fine-tune the model
history = transfer_model.fit(X_new, Y_new, epochs=5, batch_size=16, validation_split=0.2)

# Predictions on the new dataset
predictions = transfer_model.predict(X_new)

# Rescale predictions to original scale
predictions_rescaled = scaler.inverse_transform(
    np.hstack((predictions, np.zeros((len(predictions), df_numeric.shape[1] - 1))))
)
pred = predictions_rescaled[:, 0]  # Extract predicted 'learner_rating'

# Rescale actual Y values to original scale
Y_new_rescaled = scaler.inverse_transform(
    np.hstack((Y_new.reshape(-1, 1), np.zeros((len(Y_new), df_numeric.shape[1] - 1))))
)
original = Y_new_rescaled[:, 0]  # Extract actual 'learner_rating'

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(original, pred))
mae = mean_absolute_error(original, pred)

print(f"\n✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(original, color='red', label='Actual Learner Rating')
plt.plot(pred, color='blue', label='Predicted Learner Rating')
plt.title('Learner Rating Prediction (Fine-Tuned Model)')
plt.xlabel('Time')
plt.ylabel('Learner Rating')
plt.legend()
plt.show()

# Save the fine-tuned model
transfer_model.save('fine_tuned_learner_rating_model.h5')
print("✅ Fine-tuned model saved!")

