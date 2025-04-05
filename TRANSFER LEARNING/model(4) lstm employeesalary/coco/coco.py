import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the CSV file
df = pd.read_csv('coco_minimal_dataset.csv')

# Drop unnecessary index column
df = df.drop(columns=['Unnamed: 0'])

# Select relevant features
cols = ['learner_id', 'course_id', 'learner_rating']
df_for_training = df[cols].astype(float)
print("Selected columns for training:", cols)

# Normalize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_for_training)

# Split dataset into training (80%) and testing (20%)
train_data, test_data = train_test_split(df_scaled, test_size=0.2, random_state=42, shuffle=False)

# Prepare sequences for training
def create_sequences(data, n_past, n_future):
    X, Y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        Y.append(data[i + n_future - 1:i + n_future, 2])  # Predict learner_rating
    return np.array(X), np.array(Y)

# Set past time steps and future prediction horizon
n_past = 15
n_future = 1

# Create train and test sequences
trainX, trainY = create_sequences(train_data, n_past, n_future)
testX, testY = create_sequences(test_data, n_past, n_future)

# Define LSTM model
model = Sequential()
model.add(LSTM(128, activation="sigmoid", input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False))
model.add(Dense(1))  # Single output for learner_rating

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(trainX, trainY, epochs=7, batch_size=16, validation_data=(testX, testY), verbose=1)

# Make predictions
test_predictions = model.predict(testX)

# Compute RMSE and MAE
rmse = np.sqrt(mean_squared_error(testY, test_predictions))
mae = mean_absolute_error(testY, test_predictions)

print("\n✅ Model Evaluation on Test Set:")
print(f"🔹 RMSE: {rmse:.4f}")
print(f"🔹 MAE: {mae:.4f}")

# Predict one future value (learner_rating)
last_sequence = test_data[-n_past:]  # Use last part of the test set
last_sequence = np.reshape(last_sequence, (1, n_past, len(cols)))

prediction = model.predict(last_sequence)
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 2]

print("Predicted learner rating for one step ahead:", y_pred_future[0])

