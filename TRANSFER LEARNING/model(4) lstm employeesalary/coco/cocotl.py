import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv('coco_minimal_dataset.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop unnecessary column

# Select relevant features
cols = ['learner_id', 'course_id', 'learner_rating']
df_for_training = df[cols].astype(float)
print("Selected columns for training:", cols)

# Normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_scaled = scaler.transform(df_for_training)

# Define sequence lengths
n_future = 1  # Prediction horizon
n_past = 15   # Past sequences to consider

X, Y = [], []
for i in range(n_past, len(df_scaled) - n_future + 1):
    X.append(df_scaled[i - n_past:i, :])
    Y.append(df_scaled[i + n_future - 1:i + n_future, 2])  # Predict learner_rating

X, Y = np.array(X), np.array(Y)

# Split data into 80% training and 20% testing
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Load a pre-trained LSTM model (if available) OR create a base model
try:
    base_model = load_model('pretrained_lstm_model.h5')  # Load a pre-trained model if available
    print("Loaded pre-trained model successfully!")
except:
    print("No pre-trained model found. Creating a new LSTM model instead.")
    base_model = Sequential()
    base_model.add(LSTM(128, activation="sigmoid", input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False))
    base_model.add(Dense(trainY.shape[1]))
    base_model.compile(optimizer='adam', loss='mean_squared_error')
    base_model.save('pretrained_lstm_model.h5')  # Save for future use

# Freeze all layers except the last Dense layer
for layer in base_model.layers[:-1]:
    layer.trainable = False

# Compile the model again after freezing layers
base_model.compile(optimizer='adam', loss='mean_squared_error')
base_model.summary()

# Fine-tune the model (train only the last layer)
history = base_model.fit(trainX, trainY, epochs=7, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions on test data
test_predictions = base_model.predict(testX)

# Compute RMSE and MAE on test data
rmse = np.sqrt(mean_squared_error(testY, test_predictions))
mae = mean_absolute_error(testY, test_predictions)

print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")

# Predict one future value (learner_rating)
n_days_for_prediction = 1
last_sequence = df_scaled[-n_past:]
last_sequence = np.reshape(last_sequence, (1, n_past, len(cols)))

prediction = base_model.predict(last_sequence)
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 2]
print("Predicted learner rating for one step ahead:", y_pred_future[0])

