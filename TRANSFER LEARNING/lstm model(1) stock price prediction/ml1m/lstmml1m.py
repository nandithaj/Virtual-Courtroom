import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Load the dataset
file_path = "ml1m_minimal_dataset.csv"
df = pd.read_csv(file_path).head(2500)  # Select first 2500 rows


# Display the first few rows and column names
print("Dataset Preview:")
print(df.head())

# Selecting relevant numeric columns
df_numeric = df[["learner_rating", "course_id"]]  # Adjusted based on available columns

# Convert to numeric, handling errors
df_numeric["learner_rating"] = pd.to_numeric(df_numeric["learner_rating"], errors='coerce')
df_numeric["course_id"] = pd.to_numeric(df_numeric["course_id"], errors='coerce')

# Drop rows with missing or invalid values
df_numeric = df_numeric.dropna()

# Split into training and testing sets (80% training, 20% testing)
test_split = round(len(df_numeric) * 0.20)
df_for_training = df_numeric[:-test_split]
df_for_testing = df_numeric[-test_split:]

print("Training Data Shape:", df_for_training.shape)
print("Testing Data Shape:", df_for_testing.shape)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)

# Function to create X and Y datasets
def createXY(dataset, n_past):
    dataX, dataY = [], []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, :])  # Use all features for input
        dataY.append(dataset[i, 0])  # Predict 'learner_rating'
    return np.array(dataX), np.array(dataY)

# Create training and testing datasets
n_past = 30  # Number of past time steps to consider
trainX, trainY = createXY(df_for_training_scaled, n_past)
testX, testY = createXY(df_for_testing_scaled, n_past)

print("trainX Shape:", trainX.shape)
print("trainY Shape:", trainY.shape)
print("testX Shape:", testX.shape)
print("testY Shape:", testY.shape)

# Build the LSTM Model
def build_model(optimizer):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_past, df_numeric.shape[1])))  # Multivariate input
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predict 'learner_rating'
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Hyperparameter tuning
grid_model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
parameters = {'batch_size': [16, 20],
              'epochs': [8, 10],
              'optimizer': ['adam', 'Adadelta']}

grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)
grid_search = grid_search.fit(trainX, trainY)
best_params = grid_search.best_params_

# Get the best model
my_model = grid_search.best_estimator_.model

# Predictions
predictions = my_model.predict(testX)

# Rescale predictions to original scale
predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), df_numeric.shape[1] - 1)))))
pred = predictions_rescaled[:, 0]  # Extract predicted 'learner_rating'

# Rescale testY to original scale
testY_rescaled = scaler.inverse_transform(np.hstack((testY.reshape(-1, 1), np.zeros((len(testY), df_numeric.shape[1] - 1)))))
original = testY_rescaled[:, 0]  # Extract actual 'learner_rating'

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(original, pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

mae = mean_absolute_error(original, pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Plot results
plt.plot(original, color='red', label='Actual Learner Rating')
plt.plot(pred, color='blue', label='Predicted Learner Rating')
plt.title('Learner Rating Prediction')
plt.xlabel('Time')
plt.ylabel('Learner Rating')
plt.legend()
plt.show()

# Save the model
my_model.save('learner_rating_prediction_model.h5')
print('Model Saved!')

# Save the scaler
scalerfile = 'scaler_learner_rating.pkl'
pickle.dump(scaler, open(scalerfile, 'wb'))
print('Scaler Saved!')

