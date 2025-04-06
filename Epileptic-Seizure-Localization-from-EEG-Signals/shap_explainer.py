# shap_explainer.py

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, data_path, model_path, seizure_time_path, output_path):
        self.data_path = data_path
        self.model_path = model_path
        self.seizure_time_path = seizure_time_path
        self.output_path = output_path

        selected_electrodes = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']
        
        # Create a list of seizures to explain
        seizures = []
        for filename in os.listdir(seizure_time_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(seizure_time_path, filename)
                with open(file_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    row_num = 1    
                    for row in reader:
                        try:
                            start_time = float(row['Start Time (s)'])
                            end_time = float(row['End Time (s)'])
                            seizures.append((filename, start_time, end_time, row_num))
                            row_num += 1
                        except (KeyError, ValueError) as e:
                            continue

        # Function to preprocess the data file and extract the specified time range
        def process_file(file_path, start_time, end_time, sample_rate=256):
            # Load the data
            df = pd.read_csv(file_path)

            # Separate features and labels
            X = df.drop(columns=['SEIZURE_LABEL'])
            y = df['SEIZURE_LABEL']

            # Filter to remove extra electrodes
            X = X[selected_electrodes]

            # Convert time range to indices
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)

            # Select the desired time range
            X = X.iloc[start_idx:end_idx]
            y = y.iloc[start_idx:end_idx]

            return X, y
        
        # Load model
        model = tf.keras.models.load_model(model_path)

        '''seizures = [('chb01_03.csv', 3001, 3001.03, 1),
                    ('chb01_04.csv', 1468, 1468.03, 1)]'''

        # Generating explanations
        for seizure in seizures:
            data_file = os.path.join(data_path, seizure[0])
            start_time = seizure[1]
            end_time = seizure[2]

            # Process the file and extract data for the given time range
            X_explain, y_explain = process_file(data_file, start_time, end_time)

            # Reshape the data to match the model's input shape
            X_explain = np.array(X_explain).reshape(X_explain.shape[0], 23, 1, 1)

            # Select a subset of the data for SHAP background (reduce size if memory issues arise)
            X_background = X_explain[:100]  # Modify size based on memory availability

            # KernelExplainer works by using the model's predict function and a background dataset
            def model_predict(X_batch):
                # Ensure the model gets input in the correct shape (batch size, 23 features, 1, 1)
                X_batch = X_batch.reshape(-1, 23, 1, 1)  # Reshape batch to match model input shape
                return model.predict(X_batch).reshape(-1)  # Return flattened predictions

            # Initialize SHAP Kernel Explainer with the model and background data
            explainer = shap.KernelExplainer(model_predict, X_background.reshape(X_background.shape[0], -1))

            # Compute SHAP values for all samples in X_explain using batches
            batch_size = 2000  # Adjust batch size based on memory availability
            num_samples = X_explain.shape[0]

            shap_values_full = []  # List to store SHAP values for all batches

            # Process in batches
            for i in range(0, num_samples, batch_size):
                batch = X_explain[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}: {batch.shape}")
                try:
                    shap_values_batch = explainer.shap_values(batch.reshape(batch.shape[0], -1))  # SHAP computation for the batch
                    shap_values_full.append(shap_values_batch[0])  # Use first output's SHAP values
                except Exception as e:
                    print(f"Error during SHAP value computation for batch {i // batch_size + 1}: {e}")
                    raise

            # Concatenate all batched SHAP values
            shap_values_full = np.concatenate(shap_values_full, axis=0)

            # Reshape SHAP values to match the input shape (23 features per sample)
            shap_values_full = shap_values_full.reshape(num_samples, 23)

            output_folder = os.path.join(output_path, f"{seizure[0]}_{seizure[3]}")
            os.makedirs(output_folder, exist_ok=True)

            # Visualize SHAP values for the first sample
            shap.initjs()
            force_fig = shap.force_plot(
                explainer.expected_value[0],
                shap_values_full[0],  # Flatten SHAP values for visualization
                X_explain[0].reshape(-1),  # Flatten input features for visualization
                feature_names=selected_electrodes
            )
            force_fig.savefig(os.path.join(output_folder, 'force_plot.png'))
            plt.close(force_fig)

            # Summary plot for SHAP values
            plt.figure()
            shap.summary_plot(
                shap_values_full,                # Flatten SHAP values for summary plot
                X_explain.reshape(num_samples, 23),  # Flatten input features for summary plot
                feature_names=selected_electrodes
            )
            plt.savefig(os.path.join(output_folder, 'summary_plot.png'))
            plt.close()
