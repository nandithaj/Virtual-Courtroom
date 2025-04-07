# lime_explainer.py

import csv
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import gc

class LimeExplainer:
    def __init__(self, data_path, model_path, seizure_time_path, output_path):
        self.data_path = data_path
        self.model_path = model_path
        self.seizure_time_path = seizure_time_path
        self.output_path = output_path

        # 23 selected electrodes
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

        # Prediction function
        def model_predict(input_data):
            sample = input_data.reshape(-1, X.shape[1], 1, 1)
            prediction = model.predict(sample)
            probability_0 = 1 - prediction
            probability_1 = prediction
            return np.concatenate([probability_0, probability_1], axis=1)
        
        # Load model
        model = load_model(model_path)

        for seizure in seizures:
            data_file = os.path.join(data_path, seizure[0])
            start = seizure[1]
            
            # Create output folder
            output_folder = os.path.join(output_path, f'{seizure[0]}_{seizure[3]}')
            os.makedirs(output_folder, exist_ok=True)

            # Feature weights file
            electrode_weight_file = os.path.join(output_folder, 'electrode_weights.csv')
            with open(electrode_weight_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['sample'] + selected_electrodes
                writer.writerow(header)

            # Loading data
            data = pd.read_csv(data_file)

            # Time interval
            start_time = start - 30
            end_time = start + 10

            # Data preparation
            X = data[(data['time'] >= start_time) & (data['time'] <= end_time)][selected_electrodes].values
            y = data[(data['time'] >= start_time) & (data['time'] <= end_time)]['SEIZURE_LABEL'].values
            time_column = data[(data['time'] >= start_time) & (data['time'] <= end_time)]['time'].values

            # Creating LIME explainer
            explainer = LimeTabularExplainer(
                training_data=X.reshape(X.shape[0], X.shape[1]),
                mode='classification',
                feature_names=selected_electrodes
            )

            # Generating LIME explanations
            with open(electrode_weight_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                for i in range(len(X)):
                    # Generating explanation
                    explanation = explainer.explain_instance(
                        data_row=X[i].reshape(-1),
                        predict_fn=model_predict,
                        top_labels=2,
                        num_features=23
                    )

                    # Explanations
                    prediction_class = 1
                    local_exp = explanation.local_exp[prediction_class]
                    mapped_explanations_d = {explainer.feature_names[idx]: weight for idx, weight in local_exp}
                    row = [f'sample_{i}'] + [mapped_explanations_d.get(electrode, 0) for electrode in selected_electrodes]
                    writer.writerow(row)

                    plt.figure()
                    explanation.as_pyplot_figure(label=prediction_class)
                    plt.title(f"LIME Explanation for Sample {i} - Predicted Class {prediction_class}")
                    plot_filename = f"{output_folder}/lime_explanation_sample_{i}.png"
                    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
                    plt.close()

                    # Explicitly delete objects to free memory
                    del explanation
                    gc.collect()
