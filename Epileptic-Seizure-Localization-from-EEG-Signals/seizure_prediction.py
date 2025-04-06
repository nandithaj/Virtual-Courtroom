# seizure_prediction.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, lfilter, welch
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize_scalar
import seaborn as sns
from statsmodels.robust.scale import mad
import gc

class PredictSeizure:
    def __init__(self, data_path, feature_path, model_path, seizure_time_path):
        self.data_path = data_path
        self.feature_path = feature_path
        self.model_path = model_path
        self.seizure_time_path = seizure_time_path
        
        # 23 selected electrodes
        selected_electrodes = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']

        # List of csv files in the folder
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        # Seizure time approximation
        for file_name in csv_files:
            # Load EEG file
            file_path = os.path.join(data_path, file_name)
            data = pd.read_csv(file_path)

            # Get electrode names (exclude non-electrode columns if any)
            electrode_cols = data.columns

            # Bandpass Filter 
            def bandpass_filter(data, lowcut, highcut, fs, order=4):
                nyquist = 0.5 * fs
                low, high = lowcut / nyquist, highcut / nyquist
                b, a = butter(order, [low, high], btype='band')
                return lfilter(b, a, data)

            # Calculate Power using Welch’s Method
            def calculate_power(data, fs=256):
                freqs, psd = welch(data, fs, nperseg=1024)
                return freqs, psd

            # Sliding Window
            window_size = 30 * 256  # 30 seconds × 256 samples/sec
            step_size = 1 * 256  # 1 second step 

            # Process Each Window
            alpha_band = (8, 13)
            delta_band = (0.5, 4)

            dar_values, time_points = [], []

            # Main Window Processing
            for start in range(0, len(data) - window_size, step_size):
                end = start + window_size
                window_data = data.iloc[start:end]

                alpha_power_total, delta_power_total = 0, 0

                for col in electrode_cols:
                    signal = window_data[col].values
                    freqs, psd = calculate_power(signal)

                    # Alpha Power
                    alpha_idx = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
                    alpha_power = np.sum(psd[alpha_idx])

                    # Delta Power
                    delta_idx = np.where((freqs >= delta_band[0]) & (freqs <= delta_band[1]))[0]
                    delta_power = np.sum(psd[delta_idx])

                    alpha_power_total += alpha_power
                    delta_power_total += delta_power

                # Calculate Delta-to-Alpha Ratio (DAR)
                dar = delta_power_total / alpha_power_total if alpha_power_total != 0 else 0
                dar_values.append(dar)
                time_sec = start / 256  # Convert sample to seconds
                time_points.append(time_sec)

            # Initialize seizure_time as None to avoid errors if no seizure is found
            seizure_time = None

            # Peak Detection and Backtracking Logic

            # Find the peak (maximum value in the dar_values)
            peak_idx = np.argmax(dar_values)  # Find the index of the maximum value
            peak_value = dar_values[peak_idx]

            # Now check the points behind the peak, calculate the ratio between the peak and each previous point
            for j in range(peak_idx - 1, -1, -1):  # Start from the point before the peak and move backward
                ratio = peak_value / dar_values[j] if dar_values[j] != 0 else np.inf  # Avoid division by zero
                if ratio >= 1.9:  # Check if the ratio is 1.9 or greater
                    seizure_time = time_points[j]  # Mark the time of seizure onset
                    break  # Stop further backtracking as we've found the point of interest

            # Save Approximations
            if seizure_time:
                results_df = pd.DataFrame({'Time (seconds)': time_points, 'DAR Value': dar_values})
                results_df['Seizure Indicator'] = ['Yes' if time == seizure_time else 'No' for time in time_points]
                results_path = os.path.join(feature_path, 'seizure_approx')
                os.makedirs(results_path, exist_ok=True)
                results_df.to_csv(os.path.join(results_path, f'{file_name}'), index=False)
        
        # Integrated alpha-delta calculation for preictal
        # Define the output folder
        output_folder_path = os.path.join(feature_path, 'relative_powers')

        # Ensure the output folder exists
        os.makedirs(output_folder_path, exist_ok=True)

        # Define the output CSV files for calculations
        alpha_file_path = os.path.join(output_folder_path, 'preictal_alpha.csv')
        delta_file_path = os.path.join(output_folder_path, 'preictal_delta.csv')

        # Frequency bands
        alpha_range = (8, 12)  # Alpha wave: 8-12 Hz
        delta_range = (0.5, 4)  # Delta wave: 0.5-4 Hz

        # List of file names with seizure start times
        file_list = list()

        results_path = os.path.join(feature_path, 'seizure_approx')
        results_files = [f for f in os.listdir(results_path) if f.endswith('.csv')]

        for file in results_files:
            seizure_df = pd.read_csv(os.path.join(results_path, file)).query(" `Seizure Indicator` == 'Yes'")

            for i in range(len(seizure_df)):
                seizure_record = {
                    "file_name": file,
                    "start": seizure_df.iloc[i]['Time (seconds)'],
                    "seizure_id": i+1
                }
                file_list.append(seizure_record)

        # Dictionary to store computed values before writing
        computed_data = {
            "alpha": {},
            "delta": {}
        }

        # Function to calculate relative band power
        def calculate_relative_band_power(data, column, band_range, fs=256):
            signal = data[column].values
            freqs, psd = welch(signal, fs=fs)
            band_power = psd[(freqs >= band_range[0]) & (freqs <= band_range[1])].sum()
            total_power = psd.sum()
            return band_power / total_power if total_power > 0 else None

        # Process each file and create frames
        for file_info in file_list:
            file_name = file_info["file_name"]
            start_time = file_info["start"]
            seizure_id = file_info["seizure_id"]

            # Load the file as a DataFrame
            input_file_path = os.path.join(data_path, file_name)
            df = pd.read_csv(input_file_path)
            df['time'] = pd.to_numeric(df['time'])

            # Loop to create 30 frames of 1 second each
            for frame_index in range(30):
                frame_start_time = start_time - (30 - frame_index)
                frame_end_time = frame_start_time + 1

                # Filter data for the current frame
                frame_data = df[(df['time'] >= frame_start_time) & (df['time'] < frame_end_time)]
                if frame_data.empty:
                    continue

                # Get actual start and end times from the filtered frame data
                actual_start_time = frame_data['time'].iloc[0]
                actual_end_time = frame_data['time'].iloc[-1]
                time_frame = f"{actual_start_time}-{actual_end_time}"

                # Construct frame name with seizure ID
                frame_full_name = f'{file_name}_{seizure_id}_frame{frame_index + 1}'

                # Calculate alpha and delta power
                alpha_values = []
                delta_values = []
                for electrode in selected_electrodes:
                    if electrode in frame_data.columns:
                        alpha_power = calculate_relative_band_power(frame_data, electrode, alpha_range)
                        delta_power = calculate_relative_band_power(frame_data, electrode, delta_range)
                    else:
                        alpha_power, delta_power = None, None
                    alpha_values.append(alpha_power)
                    delta_values.append(delta_power)

                # Store results in dictionary
                computed_data["alpha"].setdefault((file_name, seizure_id), []).append(
                    [frame_full_name, "alpha", time_frame] + alpha_values
                )
                computed_data["delta"].setdefault((file_name, seizure_id), []).append(
                    [frame_full_name, "delta", time_frame] + delta_values
                )

        # Function to write computed data to CSV
        def write_to_csv(file_path, data_key):
            with open(file_path, 'w') as calc_file:
                # Write header
                calc_file.write('filename,band,time frame,' + ','.join(selected_electrodes) + '\n')

                # Write data in file-wise order
                for file_info in file_list:
                    file_key = (file_info["file_name"], file_info["seizure_id"])
                    if file_key in computed_data[data_key]:
                        for row in computed_data[data_key][file_key]:
                            calc_file.write(",".join([row[0], row[1], row[2]] + [f"{x:.4f}" if x is not None else "NA" for x in row[3:]]) + '\n')

        # Write to separate CSV files
        write_to_csv(alpha_file_path, "alpha")
        write_to_csv(delta_file_path, "delta")
        
        # Integrated alpha-delta calculation for preictal
        # Define the output folder
        output_folder_path = os.path.join(feature_path, 'relative_powers')

        # Ensure the output folder exists
        os.makedirs(output_folder_path, exist_ok=True)

        # Define the output CSV files for calculations
        alpha_file_path = os.path.join(output_folder_path, 'ictal_alpha.csv')
        delta_file_path = os.path.join(output_folder_path, 'ictal_delta.csv')

        # Frequency bands
        alpha_range = (8, 12)  # Alpha wave: 8-12 Hz
        delta_range = (0.5, 4)  # Delta wave: 0.5-4 Hz

        # List of file names with seizure start times
        file_list = list()

        results_path = os.path.join(feature_path, 'seizure_approx')
        results_files = [f for f in os.listdir(results_path) if f.endswith('.csv')]

        for file in results_files:
            seizure_df = pd.read_csv(os.path.join(results_path, file)).query(" `Seizure Indicator` == 'Yes'")

            for i in range(len(seizure_df)):
                seizure_record = {
                    "file_name": file,
                    "start": seizure_df.iloc[i]['Time (seconds)'],
                    "seizure_id": i+1
                }
                file_list.append(seizure_record)

        # Dictionary to store computed values before writing
        computed_data = {
            "alpha": {},
            "delta": {}
        }

        # Function to calculate relative band power
        def calculate_relative_band_power(data, column, band_range, fs=256):
            signal = data[column].values
            freqs, psd = welch(signal, fs=fs)
            band_power = psd[(freqs >= band_range[0]) & (freqs <= band_range[1])].sum()
            total_power = psd.sum()
            return band_power / total_power if total_power > 0 else None

        # Process each file and create frames
        for file_info in file_list:
            file_name = file_info["file_name"]
            start_time = file_info["start"]
            seizure_id = file_info["seizure_id"]

            # Load the file as a DataFrame
            input_file_path = os.path.join(data_path, file_name)
            df = pd.read_csv(input_file_path)
            df['time'] = pd.to_numeric(df['time'])

            # Loop to create 10 frames of 1 second each
            for frame_index in range(10):
                frame_start_time = start_time + frame_index
                frame_end_time = frame_start_time + 1

                # Filter data for the current frame
                frame_data = df[(df['time'] >= frame_start_time) & (df['time'] < frame_end_time)]
                if frame_data.empty:
                    continue

                # Get actual start and end times from the filtered frame data
                actual_start_time = frame_data['time'].iloc[0]
                actual_end_time = frame_data['time'].iloc[-1]
                time_frame = f"{actual_start_time}-{actual_end_time}"

                # Construct frame name with seizure ID
                frame_full_name = f'{file_name}_{seizure_id}_frame{frame_index + 1}'

                # Calculate alpha and delta power
                alpha_values = []
                delta_values = []
                for electrode in selected_electrodes:
                    if electrode in frame_data.columns:
                        alpha_power = calculate_relative_band_power(frame_data, electrode, alpha_range)
                        delta_power = calculate_relative_band_power(frame_data, electrode, delta_range)
                    else:
                        alpha_power, delta_power = None, None
                    alpha_values.append(alpha_power)
                    delta_values.append(delta_power)

                # Store results in dictionary
                computed_data["alpha"].setdefault((file_name, seizure_id), []).append(
                    [frame_full_name, "alpha", time_frame] + alpha_values
                )
                computed_data["delta"].setdefault((file_name, seizure_id), []).append(
                    [frame_full_name, "delta", time_frame] + delta_values
                )

        # Function to write computed data to CSV
        def write_to_csv(file_path, data_key):
            with open(file_path, 'w') as calc_file:
                # Write header
                calc_file.write('filename,band,time frame,' + ','.join(selected_electrodes) + '\n')

                # Write data in file-wise order
                for file_info in file_list:
                    file_key = (file_info["file_name"], file_info["seizure_id"])
                    if file_key in computed_data[data_key]:
                        for row in computed_data[data_key][file_key]:
                            calc_file.write(",".join([row[0], row[1], row[2]] + [f"{x:.4f}" if x is not None else "NA" for x in row[3:]]) + '\n')

        # Write to separate CSV files
        write_to_csv(alpha_file_path, "alpha")
        write_to_csv(delta_file_path, "delta")
        
        # Threshold automation
        # THRESHOLD CALCULATION METHODS

        def dynamic_mad_threshold(data):
            """MAD-based threshold with adaptive scaling"""
            data_clean = data[~np.isnan(data)]
            if len(data_clean) == 0:
                return np.nan
            median = np.median(data_clean)
            mad_val = mad(data_clean)
            if mad_val == 0:
                return median
            # Adaptive scaling based on data spread
            spread_factor = np.log1p(np.ptp(data_clean))  # Log of range
            scale_factor = 1.0 + 0.3 * spread_factor
            return median + scale_factor * mad_val

        def refined_gmm_threshold(data, components=2):
            """Improved GMM threshold with robust component selection"""
            data_clean = data[~np.isnan(data)]
            if len(data_clean) < 10:
                return np.percentile(data_clean, 90)

            try:
                gmm = GaussianMixture(n_components=min(components, len(data_clean)//3),
                                    random_state=42, n_init=3)
                gmm.fit(data_clean.reshape(-1, 1))
                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_.flatten())
                weights = gmm.weights_.flatten()

                # Sort components by mean value
                sorted_idx = np.argsort(means)
                means = means[sorted_idx]
                stds = stds[sorted_idx]
                weights = weights[sorted_idx]

                # Find optimal component (balancing weight and position)
                optimal_idx = -1  # Start with highest mean component
                for i in reversed(range(len(means))):
                    if weights[i] > 0.2:  # Only consider significant components
                        optimal_idx = i
                        break

                # Conservative estimate: mean + 0.25*std
                return means[optimal_idx] + 0.25*stds[optimal_idx]
            except:
                return np.percentile(data_clean, 85)

        def calculate_distribution_features(data):
            """Calculate key statistical features of the data"""
            data_clean = data[~np.isnan(data)]
            if len(data_clean) < 2:
                return {'skew': 0, 'kurtosis': 3, 'iqr': 0, 'tail_ratio': 1}

            q75, q25 = np.percentile(data_clean, [75, 25])
            iqr_val = q75 - q25
            median = np.median(data_clean)

            return {
                'skew': skew(data_clean),
                'kurtosis': max(kurtosis(data_clean), 1),  # Minimum kurtosis of 1
                'iqr': iqr_val,
                'tail_ratio': (q75 - median)/(median - q25 + 1e-9)  # Avoid division by zero
            }

        def auto_calibrated_ensemble(preictal_values, ictal_values=None):
            """Optimized ensemble method that automatically learns the best threshold"""
            # Calculate base thresholds
            gmm_t = refined_gmm_threshold(preictal_values)
            mad_t = dynamic_mad_threshold(preictal_values)

            # Calculate distribution features
            features = calculate_distribution_features(preictal_values)

            # Optimize the ensemble weighting
            def objective(alpha):
                threshold = alpha*gmm_t + (1-alpha)*mad_t

                # Distribution-based penalty
                p95 = np.percentile(preictal_values[~np.isnan(preictal_values)], 95)
                dist_penalty = abs(p95 - threshold)/np.ptp(preictal_values[~np.isnan(preictal_values)])

                # Performance-based optimization (if ictal data available)
                if ictal_values is not None:
                    metrics = evaluate_performance(preictal_values, ictal_values, threshold)
                    perf_score = 0.7*metrics['TP Rate'] + 0.3*(1-metrics['FP Rate'])
                else:
                    perf_score = 0

                return -(0.6*perf_score + 0.4*(1-dist_penalty))

            # Find optimal alpha between 0 and 1
            res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            optimal_alpha = res.x

            # Calculate final threshold with learned weighting
            base_threshold = optimal_alpha*gmm_t + (1-optimal_alpha)*mad_t

            # Apply automatic correction based on historical patterns
            auto_correction = 0.05 * np.log1p(features['iqr']) * np.tanh(features['skew'])
            final_threshold = base_threshold + auto_correction

            # Ensure reasonable bounds
            lower_bound = min(gmm_t, mad_t) - 0.1*np.abs(gmm_t - mad_t)
            upper_bound = max(gmm_t, mad_t) + 0.1*np.abs(gmm_t - mad_t)
            return np.clip(final_threshold, lower_bound, upper_bound)

        # PERFORMANCE EVALUATION

        def evaluate_performance(preictal_values, ictal_values, threshold):
            """Comprehensive evaluation of seizure detection performance"""
            preictal_clean = preictal_values[~np.isnan(preictal_values)]
            ictal_clean = ictal_values[~np.isnan(ictal_values)]

            # Basic counts
            fp = np.sum(preictal_clean > threshold)
            tp = np.sum(ictal_clean > threshold)
            n_preictal = len(preictal_clean)
            n_ictal = len(ictal_clean)

            # Calculate rates
            fp_rate = fp / n_preictal if n_preictal > 0 else 0
            tp_rate = tp / n_ictal if n_ictal > 0 else 0
            tn_rate = 1 - fp_rate
            fn_rate = 1 - tp_rate

            # Precision-recall metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp_rate
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Balanced accuracy
            balanced_acc = (tp_rate + tn_rate) / 2

            return {
                'Threshold': threshold,
                'FP Rate': fp_rate,
                'TP Rate': tp_rate,
                'TN Rate': tn_rate,
                'FN Rate': fn_rate,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Balanced Accuracy': balanced_acc,
                'FP Count': fp,
                'TP Count': tp,
                'Preictal Samples': n_preictal,
                'Ictal Samples': n_ictal
            }

        # MAIN PROCESSING LOOP

        def process_patient():
            try:
                # Load data
                preictal_data = pd.read_csv(os.path.join(feature_path, os.path.join('relative_powers', 'preictal_delta.csv')))
                ictal_data = pd.read_csv(os.path.join(feature_path, os.path.join('relative_powers', 'ictal_delta.csv')))

                # Flatten and clean data
                preictal_values = preictal_data.iloc[:, 3:].values.flatten()
                ictal_values = ictal_data.iloc[:, 3:].values.flatten()
                preictal_values = preictal_values[~np.isnan(preictal_values)]
                ictal_values = ictal_values[~np.isnan(ictal_values)]

                if len(preictal_values) == 0 or len(ictal_values) == 0:
                    return None

                # Calculate only the auto-calibrated threshold
                auto_threshold = auto_calibrated_ensemble(preictal_values, ictal_values)

                return auto_threshold

            except FileNotFoundError:
                print(f"Files for Patient not found, skipping.")
                return None
            except Exception as e:
                print(f"Error processing Patient")
                return None

        result = process_patient()
        if result is not None:
            with open(os.path.join(feature_path, 'threshold.txt'), 'w') as f:
                f.write(str(result))  # Convert result to string before writing
        
        # Model Training
        # List to store DataFrames for each file
        dataframes = []

        # Process specified files in the folder
        for file_name in csv_files:
            file_path = os.path.join(self.data_path, file_name)

            if file_name.endswith('.csv'):
                # Load the file and keep only the selected electrodes and seizure label
                df = pd.read_csv(file_path)
                df_filtered = df[selected_electrodes + ['SEIZURE_LABEL']]
                dataframes.append(df_filtered)

        # Combine all DataFrames into a single dataset
        df_combined = pd.concat(dataframes, ignore_index=True)

        # Free up memory by deleting individual DataFrames
        del dataframes
        gc.collect()

        # Separate features and labels
        X = df_combined.drop(columns=['SEIZURE_LABEL'])
        y = df_combined['SEIZURE_LABEL']

        # Free up memory by deleting the original DataFrame
        del df_combined
        gc.collect()

        # Reshape data for CNN
        X = np.array(X).reshape(X.shape[0], X.shape[1], 1, 1)

        # Define the split ratios
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        # Split into training and temporary (validation + test) datasets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)

        # Further split the temporary dataset into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        # Free up memory by deleting temporary arrays
        del X_temp, y_temp
        gc.collect()

        # Temporarily remove the seizure labels for testing dataset
        y_test_original = y_test.copy()
        y_test = pd.Series(np.nan, index=y_test.index)

        # Free up memory by deleting original NumPy arrays
        X_shape_1 = X.shape[1]
        del X, y
        gc.collect()

        # Define batch size
        batch_size = 32

        # Create TensorFlow datasets for training, validation, and testing
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Free up memory by deleting NumPy arrays after creating datasets
        del X_train, X_val, X_test, y_train, y_val, y_test
        gc.collect()

        # Create saved model folder
        os.makedirs(model_path, exist_ok=True)

        # Path to model checkpoint
        checkpoint_path = os.path.join(model_path, 'model_checkpoint.keras')

        # Build the CNN model
        model = Sequential([
            Input(shape=(X_shape_1, 1, 1)),
            Conv2D(32, (3, 1), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 1)),
            Dropout(0.3),
            Conv2D(64, (3, 1), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 1)),
            Dropout(0.3),
            Conv2D(128, (3, 1), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 1)),
            Dropout(0.4),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # CSV Logger
        csv_logger = CSVLogger(os.path.join(model_path, 'training_log.csv'))

        # Define the checkpoint callback to continue saving the model
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            verbose=1
        )

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=50,
            initial_epoch=0,
            validation_data=val_dataset,
            callbacks=[csv_logger, checkpoint_callback]
        )
        
        # Seizure time prediction
        # Create seizure time folder
        os.makedirs(seizure_time_path, exist_ok=True)

        # Load the trained model
        model = tf.keras.models.load_model(checkpoint_path)

        # Threshold for seizure prediction
        with open(os.path.join(feature_path, 'threshold.txt'), 'r') as f:
            prediction_threshold = round(float(f.read().strip()), 2)

        # Parameters for interval processing
        sampling_rate = 256
        merge_interval_threshold = 5  # Increase merging threshold
        min_seizure_duration = 10  # Minimum duration for a valid seizure interval

        # Process each file independently
        for file in csv_files:
            file_path = os.path.join(data_path, file)

            # Load and filter data
            df = pd.read_csv(file_path)
            df_filtered = df[selected_electrodes].copy()

            # Prepare data for prediction
            X = np.array(df_filtered).reshape(-1, len(selected_electrodes), 1, 1)

            # Predict seizure labels
            y_pred = model.predict(X)
            y_pred_binary = (y_pred > prediction_threshold).astype(int).flatten()

            # Convert predictions to DataFrame and add time column
            df_filtered['predicted_label'] = y_pred_binary
            df_filtered['time'] = np.arange(len(df_filtered)) / sampling_rate

            # Merge contiguous or close seizure intervals
            seizure_frames = df_filtered[df_filtered['predicted_label'] == 1]
            merged_seizures = []

            if not seizure_frames.empty:
                start_time = seizure_frames['time'].iloc[0]
                for i in range(1, len(seizure_frames)):
                    if seizure_frames['time'].iloc[i] - seizure_frames['time'].iloc[i - 1] <= merge_interval_threshold:
                        continue
                    else:
                        end_time = seizure_frames['time'].iloc[i - 1]
                        # Only keep intervals longer than minimum seizure duration
                        if end_time - start_time >= min_seizure_duration:
                            merged_seizures.append([start_time, end_time])
                        start_time = seizure_frames['time'].iloc[i]

                # Final interval check for the last sequence
                end_time = seizure_frames['time'].iloc[-1]
                if end_time - start_time >= min_seizure_duration:
                    merged_seizures.append([start_time, end_time])

            # Save merged seizure intervals as a new CSV for each file
            merged_seizures_df = pd.DataFrame(merged_seizures, columns=['Start Time (s)', 'End Time (s)'])

            # Save the merged seizure intervals to CSV
            csv_file_name = os.path.join(seizure_time_path, file)
            merged_seizures_df.to_csv(csv_file_name, index=False)

            # Clear predictions to ensure independence for the next file
            del df_filtered, X, y_pred, y_pred_binary, seizure_frames, merged_seizures, merged_seizures_df
