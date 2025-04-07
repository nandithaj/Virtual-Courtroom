# import joblib
# import numpy as np
# import pandas as pd

# def model_output(Avg_population_density, traffic_severity, distance_to_main_road, average_price_level, same_type, total_type, cost):
#     # Load the saved model and scaler
#     xgb_model = joblib.load("xgb_model_optuna.pkl")
#     scaler = joblib.load("scaler.pkl")

#     #  Define the feature names (must match training)
#     features = ["cost", "pop_density", "traffic_rte", "visibility", "avg_price_level", "Comp_Score"]
    
#     #  Construct input data
#     new_data = pd.DataFrame([{
#         "cost": float(cost),
#         "pop_density": float(Avg_population_density),
#         "traffic_rte": float(traffic_severity),
#         "visibility": float(distance_to_main_road),
#         "avg_price_level": float(average_price_level),
#         "Comp_Score": (0.35 * same_type) + (0.65 * total_type)
#     }])

#     #  Apply MinMaxScaler
#     new_data_scaled = scaler.transform(new_data)

#     #  Predict the true score
#     predicted_score = xgb_model.predict(new_data_scaled)

#     #  Display result
#     print("Predicted True Score:", predicted_score[0])
#     return predicted_score[0]

# if __name__=="__main__":
#     # Example usage
#     model_output(100, 2, 50, 3, 0.5, 0.8, 200000)
#     # Replace the parameters with actual values when calling the function
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def model_output(Avg_population_density, traffic_severity, distance_to_main_road, average_price_level, same_type, total_type, cost):
    # Load the saved model and scaler
    lgb_model = joblib.load("lgbm_model_optuna.pkl")  # Make sure this model is saved from training
    scaler = joblib.load("scaler.pkl")

    # Define the feature names (must match training)
    features = ["cost", "pop_density", "traffic_rte", "visibility", "avg_price_level", "Comp_Score"]

    # Construct input data
    new_data = pd.DataFrame([{
        "cost": float(cost),
        "pop_density": float(Avg_population_density),
        "traffic_rte": float(traffic_severity),
        "visibility": float(distance_to_main_road),
        "avg_price_level": float(average_price_level),
        "Comp_Score": (0.35 * same_type) + (0.65 * total_type)
    }])

    # Apply scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict the true score
    predicted_score = lgb_model.predict(new_data_scaled)

    # Display result
    print("Predicted True Score:", predicted_score[0])
    return predicted_score[0]

if __name__ == "__main__":
    # Example usage
    model_output(100, 2, 50, 3, 0.5, 0.8, 200000)

