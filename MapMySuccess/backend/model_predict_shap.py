import joblib
import numpy as np
import pandas as pd
import shap

def model_output(Avg_population_density, traffic_severity, distance_to_main_road, average_price_level, same_type, total_type, cost):
    # 🔹 Load the saved model and scaler
    xgb_model = joblib.load("xgb_model_optuna.pkl")
    scaler = joblib.load("scaler.pkl")

    # 🔹 Define the input features
    features = ["cost", "pop_density", "traffic_rte", "visibility", "avg_price_level", "Comp_Score"]

    # 🔹 Construct input data
    new_data = pd.DataFrame([{
        "cost": float(cost),
        "pop_density": float(Avg_population_density),
        "traffic_rte": float(traffic_severity),
        "visibility": float(distance_to_main_road),
        "avg_price_level": float(average_price_level),
        "Comp_Score": (0.35 * same_type) + (0.65 * total_type)
    }])

    # 🔹 Scale the data
    new_data_scaled = scaler.transform(new_data)

    # 🔹 Predict the score
    predicted_score = xgb_model.predict(new_data_scaled)

    # 🔹 Compute SHAP values
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(new_data_scaled)

    # 🔹 Display prediction and SHAP values
    print("Predicted True Score:", predicted_score[0])
    shap_df = pd.DataFrame({
        "feature": features,
        "value": new_data.iloc[0].values,
        "shap_value": shap_values.values[0]
    })
    
    #shap.plots.waterfall(shap_values[0])
    print(shap_df)

    return float(predicted_score[0])

if __name__ == "__main__":
    score, shap_details = model_output(100, 2, 50, 3, 0.5, 0.8, 200000)
    print("Final Score:", score)
    print("SHAP Explanation:\n", shap_details)
