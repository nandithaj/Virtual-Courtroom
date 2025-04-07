from flask import Flask, request, jsonify
from flask_cors import CORS
from model_predict_shap import model_output
from data_fetch import find_restaurant_details
from compModule import getCompCount
import traceback
app = Flask(__name__)
CORS(app) 

@app.route("/predict", methods=["POST"])
def data_fetch():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print("Received data:", data)

        # Ensure required fields are present
        if "latitude" not in data or "longitude" not in data:
            return jsonify({"error": "Missing latitude or longitude"}), 400

        latitude = data["latitude"]
        longitude = data["longitude"]
        cuisine = data["cuisine"]
        expected_price = int(data["expected_price"])

        ## do prediction
        Avg_population_density, traffic_severity, distance_to_main_road, average_price_level,same_type,total_type,summary = find_restaurant_details(latitude,longitude,cuisine)

        required_values = [
                Avg_population_density, traffic_severity, distance_to_main_road,
                average_price_level, same_type, total_type, expected_price
                ]

        if any(v is None for v in required_values):
            if distance_to_main_road is None:
                response = {
                    "location": {"latitude": latitude, "longitude": longitude},
                    "prediction": 0.0, 
                    "remarks": "NOROAD",
                    "summary": summary,
                    "same_type":same_type-1
                }
        else:
            prediction = model_output(Avg_population_density, traffic_severity, distance_to_main_road, average_price_level,same_type,total_type,expected_price)
            print(prediction)
            # Example response (replace with your actual logic)
            response = {
                "location": {"latitude": latitude, "longitude": longitude},
                "prediction": float(prediction),
                "remarks": "OK",
                "summary": summary,
                "same_type":same_type-1
            }

        print(response)
        return jsonify(response)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

