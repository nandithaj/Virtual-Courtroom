
# 🍽️ Restaurant Success Predictor

A machine learning-based web application that predicts the potential success of a restaurant based on its **location**, **cuisine**, and **average price**. The system leverages real-world data from Google APIs to assess key features such as population density, traffic, competitor density, road proximity, and more.

---

## 🚀 Features

- Predicts success score of a new restaurant at any location.
- Gathers real-time data using Google Places, Roads, and Routes APIs.
- Calculates competition score based on nearby restaurants.
- Supports cuisine and price input from user.
- Frontend built with **ReactJS**, interactive map for selecting location.
- Backend with **Flask**, connects to a trained **LightGBM** model.
- Uses **model serialization** via `joblib` for fast deployment.
- Parallel dataset processing (100 subsets) for efficient data gathering.

---

## 🧠 How It Works

1. **Data Source**: Initial dataset from Kaggle (Swiggy) with 1.5 lakh restaurants.
2. **Feature Enrichment**: Used Google Places API to extract:
   - Ratings and reviews
   - Latitude and longitude
   - Nearby establishments for estimating **population**
   - Distance to nearest road (visibility)
   - **Traffic** information via Google Routes API
   - Average **price level** nearby
3. **Competition Score**:
   - Calculated using total nearby restaurants and similar cuisine types.
4. **True Score**:
   - Derived using restaurant rating and rating count (confidence factor).
5. **Model**:
   - Trained and evaluated using Random Forest, XGBoost, and LightGBM.
   - LightGBM performed the best and was serialized (`lgbm_model_optuna.pkl`).
6. **UI + Backend**:
   - React frontend with map + form for cuisine & price.
   - Flask backend processes Google API data and returns prediction.

---

## 🧰 Software Requirements

| Tool            | Version/Details                |
|-----------------|--------------------------------|
| Python          | 3.8+                           |
| Flask           | For backend API                |
| joblib          | For model serialization        |
| scikit-learn    | For scaling features           |
| LightGBM        | Final ML model                 |
| Google APIs     | Places, Roads, Routes          |
| React           | Frontend framework             |

---

## 🧪 Setup Instructions

### 🔧 Backend (Python)

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run backend:

   ```bash
   python server.py
   ```

---

### 🌍 Frontend (React)

1. Navigate to the frontend folder:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start frontend server:

   ```bash
   npm start
   ```

---

## ⚙️ Model Serialization

The final LightGBM model and `MinMaxScaler` are saved using `joblib` for fast loading during prediction:

```python
joblib.dump(model, 'lgbm_model_optuna.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

They are loaded in `model_predict.py` when predicting based on API input.

---

## 🤖 Future Improvements

- Add Explainable AI (e.g. SHAP) to explain prediction drivers.
- Dynamic competitor visualization on the map.
- Allow reverse search: “Suggest best location for X cuisine and budget”.

---

## 👨‍💻 Authors

- Rahul Varghese (U2103169)
- Rebecca Liz Punnoose (U2103171)
- Richu Kurian (U2103175)
- Rohan Joseph Arun (U2103180)

---

