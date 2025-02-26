from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

app = Flask(__name__)
CORS(app)

# Load models
sarima_model = joblib.load("sarima_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        data = request.json
        start_date = pd.to_datetime(data.get("start_date", "2024-03-01"))
        max_temp = data.get("Max(Temperature)", 25)
        avg_humidity = data.get("Avg(humidity)", 60)
        rolling_avg_7 = data.get("rolling_avg_7", 0)  # Added
        residual_lag1 = data.get("residual_lag1", 0)  # Added

        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=7, freq="D")

        # Prepare exogenous variables for SARIMA
        future_exog = pd.DataFrame({
            "Max(Temperature)": [max_temp] * 7,
            "Avg(humidity)": [avg_humidity] * 7
        })

        # SARIMA Prediction
        sarima_pred = sarima_model.forecast(steps=7, exog=future_exog)

        # Ensure XGBoost input includes all trained features
        xgb_input = pd.DataFrame({
            "lag1": sarima_pred,
            "lag7": [sarima_pred[0]] * 7,
            "Max(Temperature)": [max_temp] * 7,
            "Avg(humidity)": [avg_humidity] * 7,
            "rolling_avg_7": [rolling_avg_7] * 7,
            "residual_lag1": [residual_lag1] * 7
        })

        # XGBoost Prediction
        xgb_residuals = xgb_model.predict(xgb_input)

        # Final Predictions
        final_predictions = (sarima_pred + xgb_residuals).tolist()

        return jsonify({
            "start_date": str(start_date.date()),
            "predictions": final_predictions
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run

# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
#
# app = Flask(__name__)
#
# # ✅ Load SARIMA and XGBoost Models
# sarima_model = joblib.load("sarima_model.pkl")  # Trained SARIMA model
# xgb_model = joblib.load("xgboost_model.pkl")  # Trained XGBoost model
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#
#         # Validate model selection
#         model_type = data.get("model_type", "sarima")  # Default to SARIMA
#         if model_type not in ["sarima", "xgboost", "hybrid"]:
#             return jsonify({"error": "Invalid model_type. Choose 'sarima', 'xgboost', or 'hybrid'."}), 400
#
#         # Prepare input features (handle missing values)
#         input_data = {
#             "lag1": data.get("lag1", 0),
#             "lag7": data.get("lag7", 0),
#             "Max(Temperature)": data.get("Max(Temperature)", 25),
#             "Avg(humidity)": data.get("Avg(humidity)", 60),
#             "rolling_avg_7": data.get("rolling_avg_7", 0),
#             "residual_lag1": data.get("residual_lag1", 0)
#         }
#
#         df = pd.DataFrame([input_data])  # Convert input to DataFrame
#
#         # ✅ Prepare Exogenous Data for SARIMA (shape: 7x2)
#         exog_data = df[['Max(Temperature)', 'Avg(humidity)']].values
#         exog_data = np.tile(exog_data, (7, 1))  # Expand for 7-day forecast
#
#         # ✅ Make Predictions
#         if model_type == "sarima":
#             prediction = sarima_model.get_forecast(steps=7, exog=exog_data).predicted_mean.tolist()  # ✅ Convert Series to list
#
#         elif model_type == "xgboost":
#             # ✅ Generate Data for 7 Days
#             xgb_inputs = pd.concat([df] * 7, ignore_index=True)  # Repeat input row 7 times
#             prediction = xgb_model.predict(xgb_inputs).tolist()  # ✅ Predict 7 days
#
#         else:  # Hybrid Prediction (SARIMA + XGBoost)
#             # ✅ SARIMA Prediction
#             sarima_pred = sarima_model.get_forecast(steps=7, exog=exog_data).predicted_mean.tolist()
#
#             # ✅ XGBoost Prediction (Ensure 7 Inputs)
#             xgb_inputs = pd.concat([df] * 7, ignore_index=True)  # Repeat input row 7 times
#             xgb_pred = xgb_model.predict(xgb_inputs).tolist()
#
#             # ✅ Compute Hybrid Prediction (Average of SARIMA & XGBoost)
#             prediction = [(s + x) / 2 for s, x in zip(sarima_pred, xgb_pred)]
#
#         return jsonify({"model": model_type, "prediction": prediction})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True)

