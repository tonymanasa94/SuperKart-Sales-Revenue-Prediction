
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify


# ---------------------------------------------------------
# Initialize Flask application
# ---------------------------------------------------------
super_kart_api = Flask("SuperKart Revenue Predictor")


# ---------------------------------------------------------
# Base directory
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_model_v1_0.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

model_name = type(model).__name__

print("Model loaded successfully:", model_name)


# ---------------------------------------------------------
# Load fitted preprocessor
# ---------------------------------------------------------

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "superkart_preprocessor.joblib"
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Preprocessor loaded successfully!")


# ---------------------------------------------------------
# Required original input features
# ---------------------------------------------------------
REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]


# ---------------------------------------------------------
# Helper function for preprocessing
# ---------------------------------------------------------
def preprocess_input(input_data):

    processed_data = preprocessor.transform(input_data)

    return processed_data


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------
@super_kart_api.get("/")
def home():

    return jsonify({
        "message": "Welcome to the SuperKart Sales Revenue Prediction API!",
        "model": model_name,
        "status": "running"
    })


# ---------------------------------------------------------
# Single prediction endpoint
# ---------------------------------------------------------
@super_kart_api.post("/v1/sales")
def predict_sales():

    try:

        product_data = request.get_json()

        if not product_data:
            return jsonify({
                "error": "No JSON data received."
            }), 400


        # Check for missing features
        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in product_data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing_features
            }), 400


        # Create input sample
        sample = {
            feature: product_data[feature]
            for feature in REQUIRED_FEATURES
        }


        # Convert to DataFrame
        input_data = pd.DataFrame([sample])


        # Preprocess
        processed_data = preprocess_input(input_data)


        # Make prediction
        prediction = model.predict(processed_data)[0]


        return jsonify({
            "Predicted_Product_Store_Sales_Total":
                float(prediction)
        })


    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ---------------------------------------------------------
# Batch prediction endpoint
# ---------------------------------------------------------
@super_kart_api.post("/v1/salesbatch")
def predict_sales_batch():

    try:

        # Check uploaded file
        if "file" not in request.files:

            return jsonify({
                "error": "Please upload a CSV file using the 'file' field."
            }), 400


        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No CSV file selected."
            }), 400


        # Read CSV
        input_data = pd.read_csv(file)


        # Check required columns
        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "CSV is missing required columns.",
                "missing_features": missing_features
            }), 400


        # Keep original data
        result_data = input_data.copy()


        # Use only expected input features
        model_input = input_data[
            REQUIRED_FEATURES
        ].copy()


        # Preprocess
        processed_data = preprocess_input(model_input)


        # Generate predictions
        predictions = model.predict(processed_data)


        # Add predictions
        result_data[
            "Predicted_Product_Store_Sales_Total"
        ] = predictions


        # Return records
        result = result_data.to_dict(
            orient="records"
        )

        return jsonify(result)


    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ---------------------------------------------------------
# Run Flask application
# ---------------------------------------------------------
if __name__ == "__main__":

    super_kart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
