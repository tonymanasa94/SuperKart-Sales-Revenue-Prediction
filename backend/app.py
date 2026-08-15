
import joblib
import pandas as pd
from flask import Flask, request, jsonify


# ---------------------------------------------------------
# Initialize Flask application
# ---------------------------------------------------------
super_kart_api = Flask("SuperKart Revenue Predictor")


# ---------------------------------------------------------
# Load trained model artifact
# ---------------------------------------------------------
model_artifact = joblib.load("superkart_model_v1_0.joblib")

model = model_artifact["model"]
feature_columns = model_artifact["feature_columns"]
model_name = model_artifact["model_name"]


# ---------------------------------------------------------
# Helper function for preprocessing
# ---------------------------------------------------------
def preprocess_input(input_data):

    # Convert categorical variables into dummy variables
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Make sure input columns match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert everything to numeric float
    input_data = input_data.astype(float)

    return input_data


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------
@super_kart_api.get("/")
def home():

    return jsonify({
        "message": "Welcome to the SuperKart Sales Revenue Prediction API!",
        "model": model_name
    })


# ---------------------------------------------------------
# Single prediction endpoint
# ---------------------------------------------------------
@super_kart_api.post("/v1/sales")
def predict_sales():

    try:

        # Read JSON request
        product_data = request.get_json()

        # Extract SuperKart features
        sample = {

            "Product_Weight":
                product_data["Product_Weight"],

            "Product_Sugar_Content":
                product_data["Product_Sugar_Content"],

            "Product_Allocated_Area":
                product_data["Product_Allocated_Area"],

            "Product_Type":
                product_data["Product_Type"],

            "Product_MRP":
                product_data["Product_MRP"],

            "Store_Id":
                product_data["Store_Id"],

            "Store_Establishment_Year":
                product_data["Store_Establishment_Year"],

            "Store_Size":
                product_data["Store_Size"],

            "Store_Location_City_Type":
                product_data["Store_Location_City_Type"],

            "Store_Type":
                product_data["Store_Type"]
        }


        # Convert sample into DataFrame
        input_data = pd.DataFrame([sample])


        # Apply same preprocessing used during training
        processed_data = preprocess_input(input_data)


        # Make prediction
        prediction = model.predict(processed_data)[0]


        # Return result
        return jsonify({
            "Predicted_Product_Store_Sales_Total":
                float(prediction)
        })


    except KeyError as error:

        return jsonify({
            "error": f"Missing required feature: {str(error)}"
        }), 400


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

        # Check if a file was provided
        if "file" not in request.files:

            return jsonify({
                "error": "Please upload a CSV file using the 'file' field."
            }), 400


        # Get uploaded CSV
        file = request.files["file"]


        # Read CSV
        input_data = pd.read_csv(file)


        # Keep original data for final response
        result_data = input_data.copy()


        # Apply preprocessing
        processed_data = preprocess_input(input_data)


        # Generate predictions
        predictions = model.predict(processed_data)


        # Add predictions to original DataFrame
        result_data[
            "Predicted_Product_Store_Sales_Total"
        ] = predictions


        # Convert DataFrame to dictionary
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
        port=5000,
        debug=True
    )
