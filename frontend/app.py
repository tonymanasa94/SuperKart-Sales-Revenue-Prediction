
import streamlit as st
import pandas as pd
import requests


# ---------------------------------------------------------
# Flask Backend URL
# ---------------------------------------------------------
BACKEND_URL = "http://backend:7860"


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="centered"
)


# ---------------------------------------------------------
# Application Title
# ---------------------------------------------------------
st.title("🛒 SuperKart Sales Revenue Prediction")

st.write(
    "Predict the total sales revenue of a product "
    "at a SuperKart store."
)


# =========================================================
# ONLINE / SINGLE PREDICTION
# =========================================================

st.subheader("Online Prediction")


# ---------------------------------------------------------
# Product Information
# ---------------------------------------------------------

st.markdown("### Product Information")


product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    step=0.1,
    value=10.0
)


product_sugar_content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "Regular",
        "No Sugar"
    ]
)


product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    value=0.10
)


product_type = st.selectbox(
    "Product Type",
    [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ]
)


product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    step=1.0,
    value=100.0
)


# ---------------------------------------------------------
# Store Information
# ---------------------------------------------------------

st.markdown("### Store Information")


store_id = st.text_input(
    "Store ID",
    value=""
)


store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1900,
    max_value=2100,
    step=1,
    value=2000
)


store_size = st.selectbox(
    "Store Size",
    [
        "Low",
        "Medium",
        "High"
    ]
)


store_location_city_type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)


store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Supermarket Type 1",
        "Supermarket Type 2",
        "Food Mart"
    ]
)


# ---------------------------------------------------------
# Create Input Dictionary
# ---------------------------------------------------------

input_data = {
    "Product_Weight": product_weight,

    "Product_Sugar_Content":
        product_sugar_content,

    "Product_Allocated_Area":
        product_allocated_area,

    "Product_Type":
        product_type,

    "Product_MRP":
        product_mrp,

    "Store_Id":
        store_id,

    "Store_Establishment_Year":
        store_establishment_year,

    "Store_Size":
        store_size,

    "Store_Location_City_Type":
        store_location_city_type,

    "Store_Type":
        store_type
}


# ---------------------------------------------------------
# Display Input Data
# ---------------------------------------------------------

with st.expander("View Input Data"):

    st.dataframe(
        pd.DataFrame([input_data]),
        use_container_width=True
    )


# ---------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------

if st.button(
    "Predict Sales Revenue",
    type="primary",
    use_container_width=True
):

    if store_id.strip() == "":

        st.warning(
            "Please enter a Store ID."
        )

    else:

        try:

            response = requests.post(
                f"{BACKEND_URL}/v1/sales",
                json=input_data,
                timeout=30
            )


            if response.status_code == 200:

                result = response.json()

                prediction = result[
                    "Predicted_Product_Store_Sales_Total"
                ]

                st.success(
                    "Prediction completed successfully!"
                )

                st.metric(
                    label="Predicted Product Store Sales Total",
                    value=f"${prediction:,.2f}"
                )


            else:

                try:
                    error_message = response.json()
                except Exception:
                    error_message = response.text

                st.error(
                    f"Prediction failed: {error_message}"
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the SuperKart "
                "prediction API."
            )


        except requests.exceptions.Timeout:

            st.error(
                "The prediction request timed out."
            )


        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )


# =========================================================
# BATCH PREDICTION
# =========================================================

st.divider()

st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing multiple "
    "SuperKart product/store records."
)


# ---------------------------------------------------------
# Expected CSV Columns
# ---------------------------------------------------------

with st.expander("Expected CSV Columns"):

    st.code(
        """Product_Weight
Product_Sugar_Content
Product_Allocated_Area
Product_Type
Product_MRP
Store_Id
Store_Establishment_Year
Store_Size
Store_Location_City_Type
Store_Type"""
    )


# ---------------------------------------------------------
# CSV Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Preview uploaded file
        preview_data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview")

        st.dataframe(
            preview_data.head(),
            use_container_width=True
        )

        st.write(
            f"Rows: {preview_data.shape[0]} | "
            f"Columns: {preview_data.shape[1]}"
        )


        # Reset pointer before sending file
        uploaded_file.seek(0)


        if st.button(
            "Predict Batch",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating batch predictions..."
            ):

                response = requests.post(
                    f"{BACKEND_URL}/v1/salesbatch",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv"
                        )
                    },
                    timeout=120
                )


            if response.status_code == 200:

                predictions = response.json()

                prediction_df = pd.DataFrame(
                    predictions
                )

                st.success(
                    "Batch predictions completed successfully!"
                )

                st.write("### Prediction Results")

                st.dataframe(
                    prediction_df,
                    use_container_width=True
                )


                # Convert results to CSV
                csv = prediction_df.to_csv(
                    index=False
                ).encode("utf-8")


                # Download prediction results
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=
                    "superkart_sales_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


            else:

                try:
                    error_message = response.json()
                except Exception:
                    error_message = response.text

                st.error(
                    f"Batch prediction failed: "
                    f"{error_message}"
                )


    except Exception as error:

        st.error(
            f"Unable to process the uploaded file: "
            f"{error}"
        )
