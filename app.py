# # importing random forest regressor model
import joblib
import streamlit as st
import pandas as pd



# ================================
# Load trained models
# ===================================
dt_model = joblib.load('dt_model.pkl')
lr_model = joblib.load('lr_model.pkl')
rfr_model = joblib.load('rfr_model.pkl')


# sample_house = [[
#     3,        # number of bedrooms
#     2,        # number of bathrooms
#     1200,     # living area (sqft)
#     3000,     # lot area (sqft)
#     2,        # number of floors
#     7,        # grade of the house (1–10 scale)
#     4,        # condition of the house (1–5 scale)
#     28.6139,  # Lattitude
#     77.2090,  # Longitude
#     110001,   # Postal Code
#     15,       # Distance from the airport (km)
#     4,        # Number of schools nearby
#     2,        # number of views
#     0,        # waterfront present (0 = No, 1 = Yes)
#     1000,     # Area of the house(excluding basement)
#     200,      # Area of the basement
#     1100,     # living_area_renov
#     3200      # lot_area_renov
# ]]

# pricePreds = model.predict(sample_house)




# ============================
# App
# ============================
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("House Price Prediction App")
st.write("This app uses machine learning to predicts the house price. For this you have to enter the following details of house and click on Predict button.")
st.divider()




# Input fields
# --------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0, format="%.3f")
    living_area = st.number_input("Living Area (sqft)", min_value=0.0, value=1200.0, format="%.3f")
    lot_area = st.number_input("Lot Area (sqft)", min_value=0, value=3000)
    floors = st.number_input("Number of Floors", min_value=0.0, value=2.0, format="%.3f")
    grade = st.slider("Grade of the House (1–10)", 1, 10, 7)

with col2:
    condition = st.slider("Condition of the House (1–5)", 1, 5, 4)
    latitude = st.number_input("Latitude", value=28.6139, format="%.6f")
    longitude = st.number_input("Longitude", value=77.2090, format="%.6f")
    postal = st.number_input("Postal Code", value=110001)
    airport_dist = st.number_input("Distance from Airport (km)", min_value=0.0, value=15.0)
    schools = st.number_input("Schools Nearby", min_value=0, value=4)

with col3:
    views = st.number_input("Number of Views", min_value=0, value=2)
    waterfront = st.selectbox("Waterfront Present", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    area_no_basement = st.number_input("Area (Excluding Basement)", min_value=0, value=1000)
    basement_area = st.number_input("Basement Area", min_value=0, value=200)
    living_renov = st.number_input("Living Area Renovated", min_value=0, value=1100)
    lot_renov = st.number_input("Lot Area Renovated", min_value=0, value=3200)



# -------------------------------
# Prepare input data
# -------------------------------
input_data = [[
    bedrooms, bathrooms, living_area, lot_area, floors,
    grade, condition, latitude, longitude, postal,
    airport_dist, schools, views, waterfront,
    area_no_basement, basement_area, living_renov, lot_renov
]]

st.markdown("---")
st.subheader("Predict Using Model")



# -------------------------------
# Buttons
# -------------------------------
btn1, btn2, btn3 = st.columns(3)

with btn1:
    predict_tree = st.button("Predict uding Decision Tree(R2=0.83)")

with btn2:
    predict_linear = st.button("Predict using Linear Regression(R2=0.68)")

with btn3:
    predict_rf = st.button("Predict using Random Forest(R2=0.87)")

st.markdown("---")




# -------------------------------
# Predictions Output
# -------------------------------
cl1, cl2, cl3 = st.columns(3)

with cl1:
    if predict_tree:
        pred_dt = dt_model.predict(input_data)[0]
        st.success(f"Decision Tree Price: ₹ {pred_dt:,.2f}")

with cl2:
    if predict_linear:
        pred_lr = lr_model.predict(input_data).item()
        st.success(f"Linear Regression Price: ₹ {pred_lr:,.2f}")

with cl3:
    if predict_rf:
        pred_rf = rfr_model.predict(input_data)[0]
        st.success(f"Random Forest Price: ₹ {pred_rf:,.2f}")
