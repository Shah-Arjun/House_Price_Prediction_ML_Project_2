# importing random forest regressor model

import joblib

model = joblib.load('model.pkl')

sample_house = [[
    3,        # number of bedrooms
    2,        # number of bathrooms
    1200,     # living area (sqft)
    3000,     # lot area (sqft)
    2,        # number of floors
    7,        # grade of the house (1–10 scale)
    4,        # condition of the house (1–5 scale)
    28.6139,  # Lattitude
    77.2090,  # Longitude
    110001,   # Postal Code
    15,       # Distance from the airport (km)
    4,        # Number of schools nearby
    2,        # number of views
    0,        # waterfront present (0 = No, 1 = Yes)
    1000,     # Area of the house(excluding basement)
    200,      # Area of the basement
    1100,     # living_area_renov
    3200      # lot_area_renov
]]

pricePreds = model.predict(sample_house)
print("Predicted price: ", pricePreds[0])